use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::types::*;
use cranelift_codegen::ir::{AbiParam, Function, InstBuilder, MemFlags, Signature, UserFuncName};
use cranelift_codegen::isa::CallConv;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};

use crate::jit::{CompiledKernel, JitBackend, TileKernelFn};
use crate::kernel_ir::*;

/// Decomposed vector values. Backends store vec vars as 2-3 scalar f64 values.
enum VarValues {
    Scalar(cranelift_codegen::ir::Value),
    Vec2(cranelift_codegen::ir::Value, cranelift_codegen::ir::Value),
    Vec3(cranelift_codegen::ir::Value, cranelift_codegen::ir::Value, cranelift_codegen::ir::Value),
}

fn get_scalar(val_map: &std::collections::HashMap<Var, VarValues>, var: &Var) -> cranelift_codegen::ir::Value {
    match &val_map[var] {
        VarValues::Scalar(v) => *v,
        _ => panic!("expected scalar value for {:?}", var),
    }
}

fn get_vec2(val_map: &std::collections::HashMap<Var, VarValues>, var: &Var) -> (cranelift_codegen::ir::Value, cranelift_codegen::ir::Value) {
    match &val_map[var] {
        VarValues::Vec2(x, y) => (*x, *y),
        _ => panic!("expected vec2 value for {:?}", var),
    }
}

fn get_vec3(val_map: &std::collections::HashMap<Var, VarValues>, var: &Var) -> (cranelift_codegen::ir::Value, cranelift_codegen::ir::Value, cranelift_codegen::ir::Value) {
    match &val_map[var] {
        VarValues::Vec3(x, y, z) => (*x, *y, *z),
        _ => panic!("expected vec3 value for {:?}", var),
    }
}

pub struct CraneliftBackend;

struct CraneliftKernel {
    _module: JITModule,
    fn_ptr: TileKernelFn,
}

unsafe impl Send for CraneliftKernel {}
unsafe impl Sync for CraneliftKernel {}

impl CompiledKernel for CraneliftKernel {
    fn function_ptr(&self) -> TileKernelFn {
        self.fn_ptr
    }
}

impl JitBackend for CraneliftBackend {
    fn compile(&self, kernel: &Kernel) -> Box<dyn CompiledKernel> {
        Box::new(compile_kernel(kernel))
    }
}


fn compile_kernel(kernel: &Kernel) -> CraneliftKernel {
    let mut flag_builder = settings::builder();
    flag_builder.set("opt_level", "speed").unwrap();
    let isa_builder = cranelift_codegen::isa::lookup_by_name(
        &target_lexicon::Triple::host().to_string(),
    )
    .unwrap();
    let isa = isa_builder
        .finish(settings::Flags::new(flag_builder))
        .unwrap();

    let mut module =
        JITModule::new(JITBuilder::with_isa(isa, cranelift_module::default_libcall_names()));

    let mut sig = Signature::new(CallConv::SystemV);
    sig.params.push(AbiParam::new(I64)); // output: *mut u32
    sig.params.push(AbiParam::new(I32)); // width: u32
    sig.params.push(AbiParam::new(I32)); // height: u32
    sig.params.push(AbiParam::new(F64)); // x_min: f64
    sig.params.push(AbiParam::new(F64)); // y_min: f64
    sig.params.push(AbiParam::new(F64)); // x_step: f64
    sig.params.push(AbiParam::new(F64)); // y_step: f64
    sig.params.push(AbiParam::new(I32)); // row_start: u32
    sig.params.push(AbiParam::new(I32)); // row_end: u32
    sig.params.push(AbiParam::new(I32)); // sample_index: u32
    sig.params.push(AbiParam::new(F64)); // time: f64

    let func_id = module
        .declare_function(&kernel.name, Linkage::Export, &sig)
        .unwrap();

    let mut func = Function::with_name_signature(UserFuncName::default(), sig.clone());
    let mut fb_ctx = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut func, &mut fb_ctx);

    // -- Tile loop variables --
    let v_output = builder.declare_var(I64);
    let v_width = builder.declare_var(I32);
    let v_x_min = builder.declare_var(F64);
    let v_y_min = builder.declare_var(F64);
    let v_x_step = builder.declare_var(F64);
    let v_y_step = builder.declare_var(F64);
    let v_row_start = builder.declare_var(I32);
    let v_row_end = builder.declare_var(I32);
    let v_sample_index = builder.declare_var(I32);
    let v_time = builder.declare_var(F64);
    let v_row = builder.declare_var(I32);
    let v_col = builder.declare_var(I32);

    // -- Blocks --
    let entry_block = builder.create_block();
    let outer_check = builder.create_block();
    let col_init = builder.create_block();
    let inner_check = builder.create_block();
    let body_block = builder.create_block();
    let inner_inc = builder.create_block();
    let outer_inc = builder.create_block();
    let exit_block = builder.create_block();

    // -- Entry: read params, init row --
    builder.append_block_params_for_function_params(entry_block);
    builder.switch_to_block(entry_block);
    builder.seal_block(entry_block);

    let params: Vec<_> = builder.block_params(entry_block).to_vec();
    builder.def_var(v_output, params[0]);
    builder.def_var(v_width, params[1]);
    builder.def_var(v_x_min, params[3]);
    builder.def_var(v_y_min, params[4]);
    builder.def_var(v_x_step, params[5]);
    builder.def_var(v_y_step, params[6]);
    builder.def_var(v_row_start, params[7]);
    builder.def_var(v_row_end, params[8]);
    builder.def_var(v_sample_index, params[9]);
    builder.def_var(v_time, params[10]);

    let row_start_val = builder.use_var(v_row_start);
    builder.def_var(v_row, row_start_val);
    builder.ins().jump(outer_check, &[]);

    // -- Outer check: row < row_end --
    builder.switch_to_block(outer_check);
    let row = builder.use_var(v_row);
    let row_end = builder.use_var(v_row_end);
    let done = builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, row, row_end);
    builder.ins().brif(done, exit_block, &[], col_init, &[]);

    // -- Col init: col = 0 --
    builder.switch_to_block(col_init);
    builder.seal_block(col_init);
    let zero_i32 = builder.ins().iconst(I32, 0);
    builder.def_var(v_col, zero_i32);
    builder.ins().jump(inner_check, &[]);

    // -- Inner check: col < width --
    builder.switch_to_block(inner_check);
    let col = builder.use_var(v_col);
    let width = builder.use_var(v_width);
    let col_done = builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, col, width);
    builder.ins().brif(col_done, outer_inc, &[], body_block, &[]);

    // -- Body: compute pixel coordinates, lower kernel body, store result --
    builder.switch_to_block(body_block);
    builder.seal_block(body_block);

    // cx = x_min + col * x_step
    let col = builder.use_var(v_col);
    let x_min = builder.use_var(v_x_min);
    let x_step = builder.use_var(v_x_step);
    let col_f64 = builder.ins().fcvt_from_sint(F64, col);
    let cx = builder.ins().fma(col_f64, x_step, x_min);

    // cy = y_min + row * y_step
    let row = builder.use_var(v_row);
    let y_min = builder.use_var(v_y_min);
    let y_step = builder.use_var(v_y_step);
    let row_f64 = builder.ins().fcvt_from_sint(F64, row);
    let cy = builder.ins().fma(row_f64, y_step, y_min);

    // Apply sub-pixel jitter when sample_index != 0xFFFFFFFF
    let sample_idx = builder.use_var(v_sample_index);
    let no_jitter_sentinel = builder.ins().iconst(I32, 0xFFFFFFFFu32 as i64);
    let skip_jitter = builder.ins().icmp(IntCC::Equal, sample_idx, no_jitter_sentinel);

    // Hash: h = col * 0x45d9f3b + row
    let col = builder.use_var(v_col);
    let row = builder.use_var(v_row);
    let hash_k = builder.ins().iconst(I32, 0x45d9f3bu32 as i64);
    let h = builder.ins().imul(col, hash_k);
    let h = builder.ins().iadd(h, row);
    // h = h * 0x45d9f3b + sample_index
    let h = builder.ins().imul(h, hash_k);
    let h = builder.ins().iadd(h, sample_idx);
    // h ^= h >> 16
    let sixteen = builder.ins().iconst(I32, 16);
    let h_shifted = builder.ins().ushr(h, sixteen);
    let h = builder.ins().bxor(h, h_shifted);
    // h *= 0x45d9f3b
    let h = builder.ins().imul(h, hash_k);
    // h ^= h >> 16
    let h_shifted = builder.ins().ushr(h, sixteen);
    let h = builder.ins().bxor(h, h_shifted);

    // Extract jitter: jx = (h & 0xFFFF) / 65536.0, jy = (h >> 16) / 65536.0
    let mask_16 = builder.ins().iconst(I32, 0xFFFF);
    let jx_bits = builder.ins().band(h, mask_16);
    let jy_bits = builder.ins().ushr(h, sixteen);
    let recip = builder.ins().f64const(1.0 / 65536.0);
    let jx = builder.ins().fcvt_from_uint(F64, jx_bits);
    let jx = builder.ins().fmul(jx, recip);
    let jy = builder.ins().fcvt_from_uint(F64, jy_bits);
    let jy = builder.ins().fmul(jy, recip);

    // Conditional: if skip_jitter, use 0.0; otherwise use jitter
    let zero_f64 = builder.ins().f64const(0.0);
    let jx = builder.ins().select(skip_jitter, zero_f64, jx);
    let jy = builder.ins().select(skip_jitter, zero_f64, jy);

    // Apply: cx += jx * x_step, cy += jy * y_step
    let x_step = builder.use_var(v_x_step);
    let y_step = builder.use_var(v_y_step);
    let cx = builder.ins().fma(jx, x_step, cx);
    let cy = builder.ins().fma(jy, y_step, cy);

    // Lower kernel body
    let col = builder.use_var(v_col);
    let row = builder.use_var(v_row);
    let sample_idx_for_kernel = builder.use_var(v_sample_index);
    let time = builder.use_var(v_time);
    let color = lower_kernel_body(&mut module, &mut builder, kernel, cx, cy, col, row, sample_idx_for_kernel, time);

    // Store: output[(row - row_start) * width + col] = color
    let row = builder.use_var(v_row);
    let row_start = builder.use_var(v_row_start);
    let width = builder.use_var(v_width);
    let col = builder.use_var(v_col);
    let row_off = builder.ins().isub(row, row_start);
    let row_off_wide = builder.ins().uextend(I64, row_off);
    let width_wide = builder.ins().uextend(I64, width);
    let col_wide = builder.ins().uextend(I64, col);
    let idx = builder.ins().imul(row_off_wide, width_wide);
    let idx = builder.ins().iadd(idx, col_wide);
    let four_bytes = builder.ins().iconst(I64, 4);
    let byte_off = builder.ins().imul(idx, four_bytes);
    let output = builder.use_var(v_output);
    let addr = builder.ins().iadd(output, byte_off);
    builder.ins().store(MemFlags::new(), color, addr, 0);

    builder.ins().jump(inner_inc, &[]);

    // -- Inner inc: col++ --
    builder.switch_to_block(inner_inc);
    builder.seal_block(inner_inc);
    let col = builder.use_var(v_col);
    let one = builder.ins().iconst(I32, 1);
    let col_next = builder.ins().iadd(col, one);
    builder.def_var(v_col, col_next);
    builder.ins().jump(inner_check, &[]);

    builder.seal_block(inner_check);

    // -- Outer inc: row++ --
    builder.switch_to_block(outer_inc);
    builder.seal_block(outer_inc);
    let row = builder.use_var(v_row);
    let one = builder.ins().iconst(I32, 1);
    let row_next = builder.ins().iadd(row, one);
    builder.def_var(v_row, row_next);
    builder.ins().jump(outer_check, &[]);

    builder.seal_block(outer_check);

    // -- Exit --
    builder.switch_to_block(exit_block);
    builder.seal_block(exit_block);
    builder.ins().return_(&[]);

    builder.finalize();

    let mut ctx = cranelift_codegen::Context::for_function(func);
    module.define_function(func_id, &mut ctx).unwrap();
    module.finalize_definitions().unwrap();

    let code_ptr = module.get_finalized_function(func_id);
    let fn_ptr: TileKernelFn = unsafe { std::mem::transmute(code_ptr) };

    CraneliftKernel {
        _module: module,
        fn_ptr,
    }
}

/// Lower kernel IR body into Cranelift instructions.
fn lower_kernel_body(
    module: &mut JITModule,
    builder: &mut FunctionBuilder,
    kernel: &Kernel,
    cx: cranelift_codegen::ir::Value,
    cy: cranelift_codegen::ir::Value,
    col: cranelift_codegen::ir::Value,
    row: cranelift_codegen::ir::Value,
    sample_index: cranelift_codegen::ir::Value,
    time: cranelift_codegen::ir::Value,
) -> cranelift_codegen::ir::Value {
    use std::collections::HashMap;

    let mut val_map: HashMap<Var, VarValues> = HashMap::new();
    for param in &kernel.params {
        let val = match param.name.as_str() {
            "x" => cx,
            "y" => cy,
            "px" => col,
            "py" => row,
            "sample_index" => sample_index,
            "time" => time,
            name => panic!("unknown kernel parameter name: '{name}'"),
        };
        val_map.insert(param.var, VarValues::Scalar(val));
    }

    lower_body_items(module, builder, kernel, &kernel.body, &mut val_map);

    get_scalar(&val_map, &kernel.emit)
}

fn lower_body_items(
    module: &mut JITModule,
    builder: &mut FunctionBuilder,
    kernel: &Kernel,
    body: &[BodyItem],
    val_map: &mut std::collections::HashMap<Var, VarValues>,
) {
    for item in body {
        match item {
            BodyItem::Stmt(stmt) => {
                let v = lower_inst(module, builder, kernel, &stmt.inst, &stmt.binding, val_map);
                val_map.insert(stmt.binding.var, v);
            }
            BodyItem::While(w) => {
                lower_while(module, builder, kernel, w, val_map);
            }
        }
    }
}

fn lower_while(
    module: &mut JITModule,
    builder: &mut FunctionBuilder,
    kernel: &Kernel,
    w: &While,
    val_map: &mut std::collections::HashMap<Var, VarValues>,
) {
    // Cranelift uses Variables for SSA construction (like mutable locals).
    // We declare a Variable for each carry var, def it with the initial value,
    // then update it each iteration.

    let loop_header = builder.create_block();
    let loop_body = builder.create_block();
    let loop_exit = builder.create_block();

    // Declare Variables for carry vars and initialize them.
    // Vec types expand to multiple Variables (one per component).
    let carry_vars: Vec<Vec<Variable>> = w.carry.iter().map(|cv| {
        match cv.binding.ty {
            ValType::Vec2 => {
                let vx = builder.declare_var(F64);
                let vy = builder.declare_var(F64);
                let (ix, iy) = get_vec2(val_map, &cv.init);
                builder.def_var(vx, ix);
                builder.def_var(vy, iy);
                vec![vx, vy]
            }
            ValType::Vec3 => {
                let vx = builder.declare_var(F64);
                let vy = builder.declare_var(F64);
                let vz = builder.declare_var(F64);
                let (ix, iy, iz) = get_vec3(val_map, &cv.init);
                builder.def_var(vx, ix);
                builder.def_var(vy, iy);
                builder.def_var(vz, iz);
                vec![vx, vy, vz]
            }
            _ => {
                let cl_ty = match cv.binding.ty {
                    ValType::F64 => F64,
                    ValType::U32 => I32,
                    ValType::Bool => I8,
                    _ => unreachable!(),
                };
                let cl_var = builder.declare_var(cl_ty);
                let init_val = get_scalar(val_map, &cv.init);
                builder.def_var(cl_var, init_val);
                vec![cl_var]
            }
        }
    }).collect();

    builder.ins().jump(loop_header, &[]);

    // -- Loop header: read carry vars, execute cond_body, branch --
    builder.switch_to_block(loop_header);

    // Map carry vars from Cranelift Variables
    for (i, cv) in w.carry.iter().enumerate() {
        match cv.binding.ty {
            ValType::Vec2 => {
                let x = builder.use_var(carry_vars[i][0]);
                let y = builder.use_var(carry_vars[i][1]);
                val_map.insert(cv.binding.var, VarValues::Vec2(x, y));
            }
            ValType::Vec3 => {
                let x = builder.use_var(carry_vars[i][0]);
                let y = builder.use_var(carry_vars[i][1]);
                let z = builder.use_var(carry_vars[i][2]);
                val_map.insert(cv.binding.var, VarValues::Vec3(x, y, z));
            }
            _ => {
                let val = builder.use_var(carry_vars[i][0]);
                val_map.insert(cv.binding.var, VarValues::Scalar(val));
            }
        }
    }

    // Lower cond_body
    lower_body_items(module, builder, kernel, &w.cond_body, val_map);

    // Branch on cond
    let cond_val = get_scalar(val_map, &w.cond);
    builder.ins().brif(cond_val, loop_body, &[], loop_exit, &[]);

    // -- Loop body --
    builder.switch_to_block(loop_body);
    builder.seal_block(loop_body);

    lower_body_items(module, builder, kernel, &w.body, val_map);

    // Update carry Variables with yield values
    for (i, yv) in w.yields.iter().enumerate() {
        match &val_map[yv] {
            VarValues::Scalar(v) => {
                builder.def_var(carry_vars[i][0], *v);
            }
            VarValues::Vec2(x, y) => {
                builder.def_var(carry_vars[i][0], *x);
                builder.def_var(carry_vars[i][1], *y);
            }
            VarValues::Vec3(x, y, z) => {
                builder.def_var(carry_vars[i][0], *x);
                builder.def_var(carry_vars[i][1], *y);
                builder.def_var(carry_vars[i][2], *z);
            }
        }
    }

    builder.ins().jump(loop_header, &[]);

    // Seal blocks now that all predecessors are known
    builder.seal_block(loop_header);
    builder.seal_block(loop_exit);

    // Continue after loop
    builder.switch_to_block(loop_exit);

    // Re-read carry vars for use after the loop
    for (i, cv) in w.carry.iter().enumerate() {
        match cv.binding.ty {
            ValType::Vec2 => {
                let x = builder.use_var(carry_vars[i][0]);
                let y = builder.use_var(carry_vars[i][1]);
                val_map.insert(cv.binding.var, VarValues::Vec2(x, y));
            }
            ValType::Vec3 => {
                let x = builder.use_var(carry_vars[i][0]);
                let y = builder.use_var(carry_vars[i][1]);
                let z = builder.use_var(carry_vars[i][2]);
                val_map.insert(cv.binding.var, VarValues::Vec3(x, y, z));
            }
            _ => {
                let val = builder.use_var(carry_vars[i][0]);
                val_map.insert(cv.binding.var, VarValues::Scalar(val));
            }
        }
    }
}

fn call_libm_f64_unary(
    module: &mut JITModule,
    builder: &mut FunctionBuilder,
    name: &str,
    arg: cranelift_codegen::ir::Value,
) -> cranelift_codegen::ir::Value {
    let mut sig = Signature::new(CallConv::SystemV);
    sig.params.push(AbiParam::new(F64));
    sig.returns.push(AbiParam::new(F64));
    let func_id = module.declare_function(name, Linkage::Import, &sig).unwrap();
    let func_ref = module.declare_func_in_func(func_id, builder.func);
    let call = builder.ins().call(func_ref, &[arg]);
    builder.inst_results(call)[0]
}

fn call_libm_f64_binary(
    module: &mut JITModule,
    builder: &mut FunctionBuilder,
    name: &str,
    lhs: cranelift_codegen::ir::Value,
    rhs: cranelift_codegen::ir::Value,
) -> cranelift_codegen::ir::Value {
    let mut sig = Signature::new(CallConv::SystemV);
    sig.params.push(AbiParam::new(F64));
    sig.params.push(AbiParam::new(F64));
    sig.returns.push(AbiParam::new(F64));
    let func_id = module.declare_function(name, Linkage::Import, &sig).unwrap();
    let func_ref = module.declare_func_in_func(func_id, builder.func);
    let call = builder.ins().call(func_ref, &[lhs, rhs]);
    builder.inst_results(call)[0]
}

fn lower_inst(
    module: &mut JITModule,
    builder: &mut FunctionBuilder,
    kernel: &Kernel,
    inst: &Inst,
    binding: &Binding,
    val_map: &std::collections::HashMap<Var, VarValues>,
) -> VarValues {
    match inst {
        Inst::Const(c) => VarValues::Scalar(match c {
            Const::F64(v) => builder.ins().f64const(*v),
            Const::U32(v) => builder.ins().iconst(I32, *v as i64),
            Const::Bool(v) => builder.ins().iconst(I8, if *v { 1 } else { 0 }),
        }),
        Inst::Binary { op, lhs, rhs } => {
            let l = get_scalar(val_map, lhs);
            let r = get_scalar(val_map, rhs);
            VarValues::Scalar(match (op, binding.ty) {
                (BinOp::Add, ValType::F64) => builder.ins().fadd(l, r),
                (BinOp::Sub, ValType::F64) => builder.ins().fsub(l, r),
                (BinOp::Mul, ValType::F64) => builder.ins().fmul(l, r),
                (BinOp::Div, ValType::F64) => builder.ins().fdiv(l, r),
                (BinOp::Add, ValType::U32) => builder.ins().iadd(l, r),
                (BinOp::Sub, ValType::U32) => builder.ins().isub(l, r),
                (BinOp::Mul, ValType::U32) => builder.ins().imul(l, r),
                (BinOp::Div, ValType::U32) => builder.ins().udiv(l, r),
                (BinOp::Rem, ValType::F64) => {
                    let q = builder.ins().fdiv(l, r);
                    let q_floor = builder.ins().floor(q);
                    let prod = builder.ins().fmul(q_floor, r);
                    builder.ins().fsub(l, prod)
                }
                (BinOp::Rem, ValType::U32) => builder.ins().urem(l, r),
                (BinOp::BitAnd, _) => builder.ins().band(l, r),
                (BinOp::BitOr, _) => builder.ins().bor(l, r),
                (BinOp::BitXor, _) => builder.ins().bxor(l, r),
                (BinOp::Shl, _) => builder.ins().ishl(l, r),
                (BinOp::Shr, _) => builder.ins().ushr(l, r),
                (BinOp::And, _) => builder.ins().band(l, r),
                (BinOp::Or, _) => builder.ins().bor(l, r),
                (BinOp::Min, ValType::F64) => builder.ins().fmin(l, r),
                (BinOp::Max, ValType::F64) => builder.ins().fmax(l, r),
                (BinOp::Min, ValType::U32) => builder.ins().umin(l, r),
                (BinOp::Max, ValType::U32) => builder.ins().umax(l, r),
                (BinOp::Atan2, ValType::F64) => call_libm_f64_binary(module, builder, "atan2", l, r),
                (BinOp::Pow, ValType::F64) => call_libm_f64_binary(module, builder, "pow", l, r),
                (BinOp::Hash, ValType::U32) => {
                    // h = a * 0x45d9f3b + b
                    let hash_k = builder.ins().iconst(I32, 0x45d9f3bu32 as i64);
                    let h = builder.ins().imul(l, hash_k);
                    let h = builder.ins().iadd(h, r);
                    // h ^= h >> 16
                    let sixteen = builder.ins().iconst(I32, 16);
                    let h_shifted = builder.ins().ushr(h, sixteen);
                    let h = builder.ins().bxor(h, h_shifted);
                    // h *= 0x45d9f3b
                    let h = builder.ins().imul(h, hash_k);
                    // h ^= h >> 16
                    let h_shifted = builder.ins().ushr(h, sixteen);
                    builder.ins().bxor(h, h_shifted)
                }
                _ => unreachable!("invalid binary op/type combination"),
            })
        }
        Inst::Unary { op, arg } => {
            let a = get_scalar(val_map, arg);
            VarValues::Scalar(match (op, binding.ty) {
                (UnaryOp::Neg, ValType::F64) => builder.ins().fneg(a),
                (UnaryOp::Neg, ValType::U32) => {
                    let zero = builder.ins().iconst(I32, 0);
                    builder.ins().isub(zero, a)
                }
                (UnaryOp::Not, _) => {
                    let one = builder.ins().iconst(I8, 1);
                    builder.ins().bxor(a, one)
                }
                (UnaryOp::Abs, ValType::F64) => builder.ins().fabs(a),
                (UnaryOp::Abs, ValType::U32) => a,
                (UnaryOp::Sqrt, _) => builder.ins().sqrt(a),
                (UnaryOp::Floor, _) => builder.ins().floor(a),
                (UnaryOp::Ceil, _) => builder.ins().ceil(a),
                (UnaryOp::Sin, _) => call_libm_f64_unary(module, builder, "sin", a),
                (UnaryOp::Cos, _) => call_libm_f64_unary(module, builder, "cos", a),
                (UnaryOp::Tan, _) => call_libm_f64_unary(module, builder, "tan", a),
                (UnaryOp::Asin, _) => call_libm_f64_unary(module, builder, "asin", a),
                (UnaryOp::Acos, _) => call_libm_f64_unary(module, builder, "acos", a),
                (UnaryOp::Atan, _) => call_libm_f64_unary(module, builder, "atan", a),
                (UnaryOp::Exp, _) => call_libm_f64_unary(module, builder, "exp", a),
                (UnaryOp::Exp2, _) => call_libm_f64_unary(module, builder, "exp2", a),
                (UnaryOp::Log, _) => call_libm_f64_unary(module, builder, "log", a),
                (UnaryOp::Log2, _) => call_libm_f64_unary(module, builder, "log2", a),
                (UnaryOp::Log10, _) => call_libm_f64_unary(module, builder, "log10", a),
                (UnaryOp::Round, _) => builder.ins().nearest(a),
                (UnaryOp::Trunc, _) => builder.ins().trunc(a),
                (UnaryOp::Fract, _) => {
                    let floored = builder.ins().floor(a);
                    builder.ins().fsub(a, floored)
                }
                _ => unreachable!("invalid unary op/type combination"),
            })
        }
        Inst::Cmp { op, lhs, rhs } => {
            let l = get_scalar(val_map, lhs);
            let r = get_scalar(val_map, rhs);
            let operand_ty = kernel.var_type(*lhs).unwrap();
            VarValues::Scalar(match operand_ty {
                ValType::F64 => {
                    let cc = match op {
                        CmpOp::Eq => FloatCC::Equal,
                        CmpOp::Ne => FloatCC::NotEqual,
                        CmpOp::Lt => FloatCC::LessThan,
                        CmpOp::Le => FloatCC::LessThanOrEqual,
                        CmpOp::Gt => FloatCC::GreaterThan,
                        CmpOp::Ge => FloatCC::GreaterThanOrEqual,
                    };
                    builder.ins().fcmp(cc, l, r)
                }
                ValType::U32 => {
                    let cc = match op {
                        CmpOp::Eq => IntCC::Equal,
                        CmpOp::Ne => IntCC::NotEqual,
                        CmpOp::Lt => IntCC::UnsignedLessThan,
                        CmpOp::Le => IntCC::UnsignedLessThanOrEqual,
                        CmpOp::Gt => IntCC::UnsignedGreaterThan,
                        CmpOp::Ge => IntCC::UnsignedGreaterThanOrEqual,
                    };
                    builder.ins().icmp(cc, l, r)
                }
                _ => unreachable!("cannot compare bools"),
            })
        }
        Inst::Conv { op, arg } => {
            let a = get_scalar(val_map, arg);
            VarValues::Scalar(match op {
                ConvOp::F64ToU32 => builder.ins().fcvt_to_sint(I32, a),
                ConvOp::U32ToF64 => builder.ins().fcvt_from_uint(F64, a),
                ConvOp::U32ToF64Norm => {
                    let f = builder.ins().fcvt_from_uint(F64, a);
                    let recip = builder.ins().f64const(1.0 / 4294967296.0);
                    builder.ins().fmul(f, recip)
                }
            })
        }
        Inst::Select { cond, then_val, else_val } => {
            let c = get_scalar(val_map, cond);
            match (&val_map[then_val], &val_map[else_val]) {
                (VarValues::Scalar(t), VarValues::Scalar(e)) => {
                    VarValues::Scalar(builder.ins().select(c, *t, *e))
                }
                (VarValues::Vec2(tx, ty), VarValues::Vec2(ex, ey)) => {
                    VarValues::Vec2(
                        builder.ins().select(c, *tx, *ex),
                        builder.ins().select(c, *ty, *ey),
                    )
                }
                (VarValues::Vec3(tx, ty, tz), VarValues::Vec3(ex, ey, ez)) => {
                    VarValues::Vec3(
                        builder.ins().select(c, *tx, *ex),
                        builder.ins().select(c, *ty, *ey),
                        builder.ins().select(c, *tz, *ez),
                    )
                }
                _ => panic!("select branches must have matching types"),
            }
        }
        Inst::PackArgb { r, g, b } => {
            let rv = get_scalar(val_map, r);
            let gv = get_scalar(val_map, g);
            let bv = get_scalar(val_map, b);
            let alpha = builder.ins().iconst(I32, 0xFF000000u32 as i64);
            let sixteen = builder.ins().iconst(I32, 16);
            let eight = builder.ins().iconst(I32, 8);
            let r_shifted = builder.ins().ishl(rv, sixteen);
            let g_shifted = builder.ins().ishl(gv, eight);
            let color = builder.ins().bor(alpha, r_shifted);
            let color = builder.ins().bor(color, g_shifted);
            VarValues::Scalar(builder.ins().bor(color, bv))
        }
        Inst::MakeVec2 { x, y } => {
            let xv = get_scalar(val_map, x);
            let yv = get_scalar(val_map, y);
            VarValues::Vec2(xv, yv)
        }
        Inst::MakeVec3 { x, y, z } => {
            let xv = get_scalar(val_map, x);
            let yv = get_scalar(val_map, y);
            let zv = get_scalar(val_map, z);
            VarValues::Vec3(xv, yv, zv)
        }
        Inst::VecExtract { vec, index } => {
            let val = match (&val_map[vec], index) {
                (VarValues::Vec2(x, _), 0) => *x,
                (VarValues::Vec2(_, y), 1) => *y,
                (VarValues::Vec3(x, _, _), 0) => *x,
                (VarValues::Vec3(_, y, _), 1) => *y,
                (VarValues::Vec3(_, _, z), 2) => *z,
                _ => panic!("invalid vec extract index"),
            };
            VarValues::Scalar(val)
        }
        Inst::VecBinary { op, lhs, rhs } => {
            match binding.ty {
                ValType::Vec2 => {
                    let (lx, ly) = get_vec2(val_map, lhs);
                    let (rx, ry) = get_vec2(val_map, rhs);
                    let apply = |builder: &mut FunctionBuilder, a, b| match op {
                        VecBinOp::Add => builder.ins().fadd(a, b),
                        VecBinOp::Sub => builder.ins().fsub(a, b),
                        VecBinOp::Mul => builder.ins().fmul(a, b),
                        VecBinOp::Div => builder.ins().fdiv(a, b),
                        VecBinOp::Min => builder.ins().fmin(a, b),
                        VecBinOp::Max => builder.ins().fmax(a, b),
                    };
                    VarValues::Vec2(apply(builder, lx, rx), apply(builder, ly, ry))
                }
                ValType::Vec3 => {
                    let (lx, ly, lz) = get_vec3(val_map, lhs);
                    let (rx, ry, rz) = get_vec3(val_map, rhs);
                    let apply = |builder: &mut FunctionBuilder, a, b| match op {
                        VecBinOp::Add => builder.ins().fadd(a, b),
                        VecBinOp::Sub => builder.ins().fsub(a, b),
                        VecBinOp::Mul => builder.ins().fmul(a, b),
                        VecBinOp::Div => builder.ins().fdiv(a, b),
                        VecBinOp::Min => builder.ins().fmin(a, b),
                        VecBinOp::Max => builder.ins().fmax(a, b),
                    };
                    VarValues::Vec3(apply(builder, lx, rx), apply(builder, ly, ry), apply(builder, lz, rz))
                }
                _ => unreachable!("VecBinary result must be vec type"),
            }
        }
        Inst::VecScale { scalar, vec } => {
            let s = get_scalar(val_map, scalar);
            match binding.ty {
                ValType::Vec2 => {
                    let (vx, vy) = get_vec2(val_map, vec);
                    VarValues::Vec2(builder.ins().fmul(s, vx), builder.ins().fmul(s, vy))
                }
                ValType::Vec3 => {
                    let (vx, vy, vz) = get_vec3(val_map, vec);
                    VarValues::Vec3(
                        builder.ins().fmul(s, vx),
                        builder.ins().fmul(s, vy),
                        builder.ins().fmul(s, vz),
                    )
                }
                _ => unreachable!("VecScale result must be vec type"),
            }
        }
        Inst::VecUnary { op: vec_op, arg } => {
            match binding.ty {
                ValType::Vec2 => {
                    let (ax, ay) = get_vec2(val_map, arg);
                    match vec_op {
                        VecUnaryOp::Neg => VarValues::Vec2(
                            builder.ins().fneg(ax),
                            builder.ins().fneg(ay),
                        ),
                        VecUnaryOp::Abs => VarValues::Vec2(
                            builder.ins().fabs(ax),
                            builder.ins().fabs(ay),
                        ),
                        VecUnaryOp::Normalize => {
                            let xx = builder.ins().fmul(ax, ax);
                            let yy = builder.ins().fmul(ay, ay);
                            let sum = builder.ins().fadd(xx, yy);
                            let len = builder.ins().sqrt(sum);
                            VarValues::Vec2(
                                builder.ins().fdiv(ax, len),
                                builder.ins().fdiv(ay, len),
                            )
                        }
                    }
                }
                ValType::Vec3 => {
                    let (ax, ay, az) = get_vec3(val_map, arg);
                    match vec_op {
                        VecUnaryOp::Neg => VarValues::Vec3(
                            builder.ins().fneg(ax),
                            builder.ins().fneg(ay),
                            builder.ins().fneg(az),
                        ),
                        VecUnaryOp::Abs => VarValues::Vec3(
                            builder.ins().fabs(ax),
                            builder.ins().fabs(ay),
                            builder.ins().fabs(az),
                        ),
                        VecUnaryOp::Normalize => {
                            let xx = builder.ins().fmul(ax, ax);
                            let yy = builder.ins().fmul(ay, ay);
                            let zz = builder.ins().fmul(az, az);
                            let sum = builder.ins().fadd(xx, yy);
                            let sum = builder.ins().fadd(sum, zz);
                            let len = builder.ins().sqrt(sum);
                            VarValues::Vec3(
                                builder.ins().fdiv(ax, len),
                                builder.ins().fdiv(ay, len),
                                builder.ins().fdiv(az, len),
                            )
                        }
                    }
                }
                _ => unreachable!("VecUnary result must be vec type"),
            }
        }
        Inst::VecDot { lhs, rhs } => {
            let operand_ty = kernel.var_type(*lhs).unwrap();
            VarValues::Scalar(match operand_ty {
                ValType::Vec2 => {
                    let (lx, ly) = get_vec2(val_map, lhs);
                    let (rx, ry) = get_vec2(val_map, rhs);
                    let xx = builder.ins().fmul(lx, rx);
                    let yy = builder.ins().fmul(ly, ry);
                    builder.ins().fadd(xx, yy)
                }
                ValType::Vec3 => {
                    let (lx, ly, lz) = get_vec3(val_map, lhs);
                    let (rx, ry, rz) = get_vec3(val_map, rhs);
                    let xx = builder.ins().fmul(lx, rx);
                    let yy = builder.ins().fmul(ly, ry);
                    let zz = builder.ins().fmul(lz, rz);
                    let sum = builder.ins().fadd(xx, yy);
                    builder.ins().fadd(sum, zz)
                }
                _ => unreachable!("VecDot operands must be vec type"),
            })
        }
        Inst::VecLength { arg } => {
            let operand_ty = kernel.var_type(*arg).unwrap();
            VarValues::Scalar(match operand_ty {
                ValType::Vec2 => {
                    let (ax, ay) = get_vec2(val_map, arg);
                    let xx = builder.ins().fmul(ax, ax);
                    let yy = builder.ins().fmul(ay, ay);
                    let sum = builder.ins().fadd(xx, yy);
                    builder.ins().sqrt(sum)
                }
                ValType::Vec3 => {
                    let (ax, ay, az) = get_vec3(val_map, arg);
                    let xx = builder.ins().fmul(ax, ax);
                    let yy = builder.ins().fmul(ay, ay);
                    let zz = builder.ins().fmul(az, az);
                    let sum = builder.ins().fadd(xx, yy);
                    let sum = builder.ins().fadd(sum, zz);
                    builder.ins().sqrt(sum)
                }
                _ => unreachable!("VecLength operand must be vec type"),
            })
        }
        Inst::VecCross { lhs, rhs } => {
            let (lx, ly, lz) = get_vec3(val_map, lhs);
            let (rx, ry, rz) = get_vec3(val_map, rhs);
            // cross = (ly*rz - lz*ry, lz*rx - lx*rz, lx*ry - ly*rx)
            let a = builder.ins().fmul(ly, rz);
            let b = builder.ins().fmul(lz, ry);
            let cx = builder.ins().fsub(a, b);
            let a = builder.ins().fmul(lz, rx);
            let b = builder.ins().fmul(lx, rz);
            let cy = builder.ins().fsub(a, b);
            let a = builder.ins().fmul(lx, ry);
            let b = builder.ins().fmul(ly, rx);
            let cz = builder.ins().fsub(a, b);
            VarValues::Vec3(cx, cy, cz)
        }
    }
}
