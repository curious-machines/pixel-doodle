use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::types::*;
use cranelift_codegen::ir::{AbiParam, Function, InstBuilder, MemFlags, Signature, UserFuncName};
use cranelift_codegen::isa::CallConv;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};

use crate::jit::{CompiledKernel, CompiledSimKernel, JitBackend, SimTileKernelFn, TileKernelFn, UserArgSlot};
use crate::kernel_ir::*;

/// Context for buffer operations in simulation kernels.
struct BufContext {
    v_width: Variable,
    v_buf_ptrs: Variable,
    v_buf_out_ptrs: Variable,
}

/// Decomposed vector values. Backends store vec vars as N scalar values (2, 3, or 4 components).
enum VarValues {
    Scalar(cranelift_codegen::ir::Value),
    Vec(Vec<cranelift_codegen::ir::Value>),  // 2, 3, or 4 components
}

fn get_scalar(val_map: &std::collections::HashMap<Var, VarValues>, var: &Var) -> cranelift_codegen::ir::Value {
    match &val_map[var] {
        VarValues::Scalar(v) => *v,
        _ => panic!("expected scalar value for {:?}", var),
    }
}

fn get_vec<'a>(val_map: &'a std::collections::HashMap<Var, VarValues>, var: &Var) -> &'a [cranelift_codegen::ir::Value] {
    match &val_map[var] {
        VarValues::Vec(v) => v,
        _ => panic!("expected vec value for {:?}", var),
    }
}

/// Map a kernel ScalarType to the corresponding Cranelift IR type.
fn scalar_to_cl(s: ScalarType) -> cranelift_codegen::ir::Type {
    match s {
        ScalarType::F32 => F32,
        ScalarType::F64 => F64,
        ScalarType::I8 | ScalarType::U8 | ScalarType::Bool => I8,
        ScalarType::I16 | ScalarType::U16 => I16,
        ScalarType::I32 | ScalarType::U32 => I32,
        ScalarType::I64 | ScalarType::U64 => I64,
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

struct CraneliftSimKernel {
    _module: JITModule,
    fn_ptr: SimTileKernelFn,
}

unsafe impl Send for CraneliftSimKernel {}
unsafe impl Sync for CraneliftSimKernel {}

impl CompiledSimKernel for CraneliftSimKernel {
    fn function_ptr(&self) -> SimTileKernelFn {
        self.fn_ptr
    }
}

impl JitBackend for CraneliftBackend {
    fn compile(&self, kernel: &Kernel, user_args: &[UserArgSlot]) -> Box<dyn CompiledKernel> {
        Box::new(compile_kernel(kernel, user_args))
    }
    fn compile_sim(&self, kernel: &Kernel, user_args: &[UserArgSlot]) -> Box<dyn CompiledSimKernel> {
        Box::new(compile_sim_kernel(kernel, user_args))
    }
}


fn compile_kernel(kernel: &Kernel, user_args: &[UserArgSlot]) -> CraneliftKernel {
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
    sig.params.push(AbiParam::new(I64)); // user_args: *const u8

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
    let v_user_args = builder.declare_var(I64);

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
    builder.def_var(v_user_args, params[11]);

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
    let user_args_ptr = builder.use_var(v_user_args);
    let color = lower_kernel_body(&mut module, &mut builder, kernel, cx, cy, col, row, sample_idx_for_kernel, time, user_args_ptr, user_args);

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
    user_args_ptr: cranelift_codegen::ir::Value,
    user_args: &[UserArgSlot],
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
            name => {
                let slot = user_args.iter().find(|s| s.name == name)
                    .unwrap_or_else(|| panic!("unknown kernel parameter: '{name}'"));
                let offset = builder.ins().iconst(I64, slot.offset as i64);
                let addr = builder.ins().iadd(user_args_ptr, offset);
                match slot.ty {
                    ValType::Scalar(s) => builder.ins().load(scalar_to_cl(s), MemFlags::trusted(), addr, 0),
                    _ => panic!("unsupported user-arg type {:?} for param '{name}'", slot.ty),
                }
            }
        };
        val_map.insert(param.var, VarValues::Scalar(val));
    }

    lower_body_items(module, builder, kernel, &kernel.body, &mut val_map, None);

    get_scalar(&val_map, &kernel.emit)
}

fn lower_body_items(
    module: &mut JITModule,
    builder: &mut FunctionBuilder,
    kernel: &Kernel,
    body: &[BodyItem],
    val_map: &mut std::collections::HashMap<Var, VarValues>,
    buf_ctx: Option<&BufContext>,
) {
    for item in body {
        match item {
            BodyItem::Stmt(stmt) => {
                let v = lower_inst(module, builder, kernel, &stmt.inst, &stmt.binding, val_map, buf_ctx);
                val_map.insert(stmt.binding.var, v);
            }
            BodyItem::While(w) => {
                lower_while(module, builder, kernel, w, val_map, buf_ctx);
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
    buf_ctx: Option<&BufContext>,
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
            ValType::Vec { len, elem } => {
                let cl_ty = match elem {
                    ScalarType::F64 => F64,
                    ScalarType::F32 => F32,
                    _ => panic!("unsupported vec element type in carry var"),
                };
                let init_components = get_vec(val_map, &cv.init);
                assert_eq!(init_components.len(), len as usize);
                init_components.iter().map(|iv| {
                    let v = builder.declare_var(cl_ty);
                    builder.def_var(v, *iv);
                    v
                }).collect()
            }
            _ => {
                let cl_ty = match cv.binding.ty {
                    ValType::Scalar(s) => scalar_to_cl(s),
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
        if carry_vars[i].len() > 1 {
            let components: Vec<_> = carry_vars[i].iter().map(|v| builder.use_var(*v)).collect();
            val_map.insert(cv.binding.var, VarValues::Vec(components));
        } else {
            let val = builder.use_var(carry_vars[i][0]);
            val_map.insert(cv.binding.var, VarValues::Scalar(val));
        }
    }

    // Lower cond_body
    lower_body_items(module, builder, kernel, &w.cond_body, val_map, buf_ctx);

    // Branch on cond
    let cond_val = get_scalar(val_map, &w.cond);
    builder.ins().brif(cond_val, loop_body, &[], loop_exit, &[]);

    // -- Loop body --
    builder.switch_to_block(loop_body);
    builder.seal_block(loop_body);

    lower_body_items(module, builder, kernel, &w.body, val_map, buf_ctx);

    // Update carry Variables with yield values
    for (i, yv) in w.yields.iter().enumerate() {
        match &val_map[yv] {
            VarValues::Scalar(v) => {
                builder.def_var(carry_vars[i][0], *v);
            }
            VarValues::Vec(components) => {
                for (j, c) in components.iter().enumerate() {
                    builder.def_var(carry_vars[i][j], *c);
                }
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
        if carry_vars[i].len() > 1 {
            let components: Vec<_> = carry_vars[i].iter().map(|v| builder.use_var(*v)).collect();
            val_map.insert(cv.binding.var, VarValues::Vec(components));
        } else {
            let val = builder.use_var(carry_vars[i][0]);
            val_map.insert(cv.binding.var, VarValues::Scalar(val));
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

fn call_libm_f32_unary(
    module: &mut JITModule,
    builder: &mut FunctionBuilder,
    name: &str,
    arg: cranelift_codegen::ir::Value,
) -> cranelift_codegen::ir::Value {
    let mut sig = Signature::new(CallConv::SystemV);
    sig.params.push(AbiParam::new(F32));
    sig.returns.push(AbiParam::new(F32));
    let func_id = module.declare_function(name, Linkage::Import, &sig).unwrap();
    let func_ref = module.declare_func_in_func(func_id, builder.func);
    let call = builder.ins().call(func_ref, &[arg]);
    builder.inst_results(call)[0]
}

fn call_libm_f32_binary(
    module: &mut JITModule,
    builder: &mut FunctionBuilder,
    name: &str,
    lhs: cranelift_codegen::ir::Value,
    rhs: cranelift_codegen::ir::Value,
) -> cranelift_codegen::ir::Value {
    let mut sig = Signature::new(CallConv::SystemV);
    sig.params.push(AbiParam::new(F32));
    sig.params.push(AbiParam::new(F32));
    sig.returns.push(AbiParam::new(F32));
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
    buf_ctx: Option<&BufContext>,
) -> VarValues {
    match inst {
        Inst::Const(c) => VarValues::Scalar(match c {
            Const::F32(v) => builder.ins().f32const(*v),
            Const::F64(v) => builder.ins().f64const(*v),
            Const::I8(v) => builder.ins().iconst(I8, *v as i64),
            Const::U8(v) => builder.ins().iconst(I8, *v as i64),
            Const::I16(v) => builder.ins().iconst(I16, *v as i64),
            Const::U16(v) => builder.ins().iconst(I16, *v as i64),
            Const::I32(v) => builder.ins().iconst(I32, *v as i64),
            Const::U32(v) => builder.ins().iconst(I32, *v as i64),
            Const::I64(v) => builder.ins().iconst(I64, *v as i64),
            Const::U64(v) => builder.ins().iconst(I64, *v as i64),
            Const::Bool(v) => builder.ins().iconst(I8, if *v { 1 } else { 0 }),
        }),
        Inst::Binary { op, lhs, rhs } => {
            let l = get_scalar(val_map, lhs);
            let r = get_scalar(val_map, rhs);
            VarValues::Scalar(match (op, binding.ty) {
                // Float arithmetic
                (BinOp::Add, ValType::Scalar(s)) if s.is_float() => builder.ins().fadd(l, r),
                (BinOp::Sub, ValType::Scalar(s)) if s.is_float() => builder.ins().fsub(l, r),
                (BinOp::Mul, ValType::Scalar(s)) if s.is_float() => builder.ins().fmul(l, r),
                (BinOp::Div, ValType::Scalar(s)) if s.is_float() => builder.ins().fdiv(l, r),
                (BinOp::Rem, ValType::Scalar(s)) if s.is_float() => {
                    let q = builder.ins().fdiv(l, r);
                    let q_floor = builder.ins().floor(q);
                    let prod = builder.ins().fmul(q_floor, r);
                    builder.ins().fsub(l, prod)
                }
                // Integer arithmetic
                (BinOp::Add, ValType::Scalar(s)) if s.is_integer() => builder.ins().iadd(l, r),
                (BinOp::Sub, ValType::Scalar(s)) if s.is_integer() => builder.ins().isub(l, r),
                (BinOp::Mul, ValType::Scalar(s)) if s.is_integer() => builder.ins().imul(l, r),
                (BinOp::Div, ValType::Scalar(s)) if s.is_unsigned() => builder.ins().udiv(l, r),
                (BinOp::Div, ValType::Scalar(s)) if s.is_signed() => builder.ins().sdiv(l, r),
                (BinOp::Rem, ValType::Scalar(s)) if s.is_unsigned() => builder.ins().urem(l, r),
                (BinOp::Rem, ValType::Scalar(s)) if s.is_signed() => builder.ins().srem(l, r),
                // Bitwise ops (all integer types)
                (BinOp::BitAnd, _) => builder.ins().band(l, r),
                (BinOp::BitOr, _) => builder.ins().bor(l, r),
                (BinOp::BitXor, _) => builder.ins().bxor(l, r),
                (BinOp::Shl, _) => builder.ins().ishl(l, r),
                (BinOp::Shr, ValType::Scalar(s)) if s.is_signed() => builder.ins().sshr(l, r),
                (BinOp::Shr, _) => builder.ins().ushr(l, r),
                // Logical ops
                (BinOp::And, _) => builder.ins().band(l, r),
                (BinOp::Or, _) => builder.ins().bor(l, r),
                // Min/Max
                (BinOp::Min, ValType::Scalar(s)) if s.is_float() => builder.ins().fmin(l, r),
                (BinOp::Max, ValType::Scalar(s)) if s.is_float() => builder.ins().fmax(l, r),
                (BinOp::Min, ValType::Scalar(s)) if s.is_unsigned() => builder.ins().umin(l, r),
                (BinOp::Max, ValType::Scalar(s)) if s.is_unsigned() => builder.ins().umax(l, r),
                (BinOp::Min, ValType::Scalar(s)) if s.is_signed() => builder.ins().smin(l, r),
                (BinOp::Max, ValType::Scalar(s)) if s.is_signed() => builder.ins().smax(l, r),
                // Float-only ops
                (BinOp::Atan2, ValType::Scalar(ScalarType::F64)) => call_libm_f64_binary(module, builder, "atan2", l, r),
                (BinOp::Pow, ValType::Scalar(ScalarType::F64)) => call_libm_f64_binary(module, builder, "pow", l, r),
                (BinOp::Atan2, ValType::Scalar(ScalarType::F32)) => call_libm_f32_binary(module, builder, "atan2f", l, r),
                (BinOp::Pow, ValType::Scalar(ScalarType::F32)) => call_libm_f32_binary(module, builder, "powf", l, r),
                (BinOp::Hash, ValType::Scalar(ScalarType::U32)) => {
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
                (UnaryOp::Neg, ValType::Scalar(s)) if s.is_float() => builder.ins().fneg(a),
                (UnaryOp::Neg, ValType::Scalar(s)) if s.is_signed() => builder.ins().ineg(a),
                (UnaryOp::Neg, ValType::Scalar(s)) if s.is_unsigned() => {
                    let zero = builder.ins().iconst(scalar_to_cl(s), 0);
                    builder.ins().isub(zero, a)
                }
                (UnaryOp::Not, _) => {
                    let one = builder.ins().iconst(I8, 1);
                    builder.ins().bxor(a, one)
                }
                (UnaryOp::Abs, ValType::Scalar(s)) if s.is_float() => builder.ins().fabs(a),
                (UnaryOp::Abs, ValType::Scalar(s)) if s.is_signed() => {
                    // abs(x) = select(x < 0, -x, x)
                    let cl_ty = scalar_to_cl(s);
                    let zero = builder.ins().iconst(cl_ty, 0);
                    let neg = builder.ins().ineg(a);
                    let is_neg = builder.ins().icmp(IntCC::SignedLessThan, a, zero);
                    builder.ins().select(is_neg, neg, a)
                }
                (UnaryOp::Abs, ValType::Scalar(s)) if s.is_unsigned() => a,
                (UnaryOp::Sqrt, ValType::Scalar(ScalarType::F64)) => builder.ins().sqrt(a),
                (UnaryOp::Sqrt, ValType::Scalar(ScalarType::F32)) => builder.ins().sqrt(a),
                (UnaryOp::Floor, ValType::Scalar(ScalarType::F64)) => builder.ins().floor(a),
                (UnaryOp::Floor, ValType::Scalar(ScalarType::F32)) => builder.ins().floor(a),
                (UnaryOp::Ceil, ValType::Scalar(ScalarType::F64)) => builder.ins().ceil(a),
                (UnaryOp::Ceil, ValType::Scalar(ScalarType::F32)) => builder.ins().ceil(a),
                (UnaryOp::Sin, ValType::Scalar(ScalarType::F64)) => call_libm_f64_unary(module, builder, "sin", a),
                (UnaryOp::Cos, ValType::Scalar(ScalarType::F64)) => call_libm_f64_unary(module, builder, "cos", a),
                (UnaryOp::Tan, ValType::Scalar(ScalarType::F64)) => call_libm_f64_unary(module, builder, "tan", a),
                (UnaryOp::Asin, ValType::Scalar(ScalarType::F64)) => call_libm_f64_unary(module, builder, "asin", a),
                (UnaryOp::Acos, ValType::Scalar(ScalarType::F64)) => call_libm_f64_unary(module, builder, "acos", a),
                (UnaryOp::Atan, ValType::Scalar(ScalarType::F64)) => call_libm_f64_unary(module, builder, "atan", a),
                (UnaryOp::Exp, ValType::Scalar(ScalarType::F64)) => call_libm_f64_unary(module, builder, "exp", a),
                (UnaryOp::Exp2, ValType::Scalar(ScalarType::F64)) => call_libm_f64_unary(module, builder, "exp2", a),
                (UnaryOp::Log, ValType::Scalar(ScalarType::F64)) => call_libm_f64_unary(module, builder, "log", a),
                (UnaryOp::Log2, ValType::Scalar(ScalarType::F64)) => call_libm_f64_unary(module, builder, "log2", a),
                (UnaryOp::Log10, ValType::Scalar(ScalarType::F64)) => call_libm_f64_unary(module, builder, "log10", a),
                (UnaryOp::Sin, ValType::Scalar(ScalarType::F32)) => call_libm_f32_unary(module, builder, "sinf", a),
                (UnaryOp::Cos, ValType::Scalar(ScalarType::F32)) => call_libm_f32_unary(module, builder, "cosf", a),
                (UnaryOp::Tan, ValType::Scalar(ScalarType::F32)) => call_libm_f32_unary(module, builder, "tanf", a),
                (UnaryOp::Asin, ValType::Scalar(ScalarType::F32)) => call_libm_f32_unary(module, builder, "asinf", a),
                (UnaryOp::Acos, ValType::Scalar(ScalarType::F32)) => call_libm_f32_unary(module, builder, "acosf", a),
                (UnaryOp::Atan, ValType::Scalar(ScalarType::F32)) => call_libm_f32_unary(module, builder, "atanf", a),
                (UnaryOp::Exp, ValType::Scalar(ScalarType::F32)) => call_libm_f32_unary(module, builder, "expf", a),
                (UnaryOp::Exp2, ValType::Scalar(ScalarType::F32)) => call_libm_f32_unary(module, builder, "exp2f", a),
                (UnaryOp::Log, ValType::Scalar(ScalarType::F32)) => call_libm_f32_unary(module, builder, "logf", a),
                (UnaryOp::Log2, ValType::Scalar(ScalarType::F32)) => call_libm_f32_unary(module, builder, "log2f", a),
                (UnaryOp::Log10, ValType::Scalar(ScalarType::F32)) => call_libm_f32_unary(module, builder, "log10f", a),
                (UnaryOp::Round, ValType::Scalar(ScalarType::F64)) => builder.ins().nearest(a),
                (UnaryOp::Round, ValType::Scalar(ScalarType::F32)) => builder.ins().nearest(a),
                (UnaryOp::Trunc, ValType::Scalar(ScalarType::F64)) => builder.ins().trunc(a),
                (UnaryOp::Trunc, ValType::Scalar(ScalarType::F32)) => builder.ins().trunc(a),
                (UnaryOp::Fract, ValType::Scalar(ScalarType::F64)) => {
                    let floored = builder.ins().floor(a);
                    builder.ins().fsub(a, floored)
                }
                (UnaryOp::Fract, ValType::Scalar(ScalarType::F32)) => {
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
                ValType::Scalar(s) if s.is_float() => {
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
                ValType::Scalar(s) if s.is_unsigned() => {
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
                ValType::Scalar(s) if s.is_signed() => {
                    let cc = match op {
                        CmpOp::Eq => IntCC::Equal,
                        CmpOp::Ne => IntCC::NotEqual,
                        CmpOp::Lt => IntCC::SignedLessThan,
                        CmpOp::Le => IntCC::SignedLessThanOrEqual,
                        CmpOp::Gt => IntCC::SignedGreaterThan,
                        CmpOp::Ge => IntCC::SignedGreaterThanOrEqual,
                    };
                    builder.ins().icmp(cc, l, r)
                }
                _ => unreachable!("cannot compare type {:?}", operand_ty),
            })
        }
        Inst::Conv { op, arg } => {
            let a = get_scalar(val_map, arg);
            let from = op.from;
            let to = op.to;
            let from_cl = scalar_to_cl(from);
            let to_cl = scalar_to_cl(to);
            VarValues::Scalar(if op.norm {
                // Normalizing conversion (e.g. u32 -> f64: divide by 2^32)
                assert!(from.is_unsigned() && to.is_float(), "norm only for uint->float");
                let max_val = match from {
                    ScalarType::U8 => 256.0,
                    ScalarType::U16 => 65536.0,
                    ScalarType::U32 => 4294967296.0,
                    ScalarType::U64 => 18446744073709551616.0,
                    _ => unreachable!(),
                };
                let f = builder.ins().fcvt_from_uint(to_cl, a);
                let recip = if to == ScalarType::F64 {
                    builder.ins().f64const(1.0 / max_val)
                } else {
                    builder.ins().f32const((1.0 / max_val) as f32)
                };
                builder.ins().fmul(f, recip)
            } else if from == to {
                // Identity
                a
            } else if from.is_float() && to.is_float() {
                // Float <-> float
                if from.byte_size() < to.byte_size() {
                    builder.ins().fpromote(to_cl, a)
                } else {
                    builder.ins().fdemote(to_cl, a)
                }
            } else if from.is_float() && to.is_signed() {
                builder.ins().fcvt_to_sint(to_cl, a)
            } else if from.is_float() && to.is_unsigned() {
                builder.ins().fcvt_to_uint(to_cl, a)
            } else if from.is_signed() && to.is_float() {
                builder.ins().fcvt_from_sint(to_cl, a)
            } else if from.is_unsigned() && to.is_float() {
                builder.ins().fcvt_from_uint(to_cl, a)
            } else if from.is_integer() && to.is_integer() {
                // Integer <-> integer
                if from_cl == to_cl {
                    // Same Cranelift type (e.g. i32 <-> u32): identity
                    a
                } else if from.byte_size() < to.byte_size() {
                    // Widening
                    if from.is_signed() {
                        builder.ins().sextend(to_cl, a)
                    } else {
                        builder.ins().uextend(to_cl, a)
                    }
                } else {
                    // Narrowing
                    builder.ins().ireduce(to_cl, a)
                }
            } else {
                unreachable!("unsupported conversion: {:?} -> {:?}", from, to)
            })
        }
        Inst::Select { cond, then_val, else_val } => {
            let c = get_scalar(val_map, cond);
            match (&val_map[then_val], &val_map[else_val]) {
                (VarValues::Scalar(t), VarValues::Scalar(e)) => {
                    VarValues::Scalar(builder.ins().select(c, *t, *e))
                }
                (VarValues::Vec(tv), VarValues::Vec(ev)) => {
                    assert_eq!(tv.len(), ev.len(), "select branches must have matching vec lengths");
                    let components: Vec<_> = tv.iter().zip(ev.iter())
                        .map(|(t, e)| builder.ins().select(c, *t, *e))
                        .collect();
                    VarValues::Vec(components)
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
        Inst::MakeVec(components) => {
            let vals: Vec<_> = components.iter().map(|v| get_scalar(val_map, v)).collect();
            VarValues::Vec(vals)
        }
        Inst::VecExtract { vec, index } => {
            let components = get_vec(val_map, vec);
            assert!((*index as usize) < components.len(), "vec extract index {} out of bounds (len {})", index, components.len());
            VarValues::Scalar(components[*index as usize])
        }
        Inst::VecBinary { op, lhs, rhs } => {
            let lv = get_vec(val_map, lhs);
            let rv = get_vec(val_map, rhs);
            assert_eq!(lv.len(), rv.len(), "VecBinary operands must have same length");
            let apply = |builder: &mut FunctionBuilder, a, b| match op {
                VecBinOp::Add => builder.ins().fadd(a, b),
                VecBinOp::Sub => builder.ins().fsub(a, b),
                VecBinOp::Mul => builder.ins().fmul(a, b),
                VecBinOp::Div => builder.ins().fdiv(a, b),
                VecBinOp::Min => builder.ins().fmin(a, b),
                VecBinOp::Max => builder.ins().fmax(a, b),
            };
            // Copy slices to avoid borrow conflict with builder
            let lv: Vec<_> = lv.to_vec();
            let rv: Vec<_> = rv.to_vec();
            let components: Vec<_> = lv.iter().zip(rv.iter())
                .map(|(l, r)| apply(builder, *l, *r))
                .collect();
            VarValues::Vec(components)
        }
        Inst::VecScale { scalar, vec } => {
            let s = get_scalar(val_map, scalar);
            let v = get_vec(val_map, vec).to_vec();
            let components: Vec<_> = v.iter()
                .map(|c| builder.ins().fmul(s, *c))
                .collect();
            VarValues::Vec(components)
        }
        Inst::VecUnary { op: vec_op, arg } => {
            let av = get_vec(val_map, arg).to_vec();
            match vec_op {
                VecUnaryOp::Neg => {
                    let components: Vec<_> = av.iter().map(|c| builder.ins().fneg(*c)).collect();
                    VarValues::Vec(components)
                }
                VecUnaryOp::Abs => {
                    let components: Vec<_> = av.iter().map(|c| builder.ins().fabs(*c)).collect();
                    VarValues::Vec(components)
                }
                VecUnaryOp::Normalize => {
                    // dot(v, v)
                    let mut sum = builder.ins().fmul(av[0], av[0]);
                    for c in &av[1..] {
                        let sq = builder.ins().fmul(*c, *c);
                        sum = builder.ins().fadd(sum, sq);
                    }
                    let len = builder.ins().sqrt(sum);
                    let components: Vec<_> = av.iter().map(|c| builder.ins().fdiv(*c, len)).collect();
                    VarValues::Vec(components)
                }
            }
        }
        Inst::VecDot { lhs, rhs } => {
            let lv = get_vec(val_map, lhs).to_vec();
            let rv = get_vec(val_map, rhs).to_vec();
            assert_eq!(lv.len(), rv.len(), "VecDot operands must have same length");
            let mut sum = builder.ins().fmul(lv[0], rv[0]);
            for i in 1..lv.len() {
                let prod = builder.ins().fmul(lv[i], rv[i]);
                sum = builder.ins().fadd(sum, prod);
            }
            VarValues::Scalar(sum)
        }
        Inst::VecLength { arg } => {
            let av = get_vec(val_map, arg).to_vec();
            let mut sum = builder.ins().fmul(av[0], av[0]);
            for c in &av[1..] {
                let sq = builder.ins().fmul(*c, *c);
                sum = builder.ins().fadd(sum, sq);
            }
            VarValues::Scalar(builder.ins().sqrt(sum))
        }
        Inst::VecCross { lhs, rhs } => {
            let lv = get_vec(val_map, lhs).to_vec();
            let rv = get_vec(val_map, rhs).to_vec();
            assert_eq!(lv.len(), 3, "VecCross requires vec3 operands");
            assert_eq!(rv.len(), 3, "VecCross requires vec3 operands");
            let (lx, ly, lz) = (lv[0], lv[1], lv[2]);
            let (rx, ry, rz) = (rv[0], rv[1], rv[2]);
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
            VarValues::Vec(vec![cx, cy, cz])
        }
        Inst::BufLoad { buf, x, y } => {
            // Load f64 from buffer: buf_ptrs[buf][(y * width + x)]
            // buf_ctx must be set for sim kernels
            let ctx = buf_ctx.expect("BufLoad requires simulation context");
            let xv = get_scalar(val_map, x);
            let yv = get_scalar(val_map, y);
            let width = builder.use_var(ctx.v_width);

            // Compute wrapped index: ((y % h) * w + (x % w))
            // The kernel is responsible for wrapping — we just compute the flat index
            let row_off = builder.ins().imul(yv, width);
            let idx = builder.ins().iadd(row_off, xv);
            let idx64 = builder.ins().uextend(I64, idx);
            let eight = builder.ins().iconst(I64, 8);
            let byte_off = builder.ins().imul(idx64, eight);

            // Load buffer pointer from buf_ptrs array
            let is_output = kernel.buffers[*buf as usize].is_output;
            let ptrs_base = if is_output {
                builder.use_var(ctx.v_buf_out_ptrs)
            } else {
                builder.use_var(ctx.v_buf_ptrs)
            };
            // Compute the index into the read or write pointer array
            let buf_local_idx = if is_output {
                kernel.buffers.iter().take(*buf as usize).filter(|b| b.is_output).count()
            } else {
                kernel.buffers.iter().take(*buf as usize).filter(|b| !b.is_output).count()
            };
            let ptr_off = builder.ins().iconst(I64, (buf_local_idx * 8) as i64);
            let ptr_addr = builder.ins().iadd(ptrs_base, ptr_off);
            let buf_ptr = builder.ins().load(I64, MemFlags::trusted(), ptr_addr, 0);
            let elem_addr = builder.ins().iadd(buf_ptr, byte_off);
            VarValues::Scalar(builder.ins().load(F64, MemFlags::trusted(), elem_addr, 0))
        }
        Inst::BufStore { buf, x, y, val } => {
            let ctx = buf_ctx.expect("BufStore requires simulation context");
            let xv = get_scalar(val_map, x);
            let yv = get_scalar(val_map, y);
            let value = get_scalar(val_map, val);
            let width = builder.use_var(ctx.v_width);

            let row_off = builder.ins().imul(yv, width);
            let idx = builder.ins().iadd(row_off, xv);
            let idx64 = builder.ins().uextend(I64, idx);
            let eight = builder.ins().iconst(I64, 8);
            let byte_off = builder.ins().imul(idx64, eight);

            // Output buffer pointer
            let ptrs_base = builder.use_var(ctx.v_buf_out_ptrs);
            let buf_local_idx = kernel.buffers.iter().take(*buf as usize).filter(|b| b.is_output).count();
            let ptr_off = builder.ins().iconst(I64, (buf_local_idx * 8) as i64);
            let ptr_addr = builder.ins().iadd(ptrs_base, ptr_off);
            let buf_ptr = builder.ins().load(I64, MemFlags::trusted(), ptr_addr, 0);
            let elem_addr = builder.ins().iadd(buf_ptr, byte_off);
            builder.ins().store(MemFlags::trusted(), value, elem_addr, 0);

            // Return dummy u32 0
            VarValues::Scalar(builder.ins().iconst(I32, 0))
        }
    }
}

fn compile_sim_kernel(kernel: &Kernel, user_args: &[UserArgSlot]) -> CraneliftSimKernel {
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

    // SimTileKernelFn(output, width, height, row_start, row_end, buf_ptrs, buf_out_ptrs, user_args)
    let mut sig = Signature::new(CallConv::SystemV);
    sig.params.push(AbiParam::new(I64)); // output: *mut u32
    sig.params.push(AbiParam::new(I32)); // width: u32
    sig.params.push(AbiParam::new(I32)); // height: u32
    sig.params.push(AbiParam::new(I32)); // row_start: u32
    sig.params.push(AbiParam::new(I32)); // row_end: u32
    sig.params.push(AbiParam::new(I64)); // buf_ptrs: *const *const f64
    sig.params.push(AbiParam::new(I64)); // buf_out_ptrs: *const *mut f64
    sig.params.push(AbiParam::new(I64)); // user_args: *const u8

    let func_id = module
        .declare_function(&kernel.name, Linkage::Export, &sig)
        .unwrap();

    let mut func = Function::with_name_signature(UserFuncName::default(), sig.clone());
    let mut fb_ctx = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut func, &mut fb_ctx);

    // Variables for the tile loop
    let v_output = builder.declare_var(I64);
    let v_width = builder.declare_var(I32);
    let v_height = builder.declare_var(I32);
    let v_row_start = builder.declare_var(I32);
    let v_row_end = builder.declare_var(I32);
    let v_buf_ptrs = builder.declare_var(I64);
    let v_buf_out_ptrs = builder.declare_var(I64);
    let v_row = builder.declare_var(I32);
    let v_col = builder.declare_var(I32);
    let v_user_args = builder.declare_var(I64);

    let buf_ctx = BufContext { v_width, v_buf_ptrs, v_buf_out_ptrs };

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
    builder.def_var(v_height, params[2]);
    builder.def_var(v_row_start, params[3]);
    builder.def_var(v_row_end, params[4]);
    builder.def_var(v_buf_ptrs, params[5]);
    builder.def_var(v_buf_out_ptrs, params[6]);
    builder.def_var(v_user_args, params[7]);

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

    // -- Body: lower kernel body with buffer context --
    builder.switch_to_block(body_block);
    builder.seal_block(body_block);

    let col = builder.use_var(v_col);
    let row = builder.use_var(v_row);
    let height = builder.use_var(v_height);
    let width = builder.use_var(v_width);
    let user_args_ptr = builder.use_var(v_user_args);

    // Map kernel params to values
    use std::collections::HashMap;
    let mut val_map: HashMap<Var, VarValues> = HashMap::new();
    for param in &kernel.params {
        let val = match param.name.as_str() {
            "px" => col,
            "py" => row,
            "width" => width,
            "height" => height,
            name => {
                let slot = user_args.iter().find(|s| s.name == name)
                    .unwrap_or_else(|| panic!("unknown sim kernel parameter: '{name}'"));
                let offset = builder.ins().iconst(I64, slot.offset as i64);
                let addr = builder.ins().iadd(user_args_ptr, offset);
                match slot.ty {
                    ValType::Scalar(s) => builder.ins().load(scalar_to_cl(s), MemFlags::trusted(), addr, 0),
                    _ => panic!("unsupported user-arg type {:?} for param '{name}'", slot.ty),
                }
            }
        };
        val_map.insert(param.var, VarValues::Scalar(val));
    }

    lower_body_items(&mut module, &mut builder, kernel, &kernel.body, &mut val_map, Some(&buf_ctx));

    let color = get_scalar(&val_map, &kernel.emit);

    // Store pixel: output[(row - row_start) * width + col] = color
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
    let fn_ptr: SimTileKernelFn = unsafe { std::mem::transmute(code_ptr) };

    CraneliftSimKernel {
        _module: module,
        fn_ptr,
    }
}
