use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::passes::PassBuilderOptions;
use inkwell::targets::{CodeModel, InitializationConfig, RelocMode, Target, TargetMachine};
use inkwell::values::{BasicValueEnum, FunctionValue};
use inkwell::{FloatPredicate, IntPredicate, OptimizationLevel};

use crate::jit::{CompiledKernel, JitBackend, TileKernelFn};
use crate::kernel_ir::*;

/// Decomposed vector values. Backends store vec vars as 2-3 scalar f64 values.
enum VarValues<'ctx> {
    Scalar(BasicValueEnum<'ctx>),
    Vec2(BasicValueEnum<'ctx>, BasicValueEnum<'ctx>),
    Vec3(BasicValueEnum<'ctx>, BasicValueEnum<'ctx>, BasicValueEnum<'ctx>),
}

fn get_scalar<'ctx>(val_map: &std::collections::HashMap<Var, VarValues<'ctx>>, var: &Var) -> BasicValueEnum<'ctx> {
    match &val_map[var] {
        VarValues::Scalar(v) => *v,
        _ => panic!("expected scalar value for {:?}", var),
    }
}

fn get_vec2<'ctx>(val_map: &std::collections::HashMap<Var, VarValues<'ctx>>, var: &Var) -> (BasicValueEnum<'ctx>, BasicValueEnum<'ctx>) {
    match &val_map[var] {
        VarValues::Vec2(x, y) => (*x, *y),
        _ => panic!("expected vec2 value for {:?}", var),
    }
}

fn get_vec3<'ctx>(val_map: &std::collections::HashMap<Var, VarValues<'ctx>>, var: &Var) -> (BasicValueEnum<'ctx>, BasicValueEnum<'ctx>, BasicValueEnum<'ctx>) {
    match &val_map[var] {
        VarValues::Vec3(x, y, z) => (*x, *y, *z),
        _ => panic!("expected vec3 value for {:?}", var),
    }
}

pub struct LlvmBackend;

struct LlvmKernel {
    _engine: inkwell::execution_engine::ExecutionEngine<'static>,
    fn_ptr: TileKernelFn,
}

unsafe impl Send for LlvmKernel {}
unsafe impl Sync for LlvmKernel {}

impl CompiledKernel for LlvmKernel {
    fn function_ptr(&self) -> TileKernelFn {
        self.fn_ptr
    }
}

impl JitBackend for LlvmBackend {
    fn compile(&self, kernel: &Kernel) -> Box<dyn CompiledKernel> {
        Box::new(compile_kernel(kernel))
    }
}

fn compile_kernel(kernel: &Kernel) -> LlvmKernel {
    let context: &'static Context = Box::leak(Box::new(Context::create()));
    let module = context.create_module(&kernel.name);

    Target::initialize_native(&InitializationConfig::default())
        .expect("failed to initialize native target");
    let triple = TargetMachine::get_default_triple();
    let cpu = TargetMachine::get_host_cpu_name();
    let features = TargetMachine::get_host_cpu_features();
    // Filter out SVE features on AArch64 — LLVM's O3 auto-vectorizer generates
    // scalable vector types that hit legalization bugs for simple kernels.
    // We handle parallelism via tiling, so fixed-width NEON is sufficient.
    let features_str = features.to_str().unwrap();
    let filtered_features: String = features_str
        .split(',')
        .filter(|f| !f.contains("sve") && !f.contains("sme"))
        .collect::<Vec<_>>()
        .join(",");
    // On AArch64, also explicitly disable SVE to prevent LLVM from generating
    // scalable vector types even when the host doesn't report SVE features.
    let filtered_features = if cfg!(target_arch = "aarch64") {
        format!("{filtered_features},-sve,-sve2")
    } else {
        filtered_features
    };
    let target = Target::from_triple(&triple).unwrap();
    let machine = target
        .create_target_machine(
            &triple,
            cpu.to_str().unwrap(),
            &filtered_features,
            OptimizationLevel::Aggressive,
            RelocMode::Default,
            CodeModel::JITDefault,
        )
        .unwrap();
    module.set_data_layout(&machine.get_target_data().get_data_layout());
    module.set_triple(&triple);

    let i32_type = context.i32_type();
    let f64_type = context.f64_type();
    let void_type = context.void_type();
    let ptr_type = context.ptr_type(inkwell::AddressSpace::default());

    let fn_type = void_type.fn_type(
        &[
            ptr_type.into(),
            i32_type.into(),
            i32_type.into(),
            f64_type.into(),
            f64_type.into(),
            f64_type.into(),
            f64_type.into(),
            i32_type.into(),
            i32_type.into(),
            i32_type.into(), // sample_index
            f64_type.into(), // time
        ],
        false,
    );

    let function = module.add_function(&kernel.name, fn_type, None);
    build_tile_loop(context, &module, function, kernel);

    let pass_options = PassBuilderOptions::create();
    module
        .run_passes("default<O3>", &machine, pass_options)
        .expect("failed to run LLVM optimization passes");

    let engine = module
        .create_jit_execution_engine(OptimizationLevel::Aggressive)
        .unwrap();

    let fn_ptr = unsafe {
        engine
            .get_function::<unsafe extern "C" fn(*mut u32, u32, u32, f64, f64, f64, f64, u32, u32, u32, f64)>(
                &kernel.name,
            )
            .unwrap()
            .as_raw()
    };
    let fn_ptr: TileKernelFn = unsafe { std::mem::transmute(fn_ptr) };

    LlvmKernel {
        _engine: engine,
        fn_ptr,
    }
}

fn build_tile_loop(
    context: &'static Context,
    module: &Module<'static>,
    function: FunctionValue<'static>,
    kernel: &Kernel,
) {
    let builder = context.create_builder();
    let i32_type = context.i32_type();
    let i64_type = context.i64_type();
    let f64_type = context.f64_type();

    let entry = context.append_basic_block(function, "entry");
    let outer_check = context.append_basic_block(function, "outer_check");
    let inner_pre = context.append_basic_block(function, "inner_pre");
    let inner_check = context.append_basic_block(function, "inner_check");
    let body = context.append_basic_block(function, "body");
    let inner_inc = context.append_basic_block(function, "inner_inc");
    let outer_inc = context.append_basic_block(function, "outer_inc");
    let exit = context.append_basic_block(function, "exit");

    // -- Entry --
    builder.position_at_end(entry);
    let p_output = function.get_nth_param(0).unwrap().into_pointer_value();
    let p_width = function.get_nth_param(1).unwrap().into_int_value();
    let p_x_min = function.get_nth_param(3).unwrap().into_float_value();
    let p_y_min = function.get_nth_param(4).unwrap().into_float_value();
    let p_x_step = function.get_nth_param(5).unwrap().into_float_value();
    let p_y_step = function.get_nth_param(6).unwrap().into_float_value();
    let p_row_start = function.get_nth_param(7).unwrap().into_int_value();
    let p_row_end = function.get_nth_param(8).unwrap().into_int_value();
    let p_sample_index = function.get_nth_param(9).unwrap().into_int_value();
    let p_time = function.get_nth_param(10).unwrap().into_float_value();

    let row_ptr = builder.build_alloca(i32_type, "row_ptr").unwrap();
    let col_ptr = builder.build_alloca(i32_type, "col_ptr").unwrap();

    builder.build_store(row_ptr, p_row_start).unwrap();
    builder.build_unconditional_branch(outer_check).unwrap();

    // -- Outer check --
    builder.position_at_end(outer_check);
    let row = builder.build_load(i32_type, row_ptr, "row").unwrap().into_int_value();
    let cmp = builder.build_int_compare(IntPredicate::SLT, row, p_row_end, "row_lt").unwrap();
    builder.build_conditional_branch(cmp, inner_pre, exit).unwrap();

    // -- Inner pre --
    builder.position_at_end(inner_pre);
    builder.build_store(col_ptr, i32_type.const_zero()).unwrap();
    builder.build_unconditional_branch(inner_check).unwrap();

    // -- Inner check --
    builder.position_at_end(inner_check);
    let col = builder.build_load(i32_type, col_ptr, "col").unwrap().into_int_value();
    let cmp = builder.build_int_compare(IntPredicate::SLT, col, p_width, "col_lt").unwrap();
    builder.build_conditional_branch(cmp, body, outer_inc).unwrap();

    // -- Body --
    builder.position_at_end(body);
    let col = builder.build_load(i32_type, col_ptr, "col").unwrap().into_int_value();
    let row = builder.build_load(i32_type, row_ptr, "row").unwrap().into_int_value();

    let col_f = builder.build_signed_int_to_float(col, f64_type, "col_f").unwrap();
    let col_step = builder.build_float_mul(col_f, p_x_step, "col_step").unwrap();
    let cx = builder.build_float_add(p_x_min, col_step, "cx").unwrap();

    let row_f = builder.build_signed_int_to_float(row, f64_type, "row_f").unwrap();
    let row_step = builder.build_float_mul(row_f, p_y_step, "row_step").unwrap();
    let cy = builder.build_float_add(p_y_min, row_step, "cy").unwrap();

    // Apply sub-pixel jitter when sample_index != 0xFFFFFFFF
    let no_jitter = i32_type.const_int(0xFFFFFFFF, false);
    let skip_jitter = builder.build_int_compare(IntPredicate::EQ, p_sample_index, no_jitter, "skip_jitter").unwrap();

    // Hash: h = col * 0x45d9f3b + row
    let col = builder.build_load(i32_type, col_ptr, "col_h").unwrap().into_int_value();
    let row = builder.build_load(i32_type, row_ptr, "row_h").unwrap().into_int_value();
    let hash_k = i32_type.const_int(0x45d9f3b, false);
    let h = builder.build_int_mul(col, hash_k, "h1").unwrap();
    let h = builder.build_int_add(h, row, "h2").unwrap();
    let h = builder.build_int_mul(h, hash_k, "h3").unwrap();
    let h = builder.build_int_add(h, p_sample_index, "h4").unwrap();
    let sixteen = i32_type.const_int(16, false);
    let h_shifted = builder.build_right_shift(h, sixteen, false, "h_shr1").unwrap();
    let h = builder.build_xor(h, h_shifted, "h5").unwrap();
    let h = builder.build_int_mul(h, hash_k, "h6").unwrap();
    let h_shifted = builder.build_right_shift(h, sixteen, false, "h_shr2").unwrap();
    let h = builder.build_xor(h, h_shifted, "h7").unwrap();

    // Extract jitter values
    let mask_16 = i32_type.const_int(0xFFFF, false);
    let jx_bits = builder.build_and(h, mask_16, "jx_bits").unwrap();
    let jy_bits = builder.build_right_shift(h, sixteen, false, "jy_bits").unwrap();
    let recip = f64_type.const_float(1.0 / 65536.0);
    let jx = builder.build_unsigned_int_to_float(jx_bits, f64_type, "jx_f").unwrap();
    let jx = builder.build_float_mul(jx, recip, "jx").unwrap();
    let jy = builder.build_unsigned_int_to_float(jy_bits, f64_type, "jy_f").unwrap();
    let jy = builder.build_float_mul(jy, recip, "jy").unwrap();

    // Conditional: if skip_jitter, use 0.0; otherwise use jitter
    let zero_f64 = f64_type.const_float(0.0);
    let jx = builder.build_select(skip_jitter, zero_f64, jx, "jx_sel").unwrap().into_float_value();
    let jy = builder.build_select(skip_jitter, zero_f64, jy, "jy_sel").unwrap().into_float_value();

    // Apply: cx += jx * x_step, cy += jy * y_step
    let jx_step = builder.build_float_mul(jx, p_x_step, "jx_step").unwrap();
    let cx = builder.build_float_add(cx, jx_step, "cx_j").unwrap();
    let jy_step = builder.build_float_mul(jy, p_y_step, "jy_step").unwrap();
    let cy = builder.build_float_add(cy, jy_step, "cy_j").unwrap();

    let col = builder.build_load(i32_type, col_ptr, "col_k").unwrap().into_int_value();
    let row = builder.build_load(i32_type, row_ptr, "row_k").unwrap().into_int_value();
    let color = lower_kernel_body(context, module, &builder, function, kernel, cx, cy, col, row, p_sample_index, p_time);

    // Store pixel
    let row = builder.build_load(i32_type, row_ptr, "row").unwrap().into_int_value();
    let col = builder.build_load(i32_type, col_ptr, "col").unwrap().into_int_value();
    let row_off = builder.build_int_sub(row, p_row_start, "row_off").unwrap();
    let row_off64 = builder.build_int_z_extend(row_off, i64_type, "row_off64").unwrap();
    let width64 = builder.build_int_z_extend(p_width, i64_type, "width64").unwrap();
    let col64 = builder.build_int_z_extend(col, i64_type, "col64").unwrap();
    let idx = builder.build_int_mul(row_off64, width64, "idx1").unwrap();
    let idx = builder.build_int_add(idx, col64, "idx2").unwrap();
    let ptr = unsafe { builder.build_gep(i32_type, p_output, &[idx], "ptr").unwrap() };
    builder.build_store(ptr, color).unwrap();

    builder.build_unconditional_branch(inner_inc).unwrap();

    // -- Inner inc --
    builder.position_at_end(inner_inc);
    let col = builder.build_load(i32_type, col_ptr, "col").unwrap().into_int_value();
    let col_next = builder.build_int_add(col, i32_type.const_int(1, false), "col_next").unwrap();
    builder.build_store(col_ptr, col_next).unwrap();
    builder.build_unconditional_branch(inner_check).unwrap();

    // -- Outer inc --
    builder.position_at_end(outer_inc);
    let row = builder.build_load(i32_type, row_ptr, "row").unwrap().into_int_value();
    let row_next = builder.build_int_add(row, i32_type.const_int(1, false), "row_next").unwrap();
    builder.build_store(row_ptr, row_next).unwrap();
    builder.build_unconditional_branch(outer_check).unwrap();

    // -- Exit --
    builder.position_at_end(exit);
    builder.build_return(None).unwrap();
}

fn lower_kernel_body(
    context: &'static Context,
    module: &Module<'static>,
    builder: &inkwell::builder::Builder<'static>,
    function: FunctionValue<'static>,
    kernel: &Kernel,
    cx: inkwell::values::FloatValue<'static>,
    cy: inkwell::values::FloatValue<'static>,
    col: inkwell::values::IntValue<'static>,
    row: inkwell::values::IntValue<'static>,
    sample_index: inkwell::values::IntValue<'static>,
    time: inkwell::values::FloatValue<'static>,
) -> inkwell::values::IntValue<'static> {
    use std::collections::HashMap;

    let mut val_map: HashMap<Var, VarValues<'static>> = HashMap::new();
    for param in &kernel.params {
        let val: BasicValueEnum<'static> = match param.name.as_str() {
            "x" => cx.into(),
            "y" => cy.into(),
            "px" => col.into(),
            "py" => row.into(),
            "sample_index" => sample_index.into(),
            "time" => time.into(),
            name => panic!("unknown kernel parameter name: '{name}'"),
        };
        val_map.insert(param.var, VarValues::Scalar(val));
    }

    lower_body_items(context, module, builder, function, kernel, &kernel.body, &mut val_map);

    get_scalar(&val_map, &kernel.emit).into_int_value()
}

fn lower_body_items(
    context: &'static Context,
    module: &Module<'static>,
    builder: &inkwell::builder::Builder<'static>,
    function: FunctionValue<'static>,
    kernel: &Kernel,
    body: &[BodyItem],
    val_map: &mut std::collections::HashMap<Var, VarValues<'static>>,
) {
    for item in body {
        match item {
            BodyItem::Stmt(stmt) => {
                let v = lower_inst(context, module, builder, kernel, &stmt.inst, &stmt.binding, val_map);
                val_map.insert(stmt.binding.var, v);
            }
            BodyItem::While(w) => {
                lower_while(context, module, builder, function, kernel, w, val_map);
            }
        }
    }
}

fn lower_while(
    context: &'static Context,
    module: &Module<'static>,
    builder: &inkwell::builder::Builder<'static>,
    function: FunctionValue<'static>,
    kernel: &Kernel,
    w: &While,
    val_map: &mut std::collections::HashMap<Var, VarValues<'static>>,
) {
    let i32_type = context.i32_type();
    let f64_type = context.f64_type();
    let i1_type = context.bool_type();

    // Create blocks
    let pre_block = builder.get_insert_block().unwrap();
    let loop_header = context.append_basic_block(function, "while_header");
    let loop_body = context.append_basic_block(function, "while_body");
    let loop_exit = context.append_basic_block(function, "while_exit");

    // Branch from current block to loop header
    builder.build_unconditional_branch(loop_header).unwrap();

    // -- Loop header: phi nodes for carry vars, then cond_body, then branch --
    builder.position_at_end(loop_header);

    // Create phi nodes for carry variables, expanding vec types into multiple phis
    // Each entry: (carry index, list of phi nodes for that carry var)
    let mut carry_phis: Vec<Vec<inkwell::values::PhiValue<'static>>> = Vec::new();
    for cv in &w.carry {
        let component_count = cv.binding.ty.component_count();
        let llvm_ty: inkwell::types::BasicTypeEnum = match cv.binding.ty.element_type() {
            ValType::F64 => f64_type.into(),
            ValType::U32 => i32_type.into(),
            ValType::Bool => i1_type.into(),
            _ => unreachable!(),
        };

        let mut phis = Vec::new();
        for comp in 0..component_count {
            let name = if component_count == 1 {
                cv.binding.name.clone()
            } else {
                format!("{}_{}", cv.binding.name, comp)
            };
            let phi = builder.build_phi(llvm_ty, &name).unwrap();

            // Add incoming from pre-block (initial value)
            let init_vals = &val_map[&cv.init];
            let init_val = match (init_vals, comp) {
                (VarValues::Scalar(v), 0) => *v,
                (VarValues::Vec2(x, _), 0) => *x,
                (VarValues::Vec2(_, y), 1) => *y,
                (VarValues::Vec3(x, _, _), 0) => *x,
                (VarValues::Vec3(_, y, _), 1) => *y,
                (VarValues::Vec3(_, _, z), 2) => *z,
                _ => unreachable!(),
            };
            phi.add_incoming(&[(&init_val, pre_block)]);
            phis.push(phi);
        }

        // Map carry var to phi values
        match cv.binding.ty {
            ValType::Vec2 => {
                val_map.insert(cv.binding.var, VarValues::Vec2(
                    phis[0].as_basic_value(),
                    phis[1].as_basic_value(),
                ));
            }
            ValType::Vec3 => {
                val_map.insert(cv.binding.var, VarValues::Vec3(
                    phis[0].as_basic_value(),
                    phis[1].as_basic_value(),
                    phis[2].as_basic_value(),
                ));
            }
            _ => {
                val_map.insert(cv.binding.var, VarValues::Scalar(phis[0].as_basic_value()));
            }
        }
        carry_phis.push(phis);
    }

    // Lower cond_body
    lower_body_items(context, module, builder, function, kernel, &w.cond_body, val_map);

    // Branch on cond
    let cond_val = get_scalar(val_map, &w.cond).into_int_value();
    builder.build_conditional_branch(cond_val, loop_body, loop_exit).unwrap();

    // -- Loop body: compute next values, branch back to header --
    builder.position_at_end(loop_body);

    lower_body_items(context, module, builder, function, kernel, &w.body, val_map);

    // Add incoming edges to phi nodes from loop body (yield values)
    let body_block = builder.get_insert_block().unwrap();
    for (i, phis) in carry_phis.iter().enumerate() {
        let yield_var = &w.yields[i];
        let yield_vals = &val_map[yield_var];
        for (comp, phi) in phis.iter().enumerate() {
            let yield_val = match (yield_vals, comp) {
                (VarValues::Scalar(v), 0) => *v,
                (VarValues::Vec2(x, _), 0) => *x,
                (VarValues::Vec2(_, y), 1) => *y,
                (VarValues::Vec3(x, _, _), 0) => *x,
                (VarValues::Vec3(_, y, _), 1) => *y,
                (VarValues::Vec3(_, _, z), 2) => *z,
                _ => unreachable!(),
            };
            phi.add_incoming(&[(&yield_val, body_block)]);
        }
    }

    builder.build_unconditional_branch(loop_header).unwrap();

    // -- Continue after loop --
    builder.position_at_end(loop_exit);
    // Carry vars already mapped to phi values (which are their final values on exit)
}

fn lower_inst(
    context: &'static Context,
    module: &Module<'static>,
    builder: &inkwell::builder::Builder<'static>,
    kernel: &Kernel,
    inst: &Inst,
    binding: &Binding,
    val_map: &std::collections::HashMap<Var, VarValues<'static>>,
) -> VarValues<'static> {
    let i32_type = context.i32_type();
    let f64_type = context.f64_type();
    let i1_type = context.bool_type();

    match inst {
        Inst::Const(c) => VarValues::Scalar(match c {
            Const::F64(v) => f64_type.const_float(*v).into(),
            Const::U32(v) => i32_type.const_int(*v as u64, false).into(),
            Const::Bool(v) => i1_type.const_int(if *v { 1 } else { 0 }, false).into(),
        }),
        Inst::Binary { op, lhs, rhs } => {
            let l = get_scalar(val_map, lhs);
            let r = get_scalar(val_map, rhs);
            VarValues::Scalar(match (op, binding.ty) {
                (BinOp::Add, ValType::F64) => builder.build_float_add(l.into_float_value(), r.into_float_value(), "add").unwrap().into(),
                (BinOp::Sub, ValType::F64) => builder.build_float_sub(l.into_float_value(), r.into_float_value(), "sub").unwrap().into(),
                (BinOp::Mul, ValType::F64) => builder.build_float_mul(l.into_float_value(), r.into_float_value(), "mul").unwrap().into(),
                (BinOp::Div, ValType::F64) => builder.build_float_div(l.into_float_value(), r.into_float_value(), "div").unwrap().into(),
                (BinOp::Rem, ValType::F64) => builder.build_float_rem(l.into_float_value(), r.into_float_value(), "rem").unwrap().into(),
                (BinOp::Add, ValType::U32) => builder.build_int_add(l.into_int_value(), r.into_int_value(), "add").unwrap().into(),
                (BinOp::Sub, ValType::U32) => builder.build_int_sub(l.into_int_value(), r.into_int_value(), "sub").unwrap().into(),
                (BinOp::Mul, ValType::U32) => builder.build_int_mul(l.into_int_value(), r.into_int_value(), "mul").unwrap().into(),
                (BinOp::Div, ValType::U32) => builder.build_int_unsigned_div(l.into_int_value(), r.into_int_value(), "div").unwrap().into(),
                (BinOp::Rem, ValType::U32) => builder.build_int_unsigned_rem(l.into_int_value(), r.into_int_value(), "rem").unwrap().into(),
                (BinOp::BitAnd, _) | (BinOp::And, _) => builder.build_and(l.into_int_value(), r.into_int_value(), "and").unwrap().into(),
                (BinOp::BitOr, _) | (BinOp::Or, _) => builder.build_or(l.into_int_value(), r.into_int_value(), "or").unwrap().into(),
                (BinOp::BitXor, _) => builder.build_xor(l.into_int_value(), r.into_int_value(), "xor").unwrap().into(),
                (BinOp::Shl, _) => builder.build_left_shift(l.into_int_value(), r.into_int_value(), "shl").unwrap().into(),
                (BinOp::Shr, _) => builder.build_right_shift(l.into_int_value(), r.into_int_value(), false, "shr").unwrap().into(),
                (BinOp::Min, ValType::F64) => call_f64_binary_intrinsic(module, builder, "llvm.minnum.f64", l.into_float_value(), r.into_float_value(), f64_type),
                (BinOp::Max, ValType::F64) => call_f64_binary_intrinsic(module, builder, "llvm.maxnum.f64", l.into_float_value(), r.into_float_value(), f64_type),
                (BinOp::Min, ValType::U32) => call_i32_binary_intrinsic(module, builder, "llvm.umin.i32", l.into_int_value(), r.into_int_value(), i32_type),
                (BinOp::Max, ValType::U32) => call_i32_binary_intrinsic(module, builder, "llvm.umax.i32", l.into_int_value(), r.into_int_value(), i32_type),
                (BinOp::Atan2, ValType::F64) => call_f64_binary_intrinsic(module, builder, "llvm.atan2.f64", l.into_float_value(), r.into_float_value(), f64_type),
                (BinOp::Pow, ValType::F64) => call_f64_binary_intrinsic(module, builder, "llvm.pow.f64", l.into_float_value(), r.into_float_value(), f64_type),
                (BinOp::Hash, ValType::U32) => {
                    let l = l.into_int_value();
                    let r = r.into_int_value();
                    let hash_k = i32_type.const_int(0x45d9f3b, false);
                    let sixteen = i32_type.const_int(16, false);
                    // h = a * k + b
                    let h = builder.build_int_mul(l, hash_k, "hash1").unwrap();
                    let h = builder.build_int_add(h, r, "hash2").unwrap();
                    // h ^= h >> 16
                    let h_s = builder.build_right_shift(h, sixteen, false, "hash_shr1").unwrap();
                    let h = builder.build_xor(h, h_s, "hash3").unwrap();
                    // h *= k
                    let h = builder.build_int_mul(h, hash_k, "hash4").unwrap();
                    // h ^= h >> 16
                    let h_s = builder.build_right_shift(h, sixteen, false, "hash_shr2").unwrap();
                    builder.build_xor(h, h_s, "hash5").unwrap().into()
                }
                _ => unreachable!("invalid binary op/type combination"),
            })
        }
        Inst::Unary { op, arg } => {
            let a = get_scalar(val_map, arg);
            VarValues::Scalar(match (op, binding.ty) {
                (UnaryOp::Neg, ValType::F64) => builder.build_float_neg(a.into_float_value(), "neg").unwrap().into(),
                (UnaryOp::Neg, ValType::U32) => builder.build_int_sub(i32_type.const_zero(), a.into_int_value(), "neg").unwrap().into(),
                (UnaryOp::Not, _) => builder.build_not(a.into_int_value(), "not").unwrap().into(),
                (UnaryOp::Abs, ValType::F64) => {
                    call_f64_intrinsic(module, builder, "llvm.fabs.f64", a.into_float_value(), f64_type)
                }
                (UnaryOp::Abs, ValType::U32) => a,
                (UnaryOp::Sqrt, _) => {
                    call_f64_intrinsic(module, builder, "llvm.sqrt.f64", a.into_float_value(), f64_type)
                }
                (UnaryOp::Floor, _) => {
                    call_f64_intrinsic(module, builder, "llvm.floor.f64", a.into_float_value(), f64_type)
                }
                (UnaryOp::Ceil, _) => {
                    call_f64_intrinsic(module, builder, "llvm.ceil.f64", a.into_float_value(), f64_type)
                }
                (UnaryOp::Sin, _) => call_f64_intrinsic(module, builder, "llvm.sin.f64", a.into_float_value(), f64_type),
                (UnaryOp::Cos, _) => call_f64_intrinsic(module, builder, "llvm.cos.f64", a.into_float_value(), f64_type),
                (UnaryOp::Tan, _) => call_f64_intrinsic(module, builder, "llvm.tan.f64", a.into_float_value(), f64_type),
                (UnaryOp::Asin, _) => call_f64_intrinsic(module, builder, "llvm.asin.f64", a.into_float_value(), f64_type),
                (UnaryOp::Acos, _) => call_f64_intrinsic(module, builder, "llvm.acos.f64", a.into_float_value(), f64_type),
                (UnaryOp::Atan, _) => call_f64_intrinsic(module, builder, "llvm.atan.f64", a.into_float_value(), f64_type),
                (UnaryOp::Exp, _) => call_f64_intrinsic(module, builder, "llvm.exp.f64", a.into_float_value(), f64_type),
                (UnaryOp::Exp2, _) => call_f64_intrinsic(module, builder, "llvm.exp2.f64", a.into_float_value(), f64_type),
                (UnaryOp::Log, _) => call_f64_intrinsic(module, builder, "llvm.log.f64", a.into_float_value(), f64_type),
                (UnaryOp::Log2, _) => call_f64_intrinsic(module, builder, "llvm.log2.f64", a.into_float_value(), f64_type),
                (UnaryOp::Log10, _) => call_f64_intrinsic(module, builder, "llvm.log10.f64", a.into_float_value(), f64_type),
                (UnaryOp::Round, _) => call_f64_intrinsic(module, builder, "llvm.round.f64", a.into_float_value(), f64_type),
                (UnaryOp::Trunc, _) => call_f64_intrinsic(module, builder, "llvm.trunc.f64", a.into_float_value(), f64_type),
                (UnaryOp::Fract, _) => {
                    let floored = call_f64_intrinsic(module, builder, "llvm.floor.f64", a.into_float_value(), f64_type);
                    builder.build_float_sub(a.into_float_value(), floored.into_float_value(), "fract").unwrap().into()
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
                    let pred = match op {
                        CmpOp::Eq => FloatPredicate::OEQ,
                        CmpOp::Ne => FloatPredicate::ONE,
                        CmpOp::Lt => FloatPredicate::OLT,
                        CmpOp::Le => FloatPredicate::OLE,
                        CmpOp::Gt => FloatPredicate::OGT,
                        CmpOp::Ge => FloatPredicate::OGE,
                    };
                    builder.build_float_compare(pred, l.into_float_value(), r.into_float_value(), "cmp").unwrap().into()
                }
                ValType::U32 => {
                    let pred = match op {
                        CmpOp::Eq => IntPredicate::EQ,
                        CmpOp::Ne => IntPredicate::NE,
                        CmpOp::Lt => IntPredicate::ULT,
                        CmpOp::Le => IntPredicate::ULE,
                        CmpOp::Gt => IntPredicate::UGT,
                        CmpOp::Ge => IntPredicate::UGE,
                    };
                    builder.build_int_compare(pred, l.into_int_value(), r.into_int_value(), "cmp").unwrap().into()
                }
                _ => unreachable!("cannot compare bools"),
            })
        }
        Inst::Conv { op, arg } => {
            let a = get_scalar(val_map, arg);
            VarValues::Scalar(match op {
                ConvOp::F64ToU32 => builder.build_float_to_signed_int(a.into_float_value(), i32_type, "conv").unwrap().into(),
                ConvOp::U32ToF64 => builder.build_unsigned_int_to_float(a.into_int_value(), f64_type, "conv").unwrap().into(),
                ConvOp::U32ToF64Norm => {
                    let f = builder.build_unsigned_int_to_float(a.into_int_value(), f64_type, "conv_f").unwrap();
                    let recip = f64_type.const_float(1.0 / 4294967296.0);
                    builder.build_float_mul(f, recip, "norm").unwrap().into()
                }
            })
        }
        Inst::Select { cond, then_val, else_val } => {
            let c = get_scalar(val_map, cond).into_int_value();
            match binding.ty {
                ValType::Vec2 => {
                    let (tx, ty) = get_vec2(val_map, then_val);
                    let (ex, ey) = get_vec2(val_map, else_val);
                    let rx = builder.build_select(c, tx.into_float_value(), ex.into_float_value(), "sel_x").unwrap();
                    let ry = builder.build_select(c, ty.into_float_value(), ey.into_float_value(), "sel_y").unwrap();
                    VarValues::Vec2(rx, ry)
                }
                ValType::Vec3 => {
                    let (tx, ty, tz) = get_vec3(val_map, then_val);
                    let (ex, ey, ez) = get_vec3(val_map, else_val);
                    let rx = builder.build_select(c, tx.into_float_value(), ex.into_float_value(), "sel_x").unwrap();
                    let ry = builder.build_select(c, ty.into_float_value(), ey.into_float_value(), "sel_y").unwrap();
                    let rz = builder.build_select(c, tz.into_float_value(), ez.into_float_value(), "sel_z").unwrap();
                    VarValues::Vec3(rx, ry, rz)
                }
                ValType::F64 => {
                    let t = get_scalar(val_map, then_val);
                    let e = get_scalar(val_map, else_val);
                    VarValues::Scalar(builder.build_select(c, t.into_float_value(), e.into_float_value(), "sel").unwrap())
                }
                ValType::U32 | ValType::Bool => {
                    let t = get_scalar(val_map, then_val);
                    let e = get_scalar(val_map, else_val);
                    VarValues::Scalar(builder.build_select(c, t.into_int_value(), e.into_int_value(), "sel").unwrap())
                }
            }
        }
        Inst::PackArgb { r, g, b } => {
            let rv = get_scalar(val_map, r).into_int_value();
            let gv = get_scalar(val_map, g).into_int_value();
            let bv = get_scalar(val_map, b).into_int_value();
            let alpha = i32_type.const_int(0xFF000000, false);
            let r_sh = builder.build_left_shift(rv, i32_type.const_int(16, false), "r_sh").unwrap();
            let g_sh = builder.build_left_shift(gv, i32_type.const_int(8, false), "g_sh").unwrap();
            let color = builder.build_or(alpha, r_sh, "ar").unwrap();
            let color = builder.build_or(color, g_sh, "arg").unwrap();
            VarValues::Scalar(builder.build_or(color, bv, "argb").unwrap().into())
        }

        // -- Vector construction --
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

        // -- Component extraction --
        Inst::VecExtract { vec, index } => {
            let val = match &val_map[vec] {
                VarValues::Vec2(x, y) => match index {
                    0 => *x,
                    1 => *y,
                    _ => unreachable!("invalid vec2 index"),
                },
                VarValues::Vec3(x, y, z) => match index {
                    0 => *x,
                    1 => *y,
                    2 => *z,
                    _ => unreachable!("invalid vec3 index"),
                },
                _ => unreachable!("VecExtract on non-vector"),
            };
            VarValues::Scalar(val)
        }

        // -- Component-wise binary --
        Inst::VecBinary { op, lhs, rhs } => {
            match binding.ty {
                ValType::Vec2 => {
                    let (lx, ly) = get_vec2(val_map, lhs);
                    let (rx, ry) = get_vec2(val_map, rhs);
                    let res_x = lower_vec_bin_op(module, builder, f64_type, *op, lx.into_float_value(), rx.into_float_value());
                    let res_y = lower_vec_bin_op(module, builder, f64_type, *op, ly.into_float_value(), ry.into_float_value());
                    VarValues::Vec2(res_x.into(), res_y.into())
                }
                ValType::Vec3 => {
                    let (lx, ly, lz) = get_vec3(val_map, lhs);
                    let (rx, ry, rz) = get_vec3(val_map, rhs);
                    let res_x = lower_vec_bin_op(module, builder, f64_type, *op, lx.into_float_value(), rx.into_float_value());
                    let res_y = lower_vec_bin_op(module, builder, f64_type, *op, ly.into_float_value(), ry.into_float_value());
                    let res_z = lower_vec_bin_op(module, builder, f64_type, *op, lz.into_float_value(), rz.into_float_value());
                    VarValues::Vec3(res_x.into(), res_y.into(), res_z.into())
                }
                _ => unreachable!("VecBinary on non-vector type"),
            }
        }

        // -- Scalar-vector multiply --
        Inst::VecScale { scalar, vec } => {
            let s = get_scalar(val_map, scalar).into_float_value();
            match binding.ty {
                ValType::Vec2 => {
                    let (vx, vy) = get_vec2(val_map, vec);
                    let rx = builder.build_float_mul(s, vx.into_float_value(), "scale_x").unwrap();
                    let ry = builder.build_float_mul(s, vy.into_float_value(), "scale_y").unwrap();
                    VarValues::Vec2(rx.into(), ry.into())
                }
                ValType::Vec3 => {
                    let (vx, vy, vz) = get_vec3(val_map, vec);
                    let rx = builder.build_float_mul(s, vx.into_float_value(), "scale_x").unwrap();
                    let ry = builder.build_float_mul(s, vy.into_float_value(), "scale_y").unwrap();
                    let rz = builder.build_float_mul(s, vz.into_float_value(), "scale_z").unwrap();
                    VarValues::Vec3(rx.into(), ry.into(), rz.into())
                }
                _ => unreachable!("VecScale on non-vector type"),
            }
        }

        // -- Vector unary --
        Inst::VecUnary { op, arg } => {
            match binding.ty {
                ValType::Vec2 => {
                    let (ax, ay) = get_vec2(val_map, arg);
                    match op {
                        VecUnaryOp::Neg => {
                            let rx = builder.build_float_neg(ax.into_float_value(), "neg_x").unwrap();
                            let ry = builder.build_float_neg(ay.into_float_value(), "neg_y").unwrap();
                            VarValues::Vec2(rx.into(), ry.into())
                        }
                        VecUnaryOp::Abs => {
                            let rx = call_f64_intrinsic(module, builder, "llvm.fabs.f64", ax.into_float_value(), f64_type);
                            let ry = call_f64_intrinsic(module, builder, "llvm.fabs.f64", ay.into_float_value(), f64_type);
                            VarValues::Vec2(rx, ry)
                        }
                        VecUnaryOp::Normalize => {
                            // length = sqrt(x*x + y*y)
                            let xx = builder.build_float_mul(ax.into_float_value(), ax.into_float_value(), "xx").unwrap();
                            let yy = builder.build_float_mul(ay.into_float_value(), ay.into_float_value(), "yy").unwrap();
                            let dot = builder.build_float_add(xx, yy, "dot").unwrap();
                            let len = call_f64_intrinsic(module, builder, "llvm.sqrt.f64", dot, f64_type).into_float_value();
                            let rx = builder.build_float_div(ax.into_float_value(), len, "norm_x").unwrap();
                            let ry = builder.build_float_div(ay.into_float_value(), len, "norm_y").unwrap();
                            VarValues::Vec2(rx.into(), ry.into())
                        }
                    }
                }
                ValType::Vec3 => {
                    let (ax, ay, az) = get_vec3(val_map, arg);
                    match op {
                        VecUnaryOp::Neg => {
                            let rx = builder.build_float_neg(ax.into_float_value(), "neg_x").unwrap();
                            let ry = builder.build_float_neg(ay.into_float_value(), "neg_y").unwrap();
                            let rz = builder.build_float_neg(az.into_float_value(), "neg_z").unwrap();
                            VarValues::Vec3(rx.into(), ry.into(), rz.into())
                        }
                        VecUnaryOp::Abs => {
                            let rx = call_f64_intrinsic(module, builder, "llvm.fabs.f64", ax.into_float_value(), f64_type);
                            let ry = call_f64_intrinsic(module, builder, "llvm.fabs.f64", ay.into_float_value(), f64_type);
                            let rz = call_f64_intrinsic(module, builder, "llvm.fabs.f64", az.into_float_value(), f64_type);
                            VarValues::Vec3(rx, ry, rz)
                        }
                        VecUnaryOp::Normalize => {
                            // length = sqrt(x*x + y*y + z*z)
                            let xx = builder.build_float_mul(ax.into_float_value(), ax.into_float_value(), "xx").unwrap();
                            let yy = builder.build_float_mul(ay.into_float_value(), ay.into_float_value(), "yy").unwrap();
                            let zz = builder.build_float_mul(az.into_float_value(), az.into_float_value(), "zz").unwrap();
                            let dot = builder.build_float_add(xx, yy, "dot_xy").unwrap();
                            let dot = builder.build_float_add(dot, zz, "dot").unwrap();
                            let len = call_f64_intrinsic(module, builder, "llvm.sqrt.f64", dot, f64_type).into_float_value();
                            let rx = builder.build_float_div(ax.into_float_value(), len, "norm_x").unwrap();
                            let ry = builder.build_float_div(ay.into_float_value(), len, "norm_y").unwrap();
                            let rz = builder.build_float_div(az.into_float_value(), len, "norm_z").unwrap();
                            VarValues::Vec3(rx.into(), ry.into(), rz.into())
                        }
                    }
                }
                _ => unreachable!("VecUnary on non-vector type"),
            }
        }

        // -- Dot product (vec -> f64) --
        Inst::VecDot { lhs, rhs } => {
            let lhs_ty = kernel.var_type(*lhs).unwrap();
            let result = match lhs_ty {
                ValType::Vec2 => {
                    let (lx, ly) = get_vec2(val_map, lhs);
                    let (rx, ry) = get_vec2(val_map, rhs);
                    let xx = builder.build_float_mul(lx.into_float_value(), rx.into_float_value(), "dot_xx").unwrap();
                    let yy = builder.build_float_mul(ly.into_float_value(), ry.into_float_value(), "dot_yy").unwrap();
                    builder.build_float_add(xx, yy, "dot").unwrap()
                }
                ValType::Vec3 => {
                    let (lx, ly, lz) = get_vec3(val_map, lhs);
                    let (rx, ry, rz) = get_vec3(val_map, rhs);
                    let xx = builder.build_float_mul(lx.into_float_value(), rx.into_float_value(), "dot_xx").unwrap();
                    let yy = builder.build_float_mul(ly.into_float_value(), ry.into_float_value(), "dot_yy").unwrap();
                    let zz = builder.build_float_mul(lz.into_float_value(), rz.into_float_value(), "dot_zz").unwrap();
                    let sum = builder.build_float_add(xx, yy, "dot_xy").unwrap();
                    builder.build_float_add(sum, zz, "dot").unwrap()
                }
                _ => unreachable!("VecDot on non-vector type"),
            };
            VarValues::Scalar(result.into())
        }

        // -- Vector length (vec -> f64) --
        Inst::VecLength { arg } => {
            let arg_ty = kernel.var_type(*arg).unwrap();
            let dot = match arg_ty {
                ValType::Vec2 => {
                    let (ax, ay) = get_vec2(val_map, arg);
                    let xx = builder.build_float_mul(ax.into_float_value(), ax.into_float_value(), "len_xx").unwrap();
                    let yy = builder.build_float_mul(ay.into_float_value(), ay.into_float_value(), "len_yy").unwrap();
                    builder.build_float_add(xx, yy, "len_dot").unwrap()
                }
                ValType::Vec3 => {
                    let (ax, ay, az) = get_vec3(val_map, arg);
                    let xx = builder.build_float_mul(ax.into_float_value(), ax.into_float_value(), "len_xx").unwrap();
                    let yy = builder.build_float_mul(ay.into_float_value(), ay.into_float_value(), "len_yy").unwrap();
                    let zz = builder.build_float_mul(az.into_float_value(), az.into_float_value(), "len_zz").unwrap();
                    let sum = builder.build_float_add(xx, yy, "len_xy").unwrap();
                    builder.build_float_add(sum, zz, "len_dot").unwrap()
                }
                _ => unreachable!("VecLength on non-vector type"),
            };
            let len = call_f64_intrinsic(module, builder, "llvm.sqrt.f64", dot, f64_type);
            VarValues::Scalar(len)
        }

        // -- Cross product (vec3 x vec3 -> vec3) --
        Inst::VecCross { lhs, rhs } => {
            let (lx, ly, lz) = get_vec3(val_map, lhs);
            let (rx, ry, rz) = get_vec3(val_map, rhs);
            // cross.x = ly*rz - lz*ry
            let a = builder.build_float_mul(ly.into_float_value(), rz.into_float_value(), "cross_a").unwrap();
            let b = builder.build_float_mul(lz.into_float_value(), ry.into_float_value(), "cross_b").unwrap();
            let cx = builder.build_float_sub(a, b, "cross_x").unwrap();
            // cross.y = lz*rx - lx*rz
            let a = builder.build_float_mul(lz.into_float_value(), rx.into_float_value(), "cross_c").unwrap();
            let b = builder.build_float_mul(lx.into_float_value(), rz.into_float_value(), "cross_d").unwrap();
            let cy = builder.build_float_sub(a, b, "cross_y").unwrap();
            // cross.z = lx*ry - ly*rx
            let a = builder.build_float_mul(lx.into_float_value(), ry.into_float_value(), "cross_e").unwrap();
            let b = builder.build_float_mul(ly.into_float_value(), rx.into_float_value(), "cross_f").unwrap();
            let cz = builder.build_float_sub(a, b, "cross_z").unwrap();
            VarValues::Vec3(cx.into(), cy.into(), cz.into())
        }
    }
}

/// Lower a component-wise vector binary operation on two f64 scalars.
fn lower_vec_bin_op(
    module: &Module<'static>,
    builder: &inkwell::builder::Builder<'static>,
    f64_type: inkwell::types::FloatType<'static>,
    op: VecBinOp,
    lhs: inkwell::values::FloatValue<'static>,
    rhs: inkwell::values::FloatValue<'static>,
) -> inkwell::values::FloatValue<'static> {
    match op {
        VecBinOp::Add => builder.build_float_add(lhs, rhs, "vadd").unwrap(),
        VecBinOp::Sub => builder.build_float_sub(lhs, rhs, "vsub").unwrap(),
        VecBinOp::Mul => builder.build_float_mul(lhs, rhs, "vmul").unwrap(),
        VecBinOp::Div => builder.build_float_div(lhs, rhs, "vdiv").unwrap(),
        VecBinOp::Min => call_f64_binary_intrinsic(module, builder, "llvm.minnum.f64", lhs, rhs, f64_type).into_float_value(),
        VecBinOp::Max => call_f64_binary_intrinsic(module, builder, "llvm.maxnum.f64", lhs, rhs, f64_type).into_float_value(),
    }
}

fn call_f64_binary_intrinsic(
    module: &Module<'static>,
    builder: &inkwell::builder::Builder<'static>,
    name: &str,
    lhs: inkwell::values::FloatValue<'static>,
    rhs: inkwell::values::FloatValue<'static>,
    f64_type: inkwell::types::FloatType<'static>,
) -> BasicValueEnum<'static> {
    let intrinsic = inkwell::intrinsics::Intrinsic::find(name)
        .unwrap_or_else(|| panic!("intrinsic {name} not found"));
    let decl = intrinsic.get_declaration(module, &[f64_type.into()]).unwrap();
    builder.build_call(decl, &[lhs.into(), rhs.into()], "minmax")
        .unwrap()
        .try_as_basic_value()
        .unwrap_basic()
}

fn call_i32_binary_intrinsic(
    module: &Module<'static>,
    builder: &inkwell::builder::Builder<'static>,
    name: &str,
    lhs: inkwell::values::IntValue<'static>,
    rhs: inkwell::values::IntValue<'static>,
    i32_type: inkwell::types::IntType<'static>,
) -> BasicValueEnum<'static> {
    let intrinsic = inkwell::intrinsics::Intrinsic::find(name)
        .unwrap_or_else(|| panic!("intrinsic {name} not found"));
    let decl = intrinsic.get_declaration(module, &[i32_type.into()]).unwrap();
    builder.build_call(decl, &[lhs.into(), rhs.into()], "minmax")
        .unwrap()
        .try_as_basic_value()
        .unwrap_basic()
}

fn call_f64_intrinsic(
    module: &Module<'static>,
    builder: &inkwell::builder::Builder<'static>,
    name: &str,
    arg: inkwell::values::FloatValue<'static>,
    f64_type: inkwell::types::FloatType<'static>,
) -> BasicValueEnum<'static> {
    let intrinsic = inkwell::intrinsics::Intrinsic::find(name)
        .unwrap_or_else(|| panic!("intrinsic {name} not found"));
    let decl = intrinsic.get_declaration(module, &[f64_type.into()]).unwrap();
    builder.build_call(decl, &[arg.into()], &name[5..name.len()-4])  // strip "llvm." and ".f64"
        .unwrap()
        .try_as_basic_value()
        .unwrap_basic()
}
