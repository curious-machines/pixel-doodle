use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::passes::PassBuilderOptions;
use inkwell::targets::{CodeModel, InitializationConfig, RelocMode, Target, TargetMachine};
use inkwell::values::{BasicValueEnum, FunctionValue};
use inkwell::{FloatPredicate, IntPredicate, OptimizationLevel};

use crate::jit::{CompiledKernel, JitBackend, TileKernelFn};
use crate::kernel_ir::*;

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
        .filter(|f| !f.contains("sve"))
        .collect::<Vec<_>>()
        .join(",");
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
            .get_function::<unsafe extern "C" fn(*mut u32, u32, u32, f64, f64, f64, f64, u32, u32)>(
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

    let color = lower_kernel_body(context, module, &builder, function, kernel, cx, cy);

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
) -> inkwell::values::IntValue<'static> {
    use std::collections::HashMap;

    let mut val_map: HashMap<Var, BasicValueEnum<'static>> = HashMap::new();
    val_map.insert(Var(0), cx.into());
    val_map.insert(Var(1), cy.into());

    lower_body_items(context, module, builder, function, kernel, &kernel.body, &mut val_map);

    val_map[&kernel.emit].into_int_value()
}

fn lower_body_items(
    context: &'static Context,
    module: &Module<'static>,
    builder: &inkwell::builder::Builder<'static>,
    function: FunctionValue<'static>,
    kernel: &Kernel,
    body: &[BodyItem],
    val_map: &mut std::collections::HashMap<Var, BasicValueEnum<'static>>,
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
    val_map: &mut std::collections::HashMap<Var, BasicValueEnum<'static>>,
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

    // Create phi nodes for carry variables
    let mut phi_nodes = Vec::new();
    for cv in &w.carry {
        let llvm_ty: inkwell::types::BasicTypeEnum = match cv.binding.ty {
            ScalarType::F64 => f64_type.into(),
            ScalarType::U32 => i32_type.into(),
            ScalarType::Bool => i1_type.into(),
        };
        let phi = builder.build_phi(llvm_ty, &cv.binding.name).unwrap();
        // Add incoming from pre-block (initial value)
        let init_val = val_map[&cv.init];
        phi.add_incoming(&[(&init_val, pre_block)]);
        // Map carry var to phi value
        val_map.insert(cv.binding.var, phi.as_basic_value());
        phi_nodes.push(phi);
    }

    // Lower cond_body
    for stmt in &w.cond_body {
        let v = lower_inst(context, module, builder, kernel, &stmt.inst, &stmt.binding, val_map);
        val_map.insert(stmt.binding.var, v);
    }

    // Branch on cond
    let cond_val = val_map[&w.cond].into_int_value();
    builder.build_conditional_branch(cond_val, loop_body, loop_exit).unwrap();

    // -- Loop body: compute next values, branch back to header --
    builder.position_at_end(loop_body);

    for stmt in &w.body {
        let v = lower_inst(context, module, builder, kernel, &stmt.inst, &stmt.binding, val_map);
        val_map.insert(stmt.binding.var, v);
    }

    // Add incoming edges to phi nodes from loop body (yield values)
    let body_block = builder.get_insert_block().unwrap();
    for (i, phi) in phi_nodes.iter().enumerate() {
        let yield_val = val_map[&w.yields[i]];
        phi.add_incoming(&[(&yield_val, body_block)]);
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
    val_map: &std::collections::HashMap<Var, BasicValueEnum<'static>>,
) -> BasicValueEnum<'static> {
    let i32_type = context.i32_type();
    let f64_type = context.f64_type();
    let i1_type = context.bool_type();

    match inst {
        Inst::Const(c) => match c {
            Const::F64(v) => f64_type.const_float(*v).into(),
            Const::U32(v) => i32_type.const_int(*v as u64, false).into(),
            Const::Bool(v) => i1_type.const_int(if *v { 1 } else { 0 }, false).into(),
        },
        Inst::Binary { op, lhs, rhs } => {
            let l = val_map[lhs];
            let r = val_map[rhs];
            match (op, binding.ty) {
                (BinOp::Add, ScalarType::F64) => builder.build_float_add(l.into_float_value(), r.into_float_value(), "add").unwrap().into(),
                (BinOp::Sub, ScalarType::F64) => builder.build_float_sub(l.into_float_value(), r.into_float_value(), "sub").unwrap().into(),
                (BinOp::Mul, ScalarType::F64) => builder.build_float_mul(l.into_float_value(), r.into_float_value(), "mul").unwrap().into(),
                (BinOp::Div, ScalarType::F64) => builder.build_float_div(l.into_float_value(), r.into_float_value(), "div").unwrap().into(),
                (BinOp::Rem, ScalarType::F64) => builder.build_float_rem(l.into_float_value(), r.into_float_value(), "rem").unwrap().into(),
                (BinOp::Add, ScalarType::U32) => builder.build_int_add(l.into_int_value(), r.into_int_value(), "add").unwrap().into(),
                (BinOp::Sub, ScalarType::U32) => builder.build_int_sub(l.into_int_value(), r.into_int_value(), "sub").unwrap().into(),
                (BinOp::Mul, ScalarType::U32) => builder.build_int_mul(l.into_int_value(), r.into_int_value(), "mul").unwrap().into(),
                (BinOp::Div, ScalarType::U32) => builder.build_int_unsigned_div(l.into_int_value(), r.into_int_value(), "div").unwrap().into(),
                (BinOp::Rem, ScalarType::U32) => builder.build_int_unsigned_rem(l.into_int_value(), r.into_int_value(), "rem").unwrap().into(),
                (BinOp::BitAnd, _) | (BinOp::And, _) => builder.build_and(l.into_int_value(), r.into_int_value(), "and").unwrap().into(),
                (BinOp::BitOr, _) | (BinOp::Or, _) => builder.build_or(l.into_int_value(), r.into_int_value(), "or").unwrap().into(),
                (BinOp::BitXor, _) => builder.build_xor(l.into_int_value(), r.into_int_value(), "xor").unwrap().into(),
                (BinOp::Shl, _) => builder.build_left_shift(l.into_int_value(), r.into_int_value(), "shl").unwrap().into(),
                (BinOp::Shr, _) => builder.build_right_shift(l.into_int_value(), r.into_int_value(), false, "shr").unwrap().into(),
                (BinOp::Min, ScalarType::F64) => call_f64_binary_intrinsic(module, builder, "llvm.minnum.f64", l.into_float_value(), r.into_float_value(), f64_type),
                (BinOp::Max, ScalarType::F64) => call_f64_binary_intrinsic(module, builder, "llvm.maxnum.f64", l.into_float_value(), r.into_float_value(), f64_type),
                (BinOp::Min, ScalarType::U32) => call_i32_binary_intrinsic(module, builder, "llvm.umin.i32", l.into_int_value(), r.into_int_value(), i32_type),
                (BinOp::Max, ScalarType::U32) => call_i32_binary_intrinsic(module, builder, "llvm.umax.i32", l.into_int_value(), r.into_int_value(), i32_type),
                (BinOp::Atan2, ScalarType::F64) => call_libm_f64_binary(context, module, builder, "atan2", l.into_float_value(), r.into_float_value()),
                (BinOp::Pow, ScalarType::F64) => call_f64_binary_intrinsic(module, builder, "llvm.pow.f64", l.into_float_value(), r.into_float_value(), f64_type),
                _ => unreachable!("invalid binary op/type combination"),
            }
        }
        Inst::Unary { op, arg } => {
            let a = val_map[arg];
            match (op, binding.ty) {
                (UnaryOp::Neg, ScalarType::F64) => builder.build_float_neg(a.into_float_value(), "neg").unwrap().into(),
                (UnaryOp::Neg, ScalarType::U32) => builder.build_int_sub(i32_type.const_zero(), a.into_int_value(), "neg").unwrap().into(),
                (UnaryOp::Not, _) => builder.build_not(a.into_int_value(), "not").unwrap().into(),
                (UnaryOp::Abs, ScalarType::F64) => {
                    call_f64_intrinsic(module, builder, "llvm.fabs.f64", a.into_float_value(), f64_type)
                }
                (UnaryOp::Abs, ScalarType::U32) => a,
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
                (UnaryOp::Tan, _) => call_libm_f64(context, module, builder, "tan", a.into_float_value()),
                (UnaryOp::Asin, _) => call_libm_f64(context, module, builder, "asin", a.into_float_value()),
                (UnaryOp::Acos, _) => call_libm_f64(context, module, builder, "acos", a.into_float_value()),
                (UnaryOp::Atan, _) => call_libm_f64(context, module, builder, "atan", a.into_float_value()),
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
            }
        }
        Inst::Cmp { op, lhs, rhs } => {
            let l = val_map[lhs];
            let r = val_map[rhs];
            let operand_ty = kernel.var_type(*lhs).unwrap();
            match operand_ty {
                ScalarType::F64 => {
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
                ScalarType::U32 => {
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
            }
        }
        Inst::Conv { op, arg } => {
            let a = val_map[arg];
            match op {
                ConvOp::F64ToU32 => builder.build_float_to_signed_int(a.into_float_value(), i32_type, "conv").unwrap().into(),
                ConvOp::U32ToF64 => builder.build_unsigned_int_to_float(a.into_int_value(), f64_type, "conv").unwrap().into(),
            }
        }
        Inst::Select { cond, then_val, else_val } => {
            let c = val_map[cond].into_int_value();
            let t = val_map[then_val];
            let e = val_map[else_val];
            match binding.ty {
                ScalarType::F64 => {
                    builder.build_select(c, t.into_float_value(), e.into_float_value(), "sel").unwrap()
                }
                ScalarType::U32 | ScalarType::Bool => {
                    builder.build_select(c, t.into_int_value(), e.into_int_value(), "sel").unwrap()
                }
            }
        }
        Inst::PackArgb { r, g, b } => {
            let rv = val_map[r].into_int_value();
            let gv = val_map[g].into_int_value();
            let bv = val_map[b].into_int_value();
            let alpha = i32_type.const_int(0xFF000000, false);
            let r_sh = builder.build_left_shift(rv, i32_type.const_int(16, false), "r_sh").unwrap();
            let g_sh = builder.build_left_shift(gv, i32_type.const_int(8, false), "g_sh").unwrap();
            let color = builder.build_or(alpha, r_sh, "ar").unwrap();
            let color = builder.build_or(color, g_sh, "arg").unwrap();
            builder.build_or(color, bv, "argb").unwrap().into()
        }
    }
}

fn call_libm_f64(
    context: &'static Context,
    module: &Module<'static>,
    builder: &inkwell::builder::Builder<'static>,
    name: &str,
    arg: inkwell::values::FloatValue<'static>,
) -> BasicValueEnum<'static> {
    let f64_type = context.f64_type();
    let fn_type = f64_type.fn_type(&[f64_type.into()], false);
    let func = module.get_function(name)
        .unwrap_or_else(|| module.add_function(name, fn_type, Some(inkwell::module::Linkage::External)));
    builder.build_call(func, &[arg.into()], name)
        .unwrap()
        .try_as_basic_value()
        .left()
        .unwrap()
}

fn call_libm_f64_binary(
    context: &'static Context,
    module: &Module<'static>,
    builder: &inkwell::builder::Builder<'static>,
    name: &str,
    lhs: inkwell::values::FloatValue<'static>,
    rhs: inkwell::values::FloatValue<'static>,
) -> BasicValueEnum<'static> {
    let f64_type = context.f64_type();
    let fn_type = f64_type.fn_type(&[f64_type.into(), f64_type.into()], false);
    let func = module.get_function(name)
        .unwrap_or_else(|| module.add_function(name, fn_type, Some(inkwell::module::Linkage::External)));
    builder.build_call(func, &[lhs.into(), rhs.into()], name)
        .unwrap()
        .try_as_basic_value()
        .left()
        .unwrap()
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
        .left()
        .unwrap()
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
        .left()
        .unwrap()
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
        .left()
        .unwrap()
}
