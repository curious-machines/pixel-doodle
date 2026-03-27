use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::passes::PassBuilderOptions;
use inkwell::targets::{CodeModel, InitializationConfig, RelocMode, Target, TargetMachine};
use inkwell::values::{BasicValueEnum, FunctionValue};
use inkwell::{FloatPredicate, IntPredicate, OptimizationLevel};

use crate::jit::{CompiledKernel, CompiledSimKernel, JitBackend, SimTileKernelFn, TileKernelFn, UserArgSlot};
use crate::kernel_ir::*;

/// Decomposed vector values. Backends store vec vars as N scalar values (2, 3, or 4 components).
enum VarValues<'ctx> {
    Scalar(BasicValueEnum<'ctx>),
    Vec(Vec<BasicValueEnum<'ctx>>),    // 2, 3, or 4 components
    Mat(Vec<BasicValueEnum<'ctx>>),    // N*N scalars, column-major
    Array(Vec<BasicValueEnum<'ctx>>),  // flat list of scalar LLVM values
    Struct(Vec<BasicValueEnum<'ctx>>), // flat list of scalar LLVM values
}

fn get_scalar<'ctx>(val_map: &std::collections::HashMap<Var, VarValues<'ctx>>, var: &Var) -> BasicValueEnum<'ctx> {
    match &val_map[var] {
        VarValues::Scalar(v) => *v,
        _ => panic!("expected scalar value for {:?}", var),
    }
}

fn get_vec<'a, 'ctx>(val_map: &'a std::collections::HashMap<Var, VarValues<'ctx>>, var: &Var) -> &'a [BasicValueEnum<'ctx>] {
    match &val_map[var] {
        VarValues::Vec(v) => v,
        _ => panic!("expected vec value for {:?}", var),
    }
}

fn get_mat<'a, 'ctx>(val_map: &'a std::collections::HashMap<Var, VarValues<'ctx>>, var: &Var) -> &'a [BasicValueEnum<'ctx>] {
    match &val_map[var] {
        VarValues::Mat(v) => v,
        _ => panic!("expected mat value for {:?}", var),
    }
}

fn get_array<'a, 'ctx>(val_map: &'a std::collections::HashMap<Var, VarValues<'ctx>>, var: &Var) -> &'a [BasicValueEnum<'ctx>] {
    match &val_map[var] {
        VarValues::Array(v) => v,
        _ => panic!("expected array value for {:?}", var),
    }
}

fn get_struct<'a, 'ctx>(val_map: &'a std::collections::HashMap<Var, VarValues<'ctx>>, var: &Var) -> &'a [BasicValueEnum<'ctx>] {
    match &val_map[var] {
        VarValues::Struct(v) => v,
        _ => panic!("expected struct value for {:?}", var),
    }
}

/// Flatten a VarValues into a flat list of scalar LLVM values.
fn flatten_values<'ctx>(vv: &VarValues<'ctx>) -> Vec<BasicValueEnum<'ctx>> {
    match vv {
        VarValues::Scalar(v) => vec![*v],
        VarValues::Vec(vs) | VarValues::Mat(vs) | VarValues::Array(vs) | VarValues::Struct(vs) => vs.clone(),
    }
}

/// Reconstruct a VarValues from a flat list of scalars, guided by the ValType.
fn unflatten_values<'ctx>(ty: &ValType, flat: Vec<BasicValueEnum<'ctx>>) -> VarValues<'ctx> {
    match ty {
        ValType::Scalar(_) => {
            assert_eq!(flat.len(), 1);
            VarValues::Scalar(flat[0])
        }
        ValType::Vec { .. } => VarValues::Vec(flat),
        ValType::Mat { .. } => VarValues::Mat(flat),
        ValType::Array { .. } => VarValues::Array(flat),
        ValType::Struct(_) => VarValues::Struct(flat),
    }
}

/// Return the LLVM type for each scalar in the flat representation of a ValType.
fn flat_scalar_types<'ctx>(
    context: &'ctx Context,
    ty: &ValType,
    struct_defs: &[StructDef],
) -> Vec<inkwell::types::BasicTypeEnum<'ctx>> {
    match ty {
        ValType::Scalar(s) => vec![scalar_to_llvm_type(context, *s)],
        ValType::Vec { len, elem } => vec![scalar_to_llvm_type(context, *elem); *len as usize],
        ValType::Mat { size, elem } => vec![scalar_to_llvm_type(context, *elem); (*size as usize) * (*size as usize)],
        ValType::Array { elem, size } => {
            let elem_types = flat_scalar_types(context, elem, struct_defs);
            elem_types.iter().cycle().take(elem_types.len() * *size as usize).cloned().collect()
        }
        ValType::Struct(name) => {
            let sd = struct_defs.iter().find(|s| &s.name == name)
                .expect("unknown struct type");
            sd.fields.iter()
                .flat_map(|(_, fty)| flat_scalar_types(context, fty, struct_defs))
                .collect()
        }
    }
}

pub struct LlvmBackend;

/// Buffer context for simulation kernels — holds function params for buffer pointer arrays.
struct LlvmBufContext<'ctx> {
    p_width: inkwell::values::IntValue<'ctx>,
    p_buf_ptrs: inkwell::values::PointerValue<'ctx>,
    p_buf_out_ptrs: inkwell::values::PointerValue<'ctx>,
}

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

struct LlvmSimKernel {
    _engine: inkwell::execution_engine::ExecutionEngine<'static>,
    fn_ptr: SimTileKernelFn,
}

unsafe impl Send for LlvmSimKernel {}
unsafe impl Sync for LlvmSimKernel {}

impl CompiledSimKernel for LlvmSimKernel {
    fn function_ptr(&self) -> SimTileKernelFn {
        self.fn_ptr
    }
}

impl JitBackend for LlvmBackend {
    fn compile(&self, kernel: &Kernel, user_args: &[UserArgSlot]) -> Box<dyn CompiledKernel> {
        Box::new(compile_kernel(kernel, user_args))
    }
    fn compile_sim(&self, kernel: &Kernel, user_args: &[UserArgSlot]) -> Box<dyn CompiledSimKernel> {
        Box::new(compile_sim_kernel(kernel, user_args))
    }
}

fn compile_kernel(kernel: &Kernel, user_args: &[UserArgSlot]) -> LlvmKernel {
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
            ptr_type.into(), // user_args: *const u8
            ptr_type.into(), // tex_slots: *const TextureSlot
        ],
        false,
    );

    let function = module.add_function(&kernel.name, fn_type, None);
    build_tile_loop(context, &module, function, kernel, user_args);

    let pass_options = PassBuilderOptions::create();
    module
        .run_passes("default<O3>", &machine, pass_options)
        .expect("failed to run LLVM optimization passes");

    let engine = module
        .create_jit_execution_engine(OptimizationLevel::Aggressive)
        .unwrap();

    register_llvm_tex_helpers(&engine, &module);

    let fn_ptr = unsafe {
        engine
            .get_function::<unsafe extern "C" fn(*mut u32, u32, u32, f64, f64, f64, f64, u32, u32, u32, f64, *const u8, *const u8)>(
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
    user_args: &[UserArgSlot],
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
    let p_height = function.get_nth_param(2).unwrap().into_int_value();
    let p_x_min = function.get_nth_param(3).unwrap().into_float_value();
    let p_y_min = function.get_nth_param(4).unwrap().into_float_value();
    let p_x_step = function.get_nth_param(5).unwrap().into_float_value();
    let p_y_step = function.get_nth_param(6).unwrap().into_float_value();
    let p_row_start = function.get_nth_param(7).unwrap().into_int_value();
    let p_row_end = function.get_nth_param(8).unwrap().into_int_value();
    let p_sample_index = function.get_nth_param(9).unwrap().into_int_value();
    let p_time = function.get_nth_param(10).unwrap().into_float_value();
    let p_user_args = function.get_nth_param(11).unwrap().into_pointer_value();
    let _p_tex_slots = function.get_nth_param(12).unwrap().into_pointer_value();

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
    let color = lower_kernel_body(context, module, &builder, function, kernel, cx, cy, col, row, p_sample_index, p_time, p_width, p_height, p_user_args, user_args, _p_tex_slots);

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
    width: inkwell::values::IntValue<'static>,
    height: inkwell::values::IntValue<'static>,
    user_args_ptr: inkwell::values::PointerValue<'static>,
    user_args: &[UserArgSlot],
    tex_slots_ptr: inkwell::values::PointerValue<'static>,
) -> inkwell::values::IntValue<'static> {
    use std::collections::HashMap;

    let i64_type = context.i64_type();

    let mut val_map: HashMap<Var, VarValues<'static>> = HashMap::new();
    for param in &kernel.params {
        let val: BasicValueEnum<'static> = match param.name.as_str() {
            "x" => cx.into(),
            "y" => cy.into(),
            "px" => col.into(),
            "py" => row.into(),
            "sample_index" => sample_index.into(),
            "time" => time.into(),
            "width" => width.into(),
            "height" => height.into(),
            name => {
                let slot = user_args.iter().find(|s| s.name == name)
                    .unwrap_or_else(|| panic!("unknown kernel parameter: '{name}'"));
                if slot.ty.is_mat() {
                    panic!("LLVM backend does not support Mat user args (param '{name}')");
                }
                if slot.ty.is_array() {
                    panic!("LLVM backend does not support Array user args (param '{name}')");
                }
                if matches!(slot.ty, ValType::Struct(_)) {
                    panic!("LLVM backend does not support Struct user args (param '{name}')");
                }
                let offset = i64_type.const_int(slot.offset as u64, false);
                let addr = unsafe {
                    builder.build_gep(context.i8_type(), user_args_ptr, &[offset], &format!("arg_{name}_ptr")).unwrap()
                };
                let load_ty = scalar_to_llvm_type(context, slot.ty.element_scalar());
                builder.build_load(load_ty, addr, &format!("arg_{name}")).unwrap()
            }
        };
        val_map.insert(param.var, VarValues::Scalar(val));
    }

    lower_body_items(context, module, builder, function, kernel, &kernel.body, &mut val_map, None, tex_slots_ptr);

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
    buf_ctx: Option<&LlvmBufContext<'static>>,
    tex_slots_ptr: inkwell::values::PointerValue<'static>,
) {
    for item in body {
        match item {
            BodyItem::Stmt(stmt) => {
                let v = lower_inst(context, module, builder, kernel, &stmt.inst, &stmt.binding, val_map, buf_ctx, tex_slots_ptr);
                val_map.insert(stmt.binding.var, v);
            }
            BodyItem::While(w) => {
                lower_while(context, module, builder, function, kernel, w, val_map, buf_ctx, tex_slots_ptr);
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
    buf_ctx: Option<&LlvmBufContext<'static>>,
    tex_slots_ptr: inkwell::values::PointerValue<'static>,
) {
    // Create blocks
    let pre_block = builder.get_insert_block().unwrap();
    let loop_header = context.append_basic_block(function, "while_header");
    let loop_body = context.append_basic_block(function, "while_body");
    let loop_exit = context.append_basic_block(function, "while_exit");

    // Branch from current block to loop header
    builder.build_unconditional_branch(loop_header).unwrap();

    // -- Loop header: phi nodes for carry vars, then cond_body, then branch --
    builder.position_at_end(loop_header);

    // Create phi nodes for carry variables, expanding compound types into multiple phis
    // Each entry: (carry index, list of phi nodes for that carry var)
    let mut carry_phis: Vec<Vec<inkwell::values::PhiValue<'static>>> = Vec::new();
    for cv in &w.carry {
        let llvm_types = flat_scalar_types(context, &cv.binding.ty, &kernel.struct_defs);

        let mut phis = Vec::new();
        for (comp, llvm_ty) in llvm_types.iter().enumerate() {
            let name = if llvm_types.len() == 1 {
                cv.binding.name.clone()
            } else {
                format!("{}_{}", cv.binding.name, comp)
            };
            let phi = builder.build_phi(*llvm_ty, &name).unwrap();

            // Add incoming from pre-block (initial value)
            let init_flat = flatten_values(&val_map[&cv.init]);
            phi.add_incoming(&[(&init_flat[comp], pre_block)]);
            phis.push(phi);
        }

        // Map carry var to phi values
        let components: Vec<_> = phis.iter().map(|p| p.as_basic_value()).collect();
        let vv = unflatten_values(&cv.binding.ty, components);
        val_map.insert(cv.binding.var, vv);
        carry_phis.push(phis);
    }

    // Lower cond_body
    lower_body_items(context, module, builder, function, kernel, &w.cond_body, val_map, buf_ctx, tex_slots_ptr);

    // Branch on cond
    let cond_val = get_scalar(val_map, &w.cond).into_int_value();
    builder.build_conditional_branch(cond_val, loop_body, loop_exit).unwrap();

    // -- Loop body: compute next values, branch back to header --
    builder.position_at_end(loop_body);

    lower_body_items(context, module, builder, function, kernel, &w.body, val_map, buf_ctx, tex_slots_ptr);

    // Add incoming edges to phi nodes from loop body (yield values)
    let body_block = builder.get_insert_block().unwrap();
    for (i, phis) in carry_phis.iter().enumerate() {
        let yield_var = &w.yields[i];
        let yield_flat = flatten_values(&val_map[yield_var]);
        for (comp, phi) in phis.iter().enumerate() {
            phi.add_incoming(&[(&yield_flat[comp], body_block)]);
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
    buf_ctx: Option<&LlvmBufContext<'static>>,
    tex_slots_ptr: inkwell::values::PointerValue<'static>,
) -> VarValues<'static> {
    let i32_type = context.i32_type();
    let f64_type = context.f64_type();
    let i1_type = context.bool_type();

    match inst {
        Inst::Const(c) => VarValues::Scalar(match c {
            Const::F32(v) => context.f32_type().const_float(*v as f64).into(),
            Const::F64(v) => f64_type.const_float(*v).into(),
            Const::I8(v) => context.i8_type().const_int(*v as u64, true).into(),
            Const::U8(v) => context.i8_type().const_int(*v as u64, false).into(),
            Const::I16(v) => context.i16_type().const_int(*v as u64, true).into(),
            Const::U16(v) => context.i16_type().const_int(*v as u64, false).into(),
            Const::I32(v) => i32_type.const_int(*v as u64, true).into(),
            Const::U32(v) => i32_type.const_int(*v as u64, false).into(),
            Const::I64(v) => context.i64_type().const_int(*v as u64, true).into(),
            Const::U64(v) => context.i64_type().const_int(*v, false).into(),
            Const::Bool(v) => i1_type.const_int(if *v { 1 } else { 0 }, false).into(),
        }),
        Inst::Binary { op, lhs, rhs } => {
            let l = get_scalar(val_map, lhs);
            let r = get_scalar(val_map, rhs);
            let elem = binding.ty.element_scalar();
            VarValues::Scalar(if elem.is_float() {
                match op {
                    BinOp::Add => builder.build_float_add(l.into_float_value(), r.into_float_value(), "add").unwrap().into(),
                    BinOp::Sub => builder.build_float_sub(l.into_float_value(), r.into_float_value(), "sub").unwrap().into(),
                    BinOp::Mul => builder.build_float_mul(l.into_float_value(), r.into_float_value(), "mul").unwrap().into(),
                    BinOp::Div => builder.build_float_div(l.into_float_value(), r.into_float_value(), "div").unwrap().into(),
                    BinOp::Rem => builder.build_float_rem(l.into_float_value(), r.into_float_value(), "rem").unwrap().into(),
                    BinOp::Min => call_f64_binary_intrinsic(module, builder, "llvm.minnum.f64", l.into_float_value(), r.into_float_value(), f64_type),
                    BinOp::Max => call_f64_binary_intrinsic(module, builder, "llvm.maxnum.f64", l.into_float_value(), r.into_float_value(), f64_type),
                    BinOp::Atan2 => call_f64_binary_intrinsic(module, builder, "llvm.atan2.f64", l.into_float_value(), r.into_float_value(), f64_type),
                    BinOp::Pow => call_f64_binary_intrinsic(module, builder, "llvm.pow.f64", l.into_float_value(), r.into_float_value(), f64_type),
                    _ => unreachable!("invalid float binary op {:?}", op),
                }
            } else if elem.is_integer() {
                let int_ty = scalar_to_llvm_int_type(context, elem);
                match op {
                    BinOp::Add => builder.build_int_add(l.into_int_value(), r.into_int_value(), "add").unwrap().into(),
                    BinOp::Sub => builder.build_int_sub(l.into_int_value(), r.into_int_value(), "sub").unwrap().into(),
                    BinOp::Mul => builder.build_int_mul(l.into_int_value(), r.into_int_value(), "mul").unwrap().into(),
                    BinOp::Div if elem.is_signed() => builder.build_int_signed_div(l.into_int_value(), r.into_int_value(), "div").unwrap().into(),
                    BinOp::Div => builder.build_int_unsigned_div(l.into_int_value(), r.into_int_value(), "div").unwrap().into(),
                    BinOp::Rem if elem.is_signed() => builder.build_int_signed_rem(l.into_int_value(), r.into_int_value(), "rem").unwrap().into(),
                    BinOp::Rem => builder.build_int_unsigned_rem(l.into_int_value(), r.into_int_value(), "rem").unwrap().into(),
                    BinOp::BitAnd | BinOp::And => builder.build_and(l.into_int_value(), r.into_int_value(), "and").unwrap().into(),
                    BinOp::BitOr | BinOp::Or => builder.build_or(l.into_int_value(), r.into_int_value(), "or").unwrap().into(),
                    BinOp::BitXor => builder.build_xor(l.into_int_value(), r.into_int_value(), "xor").unwrap().into(),
                    BinOp::Shl => builder.build_left_shift(l.into_int_value(), r.into_int_value(), "shl").unwrap().into(),
                    BinOp::Shr => builder.build_right_shift(l.into_int_value(), r.into_int_value(), elem.is_signed(), "shr").unwrap().into(),
                    BinOp::Min if elem.is_signed() => {
                        let name = format!("llvm.smin.i{}", elem.byte_size() * 8);
                        call_int_binary_intrinsic(module, builder, &name, l.into_int_value(), r.into_int_value(), int_ty)
                    }
                    BinOp::Min => {
                        let name = format!("llvm.umin.i{}", elem.byte_size() * 8);
                        call_int_binary_intrinsic(module, builder, &name, l.into_int_value(), r.into_int_value(), int_ty)
                    }
                    BinOp::Max if elem.is_signed() => {
                        let name = format!("llvm.smax.i{}", elem.byte_size() * 8);
                        call_int_binary_intrinsic(module, builder, &name, l.into_int_value(), r.into_int_value(), int_ty)
                    }
                    BinOp::Max => {
                        let name = format!("llvm.umax.i{}", elem.byte_size() * 8);
                        call_int_binary_intrinsic(module, builder, &name, l.into_int_value(), r.into_int_value(), int_ty)
                    }
                    BinOp::Hash => {
                        let l = l.into_int_value();
                        let r = r.into_int_value();
                        let hash_k = i32_type.const_int(0x45d9f3b, false);
                        let sixteen = i32_type.const_int(16, false);
                        let h = builder.build_int_mul(l, hash_k, "hash1").unwrap();
                        let h = builder.build_int_add(h, r, "hash2").unwrap();
                        let h_s = builder.build_right_shift(h, sixteen, false, "hash_shr1").unwrap();
                        let h = builder.build_xor(h, h_s, "hash3").unwrap();
                        let h = builder.build_int_mul(h, hash_k, "hash4").unwrap();
                        let h_s = builder.build_right_shift(h, sixteen, false, "hash_shr2").unwrap();
                        builder.build_xor(h, h_s, "hash5").unwrap().into()
                    }
                    _ => unreachable!("invalid integer binary op {:?}", op),
                }
            } else {
                // Bool
                match op {
                    BinOp::BitAnd | BinOp::And => builder.build_and(l.into_int_value(), r.into_int_value(), "and").unwrap().into(),
                    BinOp::BitOr | BinOp::Or => builder.build_or(l.into_int_value(), r.into_int_value(), "or").unwrap().into(),
                    BinOp::BitXor => builder.build_xor(l.into_int_value(), r.into_int_value(), "xor").unwrap().into(),
                    _ => unreachable!("invalid bool binary op {:?}", op),
                }
            })
        }
        Inst::Unary { op, arg } => {
            let a = get_scalar(val_map, arg);
            let elem = binding.ty.element_scalar();
            VarValues::Scalar(if elem.is_float() {
                match op {
                    UnaryOp::Neg => builder.build_float_neg(a.into_float_value(), "neg").unwrap().into(),
                    UnaryOp::Abs => call_f64_intrinsic(module, builder, "llvm.fabs.f64", a.into_float_value(), f64_type),
                    UnaryOp::Sqrt => call_f64_intrinsic(module, builder, "llvm.sqrt.f64", a.into_float_value(), f64_type),
                    UnaryOp::Floor => call_f64_intrinsic(module, builder, "llvm.floor.f64", a.into_float_value(), f64_type),
                    UnaryOp::Ceil => call_f64_intrinsic(module, builder, "llvm.ceil.f64", a.into_float_value(), f64_type),
                    UnaryOp::Sin => call_f64_intrinsic(module, builder, "llvm.sin.f64", a.into_float_value(), f64_type),
                    UnaryOp::Cos => call_f64_intrinsic(module, builder, "llvm.cos.f64", a.into_float_value(), f64_type),
                    UnaryOp::Tan => call_f64_intrinsic(module, builder, "llvm.tan.f64", a.into_float_value(), f64_type),
                    UnaryOp::Asin => call_f64_intrinsic(module, builder, "llvm.asin.f64", a.into_float_value(), f64_type),
                    UnaryOp::Acos => call_f64_intrinsic(module, builder, "llvm.acos.f64", a.into_float_value(), f64_type),
                    UnaryOp::Atan => call_f64_intrinsic(module, builder, "llvm.atan.f64", a.into_float_value(), f64_type),
                    UnaryOp::Exp => call_f64_intrinsic(module, builder, "llvm.exp.f64", a.into_float_value(), f64_type),
                    UnaryOp::Exp2 => call_f64_intrinsic(module, builder, "llvm.exp2.f64", a.into_float_value(), f64_type),
                    UnaryOp::Log => call_f64_intrinsic(module, builder, "llvm.log.f64", a.into_float_value(), f64_type),
                    UnaryOp::Log2 => call_f64_intrinsic(module, builder, "llvm.log2.f64", a.into_float_value(), f64_type),
                    UnaryOp::Log10 => call_f64_intrinsic(module, builder, "llvm.log10.f64", a.into_float_value(), f64_type),
                    UnaryOp::Round => call_f64_intrinsic(module, builder, "llvm.round.f64", a.into_float_value(), f64_type),
                    UnaryOp::Trunc => call_f64_intrinsic(module, builder, "llvm.trunc.f64", a.into_float_value(), f64_type),
                    UnaryOp::Fract => {
                        let floored = call_f64_intrinsic(module, builder, "llvm.floor.f64", a.into_float_value(), f64_type);
                        builder.build_float_sub(a.into_float_value(), floored.into_float_value(), "fract").unwrap().into()
                    }
                    UnaryOp::Not => unreachable!("Not is not valid for floats"),
                }
            } else if elem.is_integer() {
                let int_ty = scalar_to_llvm_int_type(context, elem);
                match op {
                    UnaryOp::Neg => builder.build_int_sub(int_ty.const_zero(), a.into_int_value(), "neg").unwrap().into(),
                    UnaryOp::Not => builder.build_not(a.into_int_value(), "not").unwrap().into(),
                    UnaryOp::Abs if elem.is_unsigned() => a, // unsigned abs is identity
                    UnaryOp::Abs => {
                        // signed abs: val < 0 ? -val : val
                        let zero = int_ty.const_zero();
                        let is_neg = builder.build_int_compare(IntPredicate::SLT, a.into_int_value(), zero, "is_neg").unwrap();
                        let negated = builder.build_int_sub(zero, a.into_int_value(), "negated").unwrap();
                        builder.build_select(is_neg, negated, a.into_int_value(), "abs").unwrap()
                    }
                    _ => unreachable!("invalid integer unary op {:?}", op),
                }
            } else {
                // Bool
                match op {
                    UnaryOp::Not => builder.build_not(a.into_int_value(), "not").unwrap().into(),
                    _ => unreachable!("invalid bool unary op {:?}", op),
                }
            })
        }
        Inst::Cmp { op, lhs, rhs } => {
            let l = get_scalar(val_map, lhs);
            let r = get_scalar(val_map, rhs);
            let operand_ty = kernel.var_type(*lhs).unwrap();
            let operand_scalar = operand_ty.element_scalar();
            VarValues::Scalar(if operand_scalar.is_float() {
                let pred = match op {
                    CmpOp::Eq => FloatPredicate::OEQ,
                    CmpOp::Ne => FloatPredicate::ONE,
                    CmpOp::Lt => FloatPredicate::OLT,
                    CmpOp::Le => FloatPredicate::OLE,
                    CmpOp::Gt => FloatPredicate::OGT,
                    CmpOp::Ge => FloatPredicate::OGE,
                };
                builder.build_float_compare(pred, l.into_float_value(), r.into_float_value(), "cmp").unwrap().into()
            } else if operand_scalar.is_integer() {
                let pred = match (op, operand_scalar.is_signed()) {
                    (CmpOp::Eq, _) => IntPredicate::EQ,
                    (CmpOp::Ne, _) => IntPredicate::NE,
                    (CmpOp::Lt, true) => IntPredicate::SLT,
                    (CmpOp::Le, true) => IntPredicate::SLE,
                    (CmpOp::Gt, true) => IntPredicate::SGT,
                    (CmpOp::Ge, true) => IntPredicate::SGE,
                    (CmpOp::Lt, false) => IntPredicate::ULT,
                    (CmpOp::Le, false) => IntPredicate::ULE,
                    (CmpOp::Gt, false) => IntPredicate::UGT,
                    (CmpOp::Ge, false) => IntPredicate::UGE,
                };
                builder.build_int_compare(pred, l.into_int_value(), r.into_int_value(), "cmp").unwrap().into()
            } else {
                // Bool: only Eq/Ne make sense
                let pred = match op {
                    CmpOp::Eq => IntPredicate::EQ,
                    CmpOp::Ne => IntPredicate::NE,
                    _ => unreachable!("invalid comparison op {:?} for bool", op),
                };
                builder.build_int_compare(pred, l.into_int_value(), r.into_int_value(), "cmp").unwrap().into()
            })
        }
        Inst::Conv { op, arg } => {
            let a = get_scalar(val_map, arg);
            let from = op.from;
            let to = op.to;
            VarValues::Scalar(if op.norm {
                // Normalizing conversion: unsigned int -> float, divide by 2^bits
                assert!(from.is_unsigned() && to.is_float(), "norm conv requires unsigned->float");
                let float_ty = if to == ScalarType::F64 { context.f64_type() } else { context.f32_type() };
                let f = builder.build_unsigned_int_to_float(a.into_int_value(), float_ty, "conv_f").unwrap();
                let max_val = (1u128 << (from.byte_size() * 8)) as f64;
                let recip = float_ty.const_float(1.0 / max_val);
                builder.build_float_mul(f, recip, "norm").unwrap().into()
            } else if from.is_float() && to.is_float() {
                // Float to float
                let from_size = from.byte_size();
                let to_size = to.byte_size();
                let target_ty = if to == ScalarType::F64 { context.f64_type() } else { context.f32_type() };
                if to_size > from_size {
                    builder.build_float_ext(a.into_float_value(), target_ty, "conv").unwrap().into()
                } else if to_size < from_size {
                    builder.build_float_trunc(a.into_float_value(), target_ty, "conv").unwrap().into()
                } else {
                    a // same size float-to-float is identity
                }
            } else if from.is_float() && to.is_signed() {
                let int_ty = scalar_to_llvm_int_type(context, to);
                builder.build_float_to_signed_int(a.into_float_value(), int_ty, "conv").unwrap().into()
            } else if from.is_float() && to.is_unsigned() {
                // Convert via signed to avoid UB on negative values.
                // fptoui is undefined for negatives; fptosi + bitcast wraps.
                let int_ty = scalar_to_llvm_int_type(context, to);
                builder.build_float_to_signed_int(a.into_float_value(), int_ty, "conv").unwrap().into()
            } else if from.is_signed() && to.is_float() {
                let float_ty = if to == ScalarType::F64 { context.f64_type() } else { context.f32_type() };
                builder.build_signed_int_to_float(a.into_int_value(), float_ty, "conv").unwrap().into()
            } else if from.is_unsigned() && to.is_float() {
                let float_ty = if to == ScalarType::F64 { context.f64_type() } else { context.f32_type() };
                builder.build_unsigned_int_to_float(a.into_int_value(), float_ty, "conv").unwrap().into()
            } else if from.is_integer() && to.is_integer() {
                let from_size = from.byte_size();
                let to_size = to.byte_size();
                if from_size == to_size {
                    a // same bit width, reinterpret (e.g. i32 <-> u32)
                } else if to_size < from_size {
                    let int_ty = scalar_to_llvm_int_type(context, to);
                    builder.build_int_truncate(a.into_int_value(), int_ty, "conv").unwrap().into()
                } else if from.is_signed() {
                    let int_ty = scalar_to_llvm_int_type(context, to);
                    builder.build_int_s_extend(a.into_int_value(), int_ty, "conv").unwrap().into()
                } else {
                    let int_ty = scalar_to_llvm_int_type(context, to);
                    builder.build_int_z_extend(a.into_int_value(), int_ty, "conv").unwrap().into()
                }
            } else {
                unreachable!("unsupported conv {:?}", op)
            })
        }
        Inst::Select { cond, then_val, else_val } => {
            let c = get_scalar(val_map, cond).into_int_value();
            match &binding.ty {
                ValType::Mat { .. } => {
                    let t_comps = get_mat(val_map, then_val);
                    let e_comps = get_mat(val_map, else_val);
                    let results: Vec<_> = t_comps.iter().zip(e_comps.iter()).enumerate().map(|(i, (t, e))| {
                        let name = format!("sel_{}", i);
                        if binding.ty.element_scalar().is_float() {
                            builder.build_select(c, t.into_float_value(), e.into_float_value(), &name).unwrap()
                        } else {
                            builder.build_select(c, t.into_int_value(), e.into_int_value(), &name).unwrap()
                        }
                    }).collect();
                    VarValues::Mat(results)
                }
                ValType::Vec { .. } => {
                    let t_comps = get_vec(val_map, then_val);
                    let e_comps = get_vec(val_map, else_val);
                    let results: Vec<_> = t_comps.iter().zip(e_comps.iter()).enumerate().map(|(i, (t, e))| {
                        let name = format!("sel_{}", i);
                        if binding.ty.element_scalar().is_float() {
                            builder.build_select(c, t.into_float_value(), e.into_float_value(), &name).unwrap()
                        } else {
                            builder.build_select(c, t.into_int_value(), e.into_int_value(), &name).unwrap()
                        }
                    }).collect();
                    VarValues::Vec(results)
                }
                ValType::Scalar(s) if s.is_float() => {
                    let t = get_scalar(val_map, then_val);
                    let e = get_scalar(val_map, else_val);
                    VarValues::Scalar(builder.build_select(c, t.into_float_value(), e.into_float_value(), "sel").unwrap())
                }
                ValType::Scalar(_) => {
                    // All integer types and bool
                    let t = get_scalar(val_map, then_val);
                    let e = get_scalar(val_map, else_val);
                    VarValues::Scalar(builder.build_select(c, t.into_int_value(), e.into_int_value(), "sel").unwrap())
                }
                ValType::Array { .. } => {
                    let t_comps = get_array(val_map, then_val);
                    let e_comps = get_array(val_map, else_val);
                    let results: Vec<_> = t_comps.iter().zip(e_comps.iter()).enumerate().map(|(i, (t, e))| {
                        let name = format!("sel_{}", i);
                        if t.is_float_value() {
                            builder.build_select(c, t.into_float_value(), e.into_float_value(), &name).unwrap()
                        } else {
                            builder.build_select(c, t.into_int_value(), e.into_int_value(), &name).unwrap()
                        }
                    }).collect();
                    VarValues::Array(results)
                }
                ValType::Struct(_) => {
                    let t_comps = get_struct(val_map, then_val);
                    let e_comps = get_struct(val_map, else_val);
                    let results: Vec<_> = t_comps.iter().zip(e_comps.iter()).enumerate().map(|(i, (t, e))| {
                        let name = format!("sel_{}", i);
                        if t.is_float_value() {
                            builder.build_select(c, t.into_float_value(), e.into_float_value(), &name).unwrap()
                        } else {
                            builder.build_select(c, t.into_int_value(), e.into_int_value(), &name).unwrap()
                        }
                    }).collect();
                    VarValues::Struct(results)
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
        Inst::MakeVec(components) => {
            let vals: Vec<_> = components.iter().map(|v| get_scalar(val_map, v)).collect();
            VarValues::Vec(vals)
        }

        // -- Component extraction --
        Inst::VecExtract { vec, index } => {
            let components = get_vec(val_map, vec);
            VarValues::Scalar(components[*index as usize])
        }

        // -- Component-wise binary --
        Inst::VecBinary { op, lhs, rhs } => {
            let l_comps = get_vec(val_map, lhs);
            let r_comps = get_vec(val_map, rhs);
            let results: Vec<_> = l_comps.iter().zip(r_comps.iter()).map(|(l, r)| {
                let res = lower_vec_bin_op(module, builder, f64_type, *op, l.into_float_value(), r.into_float_value());
                res.into()
            }).collect();
            VarValues::Vec(results)
        }

        // -- Scalar-vector multiply --
        Inst::VecScale { scalar, vec } => {
            let s = get_scalar(val_map, scalar).into_float_value();
            let v_comps = get_vec(val_map, vec);
            let results: Vec<_> = v_comps.iter().enumerate().map(|(i, v)| {
                let name = format!("scale_{}", i);
                let r: BasicValueEnum = builder.build_float_mul(s, v.into_float_value(), &name).unwrap().into();
                r
            }).collect();
            VarValues::Vec(results)
        }

        // -- Vector unary --
        Inst::VecUnary { op, arg } => {
            let a_comps = get_vec(val_map, arg);
            match op {
                VecUnaryOp::Neg => {
                    let results: Vec<_> = a_comps.iter().enumerate().map(|(i, a)| {
                        let name = format!("neg_{}", i);
                        let r: BasicValueEnum = builder.build_float_neg(a.into_float_value(), &name).unwrap().into();
                        r
                    }).collect();
                    VarValues::Vec(results)
                }
                VecUnaryOp::Abs => {
                    let results: Vec<_> = a_comps.iter().map(|a| {
                        call_f64_intrinsic(module, builder, "llvm.fabs.f64", a.into_float_value(), f64_type)
                    }).collect();
                    VarValues::Vec(results)
                }
                VecUnaryOp::Normalize => {
                    // length = sqrt(sum of squares)
                    let mut dot = builder.build_float_mul(
                        a_comps[0].into_float_value(), a_comps[0].into_float_value(), "sq_0"
                    ).unwrap();
                    for i in 1..a_comps.len() {
                        let sq = builder.build_float_mul(
                            a_comps[i].into_float_value(), a_comps[i].into_float_value(), &format!("sq_{}", i)
                        ).unwrap();
                        dot = builder.build_float_add(dot, sq, &format!("dot_{}", i)).unwrap();
                    }
                    let len = call_f64_intrinsic(module, builder, "llvm.sqrt.f64", dot, f64_type).into_float_value();
                    let results: Vec<_> = a_comps.iter().enumerate().map(|(i, a)| {
                        let name = format!("norm_{}", i);
                        let r: BasicValueEnum = builder.build_float_div(a.into_float_value(), len, &name).unwrap().into();
                        r
                    }).collect();
                    VarValues::Vec(results)
                }
            }
        }

        // -- Dot product (vec -> f64) --
        Inst::VecDot { lhs, rhs } => {
            let l_comps = get_vec(val_map, lhs);
            let r_comps = get_vec(val_map, rhs);
            let mut sum = builder.build_float_mul(
                l_comps[0].into_float_value(), r_comps[0].into_float_value(), "dot_0"
            ).unwrap();
            for i in 1..l_comps.len() {
                let prod = builder.build_float_mul(
                    l_comps[i].into_float_value(), r_comps[i].into_float_value(), &format!("dot_{}", i)
                ).unwrap();
                sum = builder.build_float_add(sum, prod, &format!("dot_sum_{}", i)).unwrap();
            }
            VarValues::Scalar(sum.into())
        }

        // -- Vector length (vec -> f64) --
        Inst::VecLength { arg } => {
            let a_comps = get_vec(val_map, arg);
            let mut dot = builder.build_float_mul(
                a_comps[0].into_float_value(), a_comps[0].into_float_value(), "len_sq_0"
            ).unwrap();
            for i in 1..a_comps.len() {
                let sq = builder.build_float_mul(
                    a_comps[i].into_float_value(), a_comps[i].into_float_value(), &format!("len_sq_{}", i)
                ).unwrap();
                dot = builder.build_float_add(dot, sq, &format!("len_sum_{}", i)).unwrap();
            }
            let len = call_f64_intrinsic(module, builder, "llvm.sqrt.f64", dot, f64_type);
            VarValues::Scalar(len)
        }

        // -- Cross product (vec3 x vec3 -> vec3) --
        Inst::VecCross { lhs, rhs } => {
            let l = get_vec(val_map, lhs);
            let r = get_vec(val_map, rhs);
            assert_eq!(l.len(), 3, "VecCross requires vec3");
            assert_eq!(r.len(), 3, "VecCross requires vec3");
            let (lx, ly, lz) = (l[0].into_float_value(), l[1].into_float_value(), l[2].into_float_value());
            let (rx, ry, rz) = (r[0].into_float_value(), r[1].into_float_value(), r[2].into_float_value());
            // cross.x = ly*rz - lz*ry
            let a = builder.build_float_mul(ly, rz, "cross_a").unwrap();
            let b = builder.build_float_mul(lz, ry, "cross_b").unwrap();
            let cx: BasicValueEnum = builder.build_float_sub(a, b, "cross_x").unwrap().into();
            // cross.y = lz*rx - lx*rz
            let a = builder.build_float_mul(lz, rx, "cross_c").unwrap();
            let b = builder.build_float_mul(lx, rz, "cross_d").unwrap();
            let cy: BasicValueEnum = builder.build_float_sub(a, b, "cross_y").unwrap().into();
            // cross.z = lx*ry - ly*rx
            let a = builder.build_float_mul(lx, ry, "cross_e").unwrap();
            let b = builder.build_float_mul(ly, rx, "cross_f").unwrap();
            let cz: BasicValueEnum = builder.build_float_sub(a, b, "cross_z").unwrap().into();
            VarValues::Vec(vec![cx, cy, cz])
        }

        // -- Matrix construction --
        Inst::MakeMat(cols) => {
            let mut components = Vec::new();
            for col_var in cols {
                let col = get_vec(val_map, col_var);
                components.extend_from_slice(col);
            }
            VarValues::Mat(components)
        }

        // -- Column extraction --
        Inst::MatCol { mat, index } => {
            let m = get_mat(val_map, mat);
            let size = match &binding.ty {
                ValType::Vec { len, .. } => *len as usize,
                _ => panic!("MatCol result must be a vec type"),
            };
            let start = (*index as usize) * size;
            VarValues::Vec(m[start..start + size].to_vec())
        }

        // -- Matrix transpose --
        Inst::MatTranspose { arg } => {
            let m = get_mat(val_map, arg).to_vec();
            let size = match &binding.ty {
                ValType::Mat { size, .. } => *size as usize,
                _ => panic!("MatTranspose result must be a mat type"),
            };
            let mut result = Vec::with_capacity(size * size);
            for col in 0..size {
                for row in 0..size {
                    result.push(m[row * size + col]);
                }
            }
            VarValues::Mat(result)
        }

        // -- Matrix-vector multiply --
        Inst::MatMulVec { mat, vec } => {
            let m = get_mat(val_map, mat).to_vec();
            let v = get_vec(val_map, vec).to_vec();
            let size = v.len();
            let elem = binding.ty.element_scalar();
            // result[row] = sum over col of m[col*size+row] * v[col]
            let mut result: Vec<BasicValueEnum> = (0..size).map(|i| {
                llvm_emit_mul(builder, elem, m[i], v[0])
            }).collect();
            for col in 1..size {
                for row in 0..size {
                    let product = llvm_emit_mul(builder, elem, m[col * size + row], v[col]);
                    result[row] = llvm_emit_add(builder, elem, result[row], product);
                }
            }
            VarValues::Vec(result)
        }

        // -- Matrix-matrix multiply --
        Inst::MatMul { lhs, rhs } => {
            let lhs_m = get_mat(val_map, lhs).to_vec();
            let rhs_m = get_mat(val_map, rhs).to_vec();
            let size = match &binding.ty {
                ValType::Mat { size, .. } => *size as usize,
                _ => panic!("MatMul result must be a mat type"),
            };
            let elem = binding.ty.element_scalar();
            let mut result = Vec::with_capacity(size * size);
            for col in 0..size {
                let rhs_col: Vec<_> = (0..size).map(|row| rhs_m[col * size + row]).collect();
                let mut res_col: Vec<BasicValueEnum> = (0..size).map(|i| {
                    llvm_emit_mul(builder, elem, lhs_m[i], rhs_col[0])
                }).collect();
                for k in 1..size {
                    for row in 0..size {
                        let product = llvm_emit_mul(builder, elem, lhs_m[k * size + row], rhs_col[k]);
                        res_col[row] = llvm_emit_add(builder, elem, res_col[row], product);
                    }
                }
                result.extend(res_col);
            }
            VarValues::Mat(result)
        }

        Inst::BufLoad { buf, x, y } => {
            let ctx = buf_ctx.expect("BufLoad requires simulation context");
            let i64_type = context.i64_type();
            let f64_type = context.f64_type();
            let ptr_type = context.ptr_type(inkwell::AddressSpace::default());

            let xv = get_scalar(val_map, x).into_int_value();
            let yv = get_scalar(val_map, y).into_int_value();

            // flat index = y * width + x
            let row_off = builder.build_int_mul(yv, ctx.p_width, "bl_row").unwrap();
            let idx = builder.build_int_add(row_off, xv, "bl_idx").unwrap();
            let idx64 = builder.build_int_z_extend(idx, i64_type, "bl_idx64").unwrap();

            // Load buffer pointer from the pointer array
            let is_output = kernel.buffers[*buf as usize].is_output;
            let ptrs_base = if is_output { ctx.p_buf_out_ptrs } else { ctx.p_buf_ptrs };
            let buf_local_idx = if is_output {
                kernel.buffers.iter().take(*buf as usize).filter(|b| b.is_output).count()
            } else {
                kernel.buffers.iter().take(*buf as usize).filter(|b| !b.is_output).count()
            };
            let ptr_off = i64_type.const_int(buf_local_idx as u64, false);
            let buf_ptr_ptr = unsafe { builder.build_gep(ptr_type, ptrs_base, &[ptr_off], "bl_bpp").unwrap() };
            let buf_ptr = builder.build_load(ptr_type, buf_ptr_ptr, "bl_bp").unwrap().into_pointer_value();

            // Load f64 from buf_ptr[idx]
            let elem_ptr = unsafe { builder.build_gep(f64_type, buf_ptr, &[idx64], "bl_ep").unwrap() };
            let val = builder.build_load(f64_type, elem_ptr, "bl_val").unwrap();
            VarValues::Scalar(val)
        }
        Inst::BufStore { buf, x, y, val } => {
            let ctx = buf_ctx.expect("BufStore requires simulation context");
            let i32_type = context.i32_type();
            let i64_type = context.i64_type();
            let f64_type = context.f64_type();
            let ptr_type = context.ptr_type(inkwell::AddressSpace::default());

            let xv = get_scalar(val_map, x).into_int_value();
            let yv = get_scalar(val_map, y).into_int_value();
            let value = get_scalar(val_map, val).into_float_value();

            let row_off = builder.build_int_mul(yv, ctx.p_width, "bs_row").unwrap();
            let idx = builder.build_int_add(row_off, xv, "bs_idx").unwrap();
            let idx64 = builder.build_int_z_extend(idx, i64_type, "bs_idx64").unwrap();

            // Output buffer pointer
            let buf_local_idx = kernel.buffers.iter().take(*buf as usize).filter(|b| b.is_output).count();
            let ptr_off = i64_type.const_int(buf_local_idx as u64, false);
            let buf_ptr_ptr = unsafe { builder.build_gep(ptr_type, ctx.p_buf_out_ptrs, &[ptr_off], "bs_bpp").unwrap() };
            let buf_ptr = builder.build_load(ptr_type, buf_ptr_ptr, "bs_bp").unwrap().into_pointer_value();

            let elem_ptr = unsafe { builder.build_gep(f64_type, buf_ptr, &[idx64], "bs_ep").unwrap() };
            builder.build_store(elem_ptr, value).unwrap();

            // Return dummy u32 0
            VarValues::Scalar(i32_type.const_zero().into())
        }

        // -- Array operations --
        Inst::ArrayNew(elems) => {
            let mut flat = Vec::new();
            for elem_var in elems {
                flat.extend(flatten_values(&val_map[elem_var]));
            }
            VarValues::Array(flat)
        }
        Inst::ArrayGet { array, index } => {
            let arr = get_array(val_map, array);
            let idx = get_scalar(val_map, index).into_int_value();
            let arr_ty = kernel.binding(*array).unwrap().ty.clone();
            let (elem_ty, _size) = match &arr_ty {
                ValType::Array { elem, size } => (elem.as_ref().clone(), *size as usize),
                _ => panic!("ArrayGet on non-array type"),
            };
            let elem_flat_count = elem_ty.flat_scalar_count(&kernel.struct_defs);

            // Select chain: start with element 0, conditionally swap for each subsequent index
            let mut result: Vec<BasicValueEnum> = arr[..elem_flat_count].to_vec();
            for i in 1.._size {
                let i_const = i32_type.const_int(i as u64, false);
                let cmp = builder.build_int_compare(IntPredicate::EQ, idx, i_const, &format!("aget_cmp_{}", i)).unwrap();
                let elem_start = i * elem_flat_count;
                for j in 0..elem_flat_count {
                    let candidate = arr[elem_start + j];
                    let name = format!("aget_sel_{}_{}", i, j);
                    result[j] = if candidate.is_float_value() {
                        builder.build_select(cmp, candidate.into_float_value(), result[j].into_float_value(), &name).unwrap()
                    } else {
                        builder.build_select(cmp, candidate.into_int_value(), result[j].into_int_value(), &name).unwrap()
                    };
                }
            }
            unflatten_values(&elem_ty, result)
        }
        Inst::ArraySet { array, index, val } => {
            let arr = get_array(val_map, array).to_vec();
            let idx = get_scalar(val_map, index).into_int_value();
            let new_vals = flatten_values(&val_map[val]);
            let arr_ty = kernel.binding(*array).unwrap().ty.clone();
            let (elem_ty, size) = match &arr_ty {
                ValType::Array { elem, size } => (elem.as_ref().clone(), *size as usize),
                _ => panic!("ArraySet on non-array type"),
            };
            let elem_flat_count = elem_ty.flat_scalar_count(&kernel.struct_defs);

            // For each array position, conditionally replace with new values if index matches
            let mut result = arr.clone();
            for i in 0..size {
                let i_const = i32_type.const_int(i as u64, false);
                let cmp = builder.build_int_compare(IntPredicate::EQ, idx, i_const, &format!("aset_cmp_{}", i)).unwrap();
                let elem_start = i * elem_flat_count;
                for j in 0..elem_flat_count {
                    let name = format!("aset_sel_{}_{}", i, j);
                    result[elem_start + j] = if new_vals[j].is_float_value() {
                        builder.build_select(cmp, new_vals[j].into_float_value(), result[elem_start + j].into_float_value(), &name).unwrap()
                    } else {
                        builder.build_select(cmp, new_vals[j].into_int_value(), result[elem_start + j].into_int_value(), &name).unwrap()
                    };
                }
            }
            VarValues::Array(result)
        }

        // -- Struct operations --
        Inst::StructNew(fields) => {
            let mut flat = Vec::new();
            for field_var in fields {
                flat.extend(flatten_values(&val_map[field_var]));
            }
            VarValues::Struct(flat)
        }
        Inst::StructGet { val: sval, field } => {
            let s = get_struct(val_map, sval);
            let struct_ty = kernel.binding(*sval).unwrap().ty.clone();
            let struct_name = match &struct_ty {
                ValType::Struct(name) => name.clone(),
                _ => panic!("StructGet on non-struct type"),
            };
            let sd = kernel.struct_defs.iter().find(|sd| sd.name == struct_name)
                .expect("unknown struct type");

            // Compute offset: sum of flat_scalar_count of preceding fields
            let mut offset = 0;
            for i in 0..(*field as usize) {
                offset += sd.fields[i].1.flat_scalar_count(&kernel.struct_defs);
            }
            let field_ty = &sd.fields[*field as usize].1;
            let field_count = field_ty.flat_scalar_count(&kernel.struct_defs);
            let field_vals = s[offset..offset + field_count].to_vec();
            unflatten_values(field_ty, field_vals)
        }
        Inst::StructSet { val: sval, field, new_val } => {
            let mut s = get_struct(val_map, sval).to_vec();
            let new_vals = flatten_values(&val_map[new_val]);
            let struct_ty = kernel.binding(*sval).unwrap().ty.clone();
            let struct_name = match &struct_ty {
                ValType::Struct(name) => name.clone(),
                _ => panic!("StructSet on non-struct type"),
            };
            let sd = kernel.struct_defs.iter().find(|sd| sd.name == struct_name)
                .expect("unknown struct type");

            // Compute offset: sum of flat_scalar_count of preceding fields
            let mut offset = 0;
            for i in 0..(*field as usize) {
                offset += sd.fields[i].1.flat_scalar_count(&kernel.struct_defs);
            }
            let field_count = sd.fields[*field as usize].1.flat_scalar_count(&kernel.struct_defs);
            s[offset..offset + field_count].copy_from_slice(&new_vals);
            VarValues::Struct(s)
        }
        Inst::TexLoad { tex, x, y, address } => {
            let x_val = get_scalar(val_map, x).into_int_value();
            let y_val = get_scalar(val_map, y).into_int_value();
            let helper_name = match address {
                AddressMode::Repeat => "pd_tex_load_repeat",
                AddressMode::ClampToEdge => "pd_tex_load_clamp",
            };
            emit_llvm_tex_call_i32(context, module, builder, helper_name, *tex, x_val, y_val, tex_slots_ptr)
        }
        Inst::TexSample { tex, u, v, filter, address } => {
            let u_val = get_scalar(val_map, u).into_float_value();
            let v_val = get_scalar(val_map, v).into_float_value();
            let helper_name = match (filter, address) {
                (FilterMode::Nearest, AddressMode::Repeat) => "pd_tex_sample_nearest_repeat",
                (FilterMode::Nearest, AddressMode::ClampToEdge) => "pd_tex_sample_nearest_clamp",
                (FilterMode::Bilinear, AddressMode::Repeat) => "pd_tex_sample_bilinear_repeat",
                (FilterMode::Bilinear, AddressMode::ClampToEdge) => "pd_tex_sample_bilinear_clamp",
            };
            emit_llvm_tex_call_f64(context, module, builder, helper_name, *tex, u_val, v_val, tex_slots_ptr)
        }
        Inst::TexWidth { tex } => {
            // TextureSlot: {data: *const u8 (8), width: u32 (4), height: u32 (4)} = 16 bytes
            let i32_type = context.i32_type();
            let i64_type = context.i64_type();
            let offset = i64_type.const_int((*tex as u64) * 16 + 8, false);
            let addr = unsafe {
                builder.build_gep(context.i8_type(), tex_slots_ptr, &[offset], "tw_ptr").unwrap()
            };
            VarValues::Scalar(builder.build_load(i32_type, addr, "tw").unwrap())
        }
        Inst::TexHeight { tex } => {
            let i32_type = context.i32_type();
            let i64_type = context.i64_type();
            let offset = i64_type.const_int((*tex as u64) * 16 + 12, false);
            let addr = unsafe {
                builder.build_gep(context.i8_type(), tex_slots_ptr, &[offset], "th_ptr").unwrap()
            };
            VarValues::Scalar(builder.build_load(i32_type, addr, "th").unwrap())
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

fn call_int_binary_intrinsic(
    module: &Module<'static>,
    builder: &inkwell::builder::Builder<'static>,
    name: &str,
    lhs: inkwell::values::IntValue<'static>,
    rhs: inkwell::values::IntValue<'static>,
    int_type: inkwell::types::IntType<'static>,
) -> BasicValueEnum<'static> {
    let intrinsic = inkwell::intrinsics::Intrinsic::find(name)
        .unwrap_or_else(|| panic!("intrinsic {name} not found"));
    let decl = intrinsic.get_declaration(module, &[int_type.into()]).unwrap();
    builder.build_call(decl, &[lhs.into(), rhs.into()], "minmax")
        .unwrap()
        .try_as_basic_value()
        .unwrap_basic()
}

/// Emit a multiply for the given scalar element type.
fn llvm_emit_mul(
    builder: &inkwell::builder::Builder<'static>,
    elem: ScalarType,
    a: BasicValueEnum<'static>,
    b: BasicValueEnum<'static>,
) -> BasicValueEnum<'static> {
    if elem.is_float() {
        builder.build_float_mul(a.into_float_value(), b.into_float_value(), "mmul").unwrap().into()
    } else {
        builder.build_int_mul(a.into_int_value(), b.into_int_value(), "mmul").unwrap().into()
    }
}

/// Emit an add for the given scalar element type.
fn llvm_emit_add(
    builder: &inkwell::builder::Builder<'static>,
    elem: ScalarType,
    a: BasicValueEnum<'static>,
    b: BasicValueEnum<'static>,
) -> BasicValueEnum<'static> {
    if elem.is_float() {
        builder.build_float_add(a.into_float_value(), b.into_float_value(), "madd").unwrap().into()
    } else {
        builder.build_int_add(a.into_int_value(), b.into_int_value(), "madd").unwrap().into()
    }
}

/// Map a ScalarType to the corresponding LLVM basic type.
fn scalar_to_llvm_type(context: &Context, ty: ScalarType) -> inkwell::types::BasicTypeEnum<'_> {
    match ty {
        ScalarType::F32 => context.f32_type().into(),
        ScalarType::F64 => context.f64_type().into(),
        ScalarType::I8 | ScalarType::U8 => context.i8_type().into(),
        ScalarType::I16 | ScalarType::U16 => context.i16_type().into(),
        ScalarType::I32 | ScalarType::U32 => context.i32_type().into(),
        ScalarType::I64 | ScalarType::U64 => context.i64_type().into(),
        ScalarType::Bool => context.bool_type().into(),
    }
}

/// Map a ScalarType to an LLVM integer type (panics for floats/bool).
fn scalar_to_llvm_int_type(context: &Context, ty: ScalarType) -> inkwell::types::IntType<'_> {
    match ty {
        ScalarType::I8 | ScalarType::U8 => context.i8_type(),
        ScalarType::I16 | ScalarType::U16 => context.i16_type(),
        ScalarType::I32 | ScalarType::U32 => context.i32_type(),
        ScalarType::I64 | ScalarType::U64 => context.i64_type(),
        _ => panic!("scalar_to_llvm_int_type called on non-integer type {:?}", ty),
    }
}

fn create_llvm_module_and_machine(context: &'static Context, name: &str) -> (Module<'static>, TargetMachine) {
    let module = context.create_module(name);

    Target::initialize_native(&InitializationConfig::default())
        .expect("failed to initialize native target");
    let triple = TargetMachine::get_default_triple();
    let cpu = TargetMachine::get_host_cpu_name();
    let features = TargetMachine::get_host_cpu_features();
    let features_str = features.to_str().unwrap();
    let filtered_features: String = features_str
        .split(',')
        .filter(|f| !f.contains("sve") && !f.contains("sme"))
        .collect::<Vec<_>>()
        .join(",");
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
    (module, machine)
}

/// Register texture helper function addresses with the LLVM execution engine.
fn register_llvm_tex_helpers(
    engine: &inkwell::execution_engine::ExecutionEngine<'static>,
    module: &Module<'static>,
) {
    use crate::jit;
    let helpers: &[(&str, usize)] = &[
        ("pd_tex_load_repeat", jit::pd_tex_load_repeat as *const () as usize),
        ("pd_tex_load_clamp", jit::pd_tex_load_clamp as *const () as usize),
        ("pd_tex_sample_nearest_repeat", jit::pd_tex_sample_nearest_repeat as *const () as usize),
        ("pd_tex_sample_nearest_clamp", jit::pd_tex_sample_nearest_clamp as *const () as usize),
        ("pd_tex_sample_bilinear_repeat", jit::pd_tex_sample_bilinear_repeat as *const () as usize),
        ("pd_tex_sample_bilinear_clamp", jit::pd_tex_sample_bilinear_clamp as *const () as usize),
    ];
    for (name, addr) in helpers {
        if let Some(func) = module.get_function(name) {
            engine.add_global_mapping(&func, *addr);
        }
    }
}

/// Emit an LLVM call to a texture helper that takes (slots, tex, i32, i32, out_ptr).
/// Returns VarValues::Vec of 4 f32 components.
fn emit_llvm_tex_call_i32<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    builder: &inkwell::builder::Builder<'ctx>,
    helper_name: &str,
    tex_idx: u32,
    coord_a: inkwell::values::IntValue<'ctx>,
    coord_b: inkwell::values::IntValue<'ctx>,
    tex_slots_ptr: inkwell::values::PointerValue<'ctx>,
) -> VarValues<'ctx> {
    let i32_type = context.i32_type();
    let f32_type = context.f32_type();
    let ptr_type = context.ptr_type(inkwell::AddressSpace::default());
    let void_type = context.void_type();

    // Declare or get the helper function: void(ptr, u32, i32, i32, ptr)
    let fn_type = void_type.fn_type(
        &[ptr_type.into(), i32_type.into(), i32_type.into(), i32_type.into(), ptr_type.into()],
        false,
    );
    let func = module.get_function(helper_name).unwrap_or_else(|| {
        module.add_function(helper_name, fn_type, Some(inkwell::module::Linkage::External))
    });

    // Allocate stack space for 4 × f32 = 16 bytes
    let out_alloca = builder.build_alloca(context.custom_width_int_type(128), "tex_out").unwrap();
    let tex_const = i32_type.const_int(tex_idx as u64, false);

    builder.build_call(func, &[tex_slots_ptr.into(), tex_const.into(), coord_a.into(), coord_b.into(), out_alloca.into()], "").unwrap();

    // Load back 4 f32 values
    let mut components = Vec::with_capacity(4);
    for i in 0..4u64 {
        let offset = context.i64_type().const_int(i * 4, false);
        let addr = unsafe {
            builder.build_gep(context.i8_type(), out_alloca, &[offset], &format!("tex_c{i}_ptr")).unwrap()
        };
        let val = builder.build_load(f32_type, addr, &format!("tex_c{i}")).unwrap();
        components.push(val);
    }
    VarValues::Vec(components)
}

/// Emit an LLVM call to a texture helper that takes (slots, tex, f64, f64, out_ptr).
fn emit_llvm_tex_call_f64<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    builder: &inkwell::builder::Builder<'ctx>,
    helper_name: &str,
    tex_idx: u32,
    coord_a: inkwell::values::FloatValue<'ctx>,
    coord_b: inkwell::values::FloatValue<'ctx>,
    tex_slots_ptr: inkwell::values::PointerValue<'ctx>,
) -> VarValues<'ctx> {
    let i32_type = context.i32_type();
    let f32_type = context.f32_type();
    let f64_type = context.f64_type();
    let ptr_type = context.ptr_type(inkwell::AddressSpace::default());
    let void_type = context.void_type();

    let fn_type = void_type.fn_type(
        &[ptr_type.into(), i32_type.into(), f64_type.into(), f64_type.into(), ptr_type.into()],
        false,
    );
    let func = module.get_function(helper_name).unwrap_or_else(|| {
        module.add_function(helper_name, fn_type, Some(inkwell::module::Linkage::External))
    });

    let out_alloca = builder.build_alloca(context.custom_width_int_type(128), "tex_out").unwrap();
    let tex_const = i32_type.const_int(tex_idx as u64, false);

    builder.build_call(func, &[tex_slots_ptr.into(), tex_const.into(), coord_a.into(), coord_b.into(), out_alloca.into()], "").unwrap();

    let mut components = Vec::with_capacity(4);
    for i in 0..4u64 {
        let offset = context.i64_type().const_int(i * 4, false);
        let addr = unsafe {
            builder.build_gep(context.i8_type(), out_alloca, &[offset], &format!("tex_c{i}_ptr")).unwrap()
        };
        let val = builder.build_load(f32_type, addr, &format!("tex_c{i}")).unwrap();
        components.push(val);
    }
    VarValues::Vec(components)
}

fn compile_sim_kernel(kernel: &Kernel, user_args: &[UserArgSlot]) -> LlvmSimKernel {
    let context: &'static Context = Box::leak(Box::new(Context::create()));
    let (module, machine) = create_llvm_module_and_machine(context, &kernel.name);

    let i32_type = context.i32_type();
    let void_type = context.void_type();
    let ptr_type = context.ptr_type(inkwell::AddressSpace::default());

    // SimTileKernelFn(output, width, height, row_start, row_end, buf_ptrs, buf_out_ptrs, user_args)
    let fn_type = void_type.fn_type(
        &[
            ptr_type.into(),  // output: *mut u32
            i32_type.into(),  // width: u32
            i32_type.into(),  // height: u32
            i32_type.into(),  // row_start: u32
            i32_type.into(),  // row_end: u32
            ptr_type.into(),  // buf_ptrs: *const *const f64
            ptr_type.into(),  // buf_out_ptrs: *const *mut f64
            ptr_type.into(),  // user_args: *const u8
            ptr_type.into(),  // tex_slots: *const TextureSlot
        ],
        false,
    );

    let function = module.add_function(&kernel.name, fn_type, None);
    build_sim_tile_loop(context, &module, function, kernel, user_args);

    let pass_options = PassBuilderOptions::create();
    module
        .run_passes("default<O3>", &machine, pass_options)
        .expect("failed to run LLVM optimization passes");

    let engine = module
        .create_jit_execution_engine(OptimizationLevel::Aggressive)
        .unwrap();

    register_llvm_tex_helpers(&engine, &module);

    let fn_ptr = unsafe {
        engine
            .get_function::<unsafe extern "C" fn(*mut u32, u32, u32, u32, u32, *const *const f64, *const *mut f64, *const u8, *const u8)>(
                &kernel.name,
            )
            .unwrap()
            .as_raw()
    };
    let fn_ptr: SimTileKernelFn = unsafe { std::mem::transmute(fn_ptr) };

    LlvmSimKernel {
        _engine: engine,
        fn_ptr,
    }
}

fn build_sim_tile_loop(
    context: &'static Context,
    module: &Module<'static>,
    function: FunctionValue<'static>,
    kernel: &Kernel,
    user_args: &[UserArgSlot],
) {
    let builder = context.create_builder();
    let i32_type = context.i32_type();
    let i64_type = context.i64_type();

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
    let p_height = function.get_nth_param(2).unwrap().into_int_value();
    let p_row_start = function.get_nth_param(3).unwrap().into_int_value();
    let p_row_end = function.get_nth_param(4).unwrap().into_int_value();
    let p_buf_ptrs = function.get_nth_param(5).unwrap().into_pointer_value();
    let p_buf_out_ptrs = function.get_nth_param(6).unwrap().into_pointer_value();
    let p_user_args = function.get_nth_param(7).unwrap().into_pointer_value();
    let _p_tex_slots = function.get_nth_param(8).unwrap().into_pointer_value();

    let buf_ctx = LlvmBufContext { p_width, p_buf_ptrs, p_buf_out_ptrs };

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

    // Map kernel params
    use std::collections::HashMap;
    let mut val_map: HashMap<Var, VarValues<'static>> = HashMap::new();
    for param in &kernel.params {
        let val: BasicValueEnum<'static> = match param.name.as_str() {
            "px" => col.into(),
            "py" => row.into(),
            "width" => p_width.into(),
            "height" => p_height.into(),
            name => {
                let slot = user_args.iter().find(|s| s.name == name)
                    .unwrap_or_else(|| panic!("unknown sim kernel parameter: '{name}'"));
                if slot.ty.is_mat() {
                    panic!("LLVM backend does not support Mat user args (param '{name}')");
                }
                if slot.ty.is_array() {
                    panic!("LLVM backend does not support Array user args (param '{name}')");
                }
                if matches!(slot.ty, ValType::Struct(_)) {
                    panic!("LLVM backend does not support Struct user args (param '{name}')");
                }
                let offset = i64_type.const_int(slot.offset as u64, false);
                let addr = unsafe {
                    builder.build_gep(context.i8_type(), p_user_args, &[offset], &format!("arg_{name}_ptr")).unwrap()
                };
                let load_ty = scalar_to_llvm_type(context, slot.ty.element_scalar());
                builder.build_load(load_ty, addr, &format!("arg_{name}")).unwrap()
            }
        };
        val_map.insert(param.var, VarValues::Scalar(val));
    }

    lower_body_items(context, module, &builder, function, kernel, &kernel.body, &mut val_map, Some(&buf_ctx), _p_tex_slots);

    let color = get_scalar(&val_map, &kernel.emit).into_int_value();

    // Store pixel: output[(row - row_start) * width + col]
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
