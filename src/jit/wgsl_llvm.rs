//! WGSL -> LLVM JIT backend.
//!
//! Parses WGSL compute shaders via naga and compiles them to native machine
//! code using inkwell/LLVM. The generated function iterates over all pixels,
//! executing the shader's entry point for each one.

use std::collections::HashMap;

use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::passes::PassBuilderOptions;
use inkwell::targets::{CodeModel, InitializationConfig, RelocMode, Target, TargetMachine};
use inkwell::values::{BasicValueEnum, FunctionValue, IntValue, PointerValue};
use inkwell::{FloatPredicate, IntPredicate, OptimizationLevel};

use naga::{
    BinaryOperator, Expression, Handle, MathFunction, ScalarKind, Statement, TypeInner,
    UnaryOperator,
};

use super::{CompiledWgslKernel, ParamMember, WgslKernelFn};

/// Wrapper to make `ExecutionEngine` Send + Sync.
/// SAFETY: The engine owns JIT'd code and is only accessed through the
/// function pointer stored in `CompiledWgslKernel`. The engine itself is
/// never shared mutably across threads.
struct SendSyncEngine(#[allow(dead_code)] inkwell::execution_engine::ExecutionEngine<'static>);
unsafe impl Send for SendSyncEngine {}
unsafe impl Sync for SendSyncEngine {}

/// Parse and compile a WGSL compute shader to a native function via LLVM.
pub fn compile_wgsl(source: &str) -> Result<CompiledWgslKernel, String> {
    let module = naga::front::wgsl::parse_str(source)
        .map_err(|e| format!("WGSL parse error: {e}"))?;

    let entry = module
        .entry_points
        .first()
        .ok_or_else(|| "no entry point in WGSL shader".to_string())?;

    let analysis = analyse_module(&module)?;

    // Create LLVM context, module, target machine.
    let context: &'static Context = Box::leak(Box::new(Context::create()));
    let llvm_module = context.create_module("wgsl_main");

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
    llvm_module.set_data_layout(&machine.get_target_data().get_data_layout());
    llvm_module.set_triple(&triple);

    let i32_type = context.i32_type();
    let void_type = context.void_type();
    let ptr_type = context.ptr_type(inkwell::AddressSpace::default());

    // WgslKernelFn: (params, buffers, tex_slots, width, height, stride, row_start, row_end)
    let fn_type = void_type.fn_type(
        &[
            ptr_type.into(),  // params: *const u8
            ptr_type.into(),  // buffers: *const *mut u8
            ptr_type.into(),  // tex_slots: *const u8
            i32_type.into(),  // width: u32
            i32_type.into(),  // height: u32
            i32_type.into(),  // stride: u32
            i32_type.into(),  // row_start: u32
            i32_type.into(),  // row_end: u32
        ],
        false,
    );

    let function = llvm_module.add_function("wgsl_main", fn_type, None);
    let builder = context.create_builder();

    // Build blocks for the row/col loop.
    let entry_block = context.append_basic_block(function, "entry");
    let outer_check = context.append_basic_block(function, "outer_check");
    let inner_pre = context.append_basic_block(function, "inner_pre");
    let inner_check = context.append_basic_block(function, "inner_check");
    let body_block = context.append_basic_block(function, "body");
    let inner_inc = context.append_basic_block(function, "inner_inc");
    let outer_inc = context.append_basic_block(function, "outer_inc");
    let exit_block = context.append_basic_block(function, "exit");

    // -- Entry --
    builder.position_at_end(entry_block);
    let p_params = function.get_nth_param(0).unwrap().into_pointer_value();
    let p_buffers = function.get_nth_param(1).unwrap().into_pointer_value();
    let p_tex_slots = function.get_nth_param(2).unwrap().into_pointer_value();
    let p_width = function.get_nth_param(3).unwrap().into_int_value();
    let p_height = function.get_nth_param(4).unwrap().into_int_value();
    let p_stride = function.get_nth_param(5).unwrap().into_int_value();
    let p_row_start = function.get_nth_param(6).unwrap().into_int_value();
    let p_row_end = function.get_nth_param(7).unwrap().into_int_value();

    let row_ptr = builder.build_alloca(i32_type, "row_ptr").unwrap();
    let col_ptr = builder.build_alloca(i32_type, "col_ptr").unwrap();

    builder.build_store(row_ptr, p_row_start).unwrap();
    builder.build_unconditional_branch(outer_check).unwrap();

    // -- Outer check --
    builder.position_at_end(outer_check);
    let row = builder.build_load(i32_type, row_ptr, "row").unwrap().into_int_value();
    let cmp = builder.build_int_compare(IntPredicate::ULT, row, p_row_end, "row_lt").unwrap();
    builder.build_conditional_branch(cmp, inner_pre, exit_block).unwrap();

    // -- Inner pre --
    builder.position_at_end(inner_pre);
    builder.build_store(col_ptr, i32_type.const_zero()).unwrap();
    builder.build_unconditional_branch(inner_check).unwrap();

    // -- Inner check --
    builder.position_at_end(inner_check);
    let col = builder.build_load(i32_type, col_ptr, "col").unwrap().into_int_value();
    let cmp = builder.build_int_compare(IntPredicate::ULT, col, p_width, "col_lt").unwrap();
    builder.build_conditional_branch(cmp, body_block, outer_inc).unwrap();

    // -- Body --
    builder.position_at_end(body_block);

    let return_block = context.append_basic_block(function, "shader_return");

    {
        let mut compiler = ShaderCompiler {
            naga_module: &module,
            analysis: &analysis,
            context,
            module: &llvm_module,
            builder: &builder,
            function,
            expr_values: HashMap::new(),
            local_vars: HashMap::new(),
            call_results: HashMap::new(),
            p_params,
            p_buffers,
            p_tex_slots,
            p_width,
            p_height,
            p_stride,
            row_ptr,
            col_ptr,
            workgroup_size: entry.workgroup_size,
            return_block,
            terminated: false,
            loop_stack: Vec::new(),
        };
        compiler.compile_entry_point(&entry.function)?;
    }

    // After shader body, ensure we branch to inner_inc.
    // The return_block jumps to inner_inc.
    builder.position_at_end(return_block);
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
    builder.position_at_end(exit_block);
    builder.build_return(None).unwrap();

    // Verify IR before optimization.
    if let Err(msg) = llvm_module.verify() {
        return Err(format!("LLVM IR verification failed:\n{}", msg.to_string()));
    }

    // Run O3 optimization.
    let pass_options = PassBuilderOptions::create();
    llvm_module
        .run_passes("default<O3>", &machine, pass_options)
        .expect("failed to run LLVM optimization passes");

    // Create execution engine and register texture helpers.
    let engine = llvm_module
        .create_jit_execution_engine(OptimizationLevel::Aggressive)
        .unwrap();

    register_llvm_tex_helpers(&engine, &llvm_module);

    let fn_ptr = unsafe {
        engine
            .get_function::<unsafe extern "C" fn(*const u8, *const *mut u8, *const u8, u32, u32, u32, u32, u32)>(
                "wgsl_main",
            )
            .unwrap()
            .as_raw()
    };
    let fn_ptr: WgslKernelFn = unsafe { std::mem::transmute(fn_ptr) };

    // Build binding map.
    let mut binding_map = HashMap::new();
    for (i, buf) in analysis.storage_buffers.iter().enumerate() {
        let gv = &module.global_variables[buf.global];
        if let Some(ref name) = gv.name {
            binding_map.insert(name.clone(), i);
        }
    }
    let num_storage_buffers = analysis.storage_buffers.len();
    let buffer_elem_bytes = analysis.storage_buffers.iter().map(|b| b.elem_bytes).collect();

    Ok(CompiledWgslKernel {
        _jit_handle: Box::new(SendSyncEngine(engine)),
        fn_ptr,
        binding_map,
        num_storage_buffers,
        params_members: analysis.params_members.clone(),
        buffer_elem_bytes,
    })
}

// ── Module analysis ─────────────────────────────────────────────────────

/// Info about a storage buffer binding.
struct StorageBufferInfo {
    global: Handle<naga::GlobalVariable>,
    /// Binding index (1, 2, 3, ...).
    binding: u32,
    /// Byte size per element.
    elem_bytes: u32,
    /// Number of scalar components per element (1 for scalar, 4 for vec4, etc.).
    elem_components: usize,
}

/// Info about a texture binding.
struct TextureInfo {
    global: Handle<naga::GlobalVariable>,
    /// Index into the tex_slots array (0-based, ordered by binding).
    slot_index: usize,
}

struct BindingInfo {
    params_global: Handle<naga::GlobalVariable>,
    /// Byte offset of each Params struct member.
    params_offsets: Vec<u32>,
    /// Named params members for runtime value filling.
    params_members: Vec<ParamMember>,
    /// Storage buffers, sorted by binding index.
    storage_buffers: Vec<StorageBufferInfo>,
    /// Texture bindings (texture_2d globals), sorted by binding index.
    textures: Vec<TextureInfo>,
    /// Sampler globals (just tracked so we can ignore them).
    sampler_globals: Vec<Handle<naga::GlobalVariable>>,
}

fn analyse_module(module: &naga::Module) -> Result<BindingInfo, String> {
    let mut params_global = None;
    let mut storage_buffers = Vec::new();
    let mut texture_bindings: Vec<(u32, Handle<naga::GlobalVariable>)> = Vec::new();
    let mut sampler_globals = Vec::new();

    for (handle, gv) in module.global_variables.iter() {
        let binding = match &gv.binding {
            Some(b) => b,
            None => continue,
        };
        if binding.group != 0 { continue; }

        let ty_inner = &module.types[gv.ty].inner;
        match ty_inner {
            TypeInner::Image { .. } => {
                texture_bindings.push((binding.binding, handle));
                continue;
            }
            TypeInner::Sampler { .. } => {
                sampler_globals.push(handle);
                continue;
            }
            _ => {}
        }

        if binding.binding == 0 {
            params_global = Some(handle);
        } else {
            let (elem_bytes, elem_components) = match ty_inner {
                TypeInner::Array { base, .. } => {
                    let elem = &module.types[*base].inner;
                    let bytes = match elem {
                        TypeInner::Scalar(s) => s.width as u32,
                        TypeInner::Vector { size, scalar } => *size as u32 * scalar.width as u32,
                        _ => 4,
                    };
                    let components = match elem {
                        TypeInner::Vector { size, .. } => *size as usize,
                        _ => 1,
                    };
                    (bytes, components)
                }
                _ => (4, 1),
            };
            storage_buffers.push(StorageBufferInfo {
                global: handle,
                binding: binding.binding,
                elem_bytes,
                elem_components,
            });
        }
    }

    let params_handle = params_global.ok_or("no @binding(0) uniform found")?;
    storage_buffers.sort_by_key(|b| b.binding);
    texture_bindings.sort_by_key(|(b, _)| *b);

    let textures: Vec<TextureInfo> = texture_bindings
        .iter()
        .enumerate()
        .map(|(slot_index, (_, global))| TextureInfo { global: *global, slot_index })
        .collect();

    let params_gv = &module.global_variables[params_handle];
    let params_ty = &module.types[params_gv.ty];
    let (offsets, params_members) = match &params_ty.inner {
        TypeInner::Struct { members, .. } => {
            let offsets = members.iter().map(|m| m.offset).collect();
            let named = members.iter().map(|m| ParamMember {
                name: m.name.clone().unwrap_or_default(),
                offset: m.offset,
            }).collect();
            (offsets, named)
        }
        _ => return Err("Params binding(0) is not a struct".into()),
    };

    Ok(BindingInfo {
        params_global: params_handle,
        params_offsets: offsets,
        params_members,
        storage_buffers,
        textures,
        sampler_globals,
    })
}

// ── Texture helper registration ─────────────────────────────────────────

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

// ── LLVM intrinsic helpers ──────────────────────────────────────────────

fn call_f32_intrinsic(
    module: &Module<'static>,
    builder: &inkwell::builder::Builder<'static>,
    name: &str,
    arg: inkwell::values::FloatValue<'static>,
    f32_type: inkwell::types::FloatType<'static>,
) -> BasicValueEnum<'static> {
    let intrinsic = inkwell::intrinsics::Intrinsic::find(name)
        .unwrap_or_else(|| panic!("intrinsic {name} not found"));
    let decl = intrinsic.get_declaration(module, &[f32_type.into()]).unwrap();
    let short_name = if name.len() > 9 { &name[5..name.len()-4] } else { name };
    builder.build_call(decl, &[arg.into()], short_name)
        .unwrap()
        .try_as_basic_value()
        .unwrap_basic()
}

fn call_f32_binary_intrinsic(
    module: &Module<'static>,
    builder: &inkwell::builder::Builder<'static>,
    name: &str,
    lhs: inkwell::values::FloatValue<'static>,
    rhs: inkwell::values::FloatValue<'static>,
    f32_type: inkwell::types::FloatType<'static>,
) -> BasicValueEnum<'static> {
    let intrinsic = inkwell::intrinsics::Intrinsic::find(name)
        .unwrap_or_else(|| panic!("intrinsic {name} not found"));
    let decl = intrinsic.get_declaration(module, &[f32_type.into()]).unwrap();
    builder.build_call(decl, &[lhs.into(), rhs.into()], "binop")
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

// ── Shader compiler ─────────────────────────────────────────────────────

struct ShaderCompiler<'a> {
    naga_module: &'a naga::Module,
    analysis: &'a BindingInfo,
    context: &'static Context,
    module: &'a Module<'static>,
    builder: &'a inkwell::builder::Builder<'static>,
    function: FunctionValue<'static>,
    /// Cached LLVM values for naga expressions.
    expr_values: HashMap<Handle<Expression>, Vec<BasicValueEnum<'static>>>,
    /// LLVM alloca pointers for naga LocalVariables (per component).
    local_vars: HashMap<Handle<naga::LocalVariable>, Vec<PointerValue<'static>>>,
    /// Return values captured from inlined function calls.
    call_results: HashMap<Handle<Expression>, Vec<BasicValueEnum<'static>>>,
    // Function parameters.
    p_params: PointerValue<'static>,
    p_buffers: PointerValue<'static>,
    p_tex_slots: PointerValue<'static>,
    #[allow(dead_code)]
    p_width: IntValue<'static>,
    #[allow(dead_code)]
    p_height: IntValue<'static>,
    #[allow(dead_code)]
    p_stride: IntValue<'static>,
    row_ptr: PointerValue<'static>,
    col_ptr: PointerValue<'static>,
    workgroup_size: [u32; 3],
    /// Block to jump to on early return.
    return_block: inkwell::basic_block::BasicBlock<'static>,
    /// Whether the current block has been terminated.
    terminated: bool,
    /// Stack of (break_block, continue_block) for nested loops.
    loop_stack: Vec<(inkwell::basic_block::BasicBlock<'static>, inkwell::basic_block::BasicBlock<'static>)>,
}

// Static type constants for resolve_expr_type.
static TY_F32: TypeInner = TypeInner::Scalar(naga::Scalar::F32);
static TY_U32: TypeInner = TypeInner::Scalar(naga::Scalar::U32);
static TY_I32: TypeInner = TypeInner::Scalar(naga::Scalar::I32);
static TY_BOOL: TypeInner = TypeInner::Scalar(naga::Scalar::BOOL);
static TY_F64: TypeInner = TypeInner::Scalar(naga::Scalar::F64);
static TY_I64: TypeInner = TypeInner::Scalar(naga::Scalar::I64);
static TY_U64: TypeInner = TypeInner::Scalar(naga::Scalar::U64);
static TY_VEC3_U32: TypeInner = TypeInner::Vector {
    size: naga::VectorSize::Tri,
    scalar: naga::Scalar::U32,
};

fn scalar_static_type(kind: ScalarKind, width: u8) -> &'static TypeInner {
    match (kind, width) {
        (ScalarKind::Float, 4) => &TY_F32,
        (ScalarKind::Float, 8) => &TY_F64,
        (ScalarKind::Uint, 4) => &TY_U32,
        (ScalarKind::Sint, 4) => &TY_I32,
        (ScalarKind::Uint, 8) => &TY_U64,
        (ScalarKind::Sint, 8) => &TY_I64,
        (ScalarKind::Bool, _) => &TY_BOOL,
        _ => &TY_U32,
    }
}

impl<'a> ShaderCompiler<'a> {
    fn compile_entry_point(&mut self, func: &'a naga::Function) -> Result<(), String> {
        // Declare local variables as allocas.
        for (handle, local) in func.local_variables.iter() {
            let llvm_ty = self.naga_type_to_llvm(&self.naga_module.types[local.ty].inner);
            let components = self.type_component_count(&self.naga_module.types[local.ty].inner);
            let mut alloca_ptrs = Vec::new();
            for c in 0..components {
                let alloca = self.builder.build_alloca(llvm_ty, &format!("local_{c}")).unwrap();
                self.builder.build_store(alloca, self.zero_value(llvm_ty)).unwrap();
                alloca_ptrs.push(alloca);
            }
            // Initialize from init value if present.
            if let Some(init) = local.init {
                self.eval_expr(init, func)?;
                let init_vals = self.get_expr(init).to_vec();
                for (i, ptr) in alloca_ptrs.iter().enumerate() {
                    if i < init_vals.len() {
                        self.builder.build_store(*ptr, init_vals[i]).unwrap();
                    }
                }
            }
            self.local_vars.insert(handle, alloca_ptrs);
        }

        self.lower_block(&func.body, func)?;

        // Jump to return block if not already terminated.
        if !self.terminated {
            self.builder.build_unconditional_branch(self.return_block).unwrap();
        }
        self.builder.position_at_end(self.return_block);
        self.terminated = false;

        // The return_block will be branched to inner_inc by the caller.
        Ok(())
    }

    /// Extend an i1 comparison result to i8, matching naga's Bool type.
    fn i1_to_i8(&self, val: inkwell::values::IntValue<'static>) -> BasicValueEnum<'static> {
        if val.get_type().get_bit_width() == 1 {
            self.builder.build_int_z_extend(val, self.context.i8_type(), "i1_to_i8").unwrap().into()
        } else {
            val.into()
        }
    }

    /// Ensure a value is i1 for use as a branch/select condition.
    /// If it's i8 (naga bool), truncate to i1. If already i1, pass through.
    fn to_i1(&self, val: inkwell::values::IntValue<'static>) -> inkwell::values::IntValue<'static> {
        let bits = val.get_type().get_bit_width();
        if bits == 1 {
            val
        } else {
            // Truncate i8/i32/etc to i1 (non-zero = true)
            let zero = val.get_type().const_zero();
            self.builder.build_int_compare(IntPredicate::NE, val, zero, "to_i1").unwrap()
        }
    }

    fn zero_value(&self, ty: inkwell::types::BasicTypeEnum<'static>) -> BasicValueEnum<'static> {
        match ty {
            inkwell::types::BasicTypeEnum::FloatType(ft) => ft.const_zero().into(),
            inkwell::types::BasicTypeEnum::IntType(it) => it.const_zero().into(),
            _ => self.context.i32_type().const_zero().into(),
        }
    }

    fn naga_scalar_to_llvm(&self, kind: ScalarKind, width: u8) -> inkwell::types::BasicTypeEnum<'static> {
        match (kind, width) {
            (ScalarKind::Float, 4) => self.context.f32_type().into(),
            (ScalarKind::Float, 8) => self.context.f64_type().into(),
            (ScalarKind::Sint | ScalarKind::Uint, 4) => self.context.i32_type().into(),
            (ScalarKind::Sint | ScalarKind::Uint, 8) => self.context.i64_type().into(),
            (ScalarKind::Bool, _) => self.context.i8_type().into(),
            _ => self.context.i32_type().into(),
        }
    }

    fn naga_type_to_llvm(&self, inner: &TypeInner) -> inkwell::types::BasicTypeEnum<'static> {
        match inner {
            TypeInner::Scalar(s) => self.naga_scalar_to_llvm(s.kind, s.width),
            TypeInner::Vector { scalar, .. } => self.naga_scalar_to_llvm(scalar.kind, scalar.width),
            TypeInner::Pointer { .. } | TypeInner::ValuePointer { .. } => {
                self.context.ptr_type(inkwell::AddressSpace::default()).into()
            }
            _ => self.context.i32_type().into(),
        }
    }

    fn type_component_count(&self, inner: &TypeInner) -> usize {
        match inner {
            TypeInner::Vector { size, .. } => *size as usize,
            _ => 1,
        }
    }

    fn type_byte_size(&self, inner: &TypeInner) -> u32 {
        match inner {
            TypeInner::Scalar(s) => s.width as u32,
            TypeInner::Vector { size, scalar } => *size as u32 * scalar.width as u32,
            TypeInner::Struct { span, .. } => *span,
            TypeInner::Array { stride, .. } => *stride,
            _ => 4,
        }
    }

    // ── Expression evaluation ───────────────────────────────────────────

    fn get_expr(&self, handle: Handle<Expression>) -> &[BasicValueEnum<'static>] {
        if let Some(vals) = self.expr_values.get(&handle) {
            return vals;
        }
        if let Some(vals) = self.call_results.get(&handle) {
            return vals;
        }
        panic!("expression {handle:?} not yet evaluated")
    }

    fn get_expr_scalar(&self, handle: Handle<Expression>) -> BasicValueEnum<'static> {
        self.get_expr(handle)[0]
    }

    /// Best-effort type resolution by inspecting the expression.
    fn resolve_expr_type<'c>(&'c self, handle: Handle<Expression>, func: &'c naga::Function) -> &'c TypeInner {
        match &func.expressions[handle] {
            Expression::Literal(lit) => match lit {
                naga::Literal::F32(_) => &TY_F32,
                naga::Literal::U32(_) => &TY_U32,
                naga::Literal::I32(_) => &TY_I32,
                naga::Literal::Bool(_) => &TY_BOOL,
                naga::Literal::F64(_) => &TY_F64,
                naga::Literal::I64(_) => &TY_I64,
                naga::Literal::U64(_) => &TY_U64,
                _ => &TY_F32,
            },
            Expression::Binary { left, right, .. } => {
                let lt = self.resolve_expr_type(*left, func);
                if matches!(lt, TypeInner::Vector { .. }) { lt }
                else { self.resolve_expr_type(*right, func) }
            }
            Expression::Unary { expr, .. } => self.resolve_expr_type(*expr, func),
            Expression::AccessIndex { base, index } => {
                let base_ty = self.resolve_expr_type(*base, func);
                match base_ty {
                    TypeInner::Struct { members, .. } => {
                        &self.naga_module.types[members[*index as usize].ty].inner
                    }
                    TypeInner::Vector { scalar, .. } => {
                        scalar_static_type(scalar.kind, scalar.width)
                    }
                    TypeInner::Pointer { base, .. } => {
                        let inner = &self.naga_module.types[*base].inner;
                        match inner {
                            TypeInner::Struct { members, .. } => {
                                &self.naga_module.types[members[*index as usize].ty].inner
                            }
                            _ => inner,
                        }
                    }
                    _ => base_ty,
                }
            }
            Expression::GlobalVariable(gv) => {
                &self.naga_module.types[self.naga_module.global_variables[*gv].ty].inner
            }
            Expression::LocalVariable(lv) => {
                &self.naga_module.types[func.local_variables[*lv].ty].inner
            }
            Expression::FunctionArgument(idx) => {
                if let Some(arg) = func.arguments.get(*idx as usize) {
                    return &self.naga_module.types[arg.ty].inner;
                }
                &TY_VEC3_U32
            }
            Expression::Access { base, .. } => {
                let base_ty = self.resolve_expr_type(*base, func);
                match base_ty {
                    TypeInner::Array { base: elem_ty, .. } => &self.naga_module.types[*elem_ty].inner,
                    TypeInner::Vector { scalar, .. } => scalar_static_type(scalar.kind, scalar.width),
                    _ => base_ty,
                }
            }
            Expression::As { expr, kind, convert } => {
                let width = match self.resolve_expr_type(*expr, func) {
                    TypeInner::Scalar(s) => convert.unwrap_or(s.width),
                    _ => 4,
                };
                scalar_static_type(*kind, width)
            }
            Expression::Load { pointer } => {
                let ptr_ty = self.resolve_expr_type(*pointer, func);
                match ptr_ty {
                    TypeInner::Pointer { base, .. } => &self.naga_module.types[*base].inner,
                    _ => ptr_ty,
                }
            }
            Expression::Compose { ty, .. } => &self.naga_module.types[*ty].inner,
            Expression::Select { accept, .. } => self.resolve_expr_type(*accept, func),
            Expression::Relational { .. } => &TY_BOOL,
            Expression::Math { fun, arg, .. } => {
                match fun {
                    MathFunction::Dot | MathFunction::Length | MathFunction::Distance => {
                        let arg_ty = self.resolve_expr_type(*arg, func);
                        match arg_ty {
                            TypeInner::Vector { scalar, .. } => scalar_static_type(scalar.kind, scalar.width),
                            _ => arg_ty,
                        }
                    }
                    _ => self.resolve_expr_type(*arg, func),
                }
            }
            Expression::Swizzle { size, vector, .. } => {
                if (*size as u8) == 1 {
                    let vec_ty = self.resolve_expr_type(*vector, func);
                    match vec_ty {
                        TypeInner::Vector { scalar, .. } => scalar_static_type(scalar.kind, scalar.width),
                        _ => vec_ty,
                    }
                } else {
                    self.resolve_expr_type(*vector, func)
                }
            }
            Expression::Constant(c) => {
                let constant = &self.naga_module.constants[*c];
                &self.naga_module.types[constant.ty].inner
            }
            Expression::ImageQuery { .. } => {
                static TY_VEC2_U32: TypeInner = TypeInner::Vector {
                    size: naga::VectorSize::Bi,
                    scalar: naga::Scalar::U32,
                };
                &TY_VEC2_U32
            }
            Expression::ImageSample { .. } => {
                static TY_VEC4_F32: TypeInner = TypeInner::Vector {
                    size: naga::VectorSize::Quad,
                    scalar: naga::Scalar::F32,
                };
                &TY_VEC4_F32
            }
            Expression::CallResult(func_handle) => {
                let callee = &self.naga_module.functions[*func_handle];
                if let Some(ref result) = callee.result {
                    &self.naga_module.types[result.ty].inner
                } else {
                    &TY_U32
                }
            }
            _ => &TY_U32,
        }
    }

    fn eval_expr(&mut self, handle: Handle<Expression>, func: &naga::Function) -> Result<(), String> {
        if self.expr_values.contains_key(&handle) {
            return Ok(());
        }

        let i32_type = self.context.i32_type();
        let i8_type = self.context.i8_type();
        let i64_type = self.context.i64_type();
        let f32_type = self.context.f32_type();
        let f64_type = self.context.f64_type();

        let values = match &func.expressions[handle] {
            Expression::Literal(lit) => {
                vec![match lit {
                    naga::Literal::F32(v) => f32_type.const_float(*v as f64).into(),
                    naga::Literal::U32(v) => i32_type.const_int(*v as u64, false).into(),
                    naga::Literal::I32(v) => i32_type.const_int(*v as u64, true).into(),
                    naga::Literal::Bool(v) => i8_type.const_int(*v as u64, false).into(),
                    naga::Literal::F64(v) => f64_type.const_float(*v).into(),
                    naga::Literal::I64(v) => i64_type.const_int(*v as u64, true).into(),
                    naga::Literal::U64(v) => i64_type.const_int(*v, false).into(),
                    _ => i32_type.const_zero().into(),
                }]
            }

            Expression::Constant(c) => {
                let constant = &self.naga_module.constants[*c];
                match &self.naga_module.global_expressions[constant.init] {
                    Expression::Literal(lit) => {
                        vec![match lit {
                            naga::Literal::F32(v) => f32_type.const_float(*v as f64).into(),
                            naga::Literal::U32(v) => i32_type.const_int(*v as u64, false).into(),
                            naga::Literal::I32(v) => i32_type.const_int(*v as u64, true).into(),
                            naga::Literal::Bool(v) => i8_type.const_int(*v as u64, false).into(),
                            naga::Literal::F64(v) => f64_type.const_float(*v).into(),
                            _ => return Err(format!("unsupported constant literal: {lit:?}")),
                        }]
                    }
                    _ => return Err("unsupported constant initializer".into()),
                }
            }

            Expression::ZeroValue(ty) => {
                let inner = &self.naga_module.types[*ty].inner;
                let llvm_ty = self.naga_type_to_llvm(inner);
                let count = self.type_component_count(inner);
                let zero = self.zero_value(llvm_ty);
                vec![zero; count]
            }

            Expression::FunctionArgument(idx) => {
                let col = self.builder.build_load(i32_type, self.col_ptr, "gid_col").unwrap().into_int_value();
                let row = self.builder.build_load(i32_type, self.row_ptr, "gid_row").unwrap().into_int_value();
                let z = i32_type.const_zero();

                // Determine which builtin this argument represents
                let builtin = func.arguments.get(*idx as usize)
                    .and_then(|arg| arg.binding.as_ref())
                    .and_then(|b| match b { naga::Binding::BuiltIn(bi) => Some(bi), _ => None });

                match builtin {
                    Some(naga::BuiltIn::GlobalInvocationId) | None => {
                        vec![col.into(), row.into(), z.into()]
                    }
                    Some(naga::BuiltIn::WorkGroupId) => {
                        let wg_x = i32_type.const_int(self.workgroup_size[0].max(1) as u64, false);
                        let wg_y = i32_type.const_int(self.workgroup_size[1].max(1) as u64, false);
                        let gx = self.builder.build_int_unsigned_div(col, wg_x, "wg_x").unwrap();
                        let gy = self.builder.build_int_unsigned_div(row, wg_y, "wg_y").unwrap();
                        vec![gx.into(), gy.into(), z.into()]
                    }
                    Some(naga::BuiltIn::LocalInvocationId) => {
                        let wg_x = i32_type.const_int(self.workgroup_size[0].max(1) as u64, false);
                        let wg_y = i32_type.const_int(self.workgroup_size[1].max(1) as u64, false);
                        let lx = self.builder.build_int_unsigned_rem(col, wg_x, "lid_x").unwrap();
                        let ly = self.builder.build_int_unsigned_rem(row, wg_y, "lid_y").unwrap();
                        vec![lx.into(), ly.into(), z.into()]
                    }
                    Some(other) => {
                        return Err(format!("unsupported entry point builtin: {other:?}"));
                    }
                }
            }

            Expression::GlobalVariable(gv) => {
                if *gv == self.analysis.params_global {
                    vec![self.p_params.into()]
                } else if let Some(idx) = self.analysis.storage_buffers.iter().position(|b| b.global == *gv) {
                    // Load buffer pointer from buffers array: buffers[idx]
                    let offset = i64_type.const_int((idx * 8) as u64, false);
                    let buf_ptr_addr = unsafe {
                        self.builder.build_gep(i8_type, self.p_buffers, &[offset], "buf_ptr_addr").unwrap()
                    };
                    let ptr_type = self.context.ptr_type(inkwell::AddressSpace::default());
                    let buf_ptr = self.builder.build_load(ptr_type, buf_ptr_addr, "buf_ptr").unwrap();
                    vec![buf_ptr]
                } else if self.analysis.textures.iter().any(|t| t.global == *gv) || self.analysis.sampler_globals.contains(gv) {
                    // Texture/sampler globals -- return a tag constant.
                    let tag = i64_type.const_int(gv.index() as u64, false);
                    vec![tag.into()]
                } else {
                    return Err(format!("unsupported global variable: {gv:?}"));
                }
            }

            Expression::LocalVariable(lv) => {
                // Return a tag -- loads and stores go through local_vars map.
                let tag = i64_type.const_int(lv.index() as u64, false);
                vec![tag.into()]
            }

            Expression::AccessIndex { base, index } => {
                self.eval_expr(*base, func)?;
                let base_ty = self.resolve_expr_type(*base, func);
                match base_ty {
                    TypeInner::Struct { members, .. } => {
                        // Params struct -- GEP + load at byte offset.
                        let base_ptr = self.get_expr_scalar(*base).into_pointer_value();
                        let offset = self.analysis.params_offsets[*index as usize];
                        let member_ty = &self.naga_module.types[members[*index as usize].ty].inner;
                        let llvm_ty = self.naga_type_to_llvm(member_ty);
                        let offset_val = i64_type.const_int(offset as u64, false);
                        let addr = unsafe {
                            self.builder.build_gep(i8_type, base_ptr, &[offset_val], "param_addr").unwrap()
                        };
                        let val = self.builder.build_load(llvm_ty, addr, "param_val").unwrap();
                        vec![val]
                    }
                    TypeInner::Pointer { base: base_ty_handle, .. } => {
                        let inner = &self.naga_module.types[*base_ty_handle].inner;
                        match inner {
                            TypeInner::Struct { members, .. } => {
                                let base_ptr = self.get_expr_scalar(*base).into_pointer_value();
                                let offset = self.analysis.params_offsets.get(*index as usize)
                                    .copied().unwrap_or((*index) * 4);
                                let member_ty = &self.naga_module.types[members[*index as usize].ty].inner;
                                let llvm_ty = self.naga_type_to_llvm(member_ty);
                                let offset_val = i64_type.const_int(offset as u64, false);
                                let addr = unsafe {
                                    self.builder.build_gep(i8_type, base_ptr, &[offset_val], "param_addr").unwrap()
                                };
                                let val = self.builder.build_load(llvm_ty, addr, "param_val").unwrap();
                                vec![val]
                            }
                            _ => return Err(format!("AccessIndex on pointer to non-struct: {inner:?}")),
                        }
                    }
                    TypeInner::Vector { .. } => {
                        // Check if base is a LocalVariable -- resolve through the var alloca.
                        let base_naga = &func.expressions[*base];
                        if let Expression::LocalVariable(lv) = base_naga {
                            let ptrs = self.local_vars[lv].clone();
                            let llvm_ty = self.naga_type_to_llvm(base_ty);
                            let val = self.builder.build_load(llvm_ty, ptrs[*index as usize], "lv_comp").unwrap();
                            vec![val]
                        } else {
                            let base_vals = self.get_expr(*base);
                            vec![base_vals[*index as usize]]
                        }
                    }
                    TypeInner::Scalar(_) => {
                        let base_vals = self.get_expr(*base);
                        vec![base_vals[0]]
                    }
                    _ => return Err(format!("AccessIndex on unsupported type: {base_ty:?}")),
                }
            }

            Expression::Access { base, index } => {
                self.eval_expr(*base, func)?;
                self.eval_expr(*index, func)?;
                let base_ty = self.resolve_expr_type(*base, func);
                match base_ty {
                    TypeInner::Array { base: elem_ty, .. } => {
                        let base_ptr = self.get_expr_scalar(*base).into_pointer_value();
                        let idx_val = self.get_expr_scalar(*index).into_int_value();
                        let elem_inner = &self.naga_module.types[*elem_ty].inner;
                        let elem_size = self.type_byte_size(elem_inner);
                        let idx_i64 = self.builder.build_int_z_extend(idx_val, i64_type, "idx_i64").unwrap();
                        let byte_offset = self.builder.build_int_mul(idx_i64, i64_type.const_int(elem_size as u64, false), "byte_off").unwrap();
                        let addr = unsafe {
                            self.builder.build_gep(i8_type, base_ptr, &[byte_offset], "elem_addr").unwrap()
                        };
                        let llvm_ty = self.naga_type_to_llvm(elem_inner);
                        let n_components = self.type_component_count(elem_inner);
                        if n_components == 1 {
                            let val = self.builder.build_load(llvm_ty, addr, "elem_val").unwrap();
                            vec![val]
                        } else {
                            let component_size = match elem_inner {
                                TypeInner::Vector { scalar, .. } => scalar.width as u64,
                                _ => 4,
                            };
                            (0..n_components as u64)
                                .map(|i| {
                                    let comp_off = i64_type.const_int(i * component_size, false);
                                    let comp_addr = unsafe {
                                        self.builder.build_gep(i8_type, addr, &[comp_off], &format!("comp_{i}_addr")).unwrap()
                                    };
                                    self.builder.build_load(llvm_ty, comp_addr, &format!("comp_{i}")).unwrap()
                                })
                                .collect()
                        }
                    }
                    TypeInner::Vector { .. } => {
                        let base_vals = self.get_expr(*base).to_vec();
                        let idx_val = self.get_expr_scalar(*index).into_int_value();
                        // Dynamic indexing via select chain.
                        let mut result = base_vals[0];
                        for i in 1..base_vals.len() {
                            let i_const = i32_type.const_int(i as u64, false);
                            let cmp = self.builder.build_int_compare(IntPredicate::EQ, idx_val, i_const, "idx_cmp").unwrap();
                            result = self.builder.build_select(cmp, base_vals[i], result, "sel").unwrap();
                        }
                        vec![result]
                    }
                    _ => return Err(format!("Access on unsupported type: {base_ty:?}")),
                }
            }

            Expression::Load { pointer } => {
                self.eval_expr(*pointer, func)?;
                let ptr_expr = &func.expressions[*pointer];
                match ptr_expr {
                    Expression::LocalVariable(lv) => {
                        let ptrs = self.local_vars[lv].clone();
                        let inner = &self.naga_module.types[func.local_variables[*lv].ty].inner;
                        let llvm_ty = self.naga_type_to_llvm(inner);
                        ptrs.iter().map(|p| self.builder.build_load(llvm_ty, *p, "lv_load").unwrap()).collect()
                    }
                    Expression::AccessIndex { base, index } => {
                        let base_expr = &func.expressions[*base];
                        if let Expression::LocalVariable(lv) = base_expr {
                            let ptrs = self.local_vars[lv].clone();
                            let inner = &self.naga_module.types[func.local_variables[*lv].ty].inner;
                            let llvm_ty = self.naga_type_to_llvm(inner);
                            vec![self.builder.build_load(llvm_ty, ptrs[*index as usize], "lv_comp_load").unwrap()]
                        } else {
                            self.get_expr(*pointer).to_vec()
                        }
                    }
                    _ => {
                        self.get_expr(*pointer).to_vec()
                    }
                }
            }

            Expression::Binary { op, left, right } => {
                self.eval_expr(*left, func)?;
                self.eval_expr(*right, func)?;
                let left_ty = self.resolve_expr_type(*left, func);
                let right_ty = self.resolve_expr_type(*right, func);
                let left_is_vec = matches!(left_ty, TypeInner::Vector { .. });
                let right_is_vec = matches!(right_ty, TypeInner::Vector { .. });

                // Vector binary operations: component-wise.
                if left_is_vec || right_is_vec {
                    let lhs_vals = self.get_expr(*left).to_vec();
                    let rhs_vals = self.get_expr(*right).to_vec();
                    let n = lhs_vals.len().max(rhs_vals.len());
                    let is_float = matches!(left_ty,
                        TypeInner::Vector { scalar, .. } if scalar.kind == ScalarKind::Float)
                        || matches!(right_ty,
                        TypeInner::Vector { scalar, .. } if scalar.kind == ScalarKind::Float);
                    let mut results = Vec::with_capacity(n);
                    for i in 0..n {
                        let l = lhs_vals[i % lhs_vals.len()];
                        let r = rhs_vals[i % rhs_vals.len()];
                        let v: BasicValueEnum = match op {
                            BinaryOperator::Add => if is_float { self.builder.build_float_add(l.into_float_value(), r.into_float_value(), "vadd").unwrap().into() } else { self.builder.build_int_add(l.into_int_value(), r.into_int_value(), "vadd").unwrap().into() },
                            BinaryOperator::Subtract => if is_float { self.builder.build_float_sub(l.into_float_value(), r.into_float_value(), "vsub").unwrap().into() } else { self.builder.build_int_sub(l.into_int_value(), r.into_int_value(), "vsub").unwrap().into() },
                            BinaryOperator::Multiply => if is_float { self.builder.build_float_mul(l.into_float_value(), r.into_float_value(), "vmul").unwrap().into() } else { self.builder.build_int_mul(l.into_int_value(), r.into_int_value(), "vmul").unwrap().into() },
                            BinaryOperator::Divide => if is_float { self.builder.build_float_div(l.into_float_value(), r.into_float_value(), "vdiv").unwrap().into() } else { self.builder.build_int_unsigned_div(l.into_int_value(), r.into_int_value(), "vdiv").unwrap().into() },
                            _ => return Err(format!("unsupported vector binary op: {op:?}")),
                        };
                        results.push(v);
                    }
                    self.expr_values.insert(handle, results);
                    return Ok(());
                }

                let mut lhs = self.get_expr_scalar(*left);
                let mut rhs = self.get_expr_scalar(*right);
                let is_float = matches!(left_ty, TypeInner::Scalar(s) if s.kind == ScalarKind::Float);
                let is_signed = matches!(left_ty, TypeInner::Scalar(s) if s.kind == ScalarKind::Sint);

                // Promote i8 (bool) operands to i32 when the other is i32.
                let lhs_bits = if lhs.is_int_value() { lhs.into_int_value().get_type().get_bit_width() } else { 0 };
                let rhs_bits = if rhs.is_int_value() { rhs.into_int_value().get_type().get_bit_width() } else { 0 };
                if lhs_bits == 8 && rhs_bits == 32 {
                    lhs = self.builder.build_int_z_extend(lhs.into_int_value(), i32_type, "bool_ext").unwrap().into();
                } else if rhs_bits == 8 && lhs_bits == 32 {
                    rhs = self.builder.build_int_z_extend(rhs.into_int_value(), i32_type, "bool_ext").unwrap().into();
                }

                let result: BasicValueEnum = match op {
                    BinaryOperator::Add => if is_float { self.builder.build_float_add(lhs.into_float_value(), rhs.into_float_value(), "add").unwrap().into() } else { self.builder.build_int_add(lhs.into_int_value(), rhs.into_int_value(), "add").unwrap().into() },
                    BinaryOperator::Subtract => if is_float { self.builder.build_float_sub(lhs.into_float_value(), rhs.into_float_value(), "sub").unwrap().into() } else { self.builder.build_int_sub(lhs.into_int_value(), rhs.into_int_value(), "sub").unwrap().into() },
                    BinaryOperator::Multiply => if is_float { self.builder.build_float_mul(lhs.into_float_value(), rhs.into_float_value(), "mul").unwrap().into() } else { self.builder.build_int_mul(lhs.into_int_value(), rhs.into_int_value(), "mul").unwrap().into() },
                    BinaryOperator::Divide => {
                        if is_float { self.builder.build_float_div(lhs.into_float_value(), rhs.into_float_value(), "div").unwrap().into() }
                        else if is_signed { self.builder.build_int_signed_div(lhs.into_int_value(), rhs.into_int_value(), "div").unwrap().into() }
                        else { self.builder.build_int_unsigned_div(lhs.into_int_value(), rhs.into_int_value(), "div").unwrap().into() }
                    }
                    BinaryOperator::Modulo => {
                        if is_float {
                            let div = self.builder.build_float_div(lhs.into_float_value(), rhs.into_float_value(), "div").unwrap();
                            let floored = call_f32_intrinsic(self.module, self.builder, "llvm.floor.f32", div, f32_type).into_float_value();
                            let prod = self.builder.build_float_mul(floored, rhs.into_float_value(), "prod").unwrap();
                            self.builder.build_float_sub(lhs.into_float_value(), prod, "mod").unwrap().into()
                        } else if is_signed { self.builder.build_int_signed_rem(lhs.into_int_value(), rhs.into_int_value(), "rem").unwrap().into() }
                        else { self.builder.build_int_unsigned_rem(lhs.into_int_value(), rhs.into_int_value(), "rem").unwrap().into() }
                    }
                    BinaryOperator::And => self.builder.build_and(lhs.into_int_value(), rhs.into_int_value(), "and").unwrap().into(),
                    BinaryOperator::InclusiveOr => self.builder.build_or(lhs.into_int_value(), rhs.into_int_value(), "or").unwrap().into(),
                    BinaryOperator::ExclusiveOr => self.builder.build_xor(lhs.into_int_value(), rhs.into_int_value(), "xor").unwrap().into(),
                    BinaryOperator::ShiftLeft => self.builder.build_left_shift(lhs.into_int_value(), rhs.into_int_value(), "shl").unwrap().into(),
                    BinaryOperator::ShiftRight => self.builder.build_right_shift(lhs.into_int_value(), rhs.into_int_value(), is_signed, "shr").unwrap().into(),
                    BinaryOperator::Equal => if is_float { self.i1_to_i8(self.builder.build_float_compare(FloatPredicate::OEQ, lhs.into_float_value(), rhs.into_float_value(), "eq").unwrap()) } else { self.i1_to_i8(self.builder.build_int_compare(IntPredicate::EQ, lhs.into_int_value(), rhs.into_int_value(), "eq").unwrap()) },
                    BinaryOperator::NotEqual => if is_float { self.i1_to_i8(self.builder.build_float_compare(FloatPredicate::ONE, lhs.into_float_value(), rhs.into_float_value(), "ne").unwrap()) } else { self.i1_to_i8(self.builder.build_int_compare(IntPredicate::NE, lhs.into_int_value(), rhs.into_int_value(), "ne").unwrap()) },
                    BinaryOperator::Less => {
                        if is_float { self.i1_to_i8(self.builder.build_float_compare(FloatPredicate::OLT, lhs.into_float_value(), rhs.into_float_value(), "lt").unwrap()) }
                        else if is_signed { self.i1_to_i8(self.builder.build_int_compare(IntPredicate::SLT, lhs.into_int_value(), rhs.into_int_value(), "lt").unwrap()) }
                        else { self.i1_to_i8(self.builder.build_int_compare(IntPredicate::ULT, lhs.into_int_value(), rhs.into_int_value(), "lt").unwrap()) }
                    }
                    BinaryOperator::LessEqual => {
                        if is_float { self.i1_to_i8(self.builder.build_float_compare(FloatPredicate::OLE, lhs.into_float_value(), rhs.into_float_value(), "le").unwrap()) }
                        else if is_signed { self.i1_to_i8(self.builder.build_int_compare(IntPredicate::SLE, lhs.into_int_value(), rhs.into_int_value(), "le").unwrap()) }
                        else { self.i1_to_i8(self.builder.build_int_compare(IntPredicate::ULE, lhs.into_int_value(), rhs.into_int_value(), "le").unwrap()) }
                    }
                    BinaryOperator::Greater => {
                        if is_float { self.i1_to_i8(self.builder.build_float_compare(FloatPredicate::OGT, lhs.into_float_value(), rhs.into_float_value(), "gt").unwrap()) }
                        else if is_signed { self.i1_to_i8(self.builder.build_int_compare(IntPredicate::SGT, lhs.into_int_value(), rhs.into_int_value(), "gt").unwrap()) }
                        else { self.i1_to_i8(self.builder.build_int_compare(IntPredicate::UGT, lhs.into_int_value(), rhs.into_int_value(), "gt").unwrap()) }
                    }
                    BinaryOperator::GreaterEqual => {
                        if is_float { self.i1_to_i8(self.builder.build_float_compare(FloatPredicate::OGE, lhs.into_float_value(), rhs.into_float_value(), "ge").unwrap()) }
                        else if is_signed { self.i1_to_i8(self.builder.build_int_compare(IntPredicate::SGE, lhs.into_int_value(), rhs.into_int_value(), "ge").unwrap()) }
                        else { self.i1_to_i8(self.builder.build_int_compare(IntPredicate::UGE, lhs.into_int_value(), rhs.into_int_value(), "ge").unwrap()) }
                    }
                    BinaryOperator::LogicalAnd => self.builder.build_and(lhs.into_int_value(), rhs.into_int_value(), "land").unwrap().into(),
                    BinaryOperator::LogicalOr => self.builder.build_or(lhs.into_int_value(), rhs.into_int_value(), "lor").unwrap().into(),
                };
                vec![result]
            }

            Expression::Unary { op, expr } => {
                self.eval_expr(*expr, func)?;
                let val = self.get_expr_scalar(*expr);
                let expr_ty = self.resolve_expr_type(*expr, func);
                let is_float = matches!(expr_ty, TypeInner::Scalar(s) if s.kind == ScalarKind::Float);
                let result: BasicValueEnum = match op {
                    UnaryOperator::Negate => if is_float { self.builder.build_float_neg(val.into_float_value(), "neg").unwrap().into() } else { self.builder.build_int_sub(i32_type.const_zero(), val.into_int_value(), "neg").unwrap().into() },
                    UnaryOperator::LogicalNot => { let one = i8_type.const_int(1, false); self.builder.build_xor(val.into_int_value(), one, "lnot").unwrap().into() }
                    UnaryOperator::BitwiseNot => self.builder.build_not(val.into_int_value(), "bnot").unwrap().into(),
                };
                vec![result]
            }

            Expression::As { expr, kind, convert } => {
                self.eval_expr(*expr, func)?;
                let (is_vector, src_scalar, vec_size) = {
                    let src_ty = self.resolve_expr_type(*expr, func);
                    match src_ty {
                        TypeInner::Scalar(s) => (false, *s, 0),
                        TypeInner::Vector { scalar, size } => (true, *scalar, *size as usize),
                        _ => return Err(format!("As cast on unsupported type: {src_ty:?}")),
                    }
                };
                if is_vector {
                    let vals = self.get_expr(*expr).to_vec();
                    let mut results = Vec::with_capacity(vec_size);
                    for v in &vals {
                        let r = self.cast_scalar(*v, &src_scalar, *kind, *convert)?;
                        results.push(r);
                    }
                    results
                } else {
                    let val = self.get_expr_scalar(*expr);
                    let result = self.cast_scalar(val, &src_scalar, *kind, *convert)?;
                    vec![result]
                }
            }

            Expression::Compose { components, .. } => {
                for c in components {
                    self.eval_expr(*c, func)?;
                }
                let components = match &func.expressions[handle] {
                    Expression::Compose { components, .. } => components.clone(),
                    _ => unreachable!(),
                };
                let mut vals = Vec::new();
                for c in &components {
                    vals.extend_from_slice(self.get_expr(*c));
                }
                vals
            }

            Expression::Splat { size, value } => {
                self.eval_expr(*value, func)?;
                let val = self.get_expr_scalar(*value);
                vec![val; *size as usize]
            }

            Expression::Swizzle { size, vector, pattern } => {
                self.eval_expr(*vector, func)?;
                let vec_vals = self.get_expr(*vector).to_vec();
                (0..*size as usize).map(|i| vec_vals[pattern[i] as usize]).collect()
            }

            Expression::Select { condition, accept, reject } => {
                self.eval_expr(*condition, func)?;
                self.eval_expr(*accept, func)?;
                self.eval_expr(*reject, func)?;
                let cond = self.to_i1(self.get_expr_scalar(*condition).into_int_value());
                let acc = self.get_expr(*accept).to_vec();
                let rej = self.get_expr(*reject).to_vec();
                acc.iter().zip(rej.iter())
                    .map(|(a, r)| self.builder.build_select(cond, *a, *r, "sel").unwrap())
                    .collect()
            }

            Expression::Relational { .. } => {
                vec![i8_type.const_zero().into()]
            }

            Expression::CallResult(_) => {
                if let Some(vals) = self.call_results.get(&handle) {
                    vals.clone()
                } else {
                    return Err("CallResult without preceding Call".into());
                }
            }

            Expression::ImageQuery { image, query } => {
                self.eval_expr(*image, func)?;
                match query {
                    naga::ImageQuery::Size { .. } => {
                        // textureDimensions(img) -> vec2<u32>
                        // TextureSlot: { data: *u8, width: u32, height: u32 } = 16 bytes
                        let img_expr = &func.expressions[*image];
                        let tex_idx = if let Expression::GlobalVariable(gv) = img_expr {
                            self.analysis.textures.iter().find(|t| t.global == *gv)
                                .map(|t| t.slot_index).unwrap_or(0)
                        } else { 0 };
                        // Load width at tex_slots + slot_index * 16 + 8
                        let w_off = i64_type.const_int((tex_idx * 16 + 8) as u64, false);
                        let w_addr = unsafe {
                            self.builder.build_gep(i8_type, self.p_tex_slots, &[w_off], "tex_w_addr").unwrap()
                        };
                        let w = self.builder.build_load(i32_type, w_addr, "tex_w").unwrap();
                        // Load height at tex_slots + slot_index * 16 + 12
                        let h_off = i64_type.const_int((tex_idx * 16 + 12) as u64, false);
                        let h_addr = unsafe {
                            self.builder.build_gep(i8_type, self.p_tex_slots, &[h_off], "tex_h_addr").unwrap()
                        };
                        let h = self.builder.build_load(i32_type, h_addr, "tex_h").unwrap();
                        vec![w, h]
                    }
                    _ => return Err(format!("unsupported ImageQuery: {query:?}")),
                }
            }

            Expression::ImageSample { image, sampler: _, coordinate, level, .. } => {
                self.eval_expr(*image, func)?;
                self.eval_expr(*coordinate, func)?;
                let img_expr = &func.expressions[*image];
                let tex_idx = if let Expression::GlobalVariable(gv) = img_expr {
                    self.analysis.textures.iter().find(|t| t.global == *gv)
                        .map(|t| t.slot_index).unwrap_or(0)
                } else { 0 };

                let uv = self.get_expr(*coordinate);
                let u_f32 = uv[0].into_float_value();
                let v_f32 = uv[1].into_float_value();
                // Promote f32 UV to f64 for the helper signature.
                let u_f64 = self.builder.build_float_ext(u_f32, f64_type, "u_f64").unwrap();
                let v_f64 = self.builder.build_float_ext(v_f32, f64_type, "v_f64").unwrap();

                let _ = level;

                // Call pd_tex_sample_bilinear_clamp(slots, tex, u, v, out)
                let ptr_type = self.context.ptr_type(inkwell::AddressSpace::default());
                let void_type = self.context.void_type();
                let fn_type = void_type.fn_type(
                    &[ptr_type.into(), i32_type.into(), f64_type.into(), f64_type.into(), ptr_type.into()],
                    false,
                );
                let helper = self.module.get_function("pd_tex_sample_bilinear_clamp").unwrap_or_else(|| {
                    self.module.add_function("pd_tex_sample_bilinear_clamp", fn_type, Some(inkwell::module::Linkage::External))
                });

                // Alloca for 4 x f32 = 16 bytes output.
                let out_alloca = self.builder.build_alloca(self.context.custom_width_int_type(128), "tex_out").unwrap();
                let tex_const = i32_type.const_int(tex_idx as u64, false);

                self.builder.build_call(helper, &[self.p_tex_slots.into(), tex_const.into(), u_f64.into(), v_f64.into(), out_alloca.into()], "").unwrap();

                // Load back 4 f32 values.
                let mut components = Vec::with_capacity(4);
                for i in 0..4u64 {
                    let offset = i64_type.const_int(i * 4, false);
                    let addr = unsafe {
                        self.builder.build_gep(i8_type, out_alloca, &[offset], &format!("tex_c{i}_ptr")).unwrap()
                    };
                    let val = self.builder.build_load(f32_type, addr, &format!("tex_c{i}")).unwrap();
                    components.push(val);
                }
                components
            }

            Expression::Math { fun, arg, arg1, arg2, arg3: _ } => {
                self.eval_expr(*arg, func)?;
                if let Some(a1) = arg1 { self.eval_expr(*a1, func)?; }
                if let Some(a2) = arg2 { self.eval_expr(*a2, func)?; }
                self.lower_math(*fun, *arg, *arg1, *arg2, func)?
            }

            _ => {
                return Err(format!("unsupported expression: {:?}", &func.expressions[handle]));
            }
        };

        self.expr_values.insert(handle, values);
        Ok(())
    }

    // ── Math lowering ───────────────────────────────────────────────────

    fn lower_math(
        &mut self,
        fun: MathFunction,
        arg: Handle<Expression>,
        arg1: Option<Handle<Expression>>,
        arg2: Option<Handle<Expression>>,
        func: &naga::Function,
    ) -> Result<Vec<BasicValueEnum<'static>>, String> {
        let arg_ty = self.resolve_expr_type(arg, func);
        let actual_components = self.get_expr(arg).len();
        if matches!(arg_ty, TypeInner::Vector { .. }) || actual_components > 1 {
            return self.lower_math_vector(fun, arg, arg1, arg2, func);
        }

        let val = self.get_expr_scalar(arg);
        let f32_type = self.context.f32_type();
        let i32_type = self.context.i32_type();
        let is_f32 = matches!(arg_ty, TypeInner::Scalar(s) if s.kind == ScalarKind::Float);

        let result: BasicValueEnum = match fun {
            MathFunction::Abs => {
                if is_f32 {
                    call_f32_intrinsic(self.module, self.builder, "llvm.fabs.f32", val.into_float_value(), f32_type)
                } else {
                    // Integer abs: val < 0 ? -val : val
                    let is_signed = matches!(arg_ty, TypeInner::Scalar(s) if s.kind == ScalarKind::Sint);
                    if is_signed {
                        let zero = i32_type.const_zero();
                        let is_neg = self.builder.build_int_compare(IntPredicate::SLT, val.into_int_value(), zero, "is_neg").unwrap();
                        let negated = self.builder.build_int_sub(zero, val.into_int_value(), "negated").unwrap();
                        self.builder.build_select(is_neg, negated, val.into_int_value(), "abs").unwrap()
                    } else {
                        val
                    }
                }
            }
            MathFunction::Min => {
                let rhs = self.get_expr_scalar(arg1.unwrap());
                if is_f32 {
                    call_f32_binary_intrinsic(self.module, self.builder, "llvm.minnum.f32", val.into_float_value(), rhs.into_float_value(), f32_type)
                } else {
                    let is_signed = matches!(arg_ty, TypeInner::Scalar(s) if s.kind == ScalarKind::Sint);
                    if is_signed {
                        call_int_binary_intrinsic(self.module, self.builder, "llvm.smin.i32", val.into_int_value(), rhs.into_int_value(), i32_type)
                    } else {
                        call_int_binary_intrinsic(self.module, self.builder, "llvm.umin.i32", val.into_int_value(), rhs.into_int_value(), i32_type)
                    }
                }
            }
            MathFunction::Max => {
                let rhs = self.get_expr_scalar(arg1.unwrap());
                if is_f32 {
                    call_f32_binary_intrinsic(self.module, self.builder, "llvm.maxnum.f32", val.into_float_value(), rhs.into_float_value(), f32_type)
                } else {
                    let is_signed = matches!(arg_ty, TypeInner::Scalar(s) if s.kind == ScalarKind::Sint);
                    if is_signed {
                        call_int_binary_intrinsic(self.module, self.builder, "llvm.smax.i32", val.into_int_value(), rhs.into_int_value(), i32_type)
                    } else {
                        call_int_binary_intrinsic(self.module, self.builder, "llvm.umax.i32", val.into_int_value(), rhs.into_int_value(), i32_type)
                    }
                }
            }
            MathFunction::Clamp => {
                let min_val = self.get_expr_scalar(arg1.unwrap());
                let max_val = self.get_expr_scalar(arg2.unwrap());
                if is_f32 {
                    let t = call_f32_binary_intrinsic(self.module, self.builder, "llvm.maxnum.f32", val.into_float_value(), min_val.into_float_value(), f32_type);
                    call_f32_binary_intrinsic(self.module, self.builder, "llvm.minnum.f32", t.into_float_value(), max_val.into_float_value(), f32_type)
                } else {
                    let t = call_int_binary_intrinsic(self.module, self.builder, "llvm.umax.i32", val.into_int_value(), min_val.into_int_value(), i32_type);
                    call_int_binary_intrinsic(self.module, self.builder, "llvm.umin.i32", t.into_int_value(), max_val.into_int_value(), i32_type)
                }
            }
            MathFunction::Floor => call_f32_intrinsic(self.module, self.builder, "llvm.floor.f32", val.into_float_value(), f32_type),
            MathFunction::Ceil => call_f32_intrinsic(self.module, self.builder, "llvm.ceil.f32", val.into_float_value(), f32_type),
            MathFunction::Round => call_f32_intrinsic(self.module, self.builder, "llvm.round.f32", val.into_float_value(), f32_type),
            MathFunction::Trunc => call_f32_intrinsic(self.module, self.builder, "llvm.trunc.f32", val.into_float_value(), f32_type),
            MathFunction::Fract => {
                let f = call_f32_intrinsic(self.module, self.builder, "llvm.floor.f32", val.into_float_value(), f32_type);
                self.builder.build_float_sub(val.into_float_value(), f.into_float_value(), "fract").unwrap().into()
            }
            MathFunction::Sqrt => call_f32_intrinsic(self.module, self.builder, "llvm.sqrt.f32", val.into_float_value(), f32_type),
            MathFunction::Sign => {
                let zero = f32_type.const_float(0.0);
                let one = f32_type.const_float(1.0);
                let neg_one = f32_type.const_float(-1.0);
                let gt = self.builder.build_float_compare(FloatPredicate::OGT, val.into_float_value(), zero, "gt").unwrap();
                let lt = self.builder.build_float_compare(FloatPredicate::OLT, val.into_float_value(), zero, "lt").unwrap();
                let pos = self.builder.build_select(gt, one, zero, "pos").unwrap();
                self.builder.build_select(lt, neg_one, pos.into_float_value(), "sign").unwrap()
            }
            MathFunction::Sin => call_f32_intrinsic(self.module, self.builder, "llvm.sin.f32", val.into_float_value(), f32_type),
            MathFunction::Cos => call_f32_intrinsic(self.module, self.builder, "llvm.cos.f32", val.into_float_value(), f32_type),
            MathFunction::Tan => call_f32_intrinsic(self.module, self.builder, "llvm.tan.f32", val.into_float_value(), f32_type),
            MathFunction::Asin => call_f32_intrinsic(self.module, self.builder, "llvm.asin.f32", val.into_float_value(), f32_type),
            MathFunction::Acos => call_f32_intrinsic(self.module, self.builder, "llvm.acos.f32", val.into_float_value(), f32_type),
            MathFunction::Atan => call_f32_intrinsic(self.module, self.builder, "llvm.atan.f32", val.into_float_value(), f32_type),
            MathFunction::Atan2 => {
                let rhs = self.get_expr_scalar(arg1.unwrap());
                call_f32_binary_intrinsic(self.module, self.builder, "llvm.atan2.f32", val.into_float_value(), rhs.into_float_value(), f32_type)
            }
            MathFunction::Exp => call_f32_intrinsic(self.module, self.builder, "llvm.exp.f32", val.into_float_value(), f32_type),
            MathFunction::Exp2 => call_f32_intrinsic(self.module, self.builder, "llvm.exp2.f32", val.into_float_value(), f32_type),
            MathFunction::Log => call_f32_intrinsic(self.module, self.builder, "llvm.log.f32", val.into_float_value(), f32_type),
            MathFunction::Log2 => call_f32_intrinsic(self.module, self.builder, "llvm.log2.f32", val.into_float_value(), f32_type),
            MathFunction::Pow => {
                let rhs = self.get_expr_scalar(arg1.unwrap());
                call_f32_binary_intrinsic(self.module, self.builder, "llvm.pow.f32", val.into_float_value(), rhs.into_float_value(), f32_type)
            }
            MathFunction::Mix => {
                let b = self.get_expr_scalar(arg1.unwrap()).into_float_value();
                let t = self.get_expr_scalar(arg2.unwrap()).into_float_value();
                let one = f32_type.const_float(1.0);
                let omt = self.builder.build_float_sub(one, t, "omt").unwrap();
                let at = self.builder.build_float_mul(val.into_float_value(), omt, "at").unwrap();
                let bt = self.builder.build_float_mul(b, t, "bt").unwrap();
                self.builder.build_float_add(at, bt, "mix").unwrap().into()
            }
            MathFunction::SmoothStep => {
                let edge1 = self.get_expr_scalar(arg1.unwrap()).into_float_value();
                let x = self.get_expr_scalar(arg2.unwrap()).into_float_value();
                let diff = self.builder.build_float_sub(x, val.into_float_value(), "diff").unwrap();
                let range = self.builder.build_float_sub(edge1, val.into_float_value(), "range").unwrap();
                let ratio = self.builder.build_float_div(diff, range, "ratio").unwrap();
                let zero = f32_type.const_float(0.0);
                let one = f32_type.const_float(1.0);
                let t = call_f32_binary_intrinsic(self.module, self.builder, "llvm.maxnum.f32", ratio, zero, f32_type).into_float_value();
                let t = call_f32_binary_intrinsic(self.module, self.builder, "llvm.minnum.f32", t, one, f32_type).into_float_value();
                let three = f32_type.const_float(3.0);
                let two = f32_type.const_float(2.0);
                let two_t = self.builder.build_float_mul(two, t, "two_t").unwrap();
                let s = self.builder.build_float_sub(three, two_t, "s").unwrap();
                let t2 = self.builder.build_float_mul(t, t, "t2").unwrap();
                self.builder.build_float_mul(t2, s, "smooth").unwrap().into()
            }
            MathFunction::Length => {
                call_f32_intrinsic(self.module, self.builder, "llvm.fabs.f32", val.into_float_value(), f32_type)
            }
            MathFunction::Dot => {
                let rhs = self.get_expr_scalar(arg1.unwrap()).into_float_value();
                self.builder.build_float_mul(val.into_float_value(), rhs, "dot").unwrap().into()
            }
            _ => return Err(format!("unsupported scalar math: {fun:?}")),
        };
        Ok(vec![result])
    }

    fn lower_math_vector(
        &mut self,
        fun: MathFunction,
        arg: Handle<Expression>,
        arg1: Option<Handle<Expression>>,
        arg2: Option<Handle<Expression>>,
        _func: &naga::Function,
    ) -> Result<Vec<BasicValueEnum<'static>>, String> {
        let vals = self.get_expr(arg).to_vec();
        let n = vals.len();
        let f32_type = self.context.f32_type();
        match fun {
            MathFunction::Dot => {
                let rhs = self.get_expr(arg1.unwrap()).to_vec();
                let mut sum = self.builder.build_float_mul(vals[0].into_float_value(), rhs[0].into_float_value(), "dot0").unwrap();
                for i in 1..n {
                    let p = self.builder.build_float_mul(vals[i].into_float_value(), rhs[i].into_float_value(), "dotp").unwrap();
                    sum = self.builder.build_float_add(sum, p, "dots").unwrap();
                }
                Ok(vec![sum.into()])
            }
            MathFunction::Length => {
                let mut sum = self.builder.build_float_mul(vals[0].into_float_value(), vals[0].into_float_value(), "len0").unwrap();
                for i in 1..n {
                    let p = self.builder.build_float_mul(vals[i].into_float_value(), vals[i].into_float_value(), "lenp").unwrap();
                    sum = self.builder.build_float_add(sum, p, "lens").unwrap();
                }
                let result = call_f32_intrinsic(self.module, self.builder, "llvm.sqrt.f32", sum, f32_type);
                Ok(vec![result])
            }
            MathFunction::Normalize => {
                let mut sum = self.builder.build_float_mul(vals[0].into_float_value(), vals[0].into_float_value(), "n0").unwrap();
                for i in 1..n {
                    let p = self.builder.build_float_mul(vals[i].into_float_value(), vals[i].into_float_value(), "np").unwrap();
                    sum = self.builder.build_float_add(sum, p, "ns").unwrap();
                }
                let len = call_f32_intrinsic(self.module, self.builder, "llvm.sqrt.f32", sum, f32_type).into_float_value();
                Ok(vals.iter().map(|v| self.builder.build_float_div(v.into_float_value(), len, "norm").unwrap().into()).collect())
            }
            MathFunction::Cross => {
                let rhs = self.get_expr(arg1.unwrap()).to_vec();
                let a = self.builder.build_float_mul(vals[1].into_float_value(), rhs[2].into_float_value(), "c0").unwrap();
                let b = self.builder.build_float_mul(vals[2].into_float_value(), rhs[1].into_float_value(), "c1").unwrap();
                let x = self.builder.build_float_sub(a, b, "cx").unwrap();
                let a = self.builder.build_float_mul(vals[2].into_float_value(), rhs[0].into_float_value(), "c2").unwrap();
                let b = self.builder.build_float_mul(vals[0].into_float_value(), rhs[2].into_float_value(), "c3").unwrap();
                let y = self.builder.build_float_sub(a, b, "cy").unwrap();
                let a = self.builder.build_float_mul(vals[0].into_float_value(), rhs[1].into_float_value(), "c4").unwrap();
                let b = self.builder.build_float_mul(vals[1].into_float_value(), rhs[0].into_float_value(), "c5").unwrap();
                let z = self.builder.build_float_sub(a, b, "cz").unwrap();
                Ok(vec![x.into(), y.into(), z.into()])
            }
            MathFunction::Min => {
                let rhs = self.get_expr(arg1.unwrap()).to_vec();
                Ok(vals.iter().zip(rhs.iter()).map(|(a, b)| {
                    call_f32_binary_intrinsic(self.module, self.builder, "llvm.minnum.f32", a.into_float_value(), b.into_float_value(), f32_type)
                }).collect())
            }
            MathFunction::Max => {
                let rhs = self.get_expr(arg1.unwrap()).to_vec();
                Ok(vals.iter().zip(rhs.iter()).map(|(a, b)| {
                    call_f32_binary_intrinsic(self.module, self.builder, "llvm.maxnum.f32", a.into_float_value(), b.into_float_value(), f32_type)
                }).collect())
            }
            MathFunction::Abs => {
                Ok(vals.iter().map(|v| call_f32_intrinsic(self.module, self.builder, "llvm.fabs.f32", v.into_float_value(), f32_type)).collect())
            }
            MathFunction::Clamp => {
                let min_v = self.get_expr(arg1.unwrap()).to_vec();
                let max_v = self.get_expr(arg2.unwrap()).to_vec();
                Ok(vals.iter().enumerate().map(|(i, v)| {
                    let t = call_f32_binary_intrinsic(self.module, self.builder, "llvm.maxnum.f32", v.into_float_value(), min_v[i].into_float_value(), f32_type);
                    call_f32_binary_intrinsic(self.module, self.builder, "llvm.minnum.f32", t.into_float_value(), max_v[i].into_float_value(), f32_type)
                }).collect())
            }
            MathFunction::Mix => {
                let b_vals = self.get_expr(arg1.unwrap()).to_vec();
                let t_vals = self.get_expr(arg2.unwrap()).to_vec();
                let one = f32_type.const_float(1.0);
                Ok(vals.iter().enumerate().map(|(i, a)| {
                    let t = t_vals[i % t_vals.len()].into_float_value();
                    let omt = self.builder.build_float_sub(one, t, "omt").unwrap();
                    let at = self.builder.build_float_mul(a.into_float_value(), omt, "at").unwrap();
                    let bt = self.builder.build_float_mul(b_vals[i].into_float_value(), t, "bt").unwrap();
                    self.builder.build_float_add(at, bt, "mix").unwrap().into()
                }).collect())
            }
            _ => {
                // Component-wise fallback for single-arg math.
                let mut results = Vec::with_capacity(n);
                for v in &vals {
                    let r = match fun {
                        MathFunction::Floor => call_f32_intrinsic(self.module, self.builder, "llvm.floor.f32", v.into_float_value(), f32_type),
                        MathFunction::Ceil => call_f32_intrinsic(self.module, self.builder, "llvm.ceil.f32", v.into_float_value(), f32_type),
                        MathFunction::Sqrt => call_f32_intrinsic(self.module, self.builder, "llvm.sqrt.f32", v.into_float_value(), f32_type),
                        MathFunction::Fract => {
                            let f = call_f32_intrinsic(self.module, self.builder, "llvm.floor.f32", v.into_float_value(), f32_type);
                            self.builder.build_float_sub(v.into_float_value(), f.into_float_value(), "fract").unwrap().into()
                        }
                        _ => return Err(format!("unsupported vector math: {fun:?}")),
                    };
                    results.push(r);
                }
                Ok(results)
            }
        }
    }

    /// Cast a single scalar value from one type to another.
    fn cast_scalar(
        &mut self,
        val: BasicValueEnum<'static>,
        src: &naga::Scalar,
        dst_kind: ScalarKind,
        convert: Option<u8>,
    ) -> Result<BasicValueEnum<'static>, String> {
        let dst_width = convert.unwrap_or(src.width);
        if src.kind == dst_kind && src.width == dst_width {
            return Ok(val);
        }
        let i32_type = self.context.i32_type();
        let f32_type = self.context.f32_type();

        Ok(match convert {
            Some(width) => match (src.kind, src.width, dst_kind, width) {
                (ScalarKind::Uint, 4, ScalarKind::Float, 4) => self.builder.build_unsigned_int_to_float(val.into_int_value(), f32_type, "u2f").unwrap().into(),
                (ScalarKind::Sint, 4, ScalarKind::Float, 4) => self.builder.build_signed_int_to_float(val.into_int_value(), f32_type, "s2f").unwrap().into(),
                (ScalarKind::Float, 4, ScalarKind::Sint, 4) => self.builder.build_float_to_signed_int(val.into_float_value(), i32_type, "f2s").unwrap().into(),
                (ScalarKind::Float, 4, ScalarKind::Uint, 4) => {
                    // Convert via signed to avoid UB on negative values.
                    self.builder.build_float_to_signed_int(val.into_float_value(), i32_type, "f2u").unwrap().into()
                }
                (ScalarKind::Sint, 4, ScalarKind::Uint, 4) | (ScalarKind::Uint, 4, ScalarKind::Sint, 4) => val,
                // Bool -> integer: zero-extend i8 to i32
                (ScalarKind::Bool, 1, ScalarKind::Uint, 4) | (ScalarKind::Bool, 1, ScalarKind::Sint, 4) => {
                    self.builder.build_int_z_extend(val.into_int_value(), i32_type, "b2i").unwrap().into()
                }
                // Bool -> float
                (ScalarKind::Bool, 1, ScalarKind::Float, 4) => {
                    let ext = self.builder.build_int_z_extend(val.into_int_value(), i32_type, "b2i").unwrap();
                    self.builder.build_unsigned_int_to_float(ext, f32_type, "b2f").unwrap().into()
                }
                _ if src.width == width && src.kind == dst_kind => val,
                _ => return Err(format!("unsupported cast: {:?}/{} -> {:?}/{}", src.kind, src.width, dst_kind, width)),
            },
            None => match (src.kind, src.width, dst_kind) {
                (ScalarKind::Sint, 4, ScalarKind::Uint) | (ScalarKind::Uint, 4, ScalarKind::Sint) => val,
                (ScalarKind::Float, 4, ScalarKind::Uint) | (ScalarKind::Float, 4, ScalarKind::Sint) => {
                    self.builder.build_bit_cast(val, i32_type, "f2i_bc").unwrap()
                }
                (ScalarKind::Uint, 4, ScalarKind::Float) | (ScalarKind::Sint, 4, ScalarKind::Float) => {
                    self.builder.build_bit_cast(val, f32_type, "i2f_bc").unwrap()
                }
                _ => return Err(format!("unsupported bitcast: {:?}/{} -> {:?}", src.kind, src.width, dst_kind)),
            },
        })
    }

    // ── Statement lowering ──────────────────────────────────────────────

    fn lower_block(&mut self, block: &naga::Block, func: &naga::Function) -> Result<(), String> {
        for stmt in block.iter() {
            if self.terminated { break; }
            self.lower_statement(stmt, func)?;
        }
        Ok(())
    }

    fn lower_statement(&mut self, stmt: &Statement, func: &naga::Function) -> Result<(), String> {
        match stmt {
            Statement::Emit(range) => {
                for handle in range.clone() {
                    self.eval_expr(handle, func)?;
                }
            }

            Statement::If { condition, accept, reject } => {
                self.eval_expr(*condition, func)?;
                let cond = self.to_i1(self.get_expr_scalar(*condition).into_int_value());
                let then_block = self.context.append_basic_block(self.function, "then");
                let else_block = self.context.append_basic_block(self.function, "else");
                let merge_block = self.context.append_basic_block(self.function, "merge");

                self.builder.build_conditional_branch(cond, then_block, else_block).unwrap();

                self.builder.position_at_end(then_block);
                self.terminated = false;
                self.lower_block(accept, func)?;
                let then_terminated = self.terminated;
                if !then_terminated {
                    self.builder.build_unconditional_branch(merge_block).unwrap();
                }

                self.builder.position_at_end(else_block);
                self.terminated = false;
                self.lower_block(reject, func)?;
                let else_terminated = self.terminated;
                if !else_terminated {
                    self.builder.build_unconditional_branch(merge_block).unwrap();
                }

                if then_terminated && else_terminated {
                    unsafe { merge_block.delete().unwrap(); }
                } else {
                    self.builder.position_at_end(merge_block);
                }
                self.terminated = then_terminated && else_terminated;
            }

            Statement::Return { .. } => {
                self.builder.build_unconditional_branch(self.return_block).unwrap();
                self.terminated = true;
            }

            Statement::Store { pointer, value } => {
                self.eval_expr(*value, func)?;
                self.eval_expr(*pointer, func)?;
                let src_vals = self.get_expr(*value).to_vec();
                let ptr_expr = &func.expressions[*pointer];
                match ptr_expr {
                    Expression::LocalVariable(lv) => {
                        let ptrs = self.local_vars[lv].clone();
                        for (i, p) in ptrs.iter().enumerate() {
                            if i < src_vals.len() {
                                self.builder.build_store(*p, src_vals[i]).unwrap();
                            }
                        }
                    }
                    Expression::AccessIndex { base, index } => {
                        let base_ptr_expr = &func.expressions[*base];
                        if let Expression::LocalVariable(lv) = base_ptr_expr {
                            let ptrs = self.local_vars[lv].clone();
                            let idx = *index as usize;
                            if idx < ptrs.len() {
                                self.builder.build_store(ptrs[idx], src_vals[0]).unwrap();
                            }
                        }
                    }
                    Expression::Access { base, index } => {
                        self.eval_expr(*base, func)?;
                        self.eval_expr(*index, func)?;
                        let buf_ptr = self.get_expr_scalar(*base).into_pointer_value();
                        let idx_val = self.get_expr_scalar(*index).into_int_value();
                        let i64_type = self.context.i64_type();
                        let i8_type = self.context.i8_type();
                        let idx_i64 = self.builder.build_int_z_extend(idx_val, i64_type, "idx_i64").unwrap();

                        // Determine element size.
                        let base_expr = &func.expressions[*base];
                        let (elem_bytes, n_components) = if let Expression::GlobalVariable(gv) = base_expr {
                            if let Some(info) = self.analysis.storage_buffers.iter().find(|b| b.global == *gv) {
                                (info.elem_bytes, info.elem_components)
                            } else { (4, 1) }
                        } else { (4, 1) };

                        let byte_offset = self.builder.build_int_mul(idx_i64, i64_type.const_int(elem_bytes as u64, false), "byte_off").unwrap();
                        let addr = unsafe {
                            self.builder.build_gep(i8_type, buf_ptr, &[byte_offset], "store_addr").unwrap()
                        };
                        if n_components == 1 {
                            self.builder.build_store(addr, src_vals[0]).unwrap();
                        } else {
                            let component_bytes = elem_bytes / n_components as u32;
                            for (i, v) in src_vals.iter().enumerate() {
                                let comp_off = i64_type.const_int((i as u32 * component_bytes) as u64, false);
                                let comp_addr = unsafe {
                                    self.builder.build_gep(i8_type, addr, &[comp_off], &format!("store_c{i}")).unwrap()
                                };
                                self.builder.build_store(comp_addr, *v).unwrap();
                            }
                        }
                    }
                    _ => {}
                }
            }

            Statement::Loop { body, continuing, break_if } => {
                let loop_body_block = self.context.append_basic_block(self.function, "loop_body");
                let loop_continuing_block = self.context.append_basic_block(self.function, "loop_cont");
                let loop_exit = self.context.append_basic_block(self.function, "loop_exit");

                self.builder.build_unconditional_branch(loop_body_block).unwrap();

                self.builder.position_at_end(loop_body_block);
                self.loop_stack.push((loop_exit, loop_continuing_block));
                self.terminated = false;
                self.lower_block(body, func)?;
                if !self.terminated {
                    self.builder.build_unconditional_branch(loop_continuing_block).unwrap();
                }

                self.builder.position_at_end(loop_continuing_block);
                self.terminated = false;
                self.lower_block(continuing, func)?;
                if let Some(break_cond) = break_if {
                    self.eval_expr(*break_cond, func)?;
                    let cond = self.to_i1(self.get_expr_scalar(*break_cond).into_int_value());
                    self.builder.build_conditional_branch(cond, loop_exit, loop_body_block).unwrap();
                } else if !self.terminated {
                    self.builder.build_unconditional_branch(loop_body_block).unwrap();
                }

                self.loop_stack.pop();
                self.builder.position_at_end(loop_exit);
                self.terminated = false;
            }

            Statement::Break => {
                let (exit, _) = *self.loop_stack.last().unwrap();
                self.builder.build_unconditional_branch(exit).unwrap();
                self.terminated = true;
            }

            Statement::Continue => {
                let (_, cont) = *self.loop_stack.last().unwrap();
                self.builder.build_unconditional_branch(cont).unwrap();
                self.terminated = true;
            }

            Statement::Block(block) => {
                self.lower_block(block, func)?;
            }

            Statement::Switch { selector, cases } => {
                let sel_val = self.get_expr_scalar(*selector).into_int_value();
                let i32_type = self.context.i32_type();
                let merge_block = self.context.append_basic_block(self.function, "switch_merge");
                let mut case_blocks = Vec::new();
                let mut default_idx = None;

                for (i, case) in cases.iter().enumerate() {
                    case_blocks.push(self.context.append_basic_block(self.function, &format!("case_{i}")));
                    if case.value == naga::SwitchValue::Default {
                        default_idx = Some(i);
                    }
                }
                let default_block = default_idx.map(|i| case_blocks[i]).unwrap_or(merge_block);

                // Emit comparison chain.
                for (i, case) in cases.iter().enumerate() {
                    let val = match case.value {
                        naga::SwitchValue::I32(v) => Some(i32_type.const_int(v as u64, true)),
                        naga::SwitchValue::U32(v) => Some(i32_type.const_int(v as u64, false)),
                        naga::SwitchValue::Default => None,
                    };
                    if let Some(v) = val {
                        let cmp = self.builder.build_int_compare(IntPredicate::EQ, sel_val, v, "case_cmp").unwrap();
                        let next = self.context.append_basic_block(self.function, "case_next");
                        self.builder.build_conditional_branch(cmp, case_blocks[i], next).unwrap();
                        self.builder.position_at_end(next);
                    }
                }
                self.builder.build_unconditional_branch(default_block).unwrap();

                // Lower case bodies.
                self.loop_stack.push((merge_block, merge_block));
                for (i, case) in cases.iter().enumerate() {
                    self.builder.position_at_end(case_blocks[i]);
                    self.terminated = false;
                    self.lower_block(&case.body, func)?;
                    if !self.terminated {
                        if case.fall_through && i + 1 < cases.len() {
                            self.builder.build_unconditional_branch(case_blocks[i + 1]).unwrap();
                        } else {
                            self.builder.build_unconditional_branch(merge_block).unwrap();
                        }
                    }
                }
                self.loop_stack.pop();

                self.builder.position_at_end(merge_block);
                self.terminated = false;
            }

            Statement::Call { function, arguments, result } => {
                self.inline_call(*function, arguments, *result, func)?;
            }

            _ => {} // Ignore unsupported statements.
        }
        Ok(())
    }

    /// Inline a function call: evaluate args, lower callee body, capture return value.
    fn inline_call(
        &mut self,
        callee_handle: Handle<naga::Function>,
        caller_args: &[Handle<Expression>],
        result_expr: Option<Handle<Expression>>,
        caller_func: &naga::Function,
    ) -> Result<(), String> {
        let callee = &self.naga_module.functions[callee_handle];

        // Evaluate caller argument expressions.
        let arg_values: Vec<Vec<BasicValueEnum<'static>>> = caller_args
            .iter()
            .map(|a| {
                self.eval_expr(*a, caller_func).ok();
                self.get_expr(*a).to_vec()
            })
            .collect();

        // Save caller state.
        let saved_expr_values = std::mem::take(&mut self.expr_values);
        let saved_local_vars = std::mem::take(&mut self.local_vars);
        let saved_return_block = self.return_block;
        let saved_terminated = self.terminated;

        // Create a return block for the inlined function.
        let inline_return_block = self.context.append_basic_block(self.function, "inline_ret");
        self.return_block = inline_return_block;
        self.terminated = false;

        // Seed callee FunctionArgument expressions with caller arg values.
        for (handle, expr) in callee.expressions.iter() {
            if let Expression::FunctionArgument(idx) = expr {
                self.expr_values.insert(handle, arg_values[*idx as usize].clone());
            }
        }

        // Declare callee local variables as allocas.
        for (handle, local) in callee.local_variables.iter() {
            let llvm_ty = self.naga_type_to_llvm(&self.naga_module.types[local.ty].inner);
            let components = self.type_component_count(&self.naga_module.types[local.ty].inner);
            let mut alloca_ptrs = Vec::new();
            for c in 0..components {
                let alloca = self.builder.build_alloca(llvm_ty, &format!("inline_local_{c}")).unwrap();
                self.builder.build_store(alloca, self.zero_value(llvm_ty)).unwrap();
                alloca_ptrs.push(alloca);
            }
            if let Some(init) = local.init {
                self.eval_expr(init, callee)?;
                let init_vals = self.get_expr(init).to_vec();
                for (i, ptr) in alloca_ptrs.iter().enumerate() {
                    if i < init_vals.len() {
                        self.builder.build_store(*ptr, init_vals[i]).unwrap();
                    }
                }
            }
            self.local_vars.insert(handle, alloca_ptrs);
        }

        // Use allocas to carry the return value(s) out.
        // For vector return types, allocate one alloca per component.
        let ret_allocas = if result_expr.is_some() {
            let (ret_ty, n_components) = callee.result.as_ref().map(|r| {
                let inner = &self.naga_module.types[r.ty].inner;
                (self.naga_type_to_llvm(inner), self.type_component_count(inner))
            }).unwrap_or((self.context.i32_type().into(), 1));
            let mut allocas = Vec::with_capacity(n_components);
            for i in 0..n_components {
                let alloca = self.builder.build_alloca(ret_ty, &format!("ret_val{i}")).unwrap();
                self.builder.build_store(alloca, self.zero_value(ret_ty)).unwrap();
                allocas.push(alloca);
            }
            Some((allocas, ret_ty))
        } else {
            None
        };

        // Lower callee body, intercepting Return statements.
        self.lower_block_with_return(
            &callee.body,
            callee,
            ret_allocas.as_ref().map(|(a, _)| a.as_slice()),
        )?;

        if !self.terminated {
            self.builder.build_unconditional_branch(inline_return_block).unwrap();
        }
        self.builder.position_at_end(inline_return_block);

        // Capture return values.
        let return_values = ret_allocas.map(|(allocas, ret_ty)| {
            allocas.iter().enumerate().map(|(i, alloca)| {
                self.builder.build_load(ret_ty, *alloca, &format!("ret_load{i}")).unwrap()
            }).collect::<Vec<_>>()
        });

        // Restore caller state.
        self.expr_values = saved_expr_values;
        self.local_vars = saved_local_vars;
        self.return_block = saved_return_block;
        self.terminated = saved_terminated;

        if let (Some(result_handle), Some(vals)) = (result_expr, return_values) {
            self.call_results.insert(result_handle, vals);
        }

        Ok(())
    }

    /// Lower a block, converting Return statements to store the value and jump.
    fn lower_block_with_return(
        &mut self,
        block: &naga::Block,
        func: &naga::Function,
        ret_allocas: Option<&[PointerValue<'static>]>,
    ) -> Result<(), String> {
        for stmt in block.iter() {
            if self.terminated { break; }
            match stmt {
                Statement::Return { value } => {
                    if let (Some(val_handle), Some(allocas)) = (value, ret_allocas) {
                        self.eval_expr(*val_handle, func)?;
                        let vals = self.get_expr(*val_handle);
                        for (i, alloca) in allocas.iter().enumerate() {
                            if i < vals.len() {
                                self.builder.build_store(*alloca, vals[i]).unwrap();
                            }
                        }
                    }
                    self.builder.build_unconditional_branch(self.return_block).unwrap();
                    self.terminated = true;
                }
                Statement::If { condition, accept, reject } => {
                    let cond = self.to_i1(self.get_expr_scalar(*condition).into_int_value());
                    let then_block = self.context.append_basic_block(self.function, "wr_then");
                    let else_block = self.context.append_basic_block(self.function, "wr_else");
                    let merge_block = self.context.append_basic_block(self.function, "wr_merge");

                    self.builder.build_conditional_branch(cond, then_block, else_block).unwrap();

                    self.builder.position_at_end(then_block);
                    self.terminated = false;
                    self.lower_block_with_return(accept, func, ret_allocas)?;
                    let then_term = self.terminated;
                    if !then_term { self.builder.build_unconditional_branch(merge_block).unwrap(); }

                    self.builder.position_at_end(else_block);
                    self.terminated = false;
                    self.lower_block_with_return(reject, func, ret_allocas)?;
                    let else_term = self.terminated;
                    if !else_term { self.builder.build_unconditional_branch(merge_block).unwrap(); }

                    if then_term && else_term {
                        // Both branches terminated — merge block is unreachable.
                        // Remove it to avoid unterminated block.
                        unsafe { merge_block.delete().unwrap(); }
                    } else {
                        self.builder.position_at_end(merge_block);
                    }
                    self.terminated = then_term && else_term;
                }
                Statement::Loop { body, continuing, break_if } => {
                    let loop_body_block = self.context.append_basic_block(self.function, "wr_loop_body");
                    let loop_continuing_block = self.context.append_basic_block(self.function, "wr_loop_cont");
                    let loop_exit = self.context.append_basic_block(self.function, "wr_loop_exit");

                    self.builder.build_unconditional_branch(loop_body_block).unwrap();

                    self.builder.position_at_end(loop_body_block);
                    self.loop_stack.push((loop_exit, loop_continuing_block));
                    self.terminated = false;
                    self.lower_block_with_return(body, func, ret_allocas)?;
                    if !self.terminated {
                        self.builder.build_unconditional_branch(loop_continuing_block).unwrap();
                    }

                    self.builder.position_at_end(loop_continuing_block);
                    self.terminated = false;
                    self.lower_block_with_return(continuing, func, ret_allocas)?;
                    if let Some(break_cond) = break_if {
                        let cond = self.to_i1(self.get_expr_scalar(*break_cond).into_int_value());
                        self.builder.build_conditional_branch(cond, loop_exit, loop_body_block).unwrap();
                    } else if !self.terminated {
                        self.builder.build_unconditional_branch(loop_body_block).unwrap();
                    }

                    self.loop_stack.pop();
                    self.builder.position_at_end(loop_exit);
                    self.terminated = false;
                }
                _ => self.lower_statement(stmt, func)?,
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gradient_wgsl_renders_on_cpu() {
        let source = std::fs::read_to_string(
            concat!(env!("CARGO_MANIFEST_DIR"), "/examples/basic/gradient/gradient.wgsl")
        ).unwrap();

        let compiled = compile_wgsl(&source).expect("failed to compile gradient.wgsl");

        let width: u32 = 64;
        let height: u32 = 64;
        let stride: u32 = width;

        let mut params = [0u8; 48];
        params[0..4].copy_from_slice(&width.to_le_bytes());
        params[4..8].copy_from_slice(&height.to_le_bytes());
        params[8..12].copy_from_slice(&256u32.to_le_bytes());
        params[12..16].copy_from_slice(&stride.to_le_bytes());
        params[16..20].copy_from_slice(&0.0f32.to_le_bytes());
        params[20..24].copy_from_slice(&0.0f32.to_le_bytes());
        params[24..28].copy_from_slice(&(1.0f32 / width as f32).to_le_bytes());
        params[28..32].copy_from_slice(&(1.0f32 / height as f32).to_le_bytes());

        let mut output = vec![0u32; (stride * height) as usize];
        let buffers: [*mut u8; 1] = [output.as_mut_ptr() as *mut u8];

        unsafe {
            (compiled.fn_ptr)(
                params.as_ptr(),
                buffers.as_ptr() as *const *mut u8,
                std::ptr::null(),
                width,
                height,
                stride,
                0,
                height,
            );
        }

        let p00 = output[0];
        let b00 = p00 & 0xFF;
        assert_eq!(b00, 128, "pixel (0,0) blue channel should be 128, got {b00}");

        let p_last = output[(63 * stride + 63) as usize];
        let r_last = (p_last >> 16) & 0xFF;
        let g_last = (p_last >> 8) & 0xFF;
        let b_last = p_last & 0xFF;
        assert_eq!(b_last, 128, "pixel (63,63) blue should be 128");
        assert!(r_last > 240, "pixel (63,63) red should be >240, got {r_last}");
        assert!(g_last > 240, "pixel (63,63) green should be >240, got {g_last}");

        assert_eq!(p00 >> 24, 0xFF, "alpha should be 0xFF");
        assert_eq!(p_last >> 24, 0xFF, "alpha should be 0xFF");

        let p_mid = output[(32 * stride + 32) as usize];
        let r_mid = (p_mid >> 16) & 0xFF;
        let g_mid = (p_mid >> 8) & 0xFF;
        assert!((120..136).contains(&r_mid), "pixel (32,32) red should be ~128, got {r_mid}");
        assert!((120..136).contains(&g_mid), "pixel (32,32) green should be ~128, got {g_mid}");
    }

    #[test]
    fn mandelbrot_wgsl_renders_on_cpu() {
        let source = std::fs::read_to_string(
            concat!(env!("CARGO_MANIFEST_DIR"), "/examples/basic/mandelbrot/mandelbrot.wgsl")
        ).unwrap();

        let compiled = compile_wgsl(&source).expect("failed to compile mandelbrot.wgsl");

        let width: u32 = 128;
        let height: u32 = 128;
        let stride: u32 = width;

        let x_min: f32 = -2.25;
        let y_min: f32 = -1.5;
        let x_step: f32 = 3.5 / width as f32;
        let y_step: f32 = 3.0 / height as f32;
        let max_iter: u32 = 256;

        let mut params = [0u8; 48];
        params[0..4].copy_from_slice(&width.to_le_bytes());
        params[4..8].copy_from_slice(&height.to_le_bytes());
        params[8..12].copy_from_slice(&max_iter.to_le_bytes());
        params[12..16].copy_from_slice(&stride.to_le_bytes());
        params[16..20].copy_from_slice(&x_min.to_le_bytes());
        params[20..24].copy_from_slice(&y_min.to_le_bytes());
        params[24..28].copy_from_slice(&x_step.to_le_bytes());
        params[28..32].copy_from_slice(&y_step.to_le_bytes());

        let mut output = vec![0u32; (stride * height) as usize];
        let buffers: [*mut u8; 1] = [output.as_mut_ptr() as *mut u8];

        unsafe {
            (compiled.fn_ptr)(
                params.as_ptr(),
                buffers.as_ptr() as *const *mut u8,
                std::ptr::null(),
                width,
                height,
                stride,
                0,
                height,
            );
        }

        let p_center = output[(64 * stride + 64) as usize];
        assert_eq!(p_center, 0x00000000, "center pixel should be in-set (black), got 0x{p_center:08X}");

        let p_corner = output[0];
        assert_ne!(p_corner, 0x00000000, "corner pixel should be outside the set (colored)");

        let non_black = output.iter().filter(|&&p| p != 0x00000000).count();
        let black = output.len() - non_black;
        let total = output.len();
        assert!(non_black > total / 2, "should have >50% non-black pixels, got {non_black}/{total}");
        assert!(black > total / 20, "should have >5% in-set (black) pixels, got {black}/{total}");
    }

    #[test]
    fn sdf_rings_wgsl_renders_on_cpu() {
        let source = std::fs::read_to_string(
            concat!(env!("CARGO_MANIFEST_DIR"), "/examples/sdf/sdf_rings/sdf_rings.wgsl")
        ).unwrap();

        let compiled = compile_wgsl(&source).expect("failed to compile sdf_rings.wgsl");

        let width: u32 = 64;
        let height: u32 = 64;
        let stride: u32 = width;

        let mut params = [0u8; 48];
        params[0..4].copy_from_slice(&width.to_le_bytes());
        params[4..8].copy_from_slice(&height.to_le_bytes());
        params[8..12].copy_from_slice(&256u32.to_le_bytes());
        params[12..16].copy_from_slice(&stride.to_le_bytes());
        params[16..20].copy_from_slice(&(-1.0f32).to_le_bytes());
        params[20..24].copy_from_slice(&(-1.0f32).to_le_bytes());
        params[24..28].copy_from_slice(&(2.0f32 / width as f32).to_le_bytes());
        params[28..32].copy_from_slice(&(2.0f32 / height as f32).to_le_bytes());
        params[32..36].copy_from_slice(&0u32.to_le_bytes());
        params[36..40].copy_from_slice(&1u32.to_le_bytes());

        let mut output = vec![0u32; (stride * height) as usize];
        let mut accum = vec![0.0f32; (stride * height * 4) as usize];
        let buffers: [*mut u8; 2] = [
            output.as_mut_ptr() as *mut u8,
            accum.as_mut_ptr() as *mut u8,
        ];

        unsafe {
            (compiled.fn_ptr)(
                params.as_ptr(),
                buffers.as_ptr() as *const *mut u8,
                std::ptr::null(),
                width,
                height,
                stride,
                0,
                height,
            );
        }

        let p_center = output[(32 * stride + 32) as usize];
        let a = (p_center >> 24) & 0xFF;
        assert_eq!(a, 0xFF, "alpha should be 0xFF, got 0x{a:02X}");

        let bright = output.iter().filter(|&&p| {
            let r = (p >> 16) & 0xFF;
            r > 200
        }).count();
        let dark = output.iter().filter(|&&p| {
            let r = (p >> 16) & 0xFF;
            r < 50
        }).count();
        assert!(bright > 0, "should have some bright ring pixels");
        assert!(dark > 0, "should have some dark background pixels");

        let accum_nonzero = accum.iter().filter(|&&v| v != 0.0).count();
        assert!(accum_nonzero > 0, "accum buffer should have non-zero values");
    }

    #[test]
    fn game_of_life_step_wgsl_on_cpu() {
        let source = std::fs::read_to_string(
            concat!(env!("CARGO_MANIFEST_DIR"), "/examples/sim/game_of_life/game_of_life_step.wgsl")
        ).unwrap();

        let compiled = compile_wgsl(&source).expect("failed to compile game_of_life_step.wgsl");

        let width: u32 = 16;
        let height: u32 = 16;
        let stride: u32 = width;
        let total = (width * height) as usize;

        let mut params = [0u8; 16];
        params[0..4].copy_from_slice(&width.to_le_bytes());
        params[4..8].copy_from_slice(&height.to_le_bytes());
        params[8..12].copy_from_slice(&stride.to_le_bytes());

        let mut grid_in = vec![0i32; total];
        grid_in[(8 * width + 7) as usize] = 1;
        grid_in[(8 * width + 8) as usize] = 1;
        grid_in[(8 * width + 9) as usize] = 1;

        let mut grid_out = vec![0i32; total];

        let buffers: [*mut u8; 2] = [
            grid_in.as_ptr() as *mut u8,
            grid_out.as_mut_ptr() as *mut u8,
        ];

        unsafe {
            (compiled.fn_ptr)(
                params.as_ptr(),
                buffers.as_ptr() as *const *mut u8,
                std::ptr::null(),
                width,
                height,
                stride,
                0,
                height,
            );
        }

        assert!(grid_out[(7 * width + 8) as usize] > 0, "cell (8,7) should be alive");
        assert!(grid_out[(8 * width + 8) as usize] > 0, "cell (8,8) should be alive");
        assert!(grid_out[(9 * width + 8) as usize] > 0, "cell (8,9) should be alive");
        assert!(grid_out[(8 * width + 7) as usize] <= 0, "cell (7,8) should be dead");
        assert!(grid_out[(8 * width + 9) as usize] <= 0, "cell (9,8) should be dead");
    }

    #[test]
    fn texture_test_wgsl_on_cpu() {
        use crate::jit::TextureSlot;

        let source = std::fs::read_to_string(
            concat!(env!("CARGO_MANIFEST_DIR"), "/examples/basic/texture_test/texture_test.wgsl")
        ).unwrap();

        let compiled = compile_wgsl(&source).expect("failed to compile texture_test.wgsl");

        let width: u32 = 32;
        let height: u32 = 32;
        let stride: u32 = width;

        let tex_width: u32 = 4;
        let tex_height: u32 = 4;
        let tex_data: Vec<u8> = (0..tex_width * tex_height)
            .flat_map(|_| [255u8, 0, 0, 255])
            .collect();

        let tex_slot = TextureSlot {
            data: tex_data.as_ptr(),
            width: tex_width,
            height: tex_height,
        };

        let mut params = [0u8; 48];
        params[0..4].copy_from_slice(&width.to_le_bytes());
        params[4..8].copy_from_slice(&height.to_le_bytes());
        params[8..12].copy_from_slice(&256u32.to_le_bytes());
        params[12..16].copy_from_slice(&stride.to_le_bytes());
        params[16..20].copy_from_slice(&0.0f32.to_le_bytes());
        params[20..24].copy_from_slice(&0.0f32.to_le_bytes());
        params[24..28].copy_from_slice(&(1.0f32 / width as f32).to_le_bytes());
        params[28..32].copy_from_slice(&(1.0f32 / height as f32).to_le_bytes());

        let mut output = vec![0u32; (stride * height) as usize];
        let buffers: [*mut u8; 1] = [output.as_mut_ptr() as *mut u8];

        unsafe {
            (compiled.fn_ptr)(
                params.as_ptr(),
                buffers.as_ptr() as *const *mut u8,
                &tex_slot as *const TextureSlot as *const u8,
                width,
                height,
                stride,
                0,
                height,
            );
        }

        let center = output[(16 * stride + 16) as usize];
        let a = (center >> 24) & 0xFF;
        let r = (center >> 16) & 0xFF;
        let g = (center >> 8) & 0xFF;
        let b = center & 0xFF;
        assert_eq!(a, 0xFF, "alpha should be 0xFF");
        assert!(r > 200, "red channel should be >200, got {r}");
        assert!(g < 20, "green channel should be <20, got {g}");
        assert!(b < 20, "blue channel should be <20, got {b}");

        let corner = output[0];
        let r_corner = (corner >> 16) & 0xFF;
        assert!(r_corner > 200, "corner red should be >200, got {r_corner}");
    }

    #[test]
    fn inject_vec2_wgsl_on_cpu() {
        let source = std::fs::read_to_string(
            concat!(env!("CARGO_MANIFEST_DIR"), "/examples/shared/inject_vec2.wgsl")
        ).unwrap();

        let compiled = compile_wgsl(&source).expect("failed to compile inject_vec2.wgsl");

        let width: u32 = 8;
        let height: u32 = 8;
        let stride: u32 = width;
        let total = (width * height) as usize;

        let mut params = [0u8; 48];
        params[0..4].copy_from_slice(&width.to_le_bytes());
        params[4..8].copy_from_slice(&height.to_le_bytes());
        params[8..12].copy_from_slice(&stride.to_le_bytes());
        params[16..20].copy_from_slice(&4.0f32.to_le_bytes());
        params[20..24].copy_from_slice(&4.0f32.to_le_bytes());
        params[24..28].copy_from_slice(&2.0f32.to_le_bytes());
        params[28..32].copy_from_slice(&5.0f32.to_le_bytes());
        params[32..36].copy_from_slice(&0.0f32.to_le_bytes());
        params[36..40].copy_from_slice(&0.0f32.to_le_bytes());

        let buf_in = vec![0.0f32; total * 2];
        let mut buf_out = vec![0.0f32; total * 2];

        let buffers: [*mut u8; 2] = [
            buf_in.as_ptr() as *mut u8,
            buf_out.as_mut_ptr() as *mut u8,
        ];

        unsafe {
            (compiled.fn_ptr)(
                params.as_ptr(),
                buffers.as_ptr() as *const *mut u8,
                std::ptr::null(),
                width,
                height,
                stride,
                0,
                height,
            );
        }

        let center_x = buf_out[(4 * width as usize + 4) * 2];
        let center_y = buf_out[(4 * width as usize + 4) * 2 + 1];
        assert!((center_x - 5.0).abs() < 0.01, "center x should be 5.0, got {center_x}");
        assert!(center_y.abs() < 0.01, "center y should be 0.0, got {center_y}");

        let corner_x = buf_out[0];
        assert!(corner_x.abs() < 0.01, "corner x should be 0.0, got {corner_x}");
    }
}
