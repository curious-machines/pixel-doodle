//! WGSL → Cranelift JIT backend.
//!
//! Parses WGSL compute shaders via naga and compiles them to native machine
//! code using Cranelift. The generated function iterates over all pixels,
//! executing the shader's entry point for each one.

use std::collections::HashMap;

use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::types::*;
use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlags, Signature, UserFuncName};
use cranelift_codegen::isa::CallConv;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_codegen::packed_option::ReservedValue;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};

use naga::{
    BinaryOperator, Expression, Handle, MathFunction, ScalarKind, Statement, TypeInner,
    UnaryOperator,
};

// ── Public API ──────────────────────────────────────────────────────────

/// JIT'd WGSL compute kernel.
///
/// Signature: `fn(output: *mut u32, params: *const u8, width: u32, height: u32, stride: u32)`
///
/// The function iterates all rows/cols internally.
/// JIT'd WGSL compute kernel.
///
/// `buffers` is an array of pointers, one per storage buffer binding (in binding
/// order). For pixel shaders: buffers[0] = output (u32), buffers[1] = accum (f32).
/// For sim shaders: buffers[N] corresponds to @binding(N+1).
/// `tex_slots` is a pointer to an array of TextureSlot structs (16 bytes each),
/// one per texture binding in binding order. May be null if no textures.
///
/// The kernel processes rows `[row_start, row_end)`. The `params` buffer still
/// contains the full image width/height for view mapping — row_start/row_end
/// only control which rows this call computes.
pub type WgslKernelFn = unsafe extern "C" fn(
    params: *const u8,
    buffers: *const *mut u8,
    tex_slots: *const u8,
    width: u32,
    height: u32,
    stride: u32,
    row_start: u32,
    row_end: u32,
);

/// Compiled WGSL kernel — holds the JIT module (to keep code alive) and the
/// function pointer.
pub struct CompiledWgslKernel {
    _module: JITModule,
    pub fn_ptr: WgslKernelFn,
    /// WGSL variable name → index into the buffers array, for each storage buffer.
    pub binding_map: HashMap<String, usize>,
    /// Number of storage buffers.
    pub num_storage_buffers: usize,
    /// Params struct members with names and byte offsets.
    pub params_members: Vec<ParamMember>,
}

unsafe impl Send for CompiledWgslKernel {}
unsafe impl Sync for CompiledWgslKernel {}

/// Parse and compile a WGSL compute shader to a native function.
pub fn compile_wgsl(source: &str) -> Result<CompiledWgslKernel, String> {
    let module = naga::front::wgsl::parse_str(source)
        .map_err(|e| format!("WGSL parse error: {e}"))?;

    let entry = module
        .entry_points
        .first()
        .ok_or_else(|| "no entry point in WGSL shader".to_string())?;

    let analysis = analyse_module(&module)?;

    // Set up Cranelift JIT.
    let mut flag_builder = settings::builder();
    flag_builder.set("opt_level", "speed").unwrap();
    let isa_builder = cranelift_codegen::isa::lookup_by_name(
        &target_lexicon::Triple::host().to_string(),
    )
    .map_err(|e| format!("ISA lookup: {e}"))?;
    let isa = isa_builder
        .finish(settings::Flags::new(flag_builder))
        .map_err(|e| format!("ISA finish: {e}"))?;

    let mut jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());

    // Register texture helper symbols if textures are present.
    if !analysis.textures.is_empty() {
        use crate::jit;
        jit_builder.symbol("pd_tex_sample_bilinear_clamp", jit::pd_tex_sample_bilinear_clamp as *const u8);
    }
    let mut jit_module = JITModule::new(jit_builder);

    // Build the function signature:
    // (params, buffers, tex_slots, width, height, stride, row_start, row_end) -> void
    let mut sig = Signature::new(CallConv::SystemV);
    sig.params.push(AbiParam::new(I64)); // params: *const u8
    sig.params.push(AbiParam::new(I64)); // buffers: *const *mut u8
    sig.params.push(AbiParam::new(I64)); // tex_slots: *const TextureSlot
    sig.params.push(AbiParam::new(I32)); // width: u32
    sig.params.push(AbiParam::new(I32)); // height: u32
    sig.params.push(AbiParam::new(I32)); // stride: u32
    sig.params.push(AbiParam::new(I32)); // row_start: u32
    sig.params.push(AbiParam::new(I32)); // row_end: u32

    let func_id = jit_module
        .declare_function("wgsl_main", Linkage::Local, &sig)
        .map_err(|e| format!("declare_function: {e}"))?;

    let mut func = cranelift_codegen::ir::Function::with_name_signature(
        UserFuncName::user(0, 0),
        sig.clone(),
    );

    let mut func_ctx = FunctionBuilderContext::new();
    {
        let mut builder = FunctionBuilder::new(&mut func, &mut func_ctx);

        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);
        builder.seal_block(entry_block);

        // Extract function parameters.
        let p_params = builder.block_params(entry_block)[0];
        let p_buffers = builder.block_params(entry_block)[1];
        let p_tex_slots = builder.block_params(entry_block)[2];
        let p_width = builder.block_params(entry_block)[3];
        let p_height = builder.block_params(entry_block)[4];
        let p_stride = builder.block_params(entry_block)[5];
        let p_row_start = builder.block_params(entry_block)[6];
        let p_row_end = builder.block_params(entry_block)[7];

        // Build row/col loop: for row in 0..height { for col in 0..width { ... } }
        let loop_header_row = builder.create_block();
        let loop_body_row = builder.create_block();
        let loop_header_col = builder.create_block();
        let loop_body_col = builder.create_block();
        let loop_inc_col = builder.create_block();
        let loop_inc_row = builder.create_block();
        let exit_block = builder.create_block();

        // Variables for row and col counters.
        let v_row = builder.declare_var(I32);
        let v_col = builder.declare_var(I32);

        builder.def_var(v_row, p_row_start);
        builder.ins().jump(loop_header_row, &[]);

        // Row loop header: if row >= row_end, exit.
        // NOTE: Don't seal loop headers yet — they have back-edges from increments.
        builder.switch_to_block(loop_header_row);
        let row_val = builder.use_var(v_row);
        let row_cmp = builder.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, row_val, p_row_end);
        builder.ins().brif(row_cmp, exit_block, &[], loop_body_row, &[]);

        // Row loop body: reset col = 0, enter col loop.
        builder.switch_to_block(loop_body_row);
        builder.seal_block(loop_body_row);
        let zero2 = builder.ins().iconst(I32, 0);
        builder.def_var(v_col, zero2);
        builder.ins().jump(loop_header_col, &[]);

        // Col loop header: if col >= width, go to row increment.
        builder.switch_to_block(loop_header_col);
        let col_val = builder.use_var(v_col);
        let col_cmp = builder.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, col_val, p_width);
        builder.ins().brif(col_cmp, loop_inc_row, &[], loop_body_col, &[]);

        // Col loop body: execute the shader body.
        builder.switch_to_block(loop_body_col);
        builder.seal_block(loop_body_col);

        let mut compiler = ShaderCompiler::new(
            &module,
            &analysis,
            &mut builder,
            &mut jit_module,
            p_params,
            p_buffers,
            p_tex_slots,
            p_width,
            p_height,
            p_stride,
            v_row,
            v_col,
        );
        compiler.compile_entry_point(&entry.function)?;

        // After the shader body, jump to col increment (the shader may have
        // already returned via early-exit, but if we reach here we continue).
        if !compiler.terminated {
            compiler.builder.ins().jump(loop_inc_col, &[]);
        }

        // Col increment: col += 1, jump to col header.
        builder.switch_to_block(loop_inc_col);
        builder.seal_block(loop_inc_col);
        let col_val2 = builder.use_var(v_col);
        let one = builder.ins().iconst(I32, 1);
        let col_next = builder.ins().iadd(col_val2, one);
        builder.def_var(v_col, col_next);
        builder.ins().jump(loop_header_col, &[]);
        // Now all predecessors of loop_header_col are known — seal it.
        builder.seal_block(loop_header_col);

        // Row increment: row += 1, jump to row header.
        builder.switch_to_block(loop_inc_row);
        builder.seal_block(loop_inc_row);
        let row_val2 = builder.use_var(v_row);
        let one2 = builder.ins().iconst(I32, 1);
        let row_next = builder.ins().iadd(row_val2, one2);
        builder.def_var(v_row, row_next);
        builder.ins().jump(loop_header_row, &[]);
        // Now all predecessors of loop_header_row are known — seal it.
        builder.seal_block(loop_header_row);

        // Exit.
        builder.switch_to_block(exit_block);
        builder.seal_block(exit_block);
        builder.ins().return_(&[]);

        builder.finalize();
    }

    let mut ctx = cranelift_codegen::Context::for_function(func);
    if let Err(e) = jit_module.define_function(func_id, &mut ctx) {
        // Print the Cranelift IR for debugging.
        eprintln!("Cranelift IR:\n{}", ctx.func.display());
        return Err(format!("define_function: {e:#}"));
    }
    jit_module.finalize_definitions().unwrap();

    let code_ptr = jit_module.get_finalized_function(func_id);
    let fn_ptr: WgslKernelFn = unsafe { std::mem::transmute(code_ptr) };

    // Build binding map: WGSL var name → buffer index.
    let mut binding_map = HashMap::new();
    for (i, buf) in analysis.storage_buffers.iter().enumerate() {
        let gv = &module.global_variables[buf.global];
        if let Some(ref name) = gv.name {
            binding_map.insert(name.clone(), i);
        }
    }
    let num_storage_buffers = analysis.storage_buffers.len();

    Ok(CompiledWgslKernel {
        _module: jit_module,
        fn_ptr,
        binding_map,
        num_storage_buffers,
        params_members: analysis.params_members.clone(),
    })
}

// ── Module analysis ─────────────────────────────────────────────────────

/// Binding info extracted from the naga module.
/// Info about a Params struct member.
#[derive(Clone)]
pub struct ParamMember {
    pub name: String,
    pub offset: u32,
}

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
            // Storage buffer — determine element type.
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

// ── Shader compiler ─────────────────────────────────────────────────────

struct ShaderCompiler<'a, 'b> {
    naga_module: &'a naga::Module,
    analysis: &'a BindingInfo,
    builder: &'b mut FunctionBuilder<'a>,
    jit_module: &'b mut JITModule,
    /// Cached cranelift values for naga expressions.
    expr_values: HashMap<Handle<Expression>, Vec<cranelift_codegen::ir::Value>>,
    /// Cranelift Variables for naga LocalVariables (per active function scope).
    local_vars: HashMap<Handle<naga::LocalVariable>, Vec<Variable>>,
    /// Return values captured from inlined function calls, keyed by CallResult expression handle.
    call_results: HashMap<Handle<Expression>, Vec<cranelift_codegen::ir::Value>>,
    // Function parameters.
    p_params: cranelift_codegen::ir::Value,
    p_buffers: cranelift_codegen::ir::Value,
    p_tex_slots: cranelift_codegen::ir::Value,
    #[allow(dead_code)]
    p_width: cranelift_codegen::ir::Value,
    #[allow(dead_code)]
    p_height: cranelift_codegen::ir::Value,
    #[allow(dead_code)]
    p_stride: cranelift_codegen::ir::Value,
    v_row: Variable,
    v_col: Variable,
    /// Block to jump to on early return.
    return_block: cranelift_codegen::ir::Block,
    /// Whether the current block has been terminated.
    terminated: bool,
    /// Stack of (break_block, continue_block) for nested loops.
    loop_stack: Vec<(cranelift_codegen::ir::Block, cranelift_codegen::ir::Block)>,
}

// Static type constants to avoid returning references to temporaries in resolve_expr_type.
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

impl<'a, 'b> ShaderCompiler<'a, 'b> {
    fn new(
        naga_module: &'a naga::Module,
        analysis: &'a BindingInfo,
        builder: &'b mut FunctionBuilder<'a>,
        jit_module: &'b mut JITModule,
        p_params: cranelift_codegen::ir::Value,
        p_buffers: cranelift_codegen::ir::Value,
        p_tex_slots: cranelift_codegen::ir::Value,
        p_width: cranelift_codegen::ir::Value,
        p_height: cranelift_codegen::ir::Value,
        p_stride: cranelift_codegen::ir::Value,
        v_row: Variable,
        v_col: Variable,
    ) -> Self {
        Self {
            naga_module,
            analysis,
            builder,
            jit_module,
            expr_values: HashMap::new(),
            local_vars: HashMap::new(),
            call_results: HashMap::new(),
            p_params,
            p_buffers,
            p_tex_slots,
            p_width,
            p_height,
            p_stride,
            v_row,
            v_col,
            return_block: cranelift_codegen::ir::Block::reserved_value(),
            terminated: false,
            loop_stack: Vec::new(),
        }
    }

    fn alloc_var(&mut self, ty: cranelift_codegen::ir::Type) -> Variable {
        self.builder.declare_var(ty)
    }

    fn compile_entry_point(&mut self, func: &'a naga::Function) -> Result<(), String> {
        let ret_block = self.builder.create_block();
        self.return_block = ret_block;

        // Declare local variables.
        for (handle, local) in func.local_variables.iter() {
            let cl_ty = self.naga_type_to_cl(&self.naga_module.types[local.ty].inner);
            let components = self.type_component_count(&self.naga_module.types[local.ty].inner);
            let mut vars = Vec::new();
            for _ in 0..components {
                let v = self.alloc_var(cl_ty);
                let zero = self.zero_value(cl_ty);
                self.builder.def_var(v, zero);
                vars.push(v);
            }
            // Initialize from init value if present.
            if let Some(init) = local.init {
                // The init expression is in the function's expression arena.
                self.eval_expr(init, func)?;
                let init_vals = self.get_expr(init).to_vec();
                for (i, v) in vars.iter().enumerate() {
                    if i < init_vals.len() {
                        self.builder.def_var(*v, init_vals[i]);
                    }
                }
            }
            self.local_vars.insert(handle, vars);
        }

        self.lower_block(&func.body, func)?;

        // The return block (jumped to by early-return statements).
        if !self.terminated {
            self.builder.ins().jump(ret_block, &[]);
        }
        self.builder.switch_to_block(ret_block);
        self.builder.seal_block(ret_block);
        self.terminated = false;

        Ok(())
    }

    fn zero_value(&mut self, ty: cranelift_codegen::ir::Type) -> cranelift_codegen::ir::Value {
        if ty == F32 {
            self.builder.ins().f32const(0.0)
        } else if ty == F64 {
            self.builder.ins().f64const(0.0)
        } else {
            self.builder.ins().iconst(ty, 0)
        }
    }

    fn naga_scalar_to_cl(&self, kind: ScalarKind, width: u8) -> cranelift_codegen::ir::Type {
        match (kind, width) {
            (ScalarKind::Float, 4) => F32,
            (ScalarKind::Float, 8) => F64,
            (ScalarKind::Sint | ScalarKind::Uint, 4) => I32,
            (ScalarKind::Sint | ScalarKind::Uint, 8) => I64,
            (ScalarKind::Bool, _) => I8,
            _ => I32,
        }
    }

    fn naga_type_to_cl(&self, inner: &TypeInner) -> cranelift_codegen::ir::Type {
        match inner {
            TypeInner::Scalar(s) => self.naga_scalar_to_cl(s.kind, s.width),
            TypeInner::Vector { scalar, .. } => self.naga_scalar_to_cl(scalar.kind, scalar.width),
            TypeInner::Pointer { .. } | TypeInner::ValuePointer { .. } => I64,
            _ => I32,
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

    fn get_expr(&self, handle: Handle<Expression>) -> &[cranelift_codegen::ir::Value] {
        // Check expr_values first, then call_results for CallResult expressions.
        if let Some(vals) = self.expr_values.get(&handle) {
            return vals;
        }
        if let Some(vals) = self.call_results.get(&handle) {
            return vals;
        }
        panic!("expression {handle:?} not yet evaluated")
    }

    fn get_expr_scalar(&self, handle: Handle<Expression>) -> cranelift_codegen::ir::Value {
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
            Expression::Binary { left, .. } => self.resolve_expr_type(*left, func),
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
                // Entry point with no declared arguments — arg 0 is gid
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
                // textureDimensions → vec2<u32>
                static TY_VEC2_U32: TypeInner = TypeInner::Vector {
                    size: naga::VectorSize::Bi,
                    scalar: naga::Scalar::U32,
                };
                &TY_VEC2_U32
            }
            Expression::ImageSample { .. } => {
                // textureSampleLevel → vec4<f32>
                static TY_VEC4_F32: TypeInner = TypeInner::Vector {
                    size: naga::VectorSize::Quad,
                    scalar: naga::Scalar::F32,
                };
                &TY_VEC4_F32
            }
            _ => &TY_U32,
        }
    }

    fn eval_expr(&mut self, handle: Handle<Expression>, func: &naga::Function) -> Result<(), String> {
        if self.expr_values.contains_key(&handle) {
            return Ok(());
        }

        let values = match &func.expressions[handle] {
            Expression::Literal(lit) => {
                vec![match lit {
                    naga::Literal::F32(v) => self.builder.ins().f32const(*v),
                    naga::Literal::U32(v) => self.builder.ins().iconst(I32, *v as i64),
                    naga::Literal::I32(v) => self.builder.ins().iconst(I32, *v as i64),
                    naga::Literal::Bool(v) => self.builder.ins().iconst(I8, *v as i64),
                    naga::Literal::F64(v) => self.builder.ins().f64const(*v),
                    naga::Literal::I64(v) => self.builder.ins().iconst(I64, *v),
                    naga::Literal::U64(v) => self.builder.ins().iconst(I64, *v as i64),
                    _ => self.builder.ins().iconst(I32, 0), // F16, AbstractInt, etc.
                }]
            }

            Expression::Constant(c) => {
                let constant = &self.naga_module.constants[*c];
                match &self.naga_module.global_expressions[constant.init] {
                    Expression::Literal(lit) => {
                        vec![match lit {
                            naga::Literal::F32(v) => self.builder.ins().f32const(*v),
                            naga::Literal::U32(v) => self.builder.ins().iconst(I32, *v as i64),
                            naga::Literal::I32(v) => self.builder.ins().iconst(I32, *v as i64),
                            naga::Literal::Bool(v) => self.builder.ins().iconst(I8, *v as i64),
                            naga::Literal::F64(v) => self.builder.ins().f64const(*v),
                            _ => return Err(format!("unsupported constant literal: {lit:?}")),
                        }]
                    }
                    _ => return Err("unsupported constant initializer".into()),
                }
            }

            Expression::ZeroValue(ty) => {
                let inner = &self.naga_module.types[*ty].inner;
                let cl_ty = self.naga_type_to_cl(inner);
                let count = self.type_component_count(inner);
                let zero = self.zero_value(cl_ty);
                vec![zero; count]
            }

            Expression::FunctionArgument(idx) => {
                if *idx == 0 {
                    // global_invocation_id: vec3<u32> = (col, row, 0)
                    let col = self.builder.use_var(self.v_col);
                    let row = self.builder.use_var(self.v_row);
                    let z = self.builder.ins().iconst(I32, 0);
                    vec![col, row, z]
                } else {
                    return Err(format!("unsupported function argument index: {idx}"));
                }
            }

            Expression::GlobalVariable(gv) => {
                if *gv == self.analysis.params_global {
                    vec![self.p_params]
                } else if let Some(idx) = self.analysis.storage_buffers.iter().position(|b| b.global == *gv) {
                    // Load buffer pointer from buffers array: buffers[idx]
                    let offset = self.builder.ins().iconst(I64, (idx * 8) as i64); // 8 bytes per pointer
                    let buf_ptr_addr = self.builder.ins().iadd(self.p_buffers, offset);
                    let buf_ptr = self.builder.ins().load(I64, MemFlags::trusted(), buf_ptr_addr, 0);
                    vec![buf_ptr]
                } else if self.analysis.textures.iter().any(|t| t.global == *gv) || self.analysis.sampler_globals.contains(gv) {
                    // Texture and sampler globals — return a dummy tag.
                    // Actual texture ops (ImageSample, ImageQuery) handle these directly.
                    let tag = self.builder.ins().iconst(I64, gv.index() as i64);
                    vec![tag]
                } else {
                    return Err(format!("unsupported global variable: {gv:?}"));
                }
            }

            Expression::LocalVariable(lv) => {
                // Return a tag — loads and stores go through local_vars map.
                let tag = self.builder.ins().iconst(I64, lv.index() as i64);
                vec![tag]
            }

            Expression::AccessIndex { base, index } => {
                self.eval_expr(*base, func)?;
                let base_ty = self.resolve_expr_type(*base, func);
                match base_ty {
                    TypeInner::Struct { members, .. } => {
                        let base_val = self.get_expr_scalar(*base);
                        let offset = self.analysis.params_offsets[*index as usize];
                        let member_ty = &self.naga_module.types[members[*index as usize].ty].inner;
                        let cl_ty = self.naga_type_to_cl(member_ty);
                        let val = self.builder.ins().load(cl_ty, MemFlags::trusted(), base_val, offset as i32);
                        vec![val]
                    }
                    TypeInner::Pointer { base: base_ty_handle, .. } => {
                        let inner = &self.naga_module.types[*base_ty_handle].inner;
                        match inner {
                            TypeInner::Struct { members, .. } => {
                                let base_val = self.get_expr_scalar(*base);
                                let offset = self.analysis.params_offsets.get(*index as usize)
                                    .copied().unwrap_or((*index) * 4);
                                let member_ty = &self.naga_module.types[members[*index as usize].ty].inner;
                                let cl_ty = self.naga_type_to_cl(member_ty);
                                let val = self.builder.ins().load(cl_ty, MemFlags::trusted(), base_val, offset as i32);
                                vec![val]
                            }
                            _ => return Err(format!("AccessIndex on pointer to non-struct: {inner:?}")),
                        }
                    }
                    TypeInner::Vector { .. } => {
                        // Check if base is a LocalVariable (pointer) — resolve through the var.
                        let base_naga = &func.expressions[*base];
                        if let Expression::LocalVariable(lv) = base_naga {
                            let vars = self.local_vars[lv].clone();
                            vec![self.builder.use_var(vars[*index as usize])]
                        } else {
                            let base_vals = self.get_expr(*base);
                            vec![base_vals[*index as usize]]
                        }
                    }
                    TypeInner::Scalar(_) => {
                        // Scalar accessed with index 0 — just return the value.
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
                        let base_val = self.get_expr_scalar(*base);
                        let idx_val = self.get_expr_scalar(*index);
                        let elem_inner = &self.naga_module.types[*elem_ty].inner;
                        let elem_size = self.type_byte_size(elem_inner);
                        let idx_i64 = self.builder.ins().uextend(I64, idx_val);
                        let byte_offset = self.builder.ins().imul_imm(idx_i64, elem_size as i64);
                        let addr = self.builder.ins().iadd(base_val, byte_offset);
                        let cl_ty = self.naga_type_to_cl(elem_inner);
                        let n_components = self.type_component_count(elem_inner);
                        if n_components == 1 {
                            let val = self.builder.ins().load(cl_ty, MemFlags::trusted(), addr, 0);
                            vec![val]
                        } else {
                            // Load N components (e.g. vec4<f32> = 4 × f32).
                            let component_size = match elem_inner {
                                TypeInner::Vector { scalar, .. } => scalar.width as i32,
                                _ => 4,
                            };
                            (0..n_components as i32)
                                .map(|i| self.builder.ins().load(cl_ty, MemFlags::trusted(), addr, i * component_size))
                                .collect()
                        }
                    }
                    TypeInner::Vector { .. } => {
                        let base_vals = self.get_expr(*base).to_vec();
                        let idx_val = self.get_expr_scalar(*index);
                        let mut result = base_vals[0];
                        for i in 1..base_vals.len() {
                            let i_const = self.builder.ins().iconst(I32, i as i64);
                            let cmp = self.builder.ins().icmp(IntCC::Equal, idx_val, i_const);
                            result = self.builder.ins().select(cmp, base_vals[i], result);
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
                        let vars = self.local_vars[lv].clone();
                        vars.iter().map(|v| self.builder.use_var(*v)).collect()
                    }
                    Expression::AccessIndex { base, index } => {
                        // Load a specific component from a local variable vector.
                        let base_expr = &func.expressions[*base];
                        if let Expression::LocalVariable(lv) = base_expr {
                            let vars = self.local_vars[lv].clone();
                            vec![self.builder.use_var(vars[*index as usize])]
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
                        // Splat scalar to match vector length.
                        let l = lhs_vals[i % lhs_vals.len()];
                        let r = rhs_vals[i % rhs_vals.len()];
                        let v = match op {
                            BinaryOperator::Add => if is_float { self.builder.ins().fadd(l, r) } else { self.builder.ins().iadd(l, r) },
                            BinaryOperator::Subtract => if is_float { self.builder.ins().fsub(l, r) } else { self.builder.ins().isub(l, r) },
                            BinaryOperator::Multiply => if is_float { self.builder.ins().fmul(l, r) } else { self.builder.ins().imul(l, r) },
                            BinaryOperator::Divide => if is_float { self.builder.ins().fdiv(l, r) } else { self.builder.ins().udiv(l, r) },
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
                let lhs_cl = self.builder.func.dfg.value_type(lhs);
                let rhs_cl = self.builder.func.dfg.value_type(rhs);
                if lhs_cl == I8 && rhs_cl == I32 {
                    lhs = self.builder.ins().uextend(I32, lhs);
                } else if rhs_cl == I8 && lhs_cl == I32 {
                    rhs = self.builder.ins().uextend(I32, rhs);
                }

                let result = match op {
                    BinaryOperator::Add => if is_float { self.builder.ins().fadd(lhs, rhs) } else { self.builder.ins().iadd(lhs, rhs) },
                    BinaryOperator::Subtract => if is_float { self.builder.ins().fsub(lhs, rhs) } else { self.builder.ins().isub(lhs, rhs) },
                    BinaryOperator::Multiply => if is_float { self.builder.ins().fmul(lhs, rhs) } else { self.builder.ins().imul(lhs, rhs) },
                    BinaryOperator::Divide => {
                        if is_float { self.builder.ins().fdiv(lhs, rhs) }
                        else if is_signed { self.builder.ins().sdiv(lhs, rhs) }
                        else { self.builder.ins().udiv(lhs, rhs) }
                    }
                    BinaryOperator::Modulo => {
                        if is_float {
                            let div = self.builder.ins().fdiv(lhs, rhs);
                            let floored = self.builder.ins().floor(div);
                            let prod = self.builder.ins().fmul(floored, rhs);
                            self.builder.ins().fsub(lhs, prod)
                        } else if is_signed { self.builder.ins().srem(lhs, rhs) }
                        else { self.builder.ins().urem(lhs, rhs) }
                    }
                    BinaryOperator::And => self.builder.ins().band(lhs, rhs),
                    BinaryOperator::InclusiveOr => self.builder.ins().bor(lhs, rhs),
                    BinaryOperator::ExclusiveOr => self.builder.ins().bxor(lhs, rhs),
                    BinaryOperator::ShiftLeft => self.builder.ins().ishl(lhs, rhs),
                    BinaryOperator::ShiftRight => if is_signed { self.builder.ins().sshr(lhs, rhs) } else { self.builder.ins().ushr(lhs, rhs) },
                    BinaryOperator::Equal => if is_float { self.builder.ins().fcmp(FloatCC::Equal, lhs, rhs) } else { self.builder.ins().icmp(IntCC::Equal, lhs, rhs) },
                    BinaryOperator::NotEqual => if is_float { self.builder.ins().fcmp(FloatCC::NotEqual, lhs, rhs) } else { self.builder.ins().icmp(IntCC::NotEqual, lhs, rhs) },
                    BinaryOperator::Less => {
                        if is_float { self.builder.ins().fcmp(FloatCC::LessThan, lhs, rhs) }
                        else if is_signed { self.builder.ins().icmp(IntCC::SignedLessThan, lhs, rhs) }
                        else { self.builder.ins().icmp(IntCC::UnsignedLessThan, lhs, rhs) }
                    }
                    BinaryOperator::LessEqual => {
                        if is_float { self.builder.ins().fcmp(FloatCC::LessThanOrEqual, lhs, rhs) }
                        else if is_signed { self.builder.ins().icmp(IntCC::SignedLessThanOrEqual, lhs, rhs) }
                        else { self.builder.ins().icmp(IntCC::UnsignedLessThanOrEqual, lhs, rhs) }
                    }
                    BinaryOperator::Greater => {
                        if is_float { self.builder.ins().fcmp(FloatCC::GreaterThan, lhs, rhs) }
                        else if is_signed { self.builder.ins().icmp(IntCC::SignedGreaterThan, lhs, rhs) }
                        else { self.builder.ins().icmp(IntCC::UnsignedGreaterThan, lhs, rhs) }
                    }
                    BinaryOperator::GreaterEqual => {
                        if is_float { self.builder.ins().fcmp(FloatCC::GreaterThanOrEqual, lhs, rhs) }
                        else if is_signed { self.builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, lhs, rhs) }
                        else { self.builder.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, lhs, rhs) }
                    }
                    BinaryOperator::LogicalAnd => self.builder.ins().band(lhs, rhs),
                    BinaryOperator::LogicalOr => self.builder.ins().bor(lhs, rhs),
                };
                vec![result]
            }

            Expression::Unary { op, expr } => {
                self.eval_expr(*expr, func)?;
                let val = self.get_expr_scalar(*expr);
                let expr_ty = self.resolve_expr_type(*expr, func);
                let is_float = matches!(expr_ty, TypeInner::Scalar(s) if s.kind == ScalarKind::Float);
                let result = match op {
                    UnaryOperator::Negate => if is_float { self.builder.ins().fneg(val) } else { self.builder.ins().ineg(val) },
                    UnaryOperator::LogicalNot => { let one = self.builder.ins().iconst(I8, 1); self.builder.ins().bxor(val, one) }
                    UnaryOperator::BitwiseNot => self.builder.ins().bnot(val),
                };
                vec![result]
            }

            Expression::As { expr, kind, convert } => {
                self.eval_expr(*expr, func)?;
                // Copy scalar info to avoid borrow conflict with cast_scalar.
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
                let cond = self.get_expr_scalar(*condition);
                let acc = self.get_expr(*accept).to_vec();
                let rej = self.get_expr(*reject).to_vec();
                acc.iter().zip(rej.iter())
                    .map(|(a, r)| self.builder.ins().select(cond, *a, *r))
                    .collect()
            }

            Expression::Relational { .. } => {
                vec![self.builder.ins().iconst(I8, 0)]
            }

            Expression::CallResult(_) => {
                // The value was placed by Statement::Call handling.
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
                        // textureDimensions(img) → vec2<u32>
                        // Load width and height from TextureSlot: { data: *u8, width: u32, height: u32 }
                        // TextureSlot is 16 bytes; width at offset 8, height at offset 12.
                        let img_expr = &func.expressions[*image];
                        let tex_idx = if let Expression::GlobalVariable(gv) = img_expr {
                            self.analysis.textures.iter().find(|t| t.global == *gv)
                                .map(|t| t.slot_index).unwrap_or(0)
                        } else { 0 };
                        let slot_offset = self.builder.ins().iconst(I64, (tex_idx * 16) as i64);
                        let slot_addr = self.builder.ins().iadd(self.p_tex_slots, slot_offset);
                        let w = self.builder.ins().load(I32, MemFlags::trusted(), slot_addr, 8);
                        let h = self.builder.ins().load(I32, MemFlags::trusted(), slot_addr, 12);
                        vec![w, h]
                    }
                    _ => return Err(format!("unsupported ImageQuery: {query:?}")),
                }
            }

            Expression::ImageSample { image, sampler: _, coordinate, level, .. } => {
                self.eval_expr(*image, func)?;
                self.eval_expr(*coordinate, func)?;
                // textureSampleLevel(img, sampler, uv, level) → vec4<f32>
                let img_expr = &func.expressions[*image];
                let tex_idx = if let Expression::GlobalVariable(gv) = img_expr {
                    self.analysis.textures.iter().find(|t| t.global == *gv)
                        .map(|t| t.slot_index).unwrap_or(0)
                } else { 0 };

                let uv = self.get_expr(*coordinate);
                let u_f32 = uv[0];
                let v_f32 = uv[1];
                // Promote f32 UV to f64 for the helper signature.
                let u_f64 = self.builder.ins().fpromote(F64, u_f32);
                let v_f64 = self.builder.ins().fpromote(F64, v_f32);

                let _ = level; // We always use level 0 (bilinear).

                // Call pd_tex_sample_bilinear_clamp(slots, tex, u, v, out)
                let mut sig = self.jit_module.make_signature();
                sig.params.push(AbiParam::new(I64));  // slots
                sig.params.push(AbiParam::new(I32));  // tex
                sig.params.push(AbiParam::new(F64));  // u
                sig.params.push(AbiParam::new(F64));  // v
                sig.params.push(AbiParam::new(I64));  // out
                let func_id = self.jit_module.declare_function("pd_tex_sample_bilinear_clamp", Linkage::Import, &sig).unwrap();
                let func_ref = self.jit_module.declare_func_in_func(func_id, self.builder.func);

                // Stack slot for 4 × f32 = 16 bytes.
                let ss = self.builder.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
                    cranelift_codegen::ir::StackSlotKind::ExplicitSlot, 16, 3,
                ));
                let out_addr = self.builder.ins().stack_addr(I64, ss, 0);

                let tex_const = self.builder.ins().iconst(I32, tex_idx as i64);
                self.builder.ins().call(func_ref, &[self.p_tex_slots, tex_const, u_f64, v_f64, out_addr]);

                let flags = MemFlags::new();
                let r = self.builder.ins().load(F32, flags, out_addr, 0);
                let g = self.builder.ins().load(F32, flags, out_addr, 4);
                let b = self.builder.ins().load(F32, flags, out_addr, 8);
                let a = self.builder.ins().load(F32, flags, out_addr, 12);
                vec![r, g, b, a]
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
    ) -> Result<Vec<cranelift_codegen::ir::Value>, String> {
        let arg_ty = self.resolve_expr_type(arg, func);
        if matches!(arg_ty, TypeInner::Vector { .. }) {
            return self.lower_math_vector(fun, arg, arg1, arg2, func);
        }

        let val = self.get_expr_scalar(arg);
        let is_f32 = matches!(arg_ty, TypeInner::Scalar(s) if s.kind == ScalarKind::Float);

        let result = match fun {
            MathFunction::Abs => if is_f32 { self.builder.ins().fabs(val) } else { self.builder.ins().iabs(val) },
            MathFunction::Min => {
                let rhs = self.get_expr_scalar(arg1.unwrap());
                if is_f32 { self.builder.ins().fmin(val, rhs) }
                else {
                    let is_signed = matches!(arg_ty, TypeInner::Scalar(s) if s.kind == ScalarKind::Sint);
                    if is_signed { self.builder.ins().smin(val, rhs) } else { self.builder.ins().umin(val, rhs) }
                }
            }
            MathFunction::Max => {
                let rhs = self.get_expr_scalar(arg1.unwrap());
                if is_f32 { self.builder.ins().fmax(val, rhs) }
                else {
                    let is_signed = matches!(arg_ty, TypeInner::Scalar(s) if s.kind == ScalarKind::Sint);
                    if is_signed { self.builder.ins().smax(val, rhs) } else { self.builder.ins().umax(val, rhs) }
                }
            }
            MathFunction::Clamp => {
                let min_val = self.get_expr_scalar(arg1.unwrap());
                let max_val = self.get_expr_scalar(arg2.unwrap());
                if is_f32 { let t = self.builder.ins().fmax(val, min_val); self.builder.ins().fmin(t, max_val) }
                else { let t = self.builder.ins().umax(val, min_val); self.builder.ins().umin(t, max_val) }
            }
            MathFunction::Floor => self.builder.ins().floor(val),
            MathFunction::Ceil => self.builder.ins().ceil(val),
            MathFunction::Round => self.builder.ins().nearest(val),
            MathFunction::Trunc => self.builder.ins().trunc(val),
            MathFunction::Fract => { let f = self.builder.ins().floor(val); self.builder.ins().fsub(val, f) }
            MathFunction::Sqrt => self.builder.ins().sqrt(val),
            MathFunction::Sign => {
                let zero = self.builder.ins().f32const(0.0);
                let one = self.builder.ins().f32const(1.0);
                let neg_one = self.builder.ins().f32const(-1.0);
                let gt = self.builder.ins().fcmp(FloatCC::GreaterThan, val, zero);
                let lt = self.builder.ins().fcmp(FloatCC::LessThan, val, zero);
                let pos = self.builder.ins().select(gt, one, zero);
                self.builder.ins().select(lt, neg_one, pos)
            }
            MathFunction::Sin => self.call_libm_f32("sinf", val),
            MathFunction::Cos => self.call_libm_f32("cosf", val),
            MathFunction::Tan => self.call_libm_f32("tanf", val),
            MathFunction::Asin => self.call_libm_f32("asinf", val),
            MathFunction::Acos => self.call_libm_f32("acosf", val),
            MathFunction::Atan => self.call_libm_f32("atanf", val),
            MathFunction::Atan2 => { let rhs = self.get_expr_scalar(arg1.unwrap()); self.call_libm_f32_binary("atan2f", val, rhs) }
            MathFunction::Exp => self.call_libm_f32("expf", val),
            MathFunction::Exp2 => self.call_libm_f32("exp2f", val),
            MathFunction::Log => self.call_libm_f32("logf", val),
            MathFunction::Log2 => self.call_libm_f32("log2f", val),
            MathFunction::Pow => { let rhs = self.get_expr_scalar(arg1.unwrap()); self.call_libm_f32_binary("powf", val, rhs) }
            MathFunction::Mix => {
                let b = self.get_expr_scalar(arg1.unwrap());
                let t = self.get_expr_scalar(arg2.unwrap());
                let one = self.builder.ins().f32const(1.0);
                let omt = self.builder.ins().fsub(one, t);
                let at = self.builder.ins().fmul(val, omt);
                let bt = self.builder.ins().fmul(b, t);
                self.builder.ins().fadd(at, bt)
            }
            MathFunction::SmoothStep => {
                let edge1 = self.get_expr_scalar(arg1.unwrap());
                let x = self.get_expr_scalar(arg2.unwrap());
                let diff = self.builder.ins().fsub(x, val);
                let range = self.builder.ins().fsub(edge1, val);
                let ratio = self.builder.ins().fdiv(diff, range);
                let zero = self.builder.ins().f32const(0.0);
                let one = self.builder.ins().f32const(1.0);
                let t = self.builder.ins().fmax(ratio, zero);
                let t = self.builder.ins().fmin(t, one);
                let three = self.builder.ins().f32const(3.0);
                let two = self.builder.ins().f32const(2.0);
                let two_t = self.builder.ins().fmul(two, t);
                let s = self.builder.ins().fsub(three, two_t);
                let t2 = self.builder.ins().fmul(t, t);
                self.builder.ins().fmul(t2, s)
            }
            MathFunction::Length => self.builder.ins().fabs(val),
            MathFunction::Dot => { let rhs = self.get_expr_scalar(arg1.unwrap()); self.builder.ins().fmul(val, rhs) }
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
    ) -> Result<Vec<cranelift_codegen::ir::Value>, String> {
        let vals = self.get_expr(arg).to_vec();
        let n = vals.len();
        match fun {
            MathFunction::Dot => {
                let rhs = self.get_expr(arg1.unwrap()).to_vec();
                let mut sum = self.builder.ins().fmul(vals[0], rhs[0]);
                for i in 1..n { let p = self.builder.ins().fmul(vals[i], rhs[i]); sum = self.builder.ins().fadd(sum, p); }
                Ok(vec![sum])
            }
            MathFunction::Length => {
                let mut sum = self.builder.ins().fmul(vals[0], vals[0]);
                for i in 1..n { let p = self.builder.ins().fmul(vals[i], vals[i]); sum = self.builder.ins().fadd(sum, p); }
                Ok(vec![self.builder.ins().sqrt(sum)])
            }
            MathFunction::Normalize => {
                let mut sum = self.builder.ins().fmul(vals[0], vals[0]);
                for i in 1..n { let p = self.builder.ins().fmul(vals[i], vals[i]); sum = self.builder.ins().fadd(sum, p); }
                let len = self.builder.ins().sqrt(sum);
                Ok(vals.iter().map(|v| self.builder.ins().fdiv(*v, len)).collect())
            }
            MathFunction::Cross => {
                let rhs = self.get_expr(arg1.unwrap()).to_vec();
                let a = self.builder.ins().fmul(vals[1], rhs[2]);
                let b = self.builder.ins().fmul(vals[2], rhs[1]);
                let x = self.builder.ins().fsub(a, b);
                let a = self.builder.ins().fmul(vals[2], rhs[0]);
                let b = self.builder.ins().fmul(vals[0], rhs[2]);
                let y = self.builder.ins().fsub(a, b);
                let a = self.builder.ins().fmul(vals[0], rhs[1]);
                let b = self.builder.ins().fmul(vals[1], rhs[0]);
                let z = self.builder.ins().fsub(a, b);
                Ok(vec![x, y, z])
            }
            MathFunction::Min => { let rhs = self.get_expr(arg1.unwrap()).to_vec(); Ok(vals.iter().zip(rhs.iter()).map(|(a, b)| self.builder.ins().fmin(*a, *b)).collect()) }
            MathFunction::Max => { let rhs = self.get_expr(arg1.unwrap()).to_vec(); Ok(vals.iter().zip(rhs.iter()).map(|(a, b)| self.builder.ins().fmax(*a, *b)).collect()) }
            MathFunction::Abs => Ok(vals.iter().map(|v| self.builder.ins().fabs(*v)).collect()),
            MathFunction::Clamp => {
                let min_v = self.get_expr(arg1.unwrap()).to_vec();
                let max_v = self.get_expr(arg2.unwrap()).to_vec();
                Ok(vals.iter().enumerate().map(|(i, v)| { let t = self.builder.ins().fmax(*v, min_v[i]); self.builder.ins().fmin(t, max_v[i]) }).collect())
            }
            MathFunction::Mix => {
                let b_vals = self.get_expr(arg1.unwrap()).to_vec();
                let t_vals = self.get_expr(arg2.unwrap()).to_vec();
                let one = self.builder.ins().f32const(1.0);
                Ok(vals.iter().enumerate().map(|(i, a)| {
                    let t = t_vals[i % t_vals.len()];
                    let omt = self.builder.ins().fsub(one, t);
                    let at = self.builder.ins().fmul(*a, omt);
                    let bt = self.builder.ins().fmul(b_vals[i], t);
                    self.builder.ins().fadd(at, bt)
                }).collect())
            }
            _ => {
                let mut results = Vec::with_capacity(n);
                for v in &vals {
                    let r = match fun {
                        MathFunction::Floor => self.builder.ins().floor(*v),
                        MathFunction::Ceil => self.builder.ins().ceil(*v),
                        MathFunction::Sqrt => self.builder.ins().sqrt(*v),
                        MathFunction::Fract => { let f = self.builder.ins().floor(*v); self.builder.ins().fsub(*v, f) }
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
        val: cranelift_codegen::ir::Value,
        src: &naga::Scalar,
        dst_kind: ScalarKind,
        convert: Option<u8>,
    ) -> Result<cranelift_codegen::ir::Value, String> {
        // Same type → noop.
        let dst_width = convert.unwrap_or(src.width);
        if src.kind == dst_kind && src.width == dst_width {
            return Ok(val);
        }
        Ok(match convert {
            Some(width) => match (src.kind, src.width, dst_kind, width) {
                (ScalarKind::Uint, 4, ScalarKind::Float, 4) => self.builder.ins().fcvt_from_uint(F32, val),
                (ScalarKind::Sint, 4, ScalarKind::Float, 4) => self.builder.ins().fcvt_from_sint(F32, val),
                (ScalarKind::Float, 4, ScalarKind::Sint, 4) => self.builder.ins().fcvt_to_sint(I32, val),
                (ScalarKind::Float, 4, ScalarKind::Uint, 4) => self.builder.ins().fcvt_to_uint(I32, val),
                (ScalarKind::Sint, 4, ScalarKind::Uint, 4) | (ScalarKind::Uint, 4, ScalarKind::Sint, 4) => val,
                // Bool → integer: zero-extend i8 to i32
                (ScalarKind::Bool, 1, ScalarKind::Uint, 4) | (ScalarKind::Bool, 1, ScalarKind::Sint, 4) => {
                    self.builder.ins().uextend(I32, val)
                }
                // Bool → float
                (ScalarKind::Bool, 1, ScalarKind::Float, 4) => {
                    let ext = self.builder.ins().uextend(I32, val);
                    self.builder.ins().fcvt_from_uint(F32, ext)
                }
                // i32 ↔ i32 (same width, same kind — shouldn't reach here but handle gracefully)
                _ if src.width == width && src.kind == dst_kind => val,
                _ => return Err(format!("unsupported cast: {:?}/{} → {:?}/{}", src.kind, src.width, dst_kind, width)),
            },
            None => match (src.kind, src.width, dst_kind) {
                (ScalarKind::Sint, 4, ScalarKind::Uint) | (ScalarKind::Uint, 4, ScalarKind::Sint) => val,
                (ScalarKind::Float, 4, ScalarKind::Uint) | (ScalarKind::Float, 4, ScalarKind::Sint) => self.builder.ins().bitcast(I32, MemFlags::new(), val),
                (ScalarKind::Uint, 4, ScalarKind::Float) | (ScalarKind::Sint, 4, ScalarKind::Float) => self.builder.ins().bitcast(F32, MemFlags::new(), val),
                _ => return Err(format!("unsupported bitcast: {:?}/{} → {:?}", src.kind, src.width, dst_kind)),
            },
        })
    }

    fn call_libm_f32(&mut self, name: &str, arg: cranelift_codegen::ir::Value) -> cranelift_codegen::ir::Value {
        let mut sig = self.jit_module.make_signature();
        sig.params.push(AbiParam::new(F32));
        sig.returns.push(AbiParam::new(F32));
        let func_id = self.jit_module.declare_function(name, Linkage::Import, &sig).unwrap();
        let func_ref = self.jit_module.declare_func_in_func(func_id, self.builder.func);
        let call = self.builder.ins().call(func_ref, &[arg]);
        self.builder.inst_results(call)[0]
    }

    fn call_libm_f32_binary(&mut self, name: &str, a: cranelift_codegen::ir::Value, b: cranelift_codegen::ir::Value) -> cranelift_codegen::ir::Value {
        let mut sig = self.jit_module.make_signature();
        sig.params.push(AbiParam::new(F32));
        sig.params.push(AbiParam::new(F32));
        sig.returns.push(AbiParam::new(F32));
        let func_id = self.jit_module.declare_function(name, Linkage::Import, &sig).unwrap();
        let func_ref = self.jit_module.declare_func_in_func(func_id, self.builder.func);
        let call = self.builder.ins().call(func_ref, &[a, b]);
        self.builder.inst_results(call)[0]
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
                let cond = self.get_expr_scalar(*condition);
                let then_block = self.builder.create_block();
                let else_block = self.builder.create_block();
                let merge_block = self.builder.create_block();

                self.builder.ins().brif(cond, then_block, &[], else_block, &[]);

                self.builder.switch_to_block(then_block);
                self.builder.seal_block(then_block);
                self.terminated = false;
                self.lower_block(accept, func)?;
                let then_terminated = self.terminated;
                if !then_terminated {
                    self.builder.ins().jump(merge_block, &[]);
                }

                self.builder.switch_to_block(else_block);
                self.builder.seal_block(else_block);
                self.terminated = false;
                self.lower_block(reject, func)?;
                let else_terminated = self.terminated;
                if !else_terminated {
                    self.builder.ins().jump(merge_block, &[]);
                }

                self.builder.switch_to_block(merge_block);
                self.builder.seal_block(merge_block);
                self.terminated = then_terminated && else_terminated;
            }

            Statement::Return { .. } => {
                self.builder.ins().jump(self.return_block, &[]);
                self.terminated = true;
            }

            Statement::Store { pointer, value } => {
                self.eval_expr(*value, func)?;
                self.eval_expr(*pointer, func)?;
                let src_vals = self.get_expr(*value).to_vec();
                let ptr_expr = &func.expressions[*pointer];
                match ptr_expr {
                    Expression::LocalVariable(lv) => {
                        let vars = self.local_vars[lv].clone();
                        for (i, v) in vars.iter().enumerate() {
                            if i < src_vals.len() {
                                self.builder.def_var(*v, src_vals[i]);
                            }
                        }
                    }
                    Expression::AccessIndex { base, index } => {
                        // Store to a specific component of a local variable vector.
                        let base_ptr_expr = &func.expressions[*base];
                        if let Expression::LocalVariable(lv) = base_ptr_expr {
                            let vars = self.local_vars[lv].clone();
                            let idx = *index as usize;
                            if idx < vars.len() {
                                self.builder.def_var(vars[idx], src_vals[0]);
                            }
                        }
                    }
                    Expression::Access { base, index } => {
                        self.eval_expr(*base, func)?;
                        self.eval_expr(*index, func)?;
                        let buf_ptr = self.get_expr_scalar(*base);
                        let idx_val = self.get_expr_scalar(*index);
                        let idx_i64 = self.builder.ins().uextend(I64, idx_val);

                        // Determine element size from the base global's storage buffer info.
                        let base_expr = &func.expressions[*base];
                        let (elem_bytes, n_components) = if let Expression::GlobalVariable(gv) = base_expr {
                            if let Some(info) = self.analysis.storage_buffers.iter().find(|b| b.global == *gv) {
                                (info.elem_bytes, info.elem_components)
                            } else { (4, 1) }
                        } else { (4, 1) };

                        let byte_offset = self.builder.ins().imul_imm(idx_i64, elem_bytes as i64);
                        let addr = self.builder.ins().iadd(buf_ptr, byte_offset);
                        if n_components == 1 {
                            self.builder.ins().store(MemFlags::trusted(), src_vals[0], addr, 0);
                        } else {
                            let component_bytes = elem_bytes / n_components as u32;
                            for (i, v) in src_vals.iter().enumerate() {
                                self.builder.ins().store(MemFlags::trusted(), *v, addr, (i as u32 * component_bytes) as i32);
                            }
                        }
                    }
                    _ => {}
                }
            }

            Statement::Loop { body, continuing, break_if } => {
                let loop_body_block = self.builder.create_block();
                let loop_continuing_block = self.builder.create_block();
                let loop_exit = self.builder.create_block();

                self.builder.ins().jump(loop_body_block, &[]);

                self.builder.switch_to_block(loop_body_block);
                // Don't seal yet — back-edge comes later.
                self.loop_stack.push((loop_exit, loop_continuing_block));
                self.terminated = false;
                self.lower_block(body, func)?;
                if !self.terminated {
                    self.builder.ins().jump(loop_continuing_block, &[]);
                }

                self.builder.switch_to_block(loop_continuing_block);
                self.builder.seal_block(loop_continuing_block);
                self.terminated = false;
                self.lower_block(continuing, func)?;
                if let Some(break_cond) = break_if {
                    self.eval_expr(*break_cond, func)?;
                    let cond = self.get_expr_scalar(*break_cond);
                    self.builder.ins().brif(cond, loop_exit, &[], loop_body_block, &[]);
                } else if !self.terminated {
                    self.builder.ins().jump(loop_body_block, &[]);
                }

                // Seal body after back-edge.
                self.builder.seal_block(loop_body_block);
                self.loop_stack.pop();
                self.builder.switch_to_block(loop_exit);
                self.builder.seal_block(loop_exit);
                self.terminated = false;
            }

            Statement::Break => {
                let (exit, _) = *self.loop_stack.last().unwrap();
                self.builder.ins().jump(exit, &[]);
                self.terminated = true;
            }

            Statement::Continue => {
                let (_, cont) = *self.loop_stack.last().unwrap();
                self.builder.ins().jump(cont, &[]);
                self.terminated = true;
            }

            Statement::Block(block) => {
                self.lower_block(block, func)?;
            }

            Statement::Switch { selector, cases } => {
                let sel_val = self.get_expr_scalar(*selector);
                let merge_block = self.builder.create_block();
                let mut case_blocks = Vec::new();
                let mut default_idx = None;

                for (i, case) in cases.iter().enumerate() {
                    case_blocks.push(self.builder.create_block());
                    if case.value == naga::SwitchValue::Default {
                        default_idx = Some(i);
                    }
                }
                let default_block = default_idx.map(|i| case_blocks[i]).unwrap_or(merge_block);

                // Emit comparison chain.
                for (i, case) in cases.iter().enumerate() {
                    let val = match case.value {
                        naga::SwitchValue::I32(v) => Some(self.builder.ins().iconst(I32, v as i64)),
                        naga::SwitchValue::U32(v) => Some(self.builder.ins().iconst(I32, v as i64)),
                        naga::SwitchValue::Default => None,
                    };
                    if let Some(v) = val {
                        let cmp = self.builder.ins().icmp(IntCC::Equal, sel_val, v);
                        let next = self.builder.create_block();
                        self.builder.ins().brif(cmp, case_blocks[i], &[], next, &[]);
                        self.builder.switch_to_block(next);
                        self.builder.seal_block(next);
                    }
                }
                self.builder.ins().jump(default_block, &[]);

                // Lower case bodies.
                self.loop_stack.push((merge_block, merge_block));
                for (i, case) in cases.iter().enumerate() {
                    self.builder.switch_to_block(case_blocks[i]);
                    self.builder.seal_block(case_blocks[i]);
                    self.terminated = false;
                    self.lower_block(&case.body, func)?;
                    if !self.terminated {
                        if case.fall_through && i + 1 < cases.len() {
                            self.builder.ins().jump(case_blocks[i + 1], &[]);
                        } else {
                            self.builder.ins().jump(merge_block, &[]);
                        }
                    }
                }
                self.loop_stack.pop();

                self.builder.switch_to_block(merge_block);
                self.builder.seal_block(merge_block);
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
        let arg_values: Vec<Vec<cranelift_codegen::ir::Value>> = caller_args
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
        let inline_return_block = self.builder.create_block();
        self.return_block = inline_return_block;
        self.terminated = false;

        // Seed callee FunctionArgument expressions with caller arg values.
        for (handle, expr) in callee.expressions.iter() {
            if let Expression::FunctionArgument(idx) = expr {
                self.expr_values.insert(handle, arg_values[*idx as usize].clone());
            }
        }

        // Declare callee local variables.
        for (handle, local) in callee.local_variables.iter() {
            let cl_ty = self.naga_type_to_cl(&self.naga_module.types[local.ty].inner);
            let components = self.type_component_count(&self.naga_module.types[local.ty].inner);
            let mut vars = Vec::new();
            for _ in 0..components {
                let v = self.alloc_var(cl_ty);
                let zero = self.zero_value(cl_ty);
                self.builder.def_var(v, zero);
                vars.push(v);
            }
            // Initialize from init value if present.
            if let Some(init) = local.init {
                self.eval_expr(init, callee)?;
                let init_vals = self.get_expr(init).to_vec();
                for (i, v) in vars.iter().enumerate() {
                    if i < init_vals.len() {
                        self.builder.def_var(*v, init_vals[i]);
                    }
                }
            }
            self.local_vars.insert(handle, vars);
        }

        // We need to capture return values. Use a Cranelift Variable to carry
        // the return value out of the inline body (since there may be multiple
        // return paths).
        let ret_var = if result_expr.is_some() {
            // Determine return type from callee.
            let ret_ty = callee.result.as_ref().map(|r| {
                self.naga_type_to_cl(&self.naga_module.types[r.ty].inner)
            }).unwrap_or(I32);
            let rv = self.alloc_var(ret_ty);
            let zero = self.zero_value(ret_ty);
            self.builder.def_var(rv, zero);
            Some(rv)
        } else {
            None
        };

        // Lower callee body, intercepting Return statements to capture values.
        self.lower_block_with_return(&callee.body, callee, ret_var)?;

        // Jump to the inline return block if not already terminated.
        if !self.terminated {
            self.builder.ins().jump(inline_return_block, &[]);
        }
        self.builder.switch_to_block(inline_return_block);
        self.builder.seal_block(inline_return_block);

        // Capture return value.
        let return_values = ret_var.map(|rv| vec![self.builder.use_var(rv)]);

        // Restore caller state.
        self.expr_values = saved_expr_values;
        self.local_vars = saved_local_vars;
        self.return_block = saved_return_block;
        self.terminated = saved_terminated;

        // Store the return value for CallResult expression.
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
        ret_var: Option<Variable>,
    ) -> Result<(), String> {
        for stmt in block.iter() {
            if self.terminated { break; }
            match stmt {
                Statement::Return { value } => {
                    if let (Some(val_handle), Some(rv)) = (value, ret_var) {
                        self.eval_expr(*val_handle, func)?;
                        let val = self.get_expr_scalar(*val_handle);
                        self.builder.def_var(rv, val);
                    }
                    self.builder.ins().jump(self.return_block, &[]);
                    self.terminated = true;
                }
                Statement::If { condition, accept, reject } => {
                    // Need to use lower_block_with_return for both branches.
                    let cond = self.get_expr_scalar(*condition);
                    let then_block = self.builder.create_block();
                    let else_block = self.builder.create_block();
                    let merge_block = self.builder.create_block();

                    self.builder.ins().brif(cond, then_block, &[], else_block, &[]);

                    self.builder.switch_to_block(then_block);
                    self.builder.seal_block(then_block);
                    self.terminated = false;
                    self.lower_block_with_return(accept, func, ret_var)?;
                    let then_term = self.terminated;
                    if !then_term { self.builder.ins().jump(merge_block, &[]); }

                    self.builder.switch_to_block(else_block);
                    self.builder.seal_block(else_block);
                    self.terminated = false;
                    self.lower_block_with_return(reject, func, ret_var)?;
                    let else_term = self.terminated;
                    if !else_term { self.builder.ins().jump(merge_block, &[]); }

                    self.builder.switch_to_block(merge_block);
                    self.builder.seal_block(merge_block);
                    self.terminated = then_term && else_term;
                }
                Statement::Loop { body, continuing, break_if } => {
                    let loop_body_block = self.builder.create_block();
                    let loop_continuing_block = self.builder.create_block();
                    let loop_exit = self.builder.create_block();

                    self.builder.ins().jump(loop_body_block, &[]);

                    self.builder.switch_to_block(loop_body_block);
                    self.loop_stack.push((loop_exit, loop_continuing_block));
                    self.terminated = false;
                    self.lower_block_with_return(body, func, ret_var)?;
                    if !self.terminated {
                        self.builder.ins().jump(loop_continuing_block, &[]);
                    }

                    self.builder.switch_to_block(loop_continuing_block);
                    self.builder.seal_block(loop_continuing_block);
                    self.terminated = false;
                    self.lower_block_with_return(continuing, func, ret_var)?;
                    if let Some(break_cond) = break_if {
                        let cond = self.get_expr_scalar(*break_cond);
                        self.builder.ins().brif(cond, loop_exit, &[], loop_body_block, &[]);
                    } else if !self.terminated {
                        self.builder.ins().jump(loop_body_block, &[]);
                    }

                    // Seal loop body after back-edge is established.
                    self.builder.seal_block(loop_body_block);
                    self.loop_stack.pop();

                    self.builder.switch_to_block(loop_exit);
                    self.builder.seal_block(loop_exit);
                    self.terminated = false;
                }
                // All other statements delegate to the normal handler.
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
        let stride: u32 = width; // no alignment padding needed for test

        // Build a Params struct matching the WGSL layout:
        // width: u32, height: u32, max_iter: u32, stride: u32,
        // x_min: f32, y_min: f32, x_step: f32, y_step: f32
        let mut params = [0u8; 48];
        params[0..4].copy_from_slice(&width.to_le_bytes());
        params[4..8].copy_from_slice(&height.to_le_bytes());
        params[8..12].copy_from_slice(&256u32.to_le_bytes()); // max_iter
        params[12..16].copy_from_slice(&stride.to_le_bytes());
        params[16..20].copy_from_slice(&0.0f32.to_le_bytes()); // x_min
        params[20..24].copy_from_slice(&0.0f32.to_le_bytes()); // y_min
        params[24..28].copy_from_slice(&(1.0f32 / width as f32).to_le_bytes()); // x_step
        params[28..32].copy_from_slice(&(1.0f32 / height as f32).to_le_bytes()); // y_step

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

        // Verify: pixel at (0, 0) should be near (r=0, g=0, b=128) = 0xFF000080
        let p00 = output[0];
        let b00 = p00 & 0xFF;
        assert_eq!(b00, 128, "pixel (0,0) blue channel should be 128, got {b00}");

        // Pixel at (63, 63) should have r ≈ 252, g ≈ 252, b = 128
        let p_last = output[(63 * stride + 63) as usize];
        let r_last = (p_last >> 16) & 0xFF;
        let g_last = (p_last >> 8) & 0xFF;
        let b_last = p_last & 0xFF;
        assert_eq!(b_last, 128, "pixel (63,63) blue should be 128");
        assert!(r_last > 240, "pixel (63,63) red should be >240, got {r_last}");
        assert!(g_last > 240, "pixel (63,63) green should be >240, got {g_last}");

        // Alpha should always be 0xFF
        assert_eq!(p00 >> 24, 0xFF, "alpha should be 0xFF");
        assert_eq!(p_last >> 24, 0xFF, "alpha should be 0xFF");

        // Check a middle pixel (32, 32) — should have r ≈ 128, g ≈ 128
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

        // Params: center on (-0.5, 0), range ~3.5 in x
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

        // The center of the Mandelbrot set (around -0.5, 0) should be IN the set
        // (black = 0x00000000 or with alpha 0xFF000000).
        // Pixel at center: col ≈ 64, row ≈ 64 → cx ≈ -0.5, cy ≈ 0
        let p_center = output[(64 * stride + 64) as usize];
        // In-set pixels return max_iter → iter_to_color returns 0x00000000
        assert_eq!(p_center, 0x00000000, "center pixel should be in-set (black), got 0x{p_center:08X}");

        // A point far outside (top-left corner: cx = -2.25, cy = -1.5) should escape quickly.
        let p_corner = output[0];
        // Should NOT be black (should have some color from the coloring function).
        assert_ne!(p_corner, 0x00000000, "corner pixel should be outside the set (colored)");

        // Count non-black pixels — most pixels escape, so majority should be colored.
        let non_black = output.iter().filter(|&&p| p != 0x00000000).count();
        let black = output.len() - non_black;
        let total = output.len();
        // The set covers maybe 10-20% of this view.
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

        // Params layout (12 fields × 4 bytes = 48 bytes):
        // width, height, max_iter, stride, x_min, y_min, x_step, y_step,
        // sample_index, sample_count, _pad0, _pad1
        let mut params = [0u8; 48];
        params[0..4].copy_from_slice(&width.to_le_bytes());
        params[4..8].copy_from_slice(&height.to_le_bytes());
        params[8..12].copy_from_slice(&256u32.to_le_bytes()); // max_iter
        params[12..16].copy_from_slice(&stride.to_le_bytes());
        params[16..20].copy_from_slice(&(-1.0f32).to_le_bytes()); // x_min
        params[20..24].copy_from_slice(&(-1.0f32).to_le_bytes()); // y_min
        params[24..28].copy_from_slice(&(2.0f32 / width as f32).to_le_bytes()); // x_step
        params[28..32].copy_from_slice(&(2.0f32 / height as f32).to_le_bytes()); // y_step
        // sample_index = 0 (first sample), sample_count = 1
        params[32..36].copy_from_slice(&0u32.to_le_bytes());
        params[36..40].copy_from_slice(&1u32.to_le_bytes());

        let mut output = vec![0u32; (stride * height) as usize];
        let mut accum = vec![0.0f32; (stride * height * 4) as usize]; // vec4<f32> per pixel
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

        // Center pixel (32, 32) maps to (0, 0) — origin of the rings.
        // At r=0 the ring SDF should be inside (distance = 0 - thickness).
        let p_center = output[(32 * stride + 32) as usize];
        let a = (p_center >> 24) & 0xFF;
        assert_eq!(a, 0xFF, "alpha should be 0xFF, got 0x{a:02X}");

        // There should be a mix of ring (bright) and background (dark) pixels.
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

        // Accum buffer should have been written to (non-zero values).
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

        // Params: width, height, stride, _pad (4 × u32 = 16 bytes)
        let mut params = [0u8; 16];
        params[0..4].copy_from_slice(&width.to_le_bytes());
        params[4..8].copy_from_slice(&height.to_le_bytes());
        params[8..12].copy_from_slice(&stride.to_le_bytes());

        // Set up a blinker: 3 alive cells in a row at (7,8), (8,8), (9,8)
        let mut grid_in = vec![0i32; total];
        grid_in[(8 * width + 7) as usize] = 1;
        grid_in[(8 * width + 8) as usize] = 1;
        grid_in[(8 * width + 9) as usize] = 1;

        let mut grid_out = vec![0i32; total];

        let buffers: [*mut u8; 2] = [
            grid_in.as_ptr() as *mut u8,   // binding 1: grid_in (read-only)
            grid_out.as_mut_ptr() as *mut u8, // binding 2: grid_out (read-write)
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

        // Blinker should rotate: horizontal → vertical
        // After one step, alive cells should be at (8,7), (8,8), (8,9)
        assert!(grid_out[(7 * width + 8) as usize] > 0, "cell (8,7) should be alive");
        assert!(grid_out[(8 * width + 8) as usize] > 0, "cell (8,8) should be alive");
        assert!(grid_out[(9 * width + 8) as usize] > 0, "cell (8,9) should be alive");

        // Original horizontal cells (except center) should be dead
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

        // Create a simple 4x4 test texture: solid red (RGBA8).
        let tex_width: u32 = 4;
        let tex_height: u32 = 4;
        let tex_data: Vec<u8> = (0..tex_width * tex_height)
            .flat_map(|_| [255u8, 0, 0, 255]) // red pixels
            .collect();

        let tex_slot = TextureSlot {
            data: tex_data.as_ptr(),
            width: tex_width,
            height: tex_height,
        };

        // Params: width, height, max_iter, stride, x_min, y_min, x_step, y_step
        let mut params = [0u8; 48];
        params[0..4].copy_from_slice(&width.to_le_bytes());
        params[4..8].copy_from_slice(&height.to_le_bytes());
        params[8..12].copy_from_slice(&256u32.to_le_bytes());
        params[12..16].copy_from_slice(&stride.to_le_bytes());
        // x_min/y_min/x_step/y_step — not used by texture_test.wgsl
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

        // The texture is square (4x4), screen is square (32x32), so no letterbox.
        // All pixels should be red: ARGB = 0xFF_RR_00_00 where RR ≈ 255.
        let center = output[(16 * stride + 16) as usize];
        let a = (center >> 24) & 0xFF;
        let r = (center >> 16) & 0xFF;
        let g = (center >> 8) & 0xFF;
        let b = center & 0xFF;
        assert_eq!(a, 0xFF, "alpha should be 0xFF");
        assert!(r > 200, "red channel should be >200, got {r}");
        assert!(g < 20, "green channel should be <20, got {g}");
        assert!(b < 20, "blue channel should be <20, got {b}");

        // Corner pixels should also be red (square texture → square screen → no letterbox).
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

        // Params: width(4) height(4) stride(4) _pad(4)
        //         inject_x(4) inject_y(4) radius(4) value(4)
        //         falloff_quadratic(4) component(4) _pad2(4) _pad3(4)
        // Total: 48 bytes
        let mut params = [0u8; 48];
        params[0..4].copy_from_slice(&width.to_le_bytes());
        params[4..8].copy_from_slice(&height.to_le_bytes());
        params[8..12].copy_from_slice(&stride.to_le_bytes());
        // inject at center (4, 4), radius 2, value 5.0, flat (falloff=0), component 0 (x)
        params[16..20].copy_from_slice(&4.0f32.to_le_bytes()); // inject_x
        params[20..24].copy_from_slice(&4.0f32.to_le_bytes()); // inject_y
        params[24..28].copy_from_slice(&2.0f32.to_le_bytes()); // radius
        params[28..32].copy_from_slice(&5.0f32.to_le_bytes()); // value
        params[32..36].copy_from_slice(&0.0f32.to_le_bytes()); // falloff_quadratic (flat)
        params[36..40].copy_from_slice(&0.0f32.to_le_bytes()); // component (x=0)

        // Input: all zeros (vec2<f32>, 8 bytes per pixel).
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

        // Cell at (4,4) — center, within radius → x component should be 5.0.
        let center_x = buf_out[(4 * width as usize + 4) * 2];
        let center_y = buf_out[(4 * width as usize + 4) * 2 + 1];
        assert!((center_x - 5.0).abs() < 0.01, "center x should be 5.0, got {center_x}");
        assert!(center_y.abs() < 0.01, "center y should be 0.0, got {center_y}");

        // Cell at (0,0) — outside radius → should be unchanged (0.0).
        let corner_x = buf_out[0];
        assert!(corner_x.abs() < 0.01, "corner x should be 0.0, got {corner_x}");
    }
}
