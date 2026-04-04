use std::collections::HashMap;

use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::types::*;
use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlags, UserFuncName};
use cranelift_codegen::isa::CallConv;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};

use super::ast::*;
use super::error::PdcError;
use super::runtime;
use super::span::Spanned;
use super::type_check::{EnumInfo, OverloadSet, StructInfo};
pub use super::codegen_common::*;

/// Wrapper to make `JITModule` Send + Sync.
/// SAFETY: The module owns JIT'd code and is only accessed through the
/// function pointers stored in `CompiledProgram`. The module itself is
/// never shared mutably across threads.
struct SendSyncJitModule(#[allow(dead_code)] JITModule);
unsafe impl Send for SendSyncJitModule {}
unsafe impl Sync for SendSyncJitModule {}

pub fn compile(
    program: &Program,
    types: &[PdcType],
    builtins_layout: &[(&str, PdcType)],
    user_fns: &HashMap<String, OverloadSet>,
    structs: &HashMap<String, StructInfo>,
    enums: &HashMap<String, EnumInfo>,
    fn_aliases: &HashMap<String, String>,
    op_overloads: &HashMap<String, OverloadSet>,
) -> Result<(CompiledProgram, StateLayout), PdcError> {
    let mut flag_builder = settings::builder();
    flag_builder.set("opt_level", "speed").unwrap();
    let isa_builder = cranelift_codegen::isa::lookup_by_name(
        &target_lexicon::Triple::host().to_string(),
    )
    .map_err(|e| PdcError::Codegen {
        message: format!("ISA lookup: {e}"),
    })?;
    let isa = isa_builder
        .finish(settings::Flags::new(flag_builder))
        .map_err(|e| PdcError::Codegen {
            message: format!("ISA finish: {e}"),
        })?;

    let call_conv = isa.default_call_conv();
    let pointer_type = isa.pointer_type();

    let mut jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    for (name, ptr) in runtime::runtime_symbols() {
        jit_builder.symbol(name, ptr);
    }

    let mut jit_module = JITModule::new(jit_builder);

    // Collect top-level constants (will be emitted at the start of each function)
    let mut top_level_consts: Vec<&Spanned<Stmt>> = Vec::new();
    // Collect immutable builtin declarations (emitted at the start of each function)
    let mut top_level_builtins: Vec<&Spanned<Stmt>> = Vec::new();
    // Collect top-level var declarations for state block layout
    let mut top_level_vars: Vec<(&str, PdcType)> = Vec::new();
    // Collect mutable builtin names (need to be known in all functions)
    let mut mutable_builtin_names: std::collections::HashSet<String> = std::collections::HashSet::new();
    // (qualified_name, fn_def) — qualified name is "module::fn" for modules, plain "fn" for main
    let mut fn_defs: Vec<(String, &FnDef)> = Vec::new();
    let mut user_fn_ids: HashMap<String, cranelift_module::FuncId> = HashMap::new();

    // Only collect constants whose values are pure literals or literal
    // expressions (no variable references). These are safe to emit in any
    // function since they don't depend on runtime state.
    fn is_pure_literal(expr: &Expr) -> bool {
        match expr {
            Expr::Literal(_) => true,
            Expr::UnaryOp { operand, .. } => is_pure_literal(&operand.node),
            Expr::BinaryOp { left, right, .. } => {
                is_pure_literal(&left.node) && is_pure_literal(&right.node)
            }
            _ => false,
        }
    }

    // Collect from modules (qualified names)
    for module in &program.modules {
        for stmt in &module.stmts {
            match &stmt.node {
                Stmt::FnDef(fndef) => {
                    let qualified = format!("{}::{}", module.name, fndef.name);
                    fn_defs.push((qualified, fndef));
                }
                Stmt::ConstDecl { value, .. } if is_pure_literal(&value.node) => {
                    top_level_consts.push(stmt);
                }
                Stmt::VarDecl { name, ty, value, .. } => {
                    let expr_ty = &types[value.id as usize];
                    let final_ty = ty.clone().unwrap_or_else(|| expr_ty.clone());
                    top_level_vars.push((name.as_str(), final_ty));
                }
                Stmt::BuiltinDecl { name, mutable, .. } => {
                    if *mutable {
                        mutable_builtin_names.insert(name.clone());
                    } else {
                        top_level_builtins.push(stmt);
                    }
                }
                _ => {}
            }
        }
    }

    // Collect from main program (unqualified names)
    for stmt in &program.stmts {
        match &stmt.node {
            Stmt::FnDef(fndef) => fn_defs.push((fndef.name.clone(), fndef)),
            Stmt::ConstDecl { value, .. } if is_pure_literal(&value.node) => {
                top_level_consts.push(stmt);
            }
            Stmt::VarDecl { name, ty, value, .. } => {
                let expr_ty = &types[value.id as usize];
                let final_ty = ty.clone().unwrap_or_else(|| expr_ty.clone());
                top_level_vars.push((name.as_str(), final_ty));
            }
            Stmt::BuiltinDecl { name, mutable, .. } => {
                if *mutable {
                    mutable_builtin_names.insert(name.clone());
                } else {
                    top_level_builtins.push(stmt);
                }
            }
            _ => {}
        }
    }

    // Build state layout for all top-level vars
    let state_layout = StateLayout::build(&top_level_vars);

    // Track overload counts for mangled names
    let mut overload_counts: HashMap<String, usize> = HashMap::new();
    for (qualified_name, fndef) in &fn_defs {
        let mut sig = jit_module.make_signature();
        sig.call_conv = call_conv;
        // First param: ctx pointer
        sig.params.push(AbiParam::new(pointer_type));
        // User params
        for param in &fndef.params {
            sig.params.push(AbiParam::new(pdc_type_to_cl(&param.ty, pointer_type)));
        }
        if fndef.return_type != PdcType::Void {
            sig.returns.push(AbiParam::new(pdc_type_to_cl(&fndef.return_type, pointer_type)));
        }

        let base = mangle_name(qualified_name);
        let idx = overload_counts.entry(qualified_name.clone()).or_insert(0);
        let mangled = if *idx == 0 {
            format!("pdc_userfn_{base}")
        } else {
            format!("pdc_userfn_{base}_{idx}")
        };
        *idx += 1;

        let func_id = jit_module
            .declare_function(&mangled, Linkage::Local, &sig)
            .map_err(|e| PdcError::Codegen {
                message: format!("declare user fn: {e}"),
            })?;
        // Key by mangled name so each overload gets its own entry
        user_fn_ids.insert(mangled, func_id);
    }

    // Build operator dispatch table: map (op_name, overload_idx) → FuncId.
    // The op_overloads map uses unqualified op names with signatures accumulated
    // across all modules, in the order they were registered during type checking.
    // We match each signature to the correct qualified function in fn_defs.
    let mut op_fn_ids: HashMap<(String, usize), cranelift_module::FuncId> = HashMap::new();
    for (op_name, overloads) in op_overloads {
        for (sig_idx, sig) in overloads.sigs.iter().enumerate() {
            // Find the matching function in fn_defs by operator name and parameter types
            for (qualified_name, fndef) in &fn_defs {
                let short = qualified_name.rsplit("::").next().unwrap_or(qualified_name);
                if short != op_name {
                    continue;
                }
                let params_match = fndef.params.len() == sig.params.len()
                    && fndef.params.iter().zip(&sig.params).all(|(p, s)| p.ty == *s);
                if params_match {
                    // Find the mangled name for this specific function
                    let base = mangle_name(qualified_name);
                    // Count overloads of this qualified name that appear before this one in fn_defs
                    let mut qual_idx = 0;
                    for (qn2, fd2) in &fn_defs {
                        if qn2 == qualified_name {
                            if std::ptr::eq(*fd2, *fndef) {
                                break;
                            }
                            qual_idx += 1;
                        }
                    }
                    let mangled = if qual_idx == 0 {
                        format!("pdc_userfn_{base}")
                    } else {
                        format!("pdc_userfn_{base}_{qual_idx}")
                    };
                    if let Some(&fid) = user_fn_ids.get(&mangled) {
                        op_fn_ids.insert((op_name.clone(), sig_idx), fid);
                        break;
                    }
                }
            }
        }
    }

    // Build per-module intra-module alias maps: "Rect" → "geometry::Rect" for all module fns
    let mut module_aliases: HashMap<String, HashMap<String, String>> = HashMap::new();
    for (qualified_name, _) in &fn_defs {
        if let Some(sep) = qualified_name.find("::") {
            let mod_name = &qualified_name[..sep];
            let fn_name = &qualified_name[sep + 2..];
            module_aliases.entry(mod_name.to_string())
                .or_default()
                .insert(fn_name.to_string(), qualified_name.clone());
        }
    }

    // Compile each user function
    let mut compile_overload_counts: HashMap<String, usize> = HashMap::new();
    for (qualified_name, fndef) in &fn_defs {
        let base = mangle_name(qualified_name);
        let idx = compile_overload_counts.entry(qualified_name.clone()).or_insert(0);
        let mangled = if *idx == 0 {
            format!("pdc_userfn_{base}")
        } else {
            format!("pdc_userfn_{base}_{idx}")
        };
        *idx += 1;
        let func_id = user_fn_ids[&mangled];
        let mut ctx = jit_module.make_context();
        let sig = jit_module.declarations().get_function_decl(func_id).signature.clone();
        ctx.func.signature = sig;
        ctx.func.name = UserFuncName::user(0, func_id.as_u32());

        let mut fb_ctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);
        let entry = builder.create_block();
        builder.append_block_params_for_function_params(entry);
        builder.switch_to_block(entry);
        builder.seal_block(entry);

        let ctx_ptr = builder.block_params(entry)[0];
        let builtin_map = build_builtin_map(builtins_layout);

        // For module functions, merge intra-module aliases so unqualified
        // sibling calls (e.g. Rect() inside geometry::RoundedRect) resolve.
        let combined_aliases;
        let effective_aliases = if let Some(sep) = qualified_name.find("::") {
            let mod_name = &qualified_name[..sep];
            if let Some(mod_map) = module_aliases.get(mod_name) {
                combined_aliases = fn_aliases.iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .chain(mod_map.iter().map(|(k, v)| (k.clone(), v.clone())))
                    .collect::<HashMap<_, _>>();
                &combined_aliases
            } else {
                fn_aliases
            }
        } else {
            fn_aliases
        };

        let mut cg = CodegenCtx {
            builder: &mut builder,
            module: &mut jit_module,
            ctx_ptr,
            variables: HashMap::new(),
            builtin_map,
            type_table: types,
            call_conv,
            pointer_type,
            user_fn_ids: &user_fn_ids,
            user_fns,
            structs,
            enums,
            fn_aliases: effective_aliases,
            op_overloads,
            op_fn_ids: &op_fn_ids,
            struct_vars: HashMap::new(),
            block_terminated: false,
            loop_stack: Vec::new(),
            state_layout: &state_layout,
            mutable_builtins: mutable_builtin_names.clone(),
        };

        // Define parameters as variables
        for (i, param) in fndef.params.iter().enumerate() {
            let val = cg.builder.block_params(entry)[i + 1]; // +1 for ctx_ptr
            let var = cg.new_variable(&param.name, &param.ty);
            cg.builder.def_var(var, val);
        }

        // Emit top-level constants and immutable builtins so they're accessible
        for const_stmt in &top_level_consts {
            cg.emit_stmt(const_stmt)?;
        }
        for builtin_stmt in &top_level_builtins {
            cg.emit_stmt(builtin_stmt)?;
        }

        cg.emit_block(&fndef.body)?;

        // Add implicit return if the body didn't end with a return statement
        let body_returns = fndef.body.stmts.last().is_some_and(|s| matches!(s.node, Stmt::Return(_)));
        if !body_returns {
            if fndef.return_type == PdcType::Void {
                cg.builder.ins().return_(&[]);
            } else {
                let zero = cg.default_value(&fndef.return_type);
                cg.builder.ins().return_(&[zero]);
            }
        }

        drop(cg);
        builder.finalize();

        jit_module
            .define_function(func_id, &mut ctx)
            .map_err(|e| {
                eprintln!("--- Cranelift IR for '{}' ---\n{}", fndef.name, ctx.func.display());
                PdcError::Codegen {
                    message: format!("define user fn '{}': {e}", fndef.name),
                }
            })?;
    }

    // Compile main function (top-level statements, excluding fn defs)
    let mut main_sig = jit_module.make_signature();
    main_sig.params.push(AbiParam::new(pointer_type));
    main_sig.call_conv = call_conv;

    let main_id = jit_module
        .declare_function("pdc_main", Linkage::Local, &main_sig)
        .map_err(|e| PdcError::Codegen {
            message: format!("declare main: {e}"),
        })?;

    let mut main_ctx = jit_module.make_context();
    main_ctx.func.signature = main_sig;
    main_ctx.func.name = UserFuncName::user(0, main_id.as_u32());

    let mut fb_ctx = FunctionBuilderContext::new();
    {
        let mut builder = FunctionBuilder::new(&mut main_ctx.func, &mut fb_ctx);
        let entry = builder.create_block();
        builder.append_block_params_for_function_params(entry);
        builder.switch_to_block(entry);
        builder.seal_block(entry);

        let ctx_ptr = builder.block_params(entry)[0];
        let builtin_map = build_builtin_map(builtins_layout);

        let mut cg = CodegenCtx {
            builder: &mut builder,
            module: &mut jit_module,
            ctx_ptr,
            variables: HashMap::new(),
            builtin_map,
            type_table: types,
            call_conv,
            pointer_type,
            user_fn_ids: &user_fn_ids,
            user_fns,
            structs,
            enums,
            fn_aliases,
            op_overloads,
            op_fn_ids: &op_fn_ids,
            struct_vars: HashMap::new(),
            block_terminated: false,
            loop_stack: Vec::new(),
            state_layout: &state_layout,
            mutable_builtins: mutable_builtin_names.clone(),
        };

        // Emit module-level statements (const/var init) before main code
        for module in &program.modules {
            for stmt in &module.stmts {
                if matches!(&stmt.node, Stmt::FnDef(_) | Stmt::Import { .. } | Stmt::StructDef(_) | Stmt::EnumDef(_) | Stmt::TypeAlias { .. } | Stmt::TestDef { .. }) {
                    continue;
                }
                cg.emit_stmt(stmt)?;
                if cg.block_terminated {
                    break;
                }
            }
        }

        // Emit main program statements
        for stmt in &program.stmts {
            if matches!(&stmt.node, Stmt::FnDef(_) | Stmt::Import { .. } | Stmt::StructDef(_) | Stmt::TypeAlias { .. } | Stmt::TestDef { .. }) {
                continue;
            }
            cg.emit_stmt(stmt)?;
            if cg.block_terminated {
                break;
            }
        }

        if !cg.block_terminated {
            cg.builder.ins().return_(&[]);
        }

        drop(cg);
        builder.finalize();
    }

    jit_module
        .define_function(main_id, &mut main_ctx)
        .map_err(|e| {
            eprintln!("--- Cranelift IR for main ---\n{}", main_ctx.func.display());
            PdcError::Codegen {
                message: format!("define main: {e}"),
            }
        })?;

    // Compile test functions (only from the main program, not imported modules)
    let mut test_defs: Vec<(String, &Block)> = Vec::new();
    for stmt in &program.stmts {
        if let Stmt::TestDef { name, body } = &stmt.node {
            test_defs.push((name.clone(), body));
        }
    }

    let mut test_fn_ids: Vec<(String, cranelift_module::FuncId)> = Vec::new();
    for (i, (test_name, _)) in test_defs.iter().enumerate() {
        let mut sig = jit_module.make_signature();
        sig.call_conv = call_conv;
        sig.params.push(AbiParam::new(pointer_type)); // ctx pointer

        let mangled = format!("pdc_test_{i}");
        let func_id = jit_module
            .declare_function(&mangled, Linkage::Local, &sig)
            .map_err(|e| PdcError::Codegen {
                message: format!("declare test fn '{}': {e}", test_name),
            })?;
        test_fn_ids.push((test_name.clone(), func_id));
    }

    for (i, (_test_name, body)) in test_defs.iter().enumerate() {
        let (ref name, func_id) = test_fn_ids[i];
        let mut ctx = jit_module.make_context();
        let sig = jit_module.declarations().get_function_decl(func_id).signature.clone();
        ctx.func.signature = sig;
        ctx.func.name = UserFuncName::user(0, func_id.as_u32());

        let mut fb_ctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);
        let entry = builder.create_block();
        builder.append_block_params_for_function_params(entry);
        builder.switch_to_block(entry);
        builder.seal_block(entry);

        let ctx_ptr = builder.block_params(entry)[0];
        let builtin_map = build_builtin_map(builtins_layout);

        let mut cg = CodegenCtx {
            builder: &mut builder,
            module: &mut jit_module,
            ctx_ptr,
            variables: HashMap::new(),
            builtin_map,
            type_table: types,
            call_conv,
            pointer_type,
            user_fn_ids: &user_fn_ids,
            user_fns,
            structs,
            enums,
            fn_aliases,
            op_overloads,
            op_fn_ids: &op_fn_ids,
            struct_vars: HashMap::new(),
            block_terminated: false,
            loop_stack: Vec::new(),
            state_layout: &state_layout,
            mutable_builtins: mutable_builtin_names.clone(),
        };

        // Emit top-level constants so they're accessible in test functions
        for const_stmt in &top_level_consts {
            cg.emit_stmt(const_stmt)?;
        }

        cg.emit_block(body)?;

        if !cg.block_terminated {
            cg.builder.ins().return_(&[]);
        }

        drop(cg);
        builder.finalize();

        jit_module
            .define_function(func_id, &mut ctx)
            .map_err(|e| {
                eprintln!("--- Cranelift IR for test '{}' ---\n{}", name, ctx.func.display());
                PdcError::Codegen {
                    message: format!("define test fn '{}': {e}", name),
                }
            })?;
    }

    jit_module.finalize_definitions().map_err(|e| PdcError::Codegen {
        message: format!("finalize: {e}"),
    })?;

    let code_ptr = jit_module.get_finalized_function(main_id);
    let fn_ptr: PdcSceneFn = unsafe { std::mem::transmute(code_ptr) };

    // Collect finalized test function pointers
    let mut test_fn_ptrs: Vec<(String, PdcSceneFn)> = Vec::new();
    for (test_name, func_id) in &test_fn_ids {
        let test_code_ptr = jit_module.get_finalized_function(*func_id);
        let test_fn: PdcSceneFn = unsafe { std::mem::transmute(test_code_ptr) };
        test_fn_ptrs.push((test_name.clone(), test_fn));
    }

    // Collect finalized function pointers for all user functions.
    // Non-overloaded functions are keyed by their qualified name (e.g., "add", "math::lerp").
    // Overloaded functions get additional entries keyed as "name#1", "name#2", etc.
    let mut user_fn_ptrs: HashMap<String, (*const u8, Vec<PdcType>, PdcType)> = HashMap::new();
    let mut ptr_overload_counts: HashMap<String, usize> = HashMap::new();
    for (qualified_name, fndef) in &fn_defs {
        let base = mangle_name(qualified_name);
        let idx = ptr_overload_counts.entry(qualified_name.clone()).or_insert(0);
        let mangled = if *idx == 0 {
            format!("pdc_userfn_{base}")
        } else {
            format!("pdc_userfn_{base}_{idx}")
        };
        let key = if *idx == 0 {
            qualified_name.clone()
        } else {
            format!("{}#{}", qualified_name, idx)
        };
        *idx += 1;
        let func_id = user_fn_ids[&mangled];
        let fn_code_ptr = jit_module.get_finalized_function(func_id);
        let param_types: Vec<PdcType> = fndef.params.iter().map(|p| p.ty.clone()).collect();
        user_fn_ptrs.insert(key, (fn_code_ptr, param_types, fndef.return_type.clone()));
    }

    Ok((
        CompiledProgram::new(
            fn_ptr,
            user_fn_ptrs,
            test_fn_ptrs,
            Box::new(SendSyncJitModule(jit_module)),
        ),
        state_layout,
    ))
}

fn pdc_type_to_cl(ty: &PdcType, pointer_type: cranelift_codegen::ir::Type) -> cranelift_codegen::ir::Type {
    match ty {
        PdcType::F32 => F32,
        PdcType::F64 => F64,
        PdcType::I8 | PdcType::U8 | PdcType::Bool => I8,
        PdcType::I16 | PdcType::U16 => I16,
        PdcType::I32 | PdcType::U32 => I32,
        PdcType::I64 | PdcType::U64 => I64,
        PdcType::PathHandle | PdcType::BufferHandle | PdcType::KernelHandle | PdcType::TextureHandle | PdcType::SceneHandle => I32,
        PdcType::Str => I32, // strings are handles (i32)
        PdcType::Slice(_) => pointer_type, // slices are pointers to (handle, start, len)
        PdcType::Struct(_) | PdcType::Tuple(_) => pointer_type, // compound types are pointers
        PdcType::Enum(_) => I32, // enums are u32 constants
        PdcType::Array(_) => I32, // arrays are handles (u32)
        PdcType::Void => I32,
        PdcType::FnRef { .. } => pointer_type, // function references are opaque at codegen level
        PdcType::Module(_) => I32, // module namespaces have no runtime representation
        PdcType::Unknown => pointer_type,
    }
}

struct CodegenCtx<'a, 'b> {
    builder: &'a mut FunctionBuilder<'b>,
    module: &'a mut JITModule,
    ctx_ptr: cranelift_codegen::ir::Value,
    variables: HashMap<String, (Variable, PdcType)>,
    /// Struct variables stored as stack slot pointers (reserved for future use).
    #[allow(dead_code)]
    struct_vars: HashMap<String, (cranelift_codegen::ir::StackSlot, String)>,
    builtin_map: HashMap<String, BuiltinInfo>,
    type_table: &'a [PdcType],
    call_conv: CallConv,
    pointer_type: cranelift_codegen::ir::Type,
    user_fn_ids: &'a HashMap<String, cranelift_module::FuncId>,
    user_fns: &'a HashMap<String, OverloadSet>,
    structs: &'a HashMap<String, StructInfo>,
    enums: &'a HashMap<String, EnumInfo>,
    /// Alias map: unqualified name → qualified "module::name".
    fn_aliases: &'a HashMap<String, String>,
    /// Operator overloads for user-defined operator dispatch.
    op_overloads: &'a HashMap<String, OverloadSet>,
    /// Operator function dispatch table: (op_name, overload_idx) → FuncId.
    op_fn_ids: &'a HashMap<(String, usize), cranelift_module::FuncId>,
    /// Set to true after a terminator instruction (return, break, continue).
    block_terminated: bool,
    /// Stack of loop contexts for break/continue targets.
    loop_stack: Vec<LoopContext>,
    /// State layout for module-level mutable variables.
    /// If a variable name is found here, it lives in the state block, not in SSA.
    state_layout: &'a StateLayout,
    /// Names of mutable builtins — reads/writes go directly to the builtins array.
    mutable_builtins: std::collections::HashSet<String>,
}

/// Tracks the jump targets for break and continue within a loop.
struct LoopContext {
    /// Block to jump to on `continue` (header for while/for/foreach, body for loop).
    continue_block: cranelift_codegen::ir::Block,
    /// Block to jump to on `break`.
    exit_block: cranelift_codegen::ir::Block,
}

impl<'a, 'b> CodegenCtx<'a, 'b> {
    fn new_variable(&mut self, name: &str, ty: &PdcType) -> Variable {
        let cl_type = pdc_type_to_cl(ty, self.pointer_type);
        let var = self.builder.declare_var(cl_type);
        self.variables.insert(name.to_string(), (var, ty.clone()));
        var
    }

    /// Check if a variable name is a module-level state variable.
    fn is_state_var(&self, name: &str) -> bool {
        self.state_layout.vars.contains_key(name)
    }

    /// Load the state block pointer from PdcContext (offset 16).
    fn load_state_ptr(&mut self) -> cranelift_codegen::ir::Value {
        self.builder.ins().load(
            self.pointer_type,
            MemFlags::trusted(),
            self.ctx_ptr,
            16i32, // PdcContext.state is at offset 16 (after builtins:8 + scene:8)
        )
    }

    /// Load a module-level state variable from the state block.
    fn load_state_var(&mut self, name: &str) -> Result<cranelift_codegen::ir::Value, PdcError> {
        let sv = self.state_layout.vars.get(name).ok_or_else(|| PdcError::Codegen {
            message: format!("state var '{name}' not found in layout"),
        })?;
        let cl_type = pdc_type_to_cl(&sv.ty, self.pointer_type);
        let offset = sv.offset as i32;
        let state_ptr = self.load_state_ptr();
        Ok(self.builder.ins().load(cl_type, MemFlags::trusted(), state_ptr, offset))
    }

    /// Store a value to a module-level state variable in the state block.
    fn store_state_var(&mut self, name: &str, val: cranelift_codegen::ir::Value) -> Result<(), PdcError> {
        let sv = self.state_layout.vars.get(name).ok_or_else(|| PdcError::Codegen {
            message: format!("state var '{name}' not found in layout"),
        })?;
        let offset = sv.offset as i32;
        let state_ptr = self.load_state_ptr();
        self.builder.ins().store(MemFlags::trusted(), val, state_ptr, offset);
        Ok(())
    }

    /// Check if a variable name is a mutable builtin.
    fn is_mutable_builtin(&self, name: &str) -> bool {
        self.mutable_builtins.contains(name)
    }

    /// Load a mutable builtin from the builtins array.
    /// All builtins are stored as f64; this converts to the declared type.
    fn load_mutable_builtin(&mut self, name: &str) -> Result<cranelift_codegen::ir::Value, PdcError> {
        let info = self.builtin_map.get(name).ok_or_else(|| PdcError::Codegen {
            message: format!("mutable builtin '{name}' not found in layout"),
        })?;
        let offset = info.offset;
        let declared_ty = info.ty.clone();
        let builtins_ptr = self.builder.ins().load(
            self.pointer_type, MemFlags::trusted(), self.ctx_ptr, 0,
        );
        let f64_val = self.builder.ins().load(
            F64, MemFlags::trusted(), builtins_ptr, (offset * 8) as i32,
        );
        // Bool builtins: f64 != 0.0 → I8
        if declared_ty == PdcType::Bool {
            let zero = self.builder.ins().f64const(0.0);
            return Ok(self.builder.ins().fcmp(FloatCC::NotEqual, f64_val, zero));
        }
        Ok(self.convert_value(f64_val, &PdcType::F64, &declared_ty))
    }

    /// Store a value to a mutable builtin in the builtins array.
    /// Converts the declared type back to f64 for storage.
    fn store_mutable_builtin(&mut self, name: &str, val: cranelift_codegen::ir::Value) -> Result<(), PdcError> {
        let info = self.builtin_map.get(name).ok_or_else(|| PdcError::Codegen {
            message: format!("mutable builtin '{name}' not found in layout"),
        })?;
        let offset = info.offset;
        let declared_ty = info.ty.clone();
        // Bool builtins: I8 → f64 (0.0 or 1.0)
        let f64_val = if declared_ty == PdcType::Bool {
            let i64_val = self.builder.ins().uextend(I64, val);
            self.builder.ins().fcvt_from_uint(F64, i64_val)
        } else {
            self.convert_value(val, &declared_ty, &PdcType::F64)
        };
        let builtins_ptr = self.builder.ins().load(
            self.pointer_type, MemFlags::trusted(), self.ctx_ptr, 0,
        );
        self.builder.ins().store(MemFlags::trusted(), f64_val, builtins_ptr, (offset * 8) as i32);
        Ok(())
    }

    fn node_type(&self, id: u32) -> &PdcType {
        if (id as usize) >= self.type_table.len() {
            eprintln!("WARNING: node id {} out of type table (len {})", id, self.type_table.len());
            return &PdcType::Unknown;
        }
        &self.type_table[id as usize]
    }

    fn default_value(&mut self, ty: &PdcType) -> cranelift_codegen::ir::Value {
        let cl = pdc_type_to_cl(ty, self.pointer_type);
        match cl {
            F32 => self.builder.ins().f32const(0.0),
            F64 => self.builder.ins().f64const(0.0),
            _ => self.builder.ins().iconst(cl, 0),
        }
    }

    fn emit_block(&mut self, block: &Block) -> Result<(), PdcError> {
        for stmt in &block.stmts {
            self.emit_stmt(stmt)?;
            if self.block_terminated {
                break;
            }
        }
        Ok(())
    }

    fn emit_stmt(&mut self, stmt: &Spanned<Stmt>) -> Result<(), PdcError> {
        match &stmt.node {
            Stmt::BuiltinDecl { name, ty, mutable } => {
                let info = self.builtin_map.get(name).ok_or_else(|| PdcError::Codegen {
                    message: format!("builtin '{name}' not found in layout"),
                })?;
                let offset = info.offset;
                let declared_ty = ty.clone();

                if *mutable {
                    // Mutable builtins: reads/writes go directly to the builtins
                    // array. Don't snapshot into an SSA variable.
                    self.mutable_builtins.insert(name.clone());
                } else {
                    // Immutable builtins: snapshot the value into an SSA variable.
                    let builtins_ptr = self.builder.ins().load(
                        self.pointer_type,
                        MemFlags::trusted(),
                        self.ctx_ptr,
                        0,
                    );
                    let f64_val = self.builder.ins().load(
                        F64,
                        MemFlags::trusted(),
                        builtins_ptr,
                        (offset * 8) as i32,
                    );
                    let val = if declared_ty == PdcType::Bool {
                        let zero = self.builder.ins().f64const(0.0);
                        self.builder.ins().fcmp(FloatCC::NotEqual, f64_val, zero)
                    } else {
                        self.convert_value(f64_val, &PdcType::F64, &declared_ty)
                    };
                    let var = self.new_variable(name, &declared_ty);
                    self.builder.def_var(var, val);
                }
            }
            Stmt::ConstDecl { vis: _, name, ty, value } | Stmt::VarDecl { vis: _, name, ty, value } => {
                let is_var = matches!(&stmt.node, Stmt::VarDecl { .. });
                let val = self.emit_expr(value)?;
                let expr_ty = self.node_type(value.id).clone();
                let final_ty = ty.clone().unwrap_or(expr_ty.clone());

                // State block variables: store to state block and skip SSA registration.
                // Only VarDecl can be state vars (ConstDecl are always pure-literal SSA).
                if is_var && self.is_state_var(name) {
                    let converted = self.convert_value(val, &expr_ty, &final_ty);
                    self.store_state_var(name, converted)?;
                    return Ok(());
                }

                // Struct and data-variant enum values are pointers to stack slots
                let is_pointer_type = match &final_ty {
                    PdcType::Struct(_) | PdcType::Tuple(_) => true,
                    PdcType::Enum(ename) => {
                        self.enums.get(ename).map_or(false, |info| info.variants.iter().any(|v| !v.field_types.is_empty()))
                    }
                    _ => false,
                };
                if is_pointer_type {
                    let var = self.builder.declare_var(self.pointer_type);
                    self.builder.def_var(var, val);
                    self.variables.insert(name.clone(), (var, final_ty));
                } else {
                    let converted = self.convert_value(val, &expr_ty, &final_ty);
                    let var = self.new_variable(name, &final_ty);
                    self.builder.def_var(var, converted);
                }
            }
            Stmt::IndexAssign { object, index, value } => {
                let obj_ty = self.node_type(object.id).clone();
                if let PdcType::Array(ref elem_ty) = obj_ty {
                    let arr_handle = self.emit_expr(object)?;
                    let idx = self.emit_expr(index)?;
                    let val = self.emit_expr(value)?;
                    let val_ty = self.node_type(value.id).clone();
                    let converted = self.convert_value(val, &val_ty, elem_ty);
                    let elem_cl = pdc_type_to_cl(elem_ty, self.pointer_type);
                    let elem_size = elem_cl.bytes() as u32;
                    // Inline store via data pointer
                    let data_ptr = self.emit_runtime_call_raw(
                        "pdc_array_data_ptr", &[self.ctx_ptr, arr_handle], Some(self.pointer_type))?;
                    self.emit_inline_array_store(data_ptr, idx, elem_size, converted, elem_cl);
                }
            }
            Stmt::FieldAssign { object, field, value } => {
                let kernel_handle = self.emit_expr(object)?;
                let val_ty = self.node_type(value.id).clone();

                // Emit field name as inline string data (ptr, len) — no allocation.
                let field_bytes = field.as_bytes();
                let (field_ptr, field_len) = self.emit_inline_str(field_bytes);

                if let PdcType::Enum(ref ename) = val_ty {
                    if ename == "Bind" {
                        // Destructure Bind.In(buffer) / Bind.Out(buffer) at compile time
                        if let Expr::MethodCall { method, args, .. } = &value.node {
                            let direction = if method == "In" { 0i64 } else { 1i64 };
                            let buffer_handle = self.emit_expr(&args[0])?;
                            let dir_val = self.builder.ins().iconst(I32, direction);
                            self.emit_runtime_call_raw("pdc_bind_buffer",
                                &[self.ctx_ptr, kernel_handle, buffer_handle, field_ptr, field_len, dir_val],
                                None)?;
                        }
                    }
                } else {
                    // Scalar arg: emit set_kernel_arg_f64
                    let val = self.emit_expr(value)?;
                    let converted = self.convert_value(val, &val_ty, &PdcType::F64);
                    self.emit_runtime_call_raw("pdc_set_kernel_arg_f64",
                        &[self.ctx_ptr, kernel_handle, field_ptr, field_len, converted],
                        None)?;
                }
            }
            Stmt::TupleDestructure { names, value, .. } => {
                let tuple_ptr = self.emit_expr(value)?;
                let val_ty = self.node_type(value.id).clone();
                if let PdcType::Tuple(ref elems) = val_ty {
                    for (i, name) in names.iter().enumerate() {
                        if name == "_" {
                            continue;
                        }
                        let elem_ty = &elems[i];
                        let raw = self.builder.ins().load(F64, MemFlags::trusted(), tuple_ptr, (i * 8) as i32);
                        let val = self.narrow_from_f64(raw, elem_ty);
                        let var = self.new_variable(name, elem_ty);
                        self.builder.def_var(var, val);
                    }
                }
            }
            Stmt::Assign { name, value } => {
                let val = self.emit_expr(value)?;
                let expr_ty = self.node_type(value.id).clone();

                // Mutable builtin: store to builtins array
                if self.is_mutable_builtin(name) {
                    let info_ty = self.builtin_map.get(name).unwrap().ty.clone();
                    let converted = self.convert_value(val, &expr_ty, &info_ty);
                    self.store_mutable_builtin(name, converted)?;
                    return Ok(());
                }

                // State block variable: store to state block
                if let Some(sv) = self.state_layout.vars.get(name) {
                    let converted = self.convert_value(val, &expr_ty, &sv.ty.clone());
                    self.store_state_var(name, converted)?;
                    return Ok(());
                }

                let (var, var_ty) = self.variables.get(name).cloned().ok_or_else(|| {
                    PdcError::Codegen {
                        message: format!("undefined variable '{name}'"),
                    }
                })?;
                let converted = self.convert_value(val, &expr_ty, &var_ty);
                self.builder.def_var(var, converted);
            }
            Stmt::ExprStmt(expr) => {
                self.emit_expr(expr)?;
            }
            Stmt::If {
                condition,
                then_body,
                elsif_clauses,
                else_body,
            } => {
                self.emit_if(condition, then_body, elsif_clauses, else_body)?;
            }
            Stmt::While { condition, body } => {
                self.emit_while(condition, body)?;
            }
            Stmt::For { mutable: _,
                var_name,
                start,
                end,
                inclusive,
                body,
            } => {
                self.emit_for(var_name, start, end, *inclusive, body)?;
            }
            Stmt::ForEach { mutable: _,
                var_name,
                destructure_names,
                collection,
                body,
            } => {
                let coll_ty = self.node_type(collection.id).clone();
                let elem_ty = match &coll_ty {
                    PdcType::Array(et) => *et.clone(),
                    _ => return Err(PdcError::Codegen {
                        message: "for-each requires an array".into(),
                    }),
                };

                let arr_handle = self.emit_expr(collection)?;

                // Get array length and data pointer once before the loop.
                // Inline loads avoid per-element boundary crossings.
                let len_val = self.emit_runtime_call_raw(
                    "pdc_array_len",
                    &[self.ctx_ptr, arr_handle],
                    Some(I32),
                )?;
                let data_ptr = self.emit_runtime_call_raw(
                    "pdc_array_data_ptr",
                    &[self.ctx_ptr, arr_handle],
                    Some(self.pointer_type),
                )?;

                // Index variable
                let idx_var = self.builder.declare_var(I32);
                let zero = self.builder.ins().iconst(I32, 0);
                self.builder.def_var(idx_var, zero);

                let header_block = self.builder.create_block();
                let body_block = self.builder.create_block();
                let latch_block = self.builder.create_block();
                let exit_block = self.builder.create_block();

                self.builder.ins().jump(header_block, &[]);

                // Header: check idx < len
                self.builder.switch_to_block(header_block);
                let idx_val = self.builder.use_var(idx_var);
                let cond = self.builder.ins().icmp(IntCC::SignedLessThan, idx_val, len_val);
                self.builder.ins().brif(cond, body_block, &[], exit_block, &[]);

                // Body: load element via inline pointer arithmetic
                self.builder.switch_to_block(body_block);
                self.builder.seal_block(body_block);
                self.block_terminated = false;

                let elem_cl = pdc_type_to_cl(&elem_ty, self.pointer_type);
                let elem_size = elem_cl.bytes() as u32;
                let idx_for_load = self.builder.use_var(idx_var);
                let elem_val = self.emit_inline_array_load(data_ptr, idx_for_load, elem_size, elem_cl);

                if !destructure_names.is_empty() {
                    // Destructuring: elem_val is a pointer to a tuple
                    if let PdcType::Tuple(ref tuple_elems) = elem_ty {
                        for (i, name) in destructure_names.iter().enumerate() {
                            if name == "_" {
                                continue;
                            }
                            let field_ty = &tuple_elems[i];
                            let raw = self.builder.ins().load(F64, MemFlags::trusted(), elem_val, (i * 8) as i32);
                            let val = self.narrow_from_f64(raw, field_ty);
                            let var = self.new_variable(name, field_ty);
                            self.builder.def_var(var, val);
                        }
                    }
                } else {
                    let var = self.new_variable(var_name, &elem_ty);
                    self.builder.def_var(var, elem_val);
                }

                self.loop_stack.push(LoopContext { continue_block: latch_block, exit_block });
                self.emit_block(body)?;
                self.loop_stack.pop();
                if !self.block_terminated {
                    self.builder.ins().jump(latch_block, &[]);
                }

                // Latch: increment index and jump back to header
                self.builder.switch_to_block(latch_block);
                self.builder.seal_block(latch_block);
                let idx_val = self.builder.use_var(idx_var);
                let one = self.builder.ins().iconst(I32, 1);
                let next = self.builder.ins().iadd(idx_val, one);
                self.builder.def_var(idx_var, next);
                self.builder.ins().jump(header_block, &[]);

                self.builder.seal_block(header_block);
                self.builder.switch_to_block(exit_block);
                self.builder.seal_block(exit_block);
                self.block_terminated = false;
            }
            Stmt::Loop { body } => {
                self.emit_loop(body)?;
            }
            Stmt::Break => {
                let lc = self.loop_stack.last().ok_or_else(|| PdcError::Codegen {
                    message: "break outside of loop".into(),
                })?;
                self.builder.ins().jump(lc.exit_block, &[]);
                self.block_terminated = true;
            }
            Stmt::Continue => {
                let lc = self.loop_stack.last().ok_or_else(|| PdcError::Codegen {
                    message: "continue outside of loop".into(),
                })?;
                self.builder.ins().jump(lc.continue_block, &[]);
                self.block_terminated = true;
            }
            Stmt::Return(value) => {
                if let Some(expr) = value {
                    let val = self.emit_expr(expr)?;
                    self.builder.ins().return_(&[val]);
                } else {
                    self.builder.ins().return_(&[]);
                }
                self.block_terminated = true;
            }
            Stmt::Match { scrutinee, arms } => {
                self.emit_match(scrutinee, arms)?;
            }
            Stmt::FnDef(_) | Stmt::Import { .. } | Stmt::StructDef(_) | Stmt::EnumDef(_) | Stmt::TypeAlias { .. } | Stmt::TestDef { .. } => {
                // Already handled (tests compiled separately, others registered in earlier passes)
            }
        }
        Ok(())
    }

    fn emit_match(
        &mut self,
        scrutinee: &Spanned<Expr>,
        arms: &[MatchArm],
    ) -> Result<(), PdcError> {
        let scr_val = self.emit_expr(scrutinee)?;
        let scr_ty = self.node_type(scrutinee.id).clone();

        // Determine if this is a simple enum (i32 tag) or data variant (pointer to tagged union)
        let has_data_variants = if let PdcType::Enum(ref ename) = scr_ty {
            self.enums.get(ename).map_or(false, |info| info.variants.iter().any(|v| !v.field_types.is_empty()))
        } else {
            false
        };

        // For data variant enums, load the tag from the pointer
        let tag_val = if has_data_variants {
            self.builder.ins().load(I32, MemFlags::trusted(), scr_val, 0)
        } else {
            scr_val // simple enum: the value IS the tag
        };

        let merge_block = self.builder.create_block();

        for (i, arm) in arms.iter().enumerate() {
            let is_last = i == arms.len() - 1;

            match &arm.pattern {
                MatchPattern::EnumVariant { enum_name: pat_enum, variant, bindings } => {
                    // Resolve enum name (empty = dot-shorthand, infer from scrutinee)
                    let resolved_name = if pat_enum.is_empty() {
                        match &scr_ty {
                            PdcType::Enum(name) => name.clone(),
                            _ => return Err(PdcError::Codegen { message: "match scrutinee is not an enum".into() }),
                        }
                    } else {
                        pat_enum.clone()
                    };
                    let info = self.enums.get(&resolved_name).ok_or_else(|| PdcError::Codegen {
                        message: format!("undefined enum '{resolved_name}'"),
                    })?.clone();
                    let variant_idx = info.variants.iter().position(|v| v.name == *variant)
                        .ok_or_else(|| PdcError::Codegen {
                            message: format!("enum '{resolved_name}' has no variant '{variant}'"),
                        })?;
                    let variant_val = self.builder.ins().iconst(I32, variant_idx as i64);
                    let cmp = self.builder.ins().icmp(IntCC::Equal, tag_val, variant_val);

                    let arm_block = self.builder.create_block();
                    let next_block = if is_last { merge_block } else { self.builder.create_block() };

                    self.builder.ins().brif(cmp, arm_block, &[], next_block, &[]);

                    self.builder.switch_to_block(arm_block);
                    self.builder.seal_block(arm_block);
                    self.block_terminated = false;

                    // Destructure: bind variables from the payload
                    if !bindings.is_empty() && has_data_variants {
                        let variant_info = &info.variants[variant_idx];
                        for (bi, bname) in bindings.iter().enumerate() {
                            if bname == "_" {
                                continue; // wildcard: skip binding
                            }
                            let field_ty = &variant_info.field_types[bi];
                            let raw = self.builder.ins().load(F64, MemFlags::trusted(), scr_val, (8 + bi * 8) as i32);
                            let val = self.narrow_from_f64(raw, field_ty);
                            let var = self.new_variable(bname, field_ty);
                            self.builder.def_var(var, val);
                        }
                    }

                    self.emit_block(&arm.body)?;
                    if !self.block_terminated {
                        self.builder.ins().jump(merge_block, &[]);
                    }

                    if !is_last {
                        self.builder.switch_to_block(next_block);
                        self.builder.seal_block(next_block);
                        self.block_terminated = false;
                    }
                }
                MatchPattern::Wildcard => {
                    self.block_terminated = false;
                    self.emit_block(&arm.body)?;
                    if !self.block_terminated {
                        self.builder.ins().jump(merge_block, &[]);
                    }
                }
            }
        }

        self.builder.switch_to_block(merge_block);
        self.builder.seal_block(merge_block);
        self.block_terminated = false;
        Ok(())
    }

    /// Emit array method call with size-specific runtime functions.
    fn emit_array_method(
        &mut self,
        object: &Spanned<Expr>,
        method: &str,
        args: &[Spanned<Expr>],
        elem_ty: &PdcType,
    ) -> Result<cranelift_codegen::ir::Value, PdcError> {
        let handle = self.emit_expr(object)?;
        let elem_cl = pdc_type_to_cl(elem_ty, self.pointer_type);
        let elem_size = elem_cl.bytes() as u32;

        match method {
            "push" => {
                let val = self.emit_expr(&args[0])?;
                let val_ty = self.node_type(args[0].id).clone();
                let converted = self.convert_value(val, &val_ty, elem_ty);
                // For f32/f64: bitcast to integer of same size for the push function
                let store_val = self.float_to_int_if_needed(converted, elem_cl);
                let push_name = format!("pdc_array_push_{elem_size}");
                self.emit_runtime_call_raw(&push_name, &[self.ctx_ptr, handle, store_val], None)?;
                Ok(self.builder.ins().iconst(I32, 0))
            }
            "len" => {
                self.emit_runtime_call_raw("pdc_array_len", &[self.ctx_ptr, handle], Some(I32))
            }
            "get" => {
                let idx = self.emit_expr(&args[0])?;
                let get_name = format!("pdc_array_get_{elem_size}");
                let int_type = match elem_size {
                    1 => I8, 2 => I16, 4 => I32, _ => I64,
                };
                let raw = self.emit_runtime_call_raw(&get_name, &[self.ctx_ptr, handle, idx], Some(int_type))?;
                // For f32/f64: bitcast back from integer
                Ok(self.int_to_float_if_needed(raw, elem_cl))
            }
            "set" => {
                let idx = self.emit_expr(&args[0])?;
                let val = self.emit_expr(&args[1])?;
                let val_ty = self.node_type(args[1].id).clone();
                let converted = self.convert_value(val, &val_ty, elem_ty);
                let store_val = self.float_to_int_if_needed(converted, elem_cl);
                let set_name = format!("pdc_array_set_{elem_size}");
                self.emit_runtime_call_raw(&set_name, &[self.ctx_ptr, handle, idx, store_val], None)?;
                Ok(self.builder.ins().iconst(I32, 0))
            }
            "map" => {
                self.emit_array_map(object, &args[0], elem_ty)
            }
            "slice" => {
                // Create a slice: stack-allocate (arr_handle: i32, start: i32, len: i32)
                let start_val = self.emit_expr(&args[0])?;
                let end_val = self.emit_expr(&args[1])?;
                let len_val = self.builder.ins().isub(end_val, start_val);

                let slot = self.builder.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
                    cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                    12, // 3 × i32
                    0,
                ));
                self.builder.ins().stack_store(handle, slot, 0);  // arr_handle
                self.builder.ins().stack_store(start_val, slot, 4); // start
                self.builder.ins().stack_store(len_val, slot, 8);  // length

                Ok(self.builder.ins().stack_addr(self.pointer_type, slot, 0))
            }
            _ => Err(PdcError::Codegen {
                message: format!("unknown array method '{method}'"),
            }),
        }
    }

    /// Emit array broadcasting: array op array, array op scalar, scalar op array.
    fn emit_array_broadcast(
        &mut self,
        op: BinOp,
        left: &Spanned<Expr>,
        right: &Spanned<Expr>,
        lt: &PdcType,
        rt: &PdcType,
        result_ty: &PdcType,
    ) -> Result<cranelift_codegen::ir::Value, PdcError> {
        let result_elem_ty = match result_ty {
            PdcType::Array(et) => *et.clone(),
            _ => return Err(PdcError::Codegen { message: "broadcast result must be array".into() }),
        };
        let result_elem_cl = pdc_type_to_cl(&result_elem_ty, self.pointer_type);
        let result_elem_size = result_elem_cl.bytes() as u32;

        let l_is_arr = matches!(lt, PdcType::Array(_));
        let r_is_arr = matches!(rt, PdcType::Array(_));

        let lval = self.emit_expr(left)?;
        let rval = self.emit_expr(right)?;

        // Get the length from whichever operand is an array
        let arr_handle = if l_is_arr { lval } else { rval };
        let len_val = self.emit_runtime_call_raw("pdc_array_len", &[self.ctx_ptr, arr_handle], Some(I32))?;

        // Create result array
        let size_val = self.builder.ins().iconst(I32, result_elem_size as i64);
        let new_arr = self.emit_runtime_call_raw("pdc_array_new", &[self.ctx_ptr, size_val], Some(I32))?;

        // Element types for the operation
        let l_elem_ty = match lt { PdcType::Array(et) => *et.clone(), _ => lt.clone() };
        let r_elem_ty = match rt { PdcType::Array(et) => *et.clone(), _ => rt.clone() };
        let op_result_ty = if result_elem_ty == PdcType::Bool && !matches!(op, BinOp::And | BinOp::Or) {
            // For comparison broadcasting, the scalar op result is Bool but we need
            // the numeric type for the operation
            if l_elem_ty.is_float() || r_elem_ty.is_float() { PdcType::F64 } else { PdcType::I32 }
        } else {
            result_elem_ty.clone()
        };

        // Loop
        let idx_var = self.builder.declare_var(I32);
        let zero = self.builder.ins().iconst(I32, 0);
        self.builder.def_var(idx_var, zero);

        let header_block = self.builder.create_block();
        let body_block = self.builder.create_block();
        let exit_block = self.builder.create_block();

        self.builder.ins().jump(header_block, &[]);

        self.builder.switch_to_block(header_block);
        let idx_val = self.builder.use_var(idx_var);
        let cond = self.builder.ins().icmp(IntCC::SignedLessThan, idx_val, len_val);
        self.builder.ins().brif(cond, body_block, &[], exit_block, &[]);

        self.builder.switch_to_block(body_block);
        self.builder.seal_block(body_block);

        // Get left element (or scalar)
        let l_elem = if l_is_arr {
            let l_elem_cl = pdc_type_to_cl(&l_elem_ty, self.pointer_type);
            let l_elem_size = l_elem_cl.bytes() as u32;
            let get_name = format!("pdc_array_get_{l_elem_size}");
            let int_type = match l_elem_size { 1 => I8, 2 => I16, 4 => I32, _ => I64 };
            let idx_for_get = self.builder.use_var(idx_var);
            let raw = self.emit_runtime_call_raw(&get_name, &[self.ctx_ptr, lval, idx_for_get], Some(int_type))?;
            self.int_to_float_if_needed(raw, l_elem_cl)
        } else {
            lval
        };

        // Get right element (or scalar)
        let r_elem = if r_is_arr {
            let r_elem_cl = pdc_type_to_cl(&r_elem_ty, self.pointer_type);
            let r_elem_size = r_elem_cl.bytes() as u32;
            let get_name = format!("pdc_array_get_{r_elem_size}");
            let int_type = match r_elem_size { 1 => I8, 2 => I16, 4 => I32, _ => I64 };
            let idx_for_get = self.builder.use_var(idx_var);
            let raw = self.emit_runtime_call_raw(&get_name, &[self.ctx_ptr, rval, idx_for_get], Some(int_type))?;
            self.int_to_float_if_needed(raw, r_elem_cl)
        } else {
            rval
        };

        // Apply binary op
        let elem_result = self.emit_binary_op(op, l_elem, r_elem, &l_elem_ty, &r_elem_ty, &op_result_ty)?;

        // For comparison ops, result is already Bool (I8); for arithmetic, use result_elem_ty
        let final_val = if result_elem_ty == PdcType::Bool {
            elem_result // already I8
        } else {
            self.convert_value(elem_result, &op_result_ty, &result_elem_ty)
        };

        // Push to result array
        let store_val = self.float_to_int_if_needed(final_val, result_elem_cl);
        let push_name = format!("pdc_array_push_{result_elem_size}");
        self.emit_runtime_call_raw(&push_name, &[self.ctx_ptr, new_arr, store_val], None)?;

        // Increment
        let idx_val = self.builder.use_var(idx_var);
        let one = self.builder.ins().iconst(I32, 1);
        let next = self.builder.ins().iadd(idx_val, one);
        self.builder.def_var(idx_var, next);
        self.builder.ins().jump(header_block, &[]);

        self.builder.seal_block(header_block);
        self.builder.switch_to_block(exit_block);
        self.builder.seal_block(exit_block);

        Ok(new_arr)
    }

    /// Emit array.map(fn_ref): create new array, loop, call fn, push results.
    fn emit_array_map(
        &mut self,
        array_expr: &Spanned<Expr>,
        fn_ref_expr: &Spanned<Expr>,
        elem_ty: &PdcType,
    ) -> Result<cranelift_codegen::ir::Value, PdcError> {
        let arr_handle = self.emit_expr(array_expr)?;

        // Get function name from the fn_ref expression
        let fn_name = match &fn_ref_expr.node {
            Expr::Variable(name) => name.clone(),
            _ => return Err(PdcError::Codegen {
                message: "map() argument must be a function name".into(),
            }),
        };

        // Determine result element type from the type table
        let result_ty = self.node_type(fn_ref_expr.id).clone();
        let ret_ty = match &result_ty {
            PdcType::FnRef { ret, .. } => *ret.clone(),
            _ => elem_ty.clone(),
        };
        let ret_cl = pdc_type_to_cl(&ret_ty, self.pointer_type);
        let ret_size = ret_cl.bytes() as u32;

        // Create new array for results
        let size_val = self.builder.ins().iconst(I32, ret_size as i64);
        let new_arr = self.emit_runtime_call_raw("pdc_array_new", &[self.ctx_ptr, size_val], Some(I32))?;

        // Get source array length
        let len_val = self.emit_runtime_call_raw("pdc_array_len", &[self.ctx_ptr, arr_handle], Some(I32))?;

        // Loop index
        let idx_var = self.builder.declare_var(I32);
        let zero = self.builder.ins().iconst(I32, 0);
        self.builder.def_var(idx_var, zero);

        let header_block = self.builder.create_block();
        let body_block = self.builder.create_block();
        let exit_block = self.builder.create_block();

        self.builder.ins().jump(header_block, &[]);

        // Header: check idx < len
        self.builder.switch_to_block(header_block);
        let idx_val = self.builder.use_var(idx_var);
        let cond = self.builder.ins().icmp(IntCC::SignedLessThan, idx_val, len_val);
        self.builder.ins().brif(cond, body_block, &[], exit_block, &[]);

        // Body: get element, call function, push result
        self.builder.switch_to_block(body_block);
        self.builder.seal_block(body_block);

        let elem_cl = pdc_type_to_cl(elem_ty, self.pointer_type);
        let elem_size = elem_cl.bytes() as u32;
        let get_name = format!("pdc_array_get_{elem_size}");
        let int_type = match elem_size {
            1 => I8, 2 => I16, 4 => I32, _ => I64,
        };
        let idx_for_get = self.builder.use_var(idx_var);
        let raw = self.emit_runtime_call_raw(&get_name, &[self.ctx_ptr, arr_handle, idx_for_get], Some(int_type))?;
        let elem_val = self.int_to_float_if_needed(raw, elem_cl);

        // Call the function
        let converted = self.convert_value(elem_val, elem_ty, &PdcType::F64);
        let fn_result = self.emit_fn_ref_call(&fn_name, converted, elem_ty, &ret_ty)?;

        // Push result to new array
        let store_val = self.float_to_int_if_needed(fn_result, ret_cl);
        let push_name = format!("pdc_array_push_{ret_size}");
        self.emit_runtime_call_raw(&push_name, &[self.ctx_ptr, new_arr, store_val], None)?;

        // Increment index
        let idx_val = self.builder.use_var(idx_var);
        let one = self.builder.ins().iconst(I32, 1);
        let next = self.builder.ins().iadd(idx_val, one);
        self.builder.def_var(idx_var, next);
        self.builder.ins().jump(header_block, &[]);

        self.builder.seal_block(header_block);
        self.builder.switch_to_block(exit_block);
        self.builder.seal_block(exit_block);

        Ok(new_arr)
    }

    /// Call a function by name with a single argument (used by map).
    fn emit_fn_ref_call(
        &mut self,
        fn_name: &str,
        arg: cranelift_codegen::ir::Value,
        _arg_ty: &PdcType,
        ret_ty: &PdcType,
    ) -> Result<cranelift_codegen::ir::Value, PdcError> {
        // Resolve through alias map (alias first for correct JIT symbol)
        let resolved = if let Some(qualified) = self.fn_aliases.get(fn_name) {
            qualified.clone()
        } else if self.user_fns.contains_key(fn_name) {
            fn_name.to_string()
        } else {
            fn_name.to_string()
        };

        // Check user-defined functions first
        if let Some(overloads) = self.user_fns.get(&resolved) {
            let overloads = overloads.clone();
            // Find single-arg overload
            let (overload_idx, _sig) = overloads.sigs.iter().enumerate()
                .find(|(_, sig)| sig.params.len() == 1)
                .ok_or_else(|| PdcError::Codegen {
                    message: format!("no single-arg overload for '{fn_name}'"),
                })?;
            let base = mangle_name(&resolved);
            let mangled = if overload_idx == 0 {
                format!("pdc_userfn_{base}")
            } else {
                format!("pdc_userfn_{base}_{overload_idx}")
            };
            let func_id = self.user_fn_ids[&mangled];
            let func_ref = self.module.declare_func_in_func(func_id, self.builder.func);
            let call = self.builder.ins().call(func_ref, &[self.ctx_ptr, arg]);
            let results = self.builder.inst_results(call);
            if results.is_empty() {
                return Ok(self.builder.ins().iconst(I32, 0));
            }
            return Ok(results[0]);
        }

        // Emit native Cranelift instructions for math functions where possible.
        // These avoid the overhead of an extern function call.
        match fn_name {
            "abs" => return Ok(self.builder.ins().fabs(arg)),
            "sqrt" => return Ok(self.builder.ins().sqrt(arg)),
            "floor" => return Ok(self.builder.ins().floor(arg)),
            "ceil" => return Ok(self.builder.ins().ceil(arg)),
            "round" => return Ok(self.builder.ins().nearest(arg)),
            _ => {}
        }

        // Builtin runtime function
        let runtime_name = format!("pdc_{fn_name}");
        let ret_cl = if *ret_ty != PdcType::Void {
            Some(pdc_type_to_cl(ret_ty, self.pointer_type))
        } else {
            None
        };
        self.emit_runtime_call_raw(&runtime_name, &[arg], ret_cl)
    }

    /// Bitcast float to integer of the same size for passing to runtime functions.
    fn float_to_int_if_needed(&mut self, val: cranelift_codegen::ir::Value, cl: cranelift_codegen::ir::Type) -> cranelift_codegen::ir::Value {
        match cl {
            F32 => self.builder.ins().bitcast(I32, MemFlags::new(), val),
            F64 => self.builder.ins().bitcast(I64, MemFlags::new(), val),
            _ => val,
        }
    }

    /// Bitcast integer back to float if the element type is float.
    fn int_to_float_if_needed(&mut self, val: cranelift_codegen::ir::Value, cl: cranelift_codegen::ir::Type) -> cranelift_codegen::ir::Value {
        match cl {
            F32 => self.builder.ins().bitcast(F32, MemFlags::new(), val),
            F64 => self.builder.ins().bitcast(F64, MemFlags::new(), val),
            _ => val,
        }
    }

    /// Emit an inline array element load via pointer arithmetic.
    /// `data_ptr` is the base pointer, `index` is the element index (i32),
    /// `elem_size` is bytes per element, `elem_cl` is the Cranelift type to load.
    fn emit_inline_array_load(
        &mut self,
        data_ptr: cranelift_codegen::ir::Value,
        index: cranelift_codegen::ir::Value,
        elem_size: u32,
        elem_cl: cranelift_codegen::ir::Type,
    ) -> cranelift_codegen::ir::Value {
        // offset = index * elem_size (as pointer-width)
        let idx_ext = self.builder.ins().sextend(self.pointer_type, index);
        let size_val = self.builder.ins().iconst(self.pointer_type, elem_size as i64);
        let offset = self.builder.ins().imul(idx_ext, size_val);
        let elem_ptr = self.builder.ins().iadd(data_ptr, offset);
        // Load the raw integer value
        let int_type = match elem_size {
            1 => I8, 2 => I16, 4 => I32, _ => I64,
        };
        let raw = self.builder.ins().load(int_type, MemFlags::trusted(), elem_ptr, 0);
        // Bitcast to float if needed
        self.int_to_float_if_needed(raw, elem_cl)
    }

    /// Emit an inline array element store via pointer arithmetic.
    fn emit_inline_array_store(
        &mut self,
        data_ptr: cranelift_codegen::ir::Value,
        index: cranelift_codegen::ir::Value,
        elem_size: u32,
        value: cranelift_codegen::ir::Value,
        elem_cl: cranelift_codegen::ir::Type,
    ) {
        let idx_ext = self.builder.ins().sextend(self.pointer_type, index);
        let size_val = self.builder.ins().iconst(self.pointer_type, elem_size as i64);
        let offset = self.builder.ins().imul(idx_ext, size_val);
        let elem_ptr = self.builder.ins().iadd(data_ptr, offset);
        let int_type = match elem_size {
            1 => I8, 2 => I16, 4 => I32, _ => I64,
        };
        let store_val = self.float_to_int_if_needed(value, int_type);
        self.builder.ins().store(MemFlags::trusted(), store_val, elem_ptr, 0);
    }

    /// Extract a string literal from an expression and emit it as inline (ptr, len).
    /// Panics if the expression is not a string literal.
    fn emit_str_arg(&mut self, expr: &Spanned<Expr>) -> (cranelift_codegen::ir::Value, cranelift_codegen::ir::Value) {
        if let Expr::Literal(Literal::String(ref s)) = expr.node {
            self.emit_inline_str(s.as_bytes())
        } else {
            panic!("expected string literal argument");
        }
    }

    /// Emit inline string data as a stack slot, returning (ptr, len) values.
    /// Used for passing compile-time-known strings directly to runtime functions
    /// without allocating through pdc_string_new.
    fn emit_inline_str(&mut self, bytes: &[u8]) -> (cranelift_codegen::ir::Value, cranelift_codegen::ir::Value) {
        let slot = self.builder.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
            cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
            bytes.len() as u32,
            0,
        ));
        for (i, &byte) in bytes.iter().enumerate() {
            let val = self.builder.ins().iconst(I8, byte as i64);
            self.builder.ins().stack_store(val, slot, i as i32);
        }
        let ptr = self.builder.ins().stack_addr(self.pointer_type, slot, 0);
        let len = self.builder.ins().iconst(I32, bytes.len() as i64);
        (ptr, len)
    }

    /// Emit a direct runtime function call with explicit return type.
    fn emit_runtime_call_raw(
        &mut self,
        runtime_name: &str,
        args: &[cranelift_codegen::ir::Value],
        ret_type: Option<cranelift_codegen::ir::Type>,
    ) -> Result<cranelift_codegen::ir::Value, PdcError> {
        let mut sig = self.module.make_signature();
        sig.call_conv = self.call_conv;
        for val in args {
            let ty = self.builder.func.dfg.value_type(*val);
            sig.params.push(AbiParam::new(ty));
        }
        if let Some(rt) = ret_type {
            sig.returns.push(AbiParam::new(rt));
        }

        let callee = self.module
            .declare_function(runtime_name, Linkage::Import, &sig)
            .map_err(|e| PdcError::Codegen {
                message: format!("declare {runtime_name}: {e}"),
            })?;

        let func_ref = self.module.declare_func_in_func(callee, self.builder.func);
        let call = self.builder.ins().call(func_ref, args);
        let results = self.builder.inst_results(call);

        if ret_type.is_some() && !results.is_empty() {
            Ok(results[0])
        } else {
            Ok(self.builder.ins().iconst(I32, 0))
        }
    }

    /// Construct a data variant enum value. Allocates a stack slot with
    /// [tag: i32 (4 bytes), pad (4 bytes), field0: f64, field1: f64, ...]
    /// Emit struct construction from a Call with named args.
    fn emit_struct_construct_from_call(
        &mut self,
        name: &str,
        args: &[Spanned<Expr>],
        arg_names: &[Option<String>],
        info: &StructInfo,
    ) -> Result<cranelift_codegen::ir::Value, PdcError> {
        let size = (info.fields.len() * 8) as u32;
        let slot = self.builder.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
            cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
            size,
            0,
        ));

        for (i, arg) in args.iter().enumerate() {
            let val = self.emit_expr(arg)?;
            let arg_ty = self.node_type(arg.id).clone();

            let fname = arg_names[i].as_ref().unwrap();
            let field_idx = info.fields.iter().position(|(n, _)| n == fname)
                .ok_or_else(|| PdcError::Codegen {
                    message: format!("struct '{name}' has no field '{fname}'"),
                })?;
            let offset = (field_idx * 8) as i32;
            let field_ty = &info.fields[field_idx].1;
            let converted = self.convert_value(val, &arg_ty, field_ty);
            let store_val = self.widen_to_f64(converted, field_ty);
            self.builder.ins().stack_store(store_val, slot, offset);
        }

        Ok(self.builder.ins().stack_addr(self.pointer_type, slot, 0))
    }

    fn emit_enum_construct(
        &mut self,
        enum_name: &str,
        variant_name: &str,
        args: &[Spanned<Expr>],
    ) -> Result<cranelift_codegen::ir::Value, PdcError> {
        let info = self.enums.get(enum_name).ok_or_else(|| PdcError::Codegen {
            message: format!("undefined enum '{enum_name}'"),
        })?.clone();

        let variant_idx = info.variants.iter().position(|v| v.name == variant_name)
            .ok_or_else(|| PdcError::Codegen {
                message: format!("enum '{enum_name}' has no variant '{variant_name}'"),
            })?;

        // Calculate size: 8 bytes for tag + max(variant payload sizes) * 8
        let max_fields = info.variants.iter().map(|v| v.field_types.len()).max().unwrap_or(0);
        let total_size = (8 + max_fields * 8) as u32; // 8 for tag, 8 per field

        let slot = self.builder.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
            cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
            total_size,
            0,
        ));

        // Write tag
        let tag_val = self.builder.ins().iconst(I32, variant_idx as i64);
        self.builder.ins().stack_store(tag_val, slot, 0);

        // Write fields
        for (i, arg) in args.iter().enumerate() {
            let val = self.emit_expr(arg)?;
            let arg_ty = self.node_type(arg.id).clone();
            let store_val = self.widen_to_f64(val, &arg_ty);
            self.builder.ins().stack_store(store_val, slot, (8 + i * 8) as i32);
        }

        Ok(self.builder.ins().stack_addr(self.pointer_type, slot, 0))
    }

    fn emit_if(
        &mut self,
        condition: &Spanned<Expr>,
        then_body: &Block,
        elsif_clauses: &[(Spanned<Expr>, Block)],
        else_body: &Option<Block>,
    ) -> Result<(), PdcError> {
        let cond_val = self.emit_expr(condition)?;

        let then_block = self.builder.create_block();
        let merge_block = self.builder.create_block();

        // Build chain: if → elsif → elsif → else → merge
        let mut next_block = if !elsif_clauses.is_empty() || else_body.is_some() {
            self.builder.create_block()
        } else {
            merge_block
        };

        self.builder.ins().brif(cond_val, then_block, &[], next_block, &[]);

        // Then block
        self.builder.switch_to_block(then_block);
        self.builder.seal_block(then_block);
        self.block_terminated = false;
        self.emit_block(then_body)?;
        if !self.block_terminated {
            self.builder.ins().jump(merge_block, &[]);
        }

        // Elsif clauses
        for (i, (cond, body)) in elsif_clauses.iter().enumerate() {
            self.builder.switch_to_block(next_block);
            self.builder.seal_block(next_block);

            let cond_val = self.emit_expr(cond)?;
            let elsif_block = self.builder.create_block();

            let is_last = i == elsif_clauses.len() - 1;
            next_block = if is_last && else_body.is_none() {
                merge_block
            } else {
                self.builder.create_block()
            };

            self.builder.ins().brif(cond_val, elsif_block, &[], next_block, &[]);

            self.builder.switch_to_block(elsif_block);
            self.builder.seal_block(elsif_block);
            self.block_terminated = false;
            self.emit_block(body)?;
            if !self.block_terminated {
                self.builder.ins().jump(merge_block, &[]);
            }
        }

        // Else block
        if let Some(else_b) = else_body {
            self.builder.switch_to_block(next_block);
            self.builder.seal_block(next_block);
            self.block_terminated = false;
            self.emit_block(else_b)?;
            if !self.block_terminated {
                self.builder.ins().jump(merge_block, &[]);
            }
        }

        self.builder.switch_to_block(merge_block);
        self.builder.seal_block(merge_block);
        self.block_terminated = false;
        Ok(())
    }

    fn emit_while(
        &mut self,
        condition: &Spanned<Expr>,
        body: &Block,
    ) -> Result<(), PdcError> {
        let header_block = self.builder.create_block();
        let body_block = self.builder.create_block();
        let exit_block = self.builder.create_block();

        self.builder.ins().jump(header_block, &[]);

        // Header: evaluate condition
        self.builder.switch_to_block(header_block);
        let cond_val = self.emit_expr(condition)?;
        self.builder.ins().brif(cond_val, body_block, &[], exit_block, &[]);

        // Body
        self.builder.switch_to_block(body_block);
        self.builder.seal_block(body_block);
        self.block_terminated = false;
        self.loop_stack.push(LoopContext { continue_block: header_block, exit_block });
        self.emit_block(body)?;
        self.loop_stack.pop();
        if !self.block_terminated {
            self.builder.ins().jump(header_block, &[]);
        }

        // Seal header after body (back-edge)
        self.builder.seal_block(header_block);

        self.builder.switch_to_block(exit_block);
        self.builder.seal_block(exit_block);
        self.block_terminated = false;
        Ok(())
    }

    fn emit_for(
        &mut self,
        var_name: &str,
        start: &Spanned<Expr>,
        end: &Spanned<Expr>,
        inclusive: bool,
        body: &Block,
    ) -> Result<(), PdcError> {
        let start_val = self.emit_expr(start)?;
        let end_val = self.emit_expr(end)?;

        let var = self.new_variable(var_name, &PdcType::I32);
        self.builder.def_var(var, start_val);

        let header_block = self.builder.create_block();
        let body_block = self.builder.create_block();
        let latch_block = self.builder.create_block();
        let exit_block = self.builder.create_block();

        self.builder.ins().jump(header_block, &[]);

        // Header: check i < end (exclusive) or i <= end (inclusive)
        self.builder.switch_to_block(header_block);
        let i_val = self.builder.use_var(var);
        let cc = if inclusive { IntCC::SignedLessThanOrEqual } else { IntCC::SignedLessThan };
        let cond = self.builder.ins().icmp(cc, i_val, end_val);
        self.builder.ins().brif(cond, body_block, &[], exit_block, &[]);

        // Body — continue jumps to latch (increment then re-check)
        self.builder.switch_to_block(body_block);
        self.builder.seal_block(body_block);
        self.block_terminated = false;
        self.loop_stack.push(LoopContext { continue_block: latch_block, exit_block });
        self.emit_block(body)?;
        self.loop_stack.pop();
        if !self.block_terminated {
            self.builder.ins().jump(latch_block, &[]);
        }

        // Latch: increment and jump back to header
        self.builder.switch_to_block(latch_block);
        self.builder.seal_block(latch_block);
        let i_val = self.builder.use_var(var);
        let one = self.builder.ins().iconst(I32, 1);
        let next = self.builder.ins().iadd(i_val, one);
        self.builder.def_var(var, next);
        self.builder.ins().jump(header_block, &[]);

        self.builder.seal_block(header_block);
        self.builder.switch_to_block(exit_block);
        self.builder.seal_block(exit_block);
        self.block_terminated = false;
        Ok(())
    }

    fn emit_loop(&mut self, body: &Block) -> Result<(), PdcError> {
        let body_block = self.builder.create_block();
        let exit_block = self.builder.create_block();

        self.builder.ins().jump(body_block, &[]);

        self.builder.switch_to_block(body_block);
        self.block_terminated = false;
        self.loop_stack.push(LoopContext { continue_block: body_block, exit_block });
        self.emit_block(body)?;
        self.loop_stack.pop();
        if !self.block_terminated {
            self.builder.ins().jump(body_block, &[]);
        }

        self.builder.seal_block(body_block);
        self.builder.switch_to_block(exit_block);
        self.builder.seal_block(exit_block);
        self.block_terminated = false;
        Ok(())
    }

    fn emit_expr(
        &mut self,
        expr: &Spanned<Expr>,
    ) -> Result<cranelift_codegen::ir::Value, PdcError> {
        match &expr.node {
            Expr::Literal(lit) => self.emit_literal(lit, expr.id),
            Expr::Variable(name) => {
                // Mutable builtin: load from builtins array
                if self.is_mutable_builtin(name) {
                    return self.load_mutable_builtin(name);
                }
                // State block variable: load from state block
                if self.is_state_var(name) {
                    return self.load_state_var(name);
                }

                // Function reference: emit the JIT'd function's address as a pointer
                if matches!(self.node_type(expr.id), PdcType::FnRef { .. }) {
                    let resolved = if let Some(qualified) = self.fn_aliases.get(name) {
                        qualified.clone()
                    } else {
                        name.clone()
                    };
                    let base = mangle_name(&resolved);
                    let mangled = format!("pdc_userfn_{base}");
                    if let Some(&func_id) = self.user_fn_ids.get(&mangled) {
                        let func_ref = self.module.declare_func_in_func(func_id, self.builder.func);
                        return Ok(self.builder.ins().func_addr(self.pointer_type, func_ref));
                    }
                }

                let (var, _) = self.variables.get(name).cloned().ok_or_else(|| {
                    PdcError::Codegen {
                        message: format!("undefined variable '{name}'"),
                    }
                })?;
                Ok(self.builder.use_var(var))
            }
            Expr::BinaryOp { op, left, right } => {
                let lt = self.node_type(left.id).clone();
                let rt = self.node_type(right.id).clone();
                let result_ty = self.node_type(expr.id).clone();

                // Check for user-defined operator overload
                let op_name = binop_to_op_name(*op);
                if let Some(overloads) = self.op_overloads.get(op_name) {
                    let arg_types = [lt.clone(), rt.clone()];
                    if let Some((idx, _sig)) = overloads.sigs.iter().enumerate().find(|(_, sig)| {
                        sig.params.len() == 2 && sig.params[0] == arg_types[0] && sig.params[1] == arg_types[1]
                    }) {
                        let lval = self.emit_expr(left)?;
                        let rval = self.emit_expr(right)?;
                        return self.emit_operator_call(op_name, idx, &[lval, rval], &result_ty);
                    }
                }

                // Array broadcasting
                if let PdcType::Array(_) = &result_ty {
                    return self.emit_array_broadcast(*op, left, right, &lt, &rt, &result_ty);
                }

                let lval = self.emit_expr(left)?;
                let rval = self.emit_expr(right)?;
                self.emit_binary_op(*op, lval, rval, &lt, &rt, &result_ty)
            }
            Expr::UnaryOp { op, operand } => {
                let operand_ty = self.node_type(operand.id).clone();

                // Check for user-defined unary operator overload
                let uop_name = unaryop_to_op_name(*op);
                if let Some(overloads) = self.op_overloads.get(uop_name) {
                    if let Some((idx, _sig)) = overloads.sigs.iter().enumerate().find(|(_, sig)| {
                        sig.params.len() == 1 && sig.params[0] == operand_ty
                    }) {
                        let result_ty = self.node_type(expr.id).clone();
                        let val = self.emit_expr(operand)?;
                        return self.emit_operator_call(uop_name, idx, &[val], &result_ty);
                    }
                }

                let val = self.emit_expr(operand)?;
                let ty = &operand_ty;
                match op {
                    UnaryOp::Neg => {
                        if ty.is_float() {
                            Ok(self.builder.ins().fneg(val))
                        } else {
                            let zero = self.builder.ins().iconst(I32, 0);
                            Ok(self.builder.ins().isub(zero, val))
                        }
                    }
                    UnaryOp::Not => {
                        let one = self.builder.ins().iconst(I8, 1);
                        Ok(self.builder.ins().bxor(val, one))
                    }
                    UnaryOp::BitNot => {
                        let cl = pdc_type_to_cl(ty, self.pointer_type);
                        let neg_one = self.builder.ins().iconst(cl, -1);
                        Ok(self.builder.ins().bxor(val, neg_one))
                    }
                }
            }
            Expr::Call { name, args, arg_names } => {
                let has_named = arg_names.iter().any(|n| n.is_some());
                // Struct construction with named args
                if has_named {
                    if let Some(info) = self.structs.get(name).cloned() {
                        return self.emit_struct_construct_from_call(name, args, arg_names, &info);
                    }
                }
                self.emit_call(name, args, expr.id)
            }
            Expr::MethodCall {
                object,
                method,
                args,
            } => {
                let obj_ty = self.node_type(object.id).clone();
                // Module namespaced call: math.sin(x) → math::sin(x)
                if let PdcType::Module(ref mod_name) = obj_ty {
                    // Buffer factory: Buffer.I32() → pdc_create_buffer(ctx, type_code)
                    if mod_name == "Buffer" {
                        let type_code: i64 = match method.as_str() {
                            "F32" => 0, "I32" => 1, "U32" => 2,
                            "Vec2F32" => 3, "Vec3F32" => 4, "Vec4F32" => 5,
                            _ => 0,
                        };
                        let code_val = self.builder.ins().iconst(I32, type_code);
                        return self.emit_runtime_call_raw("pdc_create_buffer",
                            &[self.ctx_ptr, code_val], Some(I32));
                    }
                    // Kernel factory: Kernel.Sim("name", "path") → pdc_load_kernel(ctx, name_ptr, name_len, path_ptr, path_len, kind)
                    if mod_name == "Kernel" {
                        let kind: i64 = match method.as_str() {
                            "Pixel" => 0, "Sim" => 1, _ => 0,
                        };
                        let (name_ptr, name_len) = self.emit_str_arg(&args[0]);
                        let (path_ptr, path_len) = self.emit_str_arg(&args[1]);
                        let kind_val = self.builder.ins().iconst(I32, kind);
                        return self.emit_runtime_call_raw("pdc_load_kernel",
                            &[self.ctx_ptr, name_ptr, name_len, path_ptr, path_len, kind_val], Some(I32));
                    }
                    let qualified = format!("{mod_name}::{method}");
                    return self.emit_call(&qualified, args, expr.id);
                }
                // Tuple len(): compile-time constant
                if let PdcType::Tuple(ref elems) = obj_ty {
                    if method == "len" {
                        return Ok(self.builder.ins().iconst(I32, elems.len() as i64));
                    }
                }
                // Slice methods
                if let PdcType::Slice(ref elem_ty) = obj_ty {
                    let slice_ptr = self.emit_expr(object)?;
                    return match method.as_str() {
                        "len" => {
                            Ok(self.builder.ins().load(I32, MemFlags::trusted(), slice_ptr, 8))
                        }
                        "get" => {
                            let idx = self.emit_expr(&args[0])?;
                            let arr_handle = self.builder.ins().load(I32, MemFlags::trusted(), slice_ptr, 0);
                            let start = self.builder.ins().load(I32, MemFlags::trusted(), slice_ptr, 4);
                            let elem_cl = pdc_type_to_cl(elem_ty, self.pointer_type);
                            let elem_size = elem_cl.bytes() as u32;
                            let get_name = format!("pdc_slice_get_{elem_size}");
                            let int_type = match elem_size { 1 => I8, 2 => I16, 4 => I32, _ => I64 };
                            let raw = self.emit_runtime_call_raw(&get_name, &[self.ctx_ptr, arr_handle, start, idx], Some(int_type))?;
                            Ok(self.int_to_float_if_needed(raw, elem_cl))
                        }
                        _ => Err(PdcError::Codegen { message: format!("unknown slice method '{method}'") }),
                    };
                }
                // String methods
                if obj_ty == PdcType::Str {
                    let handle = self.emit_expr(object)?;
                    return match method.as_str() {
                        "len" => self.emit_runtime_call_raw("pdc_string_len", &[self.ctx_ptr, handle], Some(I32)),
                        "concat" => {
                            let arg = self.emit_expr(&args[0])?;
                            self.emit_runtime_call_raw("pdc_string_concat", &[self.ctx_ptr, handle, arg], Some(I32))
                        }
                        "slice" => {
                            let start = self.emit_expr(&args[0])?;
                            let end = self.emit_expr(&args[1])?;
                            self.emit_runtime_call_raw("pdc_string_slice", &[self.ctx_ptr, handle, start, end], Some(I32))
                        }
                        "char_at" => {
                            let idx = self.emit_expr(&args[0])?;
                            self.emit_runtime_call_raw("pdc_string_char_at", &[self.ctx_ptr, handle, idx], Some(I32))
                        }
                        _ => Err(PdcError::Codegen { message: format!("unknown string method '{method}'") }),
                    };
                }
                // Scene methods: scene.run() → pdc_run_scene(ctx, handle)
                if obj_ty == PdcType::SceneHandle {
                    let handle = self.emit_expr(object)?;
                    return match method.as_str() {
                        "run" => {
                            self.emit_runtime_call_raw("pdc_run_scene",
                                &[self.ctx_ptr, handle], None)?;
                            Ok(self.builder.ins().iconst(I32, 0))
                        }
                        _ => Err(PdcError::Codegen { message: format!("Scene has no method '{method}'") }),
                    };
                }
                // Array methods with type-aware bitcasting
                if let PdcType::Array(ref elem_ty) = obj_ty {
                    return self.emit_array_method(object, method, args, elem_ty);
                }
                // Check if this is enum variant construction: EnumName.Variant(args)
                if let PdcType::Enum(ref ename) = obj_ty {
                    return self.emit_enum_construct(ename, method, args);
                }
                let mut all_args = vec![object.as_ref().clone()];
                all_args.extend(args.iter().cloned());
                self.emit_call(method, &all_args, expr.id)
            }
            Expr::StructConstruct { name, fields } => {
                let info = self.structs.get(name).ok_or_else(|| PdcError::Codegen {
                    message: format!("undefined struct '{name}'"),
                })?.clone();

                // Allocate stack slot: 8 bytes per field
                let size = (info.fields.len() * 8) as u32;
                let slot = self.builder.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
                    cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                    size,
                    0,
                ));

                // Store each field
                for (fname, fexpr) in fields {
                    let val = self.emit_expr(fexpr)?;
                    let fexpr_ty = self.node_type(fexpr.id).clone();

                    // Find field offset
                    let field_idx = info.fields.iter().position(|(n, _)| n == fname)
                        .ok_or_else(|| PdcError::Codegen {
                            message: format!("struct '{name}' has no field '{fname}'"),
                        })?;
                    let offset = (field_idx * 8) as i32;
                    let field_ty = &info.fields[field_idx].1;
                    let converted = self.convert_value(val, &fexpr_ty, field_ty);

                    // Store as f64 (all fields use 8 bytes)
                    let store_val = self.widen_to_f64(converted, field_ty);
                    self.builder.ins().stack_store(store_val, slot, offset);
                }

                // Return pointer to slot
                Ok(self.builder.ins().stack_addr(self.pointer_type, slot, 0))
            }
            Expr::Index { object, index } => {
                let obj_ty = self.node_type(object.id).clone();
                if let PdcType::Array(ref elem_ty) = obj_ty {
                    let arr_handle = self.emit_expr(object)?;
                    let idx = self.emit_expr(index)?;
                    let elem_cl = pdc_type_to_cl(elem_ty, self.pointer_type);
                    let elem_size = elem_cl.bytes() as u32;
                    // Inline load via data pointer
                    let data_ptr = self.emit_runtime_call_raw(
                        "pdc_array_data_ptr", &[self.ctx_ptr, arr_handle], Some(self.pointer_type))?;
                    Ok(self.emit_inline_array_load(data_ptr, idx, elem_size, elem_cl))
                } else if let PdcType::Slice(ref elem_ty) = obj_ty {
                    let slice_ptr = self.emit_expr(object)?;
                    let idx = self.emit_expr(index)?;
                    let arr_handle = self.builder.ins().load(I32, MemFlags::trusted(), slice_ptr, 0);
                    let start = self.builder.ins().load(I32, MemFlags::trusted(), slice_ptr, 4);
                    let elem_cl = pdc_type_to_cl(elem_ty, self.pointer_type);
                    let elem_size = elem_cl.bytes() as u32;
                    let get_name = format!("pdc_slice_get_{elem_size}");
                    let int_type = match elem_size { 1 => I8, 2 => I16, 4 => I32, _ => I64 };
                    let raw = self.emit_runtime_call_raw(&get_name, &[self.ctx_ptr, arr_handle, start, idx], Some(int_type))?;
                    Ok(self.int_to_float_if_needed(raw, elem_cl))
                } else {
                    Err(PdcError::Codegen { message: "cannot index non-array/slice type".into() })
                }
            }
            Expr::TupleConstruct { elements } => {
                let size = (elements.len() * 8) as u32;
                let slot = self.builder.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
                    cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                    size,
                    0,
                ));
                for (i, elem) in elements.iter().enumerate() {
                    let val = self.emit_expr(elem)?;
                    let elem_ty = self.node_type(elem.id).clone();
                    let store_val = self.widen_to_f64(val, &elem_ty);
                    self.builder.ins().stack_store(store_val, slot, (i * 8) as i32);
                }
                Ok(self.builder.ins().stack_addr(self.pointer_type, slot, 0))
            }
            Expr::TupleIndex { object, index } => {
                let obj_val = self.emit_expr(object)?;
                let result_ty = self.node_type(expr.id).clone();
                let raw = self.builder.ins().load(F64, MemFlags::trusted(), obj_val, (*index * 8) as i32);
                Ok(self.narrow_from_f64(raw, &result_ty))
            }
            Expr::Ternary { condition, then_expr, else_expr } => {
                let cond_val = self.emit_expr(condition)?;
                let result_ty = self.node_type(expr.id).clone();
                let cl = pdc_type_to_cl(&result_ty, self.pointer_type);

                let result_var = self.builder.declare_var(cl);
                let default = self.default_value(&result_ty);
                self.builder.def_var(result_var, default);

                let then_block = self.builder.create_block();
                let else_block = self.builder.create_block();
                let merge_block = self.builder.create_block();

                self.builder.ins().brif(cond_val, then_block, &[], else_block, &[]);

                self.builder.switch_to_block(then_block);
                self.builder.seal_block(then_block);
                let then_val = self.emit_expr(then_expr)?;
                let then_ty = self.node_type(then_expr.id).clone();
                let then_converted = self.convert_value(then_val, &then_ty, &result_ty);
                self.builder.def_var(result_var, then_converted);
                self.builder.ins().jump(merge_block, &[]);

                self.builder.switch_to_block(else_block);
                self.builder.seal_block(else_block);
                let else_val = self.emit_expr(else_expr)?;
                let else_ty = self.node_type(else_expr.id).clone();
                let else_converted = self.convert_value(else_val, &else_ty, &result_ty);
                self.builder.def_var(result_var, else_converted);
                self.builder.ins().jump(merge_block, &[]);

                self.builder.switch_to_block(merge_block);
                self.builder.seal_block(merge_block);
                Ok(self.builder.use_var(result_var))
            }
            Expr::FieldAccess { object, field } => {
                let obj_ty = self.node_type(object.id).clone();

                // Module namespaced field: math.PI → math::PI (emit as variable read)
                if let PdcType::Module(ref mod_name) = obj_ty {
                    let qualified = format!("{mod_name}::{field}");
                    // Try qualified name first, fall back to unqualified (for imported consts)
                    let var_name = if self.variables.contains_key(&qualified) {
                        qualified
                    } else {
                        field.clone()
                    };
                    let (var, _) = self.variables.get(&var_name).cloned().ok_or_else(|| {
                        PdcError::Codegen {
                            message: format!("module '{mod_name}' has no member '{field}'"),
                        }
                    })?;
                    return Ok(self.builder.use_var(var));
                }

                // Enum variant access: EnumName.Variant → integer constant
                if let PdcType::Enum(ref ename) = obj_ty {
                    let info = self.enums.get(ename).ok_or_else(|| PdcError::Codegen {
                        message: format!("undefined enum '{ename}'"),
                    })?.clone();
                    let idx = info.variants.iter().position(|v| v.name == *field)
                        .ok_or_else(|| PdcError::Codegen {
                            message: format!("enum '{ename}' has no variant '{field}'"),
                        })?;
                    let variant = &info.variants[idx];
                    if variant.field_types.is_empty() {
                        // Simple enum variant: just the tag value
                        return Ok(self.builder.ins().iconst(I32, idx as i64));
                    } else {
                        // Data variant without args — shouldn't happen (use MethodCall)
                        return Ok(self.builder.ins().iconst(I32, idx as i64));
                    }
                }

                // Scene read-only virtual properties
                if obj_ty == PdcType::SceneHandle {
                    let handle = self.emit_expr(object)?;
                    return match field.as_str() {
                        "tiles_x" => {
                            self.emit_runtime_call_raw("pdc_scene_tiles_x",
                                &[self.ctx_ptr, handle], Some(F64))
                        }
                        "num_paths" => {
                            self.emit_runtime_call_raw("pdc_scene_num_paths",
                                &[self.ctx_ptr, handle], Some(F64))
                        }
                        _ => {
                            // Buffer property: scene.<name> → pdc_scene_buffer(ctx, handle, name_ptr, name_len)
                            let (name_ptr, name_len) = self.emit_inline_str(field.as_bytes());
                            self.emit_runtime_call_raw("pdc_scene_buffer",
                                &[self.ctx_ptr, handle, name_ptr, name_len], Some(I32))
                        }
                    };
                }

                let obj_val = self.emit_expr(object)?;

                if let PdcType::Struct(ref sname) = obj_ty {
                    let info = self.structs.get(sname).ok_or_else(|| PdcError::Codegen {
                        message: format!("undefined struct '{sname}'"),
                    })?.clone();

                    let field_idx = info.fields.iter().position(|(n, _)| n == field)
                        .ok_or_else(|| PdcError::Codegen {
                            message: format!("struct '{sname}' has no field '{field}'"),
                        })?;
                    let offset = (field_idx * 8) as i32;
                    let field_ty = &info.fields[field_idx].1;

                    // Load f64 from struct memory, then narrow if needed
                    let raw = self.builder.ins().load(F64, MemFlags::trusted(), obj_val, offset);
                    Ok(self.narrow_from_f64(raw, field_ty))
                } else {
                    Err(PdcError::Codegen {
                        message: format!("cannot access field '{field}' on non-struct type"),
                    })
                }
            }
            Expr::DotShorthand(variant) => {
                // The type checker resolved this to an Enum type — look up the variant index.
                let ty = self.node_type(expr.id).clone();
                if let PdcType::Enum(ref ename) = ty {
                    let info = self.enums.get(ename).ok_or_else(|| PdcError::Codegen {
                        message: format!("undefined enum '{ename}'"),
                    })?.clone();
                    let idx = info.variants.iter().position(|v| v.name == *variant)
                        .ok_or_else(|| PdcError::Codegen {
                            message: format!("enum '{ename}' has no variant '{variant}'"),
                        })?;
                    Ok(self.builder.ins().iconst(I32, idx as i64))
                } else {
                    Err(PdcError::Codegen {
                        message: format!("dot-shorthand '.{variant}' was not resolved to an enum type"),
                    })
                }
            }
        }
    }

    fn emit_literal(
        &mut self,
        lit: &Literal,
        id: u32,
    ) -> Result<cranelift_codegen::ir::Value, PdcError> {
        let ty = self.node_type(id);
        let cl = pdc_type_to_cl(ty, self.pointer_type);
        match lit {
            Literal::Int(v) => match cl {
                F32 => Ok(self.builder.ins().f32const(*v as f32)),
                F64 => Ok(self.builder.ins().f64const(*v as f64)),
                _ => Ok(self.builder.ins().iconst(cl, *v)),
            },
            Literal::Float(v) => match cl {
                F32 => Ok(self.builder.ins().f32const(*v as f32)),
                _ => Ok(self.builder.ins().f64const(*v)),
            },
            Literal::Bool(v) => Ok(self.builder.ins().iconst(I8, *v as i64)),
            Literal::String(s) => {
                // Store string data on the stack and call pdc_string_new
                let bytes = s.as_bytes();
                let len = bytes.len();
                if len == 0 {
                    let null_ptr = self.builder.ins().iconst(self.pointer_type, 0);
                    let len_val = self.builder.ins().iconst(I32, 0);
                    self.emit_runtime_call_raw("pdc_string_new", &[self.ctx_ptr, null_ptr, len_val], Some(I32))
                } else {
                    let slot = self.builder.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
                        cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                        len as u32,
                        0,
                    ));
                    for (i, &byte) in bytes.iter().enumerate() {
                        let val = self.builder.ins().iconst(I8, byte as i64);
                        self.builder.ins().stack_store(val, slot, i as i32);
                    }
                    let ptr = self.builder.ins().stack_addr(self.pointer_type, slot, 0);
                    let len_val = self.builder.ins().iconst(I32, len as i64);
                    self.emit_runtime_call_raw("pdc_string_new", &[self.ctx_ptr, ptr, len_val], Some(I32))
                }
            }
        }
    }

    fn emit_call(
        &mut self,
        name: &str,
        args: &[Spanned<Expr>],
        _call_id: u32,
    ) -> Result<cranelift_codegen::ir::Value, PdcError> {
        // Test assertion intrinsics
        if name == "assert_eq" && args.len() == 2 {
            let a = self.emit_expr(&args[0])?;
            let b = self.emit_expr(&args[1])?;
            let ty = self.node_type(args[0].id).clone();
            let runtime_name = match abi_class(&ty) {
                'd' => "pdc_assert_eq_f64",
                's' => "pdc_assert_eq_f32",
                _ => "pdc_assert_eq_i64",
            };
            return self.emit_runtime_call_raw(runtime_name, &[self.ctx_ptr, a, b], None);
        }
        if name == "assert_near" && args.len() == 3 {
            let a = self.emit_expr(&args[0])?;
            let b = self.emit_expr(&args[1])?;
            let eps = self.emit_expr(&args[2])?;
            return self.emit_runtime_call_raw("pdc_assert_near", &[self.ctx_ptr, a, b, eps], None);
        }
        if name == "assert_true" && args.len() == 1 {
            let cond = self.emit_expr(&args[0])?;
            // Promote bool (i8) to i64 for the runtime call
            let val = self.builder.ins().sextend(I64, cond);
            return self.emit_runtime_call_raw("pdc_assert_true", &[self.ctx_ptr, val], None);
        }

        // Array functions called as push(arr, val), get(arr, idx), etc.
        if matches!(name, "push" | "get" | "set" | "len") && !args.is_empty() {
            let first_ty = self.node_type(args[0].id).clone();
            if let PdcType::Array(ref elem_ty) = first_ty {
                let elem_ty = *elem_ty.clone();
                // Desugar: push(arr, val) → arr.push(val)
                return self.emit_array_method(&args[0], name, &args[1..], &elem_ty);
            }
        }

        // Type cast
        let cast_ty = match name {
            "f32" => Some(PdcType::F32),
            "f64" => Some(PdcType::F64),
            "i32" => Some(PdcType::I32),
            "u32" => Some(PdcType::U32),
            _ => None,
        };
        if let Some(target) = cast_ty {
            let val = self.emit_expr(&args[0])?;
            let from_ty = self.node_type(args[0].id).clone();
            return Ok(self.convert_value(val, &from_ty, &target));
        }

        // User-defined function (with overload resolution)
        // Resolve through alias map first: "Circle" → "geometry::Circle" if imported
        // Alias takes priority because it maps to the qualified name used for JIT symbols
        let resolved_name = if let Some(qualified) = self.fn_aliases.get(name) {
            qualified.clone()
        } else if self.user_fns.contains_key(name) {
            name.to_string()
        } else {
            String::new() // will fall through to runtime function
        };

        if let Some(overloads) = self.user_fns.get(&resolved_name) {
            let overloads = overloads.clone();
            // Collect argument types for overload resolution
            let arg_types: Vec<PdcType> = args.iter().map(|a| self.node_type(a.id).clone()).collect();

            // Find matching overload (supports default params: args can be fewer than params)
            let (overload_idx, sig) = overloads.sigs.iter().enumerate()
                .find(|(_, sig)| {
                    let n = arg_types.len();
                    n >= sig.required && n <= sig.params.len() &&
                    sig.params[..n].iter().zip(&arg_types).all(|(p, a)| {
                        p == a || (p.is_numeric() && a.is_numeric()) ||
                        matches!((p, a), (PdcType::Array(_), PdcType::Array(_)))
                    })
                })
                .map(|(i, s)| (i, s.clone()))
                .ok_or_else(|| PdcError::Codegen {
                    message: format!("no matching overload for '{name}'"),
                })?;

            let base = mangle_name(&resolved_name);
            let mangled = if overload_idx == 0 {
                format!("pdc_userfn_{base}")
            } else {
                format!("pdc_userfn_{base}_{overload_idx}")
            };
            let func_id = self.user_fn_ids[&mangled];

            let mut arg_vals = vec![self.ctx_ptr];
            // Emit provided arguments
            for (i, arg) in args.iter().enumerate() {
                let val = self.emit_expr(arg)?;
                let from_ty = self.node_type(arg.id).clone();
                let to_ty = &sig.params[i];
                let converted = self.convert_value(val, &from_ty, to_ty);
                arg_vals.push(converted);
            }
            // Emit default expressions for missing arguments
            for i in args.len()..sig.params.len() {
                let default_idx = i - sig.required;
                let default_expr = &sig.defaults[default_idx];
                let val = self.emit_expr(default_expr)?;
                let from_ty = self.node_type(default_expr.id).clone();
                let to_ty = &sig.params[i];
                let converted = self.convert_value(val, &from_ty, &to_ty);
                arg_vals.push(converted);
            }

            let func_ref = self.module.declare_func_in_func(func_id, self.builder.func);
            let call = self.builder.ins().call(func_ref, &arg_vals);
            let results = self.builder.inst_results(call);
            if results.is_empty() {
                return Ok(self.builder.ins().iconst(I32, 0));
            }
            let val = results[0];
            // Copy struct/tuple return values into the caller's frame so the
            // pointer remains valid after the callee's stack is reclaimed.
            let ret_ty = self.node_type(_call_id).clone();
            if let Some(size) = self.compound_type_size(&ret_ty) {
                return Ok(self.copy_struct_to_caller(val, size));
            }
            return Ok(val);
        }

        // Array constructor: Array<type>() — pass element_size to pdc_array_new
        if name.starts_with("Array<") {
            let result_ty = self.node_type(_call_id).clone();
            let elem_size = if let PdcType::Array(ref et) = result_ty {
                pdc_type_to_cl(et, self.pointer_type).bytes() as u32
            } else {
                8
            };
            let size_val = self.builder.ins().iconst(I32, elem_size as i64);
            return self.emit_runtime_call_raw("pdc_array_new", &[self.ctx_ptr, size_val], Some(I32));
        }

        // Emit native Cranelift instructions for 2-arg math functions.
        if args.len() == 2 {
            match name {
                "min" => {
                    let a = self.emit_expr(&args[0])?;
                    let b = self.emit_expr(&args[1])?;
                    let a_ty = self.node_type(args[0].id).clone();
                    let b_ty = self.node_type(args[1].id).clone();
                    if a_ty.is_float() || b_ty.is_float() {
                        let a = self.convert_value(a, &a_ty, &PdcType::F64);
                        let b = self.convert_value(b, &b_ty, &PdcType::F64);
                        return Ok(self.builder.ins().fmin(a, b));
                    }
                }
                "max" => {
                    let a = self.emit_expr(&args[0])?;
                    let b = self.emit_expr(&args[1])?;
                    let a_ty = self.node_type(args[0].id).clone();
                    let b_ty = self.node_type(args[1].id).clone();
                    if a_ty.is_float() || b_ty.is_float() {
                        let a = self.convert_value(a, &a_ty, &PdcType::F64);
                        let b = self.convert_value(b, &b_ty, &PdcType::F64);
                        return Ok(self.builder.ins().fmax(a, b));
                    }
                }
                _ => {}
            }
        }

        // Texture("name", "path") and Scene("name", "path") take inline string args
        if (name == "Texture" || name == "Scene") && args.len() == 2 {
            let runtime_name = if name == "Texture" { "pdc_load_texture" } else { "pdc_load_scene" };
            let (name_ptr, name_len) = self.emit_str_arg(&args[0]);
            let (path_ptr, path_len) = self.emit_str_arg(&args[1]);
            return self.emit_runtime_call_raw(runtime_name,
                &[self.ctx_ptr, name_ptr, name_len, path_ptr, path_len], Some(I32));
        }

        // Runtime function call
        let runtime_name = {
            match name {
                "Path" => "pdc_path".to_string(),
                "display_buffer" => "pdc_display_buffer".to_string(),
                "swap" => "pdc_swap_buffers".to_string(),
                "run" => "pdc_run_kernel".to_string(),
                "render" => "pdc_render_kernel".to_string(),
                "push" => "pdc_array_push".to_string(),
                "len" => "pdc_array_len".to_string(),
                "get" => "pdc_array_get".to_string(),
                "set" => "pdc_array_set".to_string(),
                // Styled overloads
                "fill" if args.len() == 3 => "pdc_fill_styled".to_string(),
                "stroke" if args.len() == 5 => "pdc_stroke_styled".to_string(),
                other => format!("pdc_{}", other),
            }
        };

        let takes_ctx = matches!(
            name,
            "Path"
            | "move_to" | "line_to" | "quad_to" | "cubic_to" | "close" | "fill" | "stroke"
            | "fill_styled" | "stroke_styled"
            | "push" | "len" | "get" | "set"
            | "display_buffer" | "swap" | "run" | "render"
            | "display"
            | "set_keypress" | "set_keydown" | "set_keyup"
            | "clear_keypress" | "clear_keydown" | "clear_keyup"
            | "set_mousedown" | "set_mouseup" | "set_click"
            | "clear_mousedown" | "clear_mouseup" | "clear_click"
        );

        let mut arg_vals = Vec::new();
        if takes_ctx {
            arg_vals.push(self.ctx_ptr);
        }
        for arg in args {
            let val = self.emit_expr(arg)?;
            let arg_ty = self.node_type(arg.id).clone();
            let converted = self.convert_for_call(val, &arg_ty, name);
            arg_vals.push(converted);
        }

        let mut sig = self.module.make_signature();
        sig.call_conv = self.call_conv;
        for val in &arg_vals {
            let ty = self.builder.func.dfg.value_type(*val);
            sig.params.push(AbiParam::new(ty));
        }

        let ret_type = self.call_return_type(name);
        if let Some(rt) = ret_type {
            sig.returns.push(AbiParam::new(rt));
        }

        let callee = self
            .module
            .declare_function(&runtime_name, Linkage::Import, &sig)
            .map_err(|e| PdcError::Codegen {
                message: format!("declare runtime function '{runtime_name}': {e}"),
            })?;

        let func_ref = self.module.declare_func_in_func(callee, self.builder.func);
        let call = self.builder.ins().call(func_ref, &arg_vals);

        if ret_type.is_some() {
            let raw = self.builder.inst_results(call)[0];
            // Runtime functions return i32 for bools; narrow to i8 for PDC's Bool type.
            if self.node_type(_call_id) == &PdcType::Bool {
                Ok(self.builder.ins().ireduce(I8, raw))
            } else {
                Ok(raw)
            }
        } else {
            Ok(self.builder.ins().iconst(I32, 0))
        }
    }

    fn call_return_type(&self, name: &str) -> Option<cranelift_codegen::ir::Type> {
        match name {
            "Path" | "len" | "Texture" | "Scene"
            | "render" => Some(I32),
            "get" => Some(F64),
            "move_to" | "line_to" | "quad_to" | "cubic_to" | "close" | "fill" | "stroke"
            | "fill_styled" | "stroke_styled"
            | "push" | "set"
            | "display_buffer" | "swap" | "run"
            | "display" => None,
            _ => Some(F64),
        }
    }

    fn convert_for_call(
        &mut self,
        val: cranelift_codegen::ir::Value,
        from: &PdcType,
        func_name: &str,
    ) -> cranelift_codegen::ir::Value {
        // stroke's width param needs f32 (but NOT the color param)
        let is_stroke_float = func_name == "stroke" && from.is_float();
        if is_stroke_float && *from != PdcType::F32 {
            return self.builder.ins().fdemote(F32, val);
        }

        // Runtime functions expect i32 for bool arguments.
        if *from == PdcType::Bool {
            return self.builder.ins().uextend(I32, val);
        }

        val
    }

    /// Widen any scalar value to f64 for storage in struct memory (8 bytes per field).
    fn widen_to_f64(&mut self, val: cranelift_codegen::ir::Value, ty: &PdcType) -> cranelift_codegen::ir::Value {
        match ty {
            PdcType::F64 => val,
            PdcType::F32 => self.builder.ins().fpromote(F64, val),
            PdcType::I32 | PdcType::U32 | PdcType::PathHandle | PdcType::BufferHandle | PdcType::KernelHandle | PdcType::TextureHandle | PdcType::SceneHandle => {
                self.builder.ins().fcvt_from_sint(F64, val)
            }
            PdcType::Bool => {
                let i32_val = self.builder.ins().uextend(I32, val);
                self.builder.ins().fcvt_from_sint(F64, i32_val)
            }
            _ => val,
        }
    }

    /// Narrow f64 from struct memory to the actual field type.
    fn narrow_from_f64(&mut self, val: cranelift_codegen::ir::Value, ty: &PdcType) -> cranelift_codegen::ir::Value {
        match ty {
            PdcType::F64 => val,
            PdcType::F32 => self.builder.ins().fdemote(F32, val),
            PdcType::I32 | PdcType::U32 | PdcType::PathHandle | PdcType::BufferHandle | PdcType::KernelHandle | PdcType::TextureHandle | PdcType::SceneHandle => {
                self.builder.ins().fcvt_to_sint_sat(I32, val)
            }
            PdcType::Bool => {
                let i32_val = self.builder.ins().fcvt_to_sint_sat(I32, val);
                self.builder.ins().ireduce(I8, i32_val)
            }
            _ => val,
        }
    }

    fn convert_value(
        &mut self,
        val: cranelift_codegen::ir::Value,
        from: &PdcType,
        to: &PdcType,
    ) -> cranelift_codegen::ir::Value {
        if from == to {
            return val;
        }

        let from_cl = pdc_type_to_cl(from, self.pointer_type);
        let to_cl = pdc_type_to_cl(to, self.pointer_type);

        // Bool → float: zero-extend to I32, then int-to-float
        if *from == PdcType::Bool && to.is_float() {
            let i32_val = self.builder.ins().uextend(I32, val);
            let f64_val = self.builder.ins().fcvt_from_sint(F64, i32_val);
            return if to_cl == F32 {
                self.builder.ins().fdemote(F32, f64_val)
            } else {
                f64_val
            };
        }

        // Float → bool: compare != 0.0
        if from.is_float() && *to == PdcType::Bool {
            let f64_val = if from_cl == F32 {
                self.builder.ins().fpromote(F64, val)
            } else {
                val
            };
            let zero = self.builder.ins().f64const(0.0);
            return self.builder.ins().fcmp(cranelift_codegen::ir::condcodes::FloatCC::NotEqual, f64_val, zero);
        }

        // Bool → int: zero-extend
        if *from == PdcType::Bool && to.is_int() {
            return if to_cl.bytes() > 1 {
                self.builder.ins().uextend(to_cl, val)
            } else {
                val
            };
        }

        // Int → bool: compare != 0
        if from.is_int() && *to == PdcType::Bool {
            let zero = self.builder.ins().iconst(from_cl, 0);
            return self.builder.ins().icmp(cranelift_codegen::ir::condcodes::IntCC::NotEqual, val, zero);
        }

        // Float ↔ float
        if from.is_float() && to.is_float() {
            return if to_cl == F64 {
                self.builder.ins().fpromote(F64, val)
            } else {
                self.builder.ins().fdemote(F32, val)
            };
        }

        // Int → float
        if from.is_int() && to.is_float() {
            // First widen to I64 if needed, then convert to f64, then demote if f32
            let wide = if from_cl.bytes() < 8 {
                self.builder.ins().sextend(I64, val)
            } else {
                val
            };
            let f64_val = self.builder.ins().fcvt_from_sint(F64, wide);
            return if to_cl == F32 {
                self.builder.ins().fdemote(F32, f64_val)
            } else {
                f64_val
            };
        }

        // Float → int
        if from.is_float() && to.is_int() {
            let f64_val = if from_cl == F32 {
                self.builder.ins().fpromote(F64, val)
            } else {
                val
            };
            let i64_val = self.builder.ins().fcvt_to_sint_sat(I64, f64_val);
            return if to_cl.bytes() < 8 {
                self.builder.ins().ireduce(to_cl, i64_val)
            } else {
                i64_val
            };
        }

        // Int → int (widen or narrow)
        if from.is_int() && to.is_int() {
            return if to_cl.bytes() > from_cl.bytes() {
                self.builder.ins().sextend(to_cl, val)
            } else if to_cl.bytes() < from_cl.bytes() {
                self.builder.ins().ireduce(to_cl, val)
            } else {
                val // same size, different signedness
            };
        }

        val
    }

    /// Copy a struct/tuple from a callee's stack frame into the caller's frame.
    ///
    /// Function return values that are struct pointers point into the callee's
    /// stack frame, which is dead after the call returns. This method copies
    /// the data into a fresh stack slot owned by the current function so the
    /// pointer remains valid.
    fn copy_struct_to_caller(
        &mut self,
        src_ptr: cranelift_codegen::ir::Value,
        size_bytes: u32,
    ) -> cranelift_codegen::ir::Value {
        let slot = self.builder.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
            cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
            size_bytes,
            0,
        ));
        let num_words = size_bytes / 8;
        for i in 0..num_words {
            let offset = (i * 8) as i32;
            let val = self.builder.ins().load(F64, MemFlags::trusted(), src_ptr, offset);
            self.builder.ins().stack_store(val, slot, offset);
        }
        self.builder.ins().stack_addr(self.pointer_type, slot, 0)
    }

    /// Get the size in bytes for a struct or tuple type, or None if not compound.
    fn compound_type_size(&self, ty: &PdcType) -> Option<u32> {
        match ty {
            PdcType::Struct(name) => {
                self.structs.get(name).map(|info| (info.fields.len() * 8) as u32)
            }
            PdcType::Tuple(elems) => Some((elems.len() * 8) as u32),
            _ => None,
        }
    }

    fn emit_operator_call(
        &mut self,
        op_name: &str,
        overload_idx: usize,
        args: &[cranelift_codegen::ir::Value],
        result_type: &PdcType,
    ) -> Result<cranelift_codegen::ir::Value, PdcError> {
        // Look up the FuncId from the operator dispatch table
        let func_id = *self.op_fn_ids.get(&(op_name.to_string(), overload_idx))
            .ok_or_else(|| PdcError::Codegen {
                message: format!("operator function '{}' overload {} not found", op_name, overload_idx),
            })?;

        let mut call_args = vec![self.ctx_ptr];
        call_args.extend_from_slice(args);

        let func_ref = self.module.declare_func_in_func(func_id, self.builder.func);
        let call = self.builder.ins().call(func_ref, &call_args);
        let results = self.builder.inst_results(call);
        if results.is_empty() {
            Ok(self.builder.ins().iconst(I32, 0))
        } else {
            let val = results[0];
            if let Some(size) = self.compound_type_size(result_type) {
                Ok(self.copy_struct_to_caller(val, size))
            } else {
                Ok(val)
            }
        }
    }

    fn emit_binary_op(
        &mut self,
        op: BinOp,
        lval: cranelift_codegen::ir::Value,
        rval: cranelift_codegen::ir::Value,
        lt: &PdcType,
        rt: &PdcType,
        result_ty: &PdcType,
    ) -> Result<cranelift_codegen::ir::Value, PdcError> {
        // String operations
        if *lt == PdcType::Str && *rt == PdcType::Str {
            return match op {
                BinOp::Add => {
                    self.emit_runtime_call_raw("pdc_string_concat", &[self.ctx_ptr, lval, rval], Some(I32))
                }
                BinOp::Eq => {
                    self.emit_runtime_call_raw("pdc_string_eq", &[self.ctx_ptr, lval, rval], Some(I8))
                }
                BinOp::NotEq => {
                    let eq = self.emit_runtime_call_raw("pdc_string_eq", &[self.ctx_ptr, lval, rval], Some(I8))?;
                    let one = self.builder.ins().iconst(I8, 1);
                    Ok(self.builder.ins().bxor(eq, one))
                }
                _ => Err(PdcError::Codegen { message: "unsupported string operator".into() }),
            };
        }

        // Logical operators — bool operands, bool result
        if matches!(op, BinOp::And | BinOp::Or) {
            let lval = self.convert_value(lval, lt, result_ty);
            let rval = self.convert_value(rval, rt, result_ty);
            return Ok(match op {
                BinOp::And => self.builder.ins().band(lval, rval),
                BinOp::Or => self.builder.ins().bor(lval, rval),
                _ => unreachable!(),
            });
        }

        if result_ty == &PdcType::Bool {
            let cmp_type = if lt.is_float() || rt.is_float() {
                if lt == &PdcType::F32 && rt == &PdcType::F32 {
                    PdcType::F32
                } else {
                    PdcType::F64
                }
            } else {
                PdcType::I32
            };
            // Convert operands from their original types to the comparison type,
            // not from result_ty (Bool), to avoid double-conversion.
            let lval = self.convert_value(lval, lt, &cmp_type);
            let rval = self.convert_value(rval, rt, &cmp_type);

            return if cmp_type.is_float() {
                let cc = match op {
                    BinOp::Eq => FloatCC::Equal,
                    BinOp::NotEq => FloatCC::NotEqual,
                    BinOp::Lt => FloatCC::LessThan,
                    BinOp::LtEq => FloatCC::LessThanOrEqual,
                    BinOp::Gt => FloatCC::GreaterThan,
                    BinOp::GtEq => FloatCC::GreaterThanOrEqual,
                    _ => unreachable!(),
                };
                Ok(self.builder.ins().fcmp(cc, lval, rval))
            } else {
                let cc = match op {
                    BinOp::Eq => IntCC::Equal,
                    BinOp::NotEq => IntCC::NotEqual,
                    BinOp::Lt => IntCC::SignedLessThan,
                    BinOp::LtEq => IntCC::SignedLessThanOrEqual,
                    BinOp::Gt => IntCC::SignedGreaterThan,
                    BinOp::GtEq => IntCC::SignedGreaterThanOrEqual,
                    _ => unreachable!(),
                };
                Ok(self.builder.ins().icmp(cc, lval, rval))
            };
        }

        // For arithmetic and bitwise operators, convert operands to result type.
        let lval = self.convert_value(lval, lt, result_ty);
        let rval = self.convert_value(rval, rt, result_ty);

        // Bitwise operators — always integer
        match op {
            BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor | BinOp::Shl | BinOp::Shr => {
                return Ok(match op {
                    BinOp::BitAnd => self.builder.ins().band(lval, rval),
                    BinOp::BitOr => self.builder.ins().bor(lval, rval),
                    BinOp::BitXor => self.builder.ins().bxor(lval, rval),
                    BinOp::Shl => self.builder.ins().ishl(lval, rval),
                    BinOp::Shr => self.builder.ins().sshr(lval, rval),
                    _ => unreachable!(),
                });
            }
            _ => {}
        }

        if result_ty.is_float() {
            Ok(match op {
                BinOp::Add => self.builder.ins().fadd(lval, rval),
                BinOp::Sub => self.builder.ins().fsub(lval, rval),
                BinOp::Mul => self.builder.ins().fmul(lval, rval),
                BinOp::Div => self.builder.ins().fdiv(lval, rval),
                BinOp::Mod => {
                    let div = self.builder.ins().fdiv(lval, rval);
                    let floored = self.builder.ins().floor(div);
                    let prod = self.builder.ins().fmul(floored, rval);
                    self.builder.ins().fsub(lval, prod)
                }
                BinOp::Pow => {
                    // float ** float → call pow runtime function
                    self.emit_runtime_call_raw("pdc_pow", &[lval, rval], Some(F64))?
                }
                _ => unreachable!(),
            })
        } else {
            Ok(match op {
                BinOp::Add => self.builder.ins().iadd(lval, rval),
                BinOp::Sub => self.builder.ins().isub(lval, rval),
                BinOp::Mul => self.builder.ins().imul(lval, rval),
                BinOp::Div => self.builder.ins().sdiv(lval, rval),
                BinOp::Mod => self.builder.ins().srem(lval, rval),
                BinOp::Pow => {
                    // int ** int → convert to f64, call pow, convert back
                    let lf = self.builder.ins().fcvt_from_sint(F64, lval);
                    let rf = self.builder.ins().fcvt_from_sint(F64, rval);
                    let result_f64 = self.emit_runtime_call_raw("pdc_pow", &[lf, rf], Some(F64))?;
                    let result_cl = pdc_type_to_cl(result_ty, self.pointer_type);
                    self.builder.ins().fcvt_to_sint_sat(result_cl, result_f64)
                }
                _ => unreachable!(),
            })
        }
    }
}
