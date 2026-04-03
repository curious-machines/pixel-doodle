//! PDC -> LLVM JIT backend.
//!
//! Translates the type-checked PDC AST into native code via inkwell/LLVM.
//! Mirrors the Cranelift backend in `codegen.rs` but uses LLVM IR and
//! alloca-based SSA (mem2reg promotes automatically).

use std::collections::HashMap;

use inkwell::basic_block::BasicBlock;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::passes::PassBuilderOptions;
use inkwell::targets::{CodeModel, InitializationConfig, RelocMode, Target, TargetMachine};
use inkwell::types::{BasicMetadataTypeEnum, BasicType, BasicTypeEnum};
use inkwell::values::{BasicMetadataValueEnum, BasicValueEnum, FunctionValue, IntValue, FloatValue, PointerValue};
use inkwell::{AddressSpace, FloatPredicate, IntPredicate, OptimizationLevel};

use super::ast::*;
use super::error::PdcError;
use super::runtime;
use super::span::Spanned;
use super::type_check::{EnumInfo, OverloadSet, StructInfo};
pub use super::codegen_common::*;

/// Wrapper to make `ExecutionEngine` Send + Sync.
/// SAFETY: The engine owns JIT'd code and is only accessed through the
/// function pointers stored in `CompiledProgram`. The engine itself is
/// never shared mutably across threads.
struct SendSyncEngine(#[allow(dead_code)] inkwell::execution_engine::ExecutionEngine<'static>);
unsafe impl Send for SendSyncEngine {}
unsafe impl Sync for SendSyncEngine {}

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
    // Create LLVM context, module, target machine.
    let context: &'static Context = Box::leak(Box::new(Context::create()));
    let llvm_module = context.create_module("pdc_module");

    Target::initialize_native(&InitializationConfig::default())
        .map_err(|e| PdcError::Codegen { message: format!("LLVM native init: {e}") })?;
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
    let target = Target::from_triple(&triple).map_err(|e| PdcError::Codegen {
        message: format!("LLVM target: {e}"),
    })?;
    let machine = target
        .create_target_machine(
            &triple,
            cpu.to_str().unwrap(),
            &filtered_features,
            OptimizationLevel::Aggressive,
            RelocMode::Default,
            CodeModel::JITDefault,
        )
        .ok_or_else(|| PdcError::Codegen {
            message: "failed to create LLVM target machine".into(),
        })?;
    llvm_module.set_data_layout(&machine.get_target_data().get_data_layout());
    llvm_module.set_triple(&triple);

    let ptr_type = context.ptr_type(AddressSpace::default());

    // Collect top-level constants, vars, and function definitions
    let mut top_level_consts: Vec<&Spanned<Stmt>> = Vec::new();
    let mut top_level_builtins: Vec<&Spanned<Stmt>> = Vec::new();
    let mut top_level_vars: Vec<(&str, PdcType)> = Vec::new();
    let mut mutable_builtin_names: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut fn_defs: Vec<(String, &FnDef)> = Vec::new();

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

    // Declare all user functions in the LLVM module
    let mut user_fn_llvm: HashMap<String, FunctionValue<'static>> = HashMap::new();
    let mut overload_counts: HashMap<String, usize> = HashMap::new();

    for (qualified_name, fndef) in &fn_defs {
        let base = mangle_name(qualified_name);
        let idx = overload_counts.entry(qualified_name.clone()).or_insert(0);
        let mangled = if *idx == 0 {
            format!("pdc_userfn_{base}")
        } else {
            format!("pdc_userfn_{base}_{idx}")
        };
        *idx += 1;

        let mut param_types: Vec<BasicMetadataTypeEnum<'static>> = vec![ptr_type.into()]; // ctx
        let sret = is_compound_return(&fndef.return_type);
        if sret {
            param_types.push(ptr_type.into()); // output pointer for struct/tuple return
        }
        for param in &fndef.params {
            param_types.push(pdc_type_to_llvm(&param.ty, context).into());
        }

        let fn_type = if fndef.return_type == PdcType::Void || sret {
            context.void_type().fn_type(&param_types, false)
        } else {
            pdc_type_to_llvm(&fndef.return_type, context).fn_type(&param_types, false)
        };

        let func = llvm_module.add_function(&mangled, fn_type, None);
        user_fn_llvm.insert(mangled, func);
    }

    // Build operator dispatch table
    let mut op_fn_map: HashMap<(String, usize), FunctionValue<'static>> = HashMap::new();
    for (op_name, overloads) in op_overloads {
        for (sig_idx, sig) in overloads.sigs.iter().enumerate() {
            for (qualified_name, fndef) in &fn_defs {
                let short = qualified_name.rsplit("::").next().unwrap_or(qualified_name);
                if short != op_name { continue; }
                let params_match = fndef.params.len() == sig.params.len()
                    && fndef.params.iter().zip(&sig.params).all(|(p, s)| p.ty == *s);
                if params_match {
                    let base = mangle_name(qualified_name);
                    let mut qual_idx = 0;
                    for (qn2, fd2) in &fn_defs {
                        if qn2 == qualified_name {
                            if std::ptr::eq(*fd2, *fndef) { break; }
                            qual_idx += 1;
                        }
                    }
                    let mangled = if qual_idx == 0 {
                        format!("pdc_userfn_{base}")
                    } else {
                        format!("pdc_userfn_{base}_{qual_idx}")
                    };
                    if let Some(&func) = user_fn_llvm.get(&mangled) {
                        op_fn_map.insert((op_name.clone(), sig_idx), func);
                        break;
                    }
                }
            }
        }
    }

    // Build per-module intra-module alias maps
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
    let builder = context.create_builder();
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

        let function = user_fn_llvm[&mangled];
        let entry = context.append_basic_block(function, "entry");
        builder.position_at_end(entry);

        let is_sret = is_compound_return(&fndef.return_type);
        let ctx_ptr = function.get_nth_param(0).unwrap().into_pointer_value();
        let sret_ptr = if is_sret {
            Some(function.get_nth_param(1).unwrap().into_pointer_value())
        } else {
            None
        };
        let param_offset: u32 = if is_sret { 2 } else { 1 };
        let builtin_map = build_builtin_map(builtins_layout);

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

        let mut cg = LlvmCodegenCtx {
            context,
            module: &llvm_module,
            builder: &builder,
            function,
            ctx_ptr,
            variables: HashMap::new(),
            builtin_map,
            type_table: types,
            user_fn_llvm: &user_fn_llvm,
            user_fns,
            structs,
            enums,
            fn_aliases: effective_aliases,
            op_overloads,
            op_fn_map: &op_fn_map,
            block_terminated: false,
            loop_stack: Vec::new(),
            sret_ptr,
            fn_return_type: fndef.return_type.clone(),
            state_layout: &state_layout,
            mutable_builtins: mutable_builtin_names.clone(),
        };

        // Define parameters as allocas
        for (i, param) in fndef.params.iter().enumerate() {
            let val = function.get_nth_param(i as u32 + param_offset).unwrap();
            cg.define_variable(&param.name, &param.ty, val);
        }

        // Emit top-level constants and immutable builtins
        for const_stmt in &top_level_consts {
            cg.emit_stmt(const_stmt)?;
        }
        for builtin_stmt in &top_level_builtins {
            cg.emit_stmt(builtin_stmt)?;
        }

        cg.emit_block(&fndef.body)?;

        // Implicit return
        let body_returns = fndef.body.stmts.last().is_some_and(|s| matches!(s.node, Stmt::Return(_)));
        if !body_returns && !cg.block_terminated {
            if fndef.return_type == PdcType::Void || is_sret {
                cg.builder.build_return(None).unwrap();
            } else {
                let zero = cg.default_value(&fndef.return_type);
                cg.builder.build_return(Some(&zero)).unwrap();
            }
        }
    }

    // Compile main function
    let main_fn_type = context.void_type().fn_type(&[ptr_type.into()], false);
    let main_function = llvm_module.add_function("pdc_main", main_fn_type, None);
    let main_entry = context.append_basic_block(main_function, "entry");
    builder.position_at_end(main_entry);

    {
        let ctx_ptr = main_function.get_nth_param(0).unwrap().into_pointer_value();
        let builtin_map = build_builtin_map(builtins_layout);

        let mut cg = LlvmCodegenCtx {
            context,
            module: &llvm_module,
            builder: &builder,
            function: main_function,
            ctx_ptr,
            variables: HashMap::new(),
            builtin_map,
            type_table: types,
            user_fn_llvm: &user_fn_llvm,
            user_fns,
            structs,
            enums,
            fn_aliases,
            op_overloads,
            op_fn_map: &op_fn_map,
            block_terminated: false,
            loop_stack: Vec::new(),
            sret_ptr: None,
            fn_return_type: PdcType::Void,
            state_layout: &state_layout,
            mutable_builtins: mutable_builtin_names.clone(),
        };

        // Module-level statements
        for module in &program.modules {
            for stmt in &module.stmts {
                if matches!(&stmt.node, Stmt::FnDef(_) | Stmt::Import { .. } | Stmt::StructDef(_) | Stmt::EnumDef(_) | Stmt::TypeAlias { .. } | Stmt::TestDef { .. }) {
                    continue;
                }
                cg.emit_stmt(stmt)?;
                if cg.block_terminated { break; }
            }
        }

        // Main program statements
        for stmt in &program.stmts {
            if matches!(&stmt.node, Stmt::FnDef(_) | Stmt::Import { .. } | Stmt::StructDef(_) | Stmt::TypeAlias { .. } | Stmt::TestDef { .. }) {
                continue;
            }
            cg.emit_stmt(stmt)?;
            if cg.block_terminated { break; }
        }

        if !cg.block_terminated {
            cg.builder.build_return(None).unwrap();
        }
    }

    // Compile test functions
    let mut test_defs: Vec<(String, &Block)> = Vec::new();
    for stmt in &program.stmts {
        if let Stmt::TestDef { name, body } = &stmt.node {
            test_defs.push((name.clone(), body));
        }
    }

    let mut test_fn_llvm: Vec<(String, FunctionValue<'static>)> = Vec::new();
    for (i, (test_name, _)) in test_defs.iter().enumerate() {
        let mangled = format!("pdc_test_{i}");
        let fn_type = context.void_type().fn_type(&[ptr_type.into()], false);
        let func = llvm_module.add_function(&mangled, fn_type, None);
        test_fn_llvm.push((test_name.clone(), func));
    }

    for (i, (_test_name, body)) in test_defs.iter().enumerate() {
        let (_, func) = test_fn_llvm[i];
        let entry = context.append_basic_block(func, "entry");
        builder.position_at_end(entry);

        let ctx_ptr = func.get_nth_param(0).unwrap().into_pointer_value();
        let builtin_map = build_builtin_map(builtins_layout);

        let mut cg = LlvmCodegenCtx {
            context,
            module: &llvm_module,
            builder: &builder,
            function: func,
            ctx_ptr,
            variables: HashMap::new(),
            builtin_map,
            type_table: types,
            user_fn_llvm: &user_fn_llvm,
            user_fns,
            structs,
            enums,
            fn_aliases,
            op_overloads,
            op_fn_map: &op_fn_map,
            block_terminated: false,
            loop_stack: Vec::new(),
            sret_ptr: None,
            fn_return_type: PdcType::Void,
            state_layout: &state_layout,
            mutable_builtins: mutable_builtin_names.clone(),
        };

        for const_stmt in &top_level_consts {
            cg.emit_stmt(const_stmt)?;
        }

        cg.emit_block(body)?;

        if !cg.block_terminated {
            cg.builder.build_return(None).unwrap();
        }
    }

    // Run optimization passes
    let pass_options = PassBuilderOptions::create();
    llvm_module
        .run_passes("default<O3>", &machine, pass_options)
        .map_err(|e| PdcError::Codegen {
            message: format!("LLVM optimization: {e}"),
        })?;

    // Create execution engine and register runtime symbols
    let engine = llvm_module
        .create_jit_execution_engine(OptimizationLevel::Aggressive)
        .map_err(|e| PdcError::Codegen {
            message: format!("LLVM JIT engine: {e}"),
        })?;

    // Register all runtime symbols
    for (name, addr) in runtime::runtime_symbols() {
        if let Some(func) = llvm_module.get_function(name) {
            engine.add_global_mapping(&func, addr as usize);
        }
    }

    // Extract main function pointer
    let fn_ptr: PdcSceneFn = unsafe {
        let raw = engine.get_function_address("pdc_main")
            .map_err(|e| PdcError::Codegen { message: format!("get pdc_main: {e}") })?;
        std::mem::transmute(raw)
    };

    // Extract test function pointers
    let mut test_fn_ptrs: Vec<(String, PdcSceneFn)> = Vec::new();
    for (test_name, _func) in &test_fn_llvm {
        let mangled = format!("pdc_test_{}", test_fn_ptrs.len());
        let test_fn: PdcSceneFn = unsafe {
            let raw = engine.get_function_address(&mangled)
                .map_err(|e| PdcError::Codegen { message: format!("get test fn: {e}") })?;
            std::mem::transmute(raw)
        };
        test_fn_ptrs.push((test_name.clone(), test_fn));
    }

    // Extract user function pointers
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
        let fn_code_ptr = engine.get_function_address(&mangled)
            .map_err(|e| PdcError::Codegen { message: format!("get user fn: {e}") })?;
        let param_types: Vec<PdcType> = fndef.params.iter().map(|p| p.ty.clone()).collect();
        user_fn_ptrs.insert(key, (fn_code_ptr as *const u8, param_types, fndef.return_type.clone()));
    }

    Ok((
        CompiledProgram::new(
            fn_ptr,
            user_fn_ptrs,
            test_fn_ptrs,
            Box::new(SendSyncEngine(engine)),
        ),
        state_layout,
    ))
}

// ---------------------------------------------------------------------------
// Type mapping
// ---------------------------------------------------------------------------

fn is_compound_return(ty: &PdcType) -> bool {
    matches!(ty, PdcType::Struct(_) | PdcType::Tuple(_))
}

fn pdc_type_to_llvm<'ctx>(ty: &PdcType, ctx: &'ctx Context) -> BasicTypeEnum<'ctx> {
    match ty {
        PdcType::F32 => ctx.f32_type().into(),
        PdcType::F64 => ctx.f64_type().into(),
        PdcType::I8 | PdcType::U8 | PdcType::Bool => ctx.i8_type().into(),
        PdcType::I16 | PdcType::U16 => ctx.i16_type().into(),
        PdcType::I32 | PdcType::U32 => ctx.i32_type().into(),
        PdcType::I64 | PdcType::U64 => ctx.i64_type().into(),
        PdcType::PathHandle | PdcType::BufferHandle | PdcType::KernelHandle => ctx.i32_type().into(),
        PdcType::Str => ctx.i32_type().into(),
        PdcType::Array(_) => ctx.i32_type().into(),
        PdcType::Enum(_) => ctx.i32_type().into(),
        PdcType::Slice(_) | PdcType::Struct(_) | PdcType::Tuple(_) | PdcType::FnRef { .. } | PdcType::Unknown => {
            ctx.ptr_type(AddressSpace::default()).into()
        }
        PdcType::Void => ctx.i32_type().into(),
        PdcType::Module(_) => ctx.i32_type().into(),
    }
}

fn pdc_type_byte_size(ty: &PdcType) -> u32 {
    match ty {
        PdcType::I8 | PdcType::U8 | PdcType::Bool => 1,
        PdcType::I16 | PdcType::U16 => 2,
        PdcType::F32 | PdcType::I32 | PdcType::U32 | PdcType::PathHandle | PdcType::BufferHandle | PdcType::KernelHandle
        | PdcType::Str | PdcType::Array(_) | PdcType::Enum(_) => 4,
        PdcType::F64 | PdcType::I64 | PdcType::U64 => 8,
        _ => 8,
    }
}

// ---------------------------------------------------------------------------
// Codegen context
// ---------------------------------------------------------------------------

struct LlvmCodegenCtx<'a> {
    context: &'static Context,
    module: &'a Module<'static>,
    builder: &'a Builder<'static>,
    function: FunctionValue<'static>,
    ctx_ptr: PointerValue<'static>,
    /// Variables stored as (alloca_ptr, pdc_type).
    variables: HashMap<String, (PointerValue<'static>, PdcType)>,
    builtin_map: HashMap<String, BuiltinInfo>,
    type_table: &'a [PdcType],
    user_fn_llvm: &'a HashMap<String, FunctionValue<'static>>,
    user_fns: &'a HashMap<String, OverloadSet>,
    structs: &'a HashMap<String, StructInfo>,
    enums: &'a HashMap<String, EnumInfo>,
    fn_aliases: &'a HashMap<String, String>,
    op_overloads: &'a HashMap<String, OverloadSet>,
    op_fn_map: &'a HashMap<(String, usize), FunctionValue<'static>>,
    block_terminated: bool,
    loop_stack: Vec<LlvmLoopContext>,
    /// Output pointer for functions that return structs/tuples (sret convention).
    sret_ptr: Option<PointerValue<'static>>,
    fn_return_type: PdcType,
    /// State layout for module-level mutable variables.
    state_layout: &'a StateLayout,
    /// Names of mutable builtins — reads/writes go directly to the builtins array.
    mutable_builtins: std::collections::HashSet<String>,
}

struct LlvmLoopContext {
    continue_block: BasicBlock<'static>,
    exit_block: BasicBlock<'static>,
}

impl<'a> LlvmCodegenCtx<'a> {
    /// Check if a variable name is a module-level state variable.
    fn is_state_var(&self, name: &str) -> bool {
        self.state_layout.vars.contains_key(name)
    }

    /// Load the state block pointer from PdcContext (offset 16).
    fn load_state_ptr(&self) -> PointerValue<'static> {
        let ptr_type = self.context.ptr_type(AddressSpace::default());
        let state_ptr_ptr = unsafe {
            self.builder.build_gep(
                self.context.i8_type(),
                self.ctx_ptr,
                &[self.context.i32_type().const_int(16, false).into()],
                "state_ptr_ptr",
            ).unwrap()
        };
        self.builder.build_load(ptr_type, state_ptr_ptr, "state_ptr").unwrap().into_pointer_value()
    }

    /// Load a module-level state variable from the state block.
    fn load_state_var(&self, name: &str) -> Result<BasicValueEnum<'static>, PdcError> {
        let sv = self.state_layout.vars.get(name).ok_or_else(|| PdcError::Codegen {
            message: format!("state var '{name}' not found in layout"),
        })?;
        let llvm_ty = self.llvm_var_type(&sv.ty);
        let state_ptr = self.load_state_ptr();
        let var_ptr = unsafe {
            self.builder.build_gep(
                self.context.i8_type(),
                state_ptr,
                &[self.context.i32_type().const_int(sv.offset as u64, false).into()],
                &format!("state_{name}_ptr"),
            ).unwrap()
        };
        Ok(self.builder.build_load(llvm_ty, var_ptr, name).unwrap())
    }

    /// Store a value to a module-level state variable in the state block.
    fn store_state_var(&self, name: &str, val: BasicValueEnum<'static>) -> Result<(), PdcError> {
        let sv = self.state_layout.vars.get(name).ok_or_else(|| PdcError::Codegen {
            message: format!("state var '{name}' not found in layout"),
        })?;
        let state_ptr = self.load_state_ptr();
        let var_ptr = unsafe {
            self.builder.build_gep(
                self.context.i8_type(),
                state_ptr,
                &[self.context.i32_type().const_int(sv.offset as u64, false).into()],
                &format!("state_{name}_ptr"),
            ).unwrap()
        };
        self.builder.build_store(var_ptr, val).unwrap();
        Ok(())
    }

    /// Check if a variable name is a mutable builtin.
    fn is_mutable_builtin(&self, name: &str) -> bool {
        self.mutable_builtins.contains(name)
    }

    /// Load a mutable builtin from the builtins array.
    fn load_mutable_builtin(&mut self, name: &str) -> Result<BasicValueEnum<'static>, PdcError> {
        let info = self.builtin_map.get(name).ok_or_else(|| PdcError::Codegen {
            message: format!("mutable builtin '{name}' not found in layout"),
        })?;
        let offset = info.offset;
        let declared_ty = info.ty.clone();
        let builtins_ptr = self.builder.build_load(
            self.context.ptr_type(AddressSpace::default()),
            self.ctx_ptr,
            "builtins_ptr",
        ).unwrap().into_pointer_value();
        let f64_ty = self.context.f64_type();
        let elem_ptr = unsafe {
            self.builder.build_gep(
                f64_ty, builtins_ptr,
                &[self.context.i32_type().const_int(offset as u64, false).into()],
                &format!("builtin_{name}_gep"),
            ).unwrap()
        };
        let f64_val = self.builder.build_load(f64_ty, elem_ptr, name).unwrap();
        Ok(self.convert_value(f64_val, &PdcType::F64, &declared_ty))
    }

    /// Store a value to a mutable builtin in the builtins array.
    fn store_mutable_builtin(&mut self, name: &str, val: BasicValueEnum<'static>) -> Result<(), PdcError> {
        let info = self.builtin_map.get(name).ok_or_else(|| PdcError::Codegen {
            message: format!("mutable builtin '{name}' not found in layout"),
        })?;
        let offset = info.offset;
        let declared_ty = info.ty.clone();
        let f64_val = self.convert_value(val, &declared_ty, &PdcType::F64);
        let builtins_ptr = self.builder.build_load(
            self.context.ptr_type(AddressSpace::default()),
            self.ctx_ptr,
            "builtins_ptr",
        ).unwrap().into_pointer_value();
        let f64_ty = self.context.f64_type();
        let elem_ptr = unsafe {
            self.builder.build_gep(
                f64_ty, builtins_ptr,
                &[self.context.i32_type().const_int(offset as u64, false).into()],
                &format!("builtin_{name}_gep"),
            ).unwrap()
        };
        self.builder.build_store(elem_ptr, f64_val).unwrap();
        Ok(())
    }

    /// Returns true if `ty` is an enum with at least one data-carrying variant.
    fn is_data_enum(&self, ty: &PdcType) -> bool {
        if let PdcType::Enum(ename) = ty {
            self.enums.get(ename).map_or(false, |info| info.variants.iter().any(|v| !v.field_types.is_empty()))
        } else {
            false
        }
    }

    /// Returns the LLVM type to use for a PDC type, accounting for data-carrying enums.
    fn llvm_var_type(&self, ty: &PdcType) -> BasicTypeEnum<'static> {
        if self.is_data_enum(ty) {
            self.context.ptr_type(AddressSpace::default()).into()
        } else {
            pdc_type_to_llvm(ty, self.context)
        }
    }

    fn define_variable(&mut self, name: &str, ty: &PdcType, val: BasicValueEnum<'static>) {
        let llvm_ty = self.llvm_var_type(ty);
        let alloca = self.builder.build_alloca(llvm_ty, name).unwrap();
        self.builder.build_store(alloca, val).unwrap();
        self.variables.insert(name.to_string(), (alloca, ty.clone()));
    }

    fn load_variable(&self, name: &str) -> Result<BasicValueEnum<'static>, PdcError> {
        let (alloca, ty) = self.variables.get(name).ok_or_else(|| PdcError::Codegen {
            message: format!("undefined variable '{name}'"),
        })?;
        let llvm_ty = self.llvm_var_type(ty);
        Ok(self.builder.build_load(llvm_ty, *alloca, name).unwrap())
    }

    fn store_variable(&self, name: &str, val: BasicValueEnum<'static>) -> Result<(), PdcError> {
        let (alloca, _) = self.variables.get(name).ok_or_else(|| PdcError::Codegen {
            message: format!("undefined variable '{name}'"),
        })?;
        self.builder.build_store(*alloca, val).unwrap();
        Ok(())
    }

    fn node_type(&self, id: u32) -> &PdcType {
        if (id as usize) >= self.type_table.len() {
            eprintln!("WARNING: node id {} out of type table (len {})", id, self.type_table.len());
            return &PdcType::Unknown;
        }
        &self.type_table[id as usize]
    }

    fn default_value(&self, ty: &PdcType) -> BasicValueEnum<'static> {
        let llvm_ty = pdc_type_to_llvm(ty, self.context);
        match llvm_ty {
            BasicTypeEnum::FloatType(ft) => ft.const_zero().into(),
            BasicTypeEnum::IntType(it) => it.const_zero().into(),
            BasicTypeEnum::PointerType(pt) => pt.const_zero().into(),
            _ => self.context.i32_type().const_zero().into(),
        }
    }

    fn i32_const(&self, val: i64) -> IntValue<'static> {
        self.context.i32_type().const_int(val as u64, val < 0)
    }

    fn i8_const(&self, val: i64) -> IntValue<'static> {
        self.context.i8_type().const_int(val as u64, val < 0)
    }

    fn f64_const(&self, val: f64) -> FloatValue<'static> {
        self.context.f64_type().const_float(val)
    }

    fn f32_const(&self, val: f32) -> FloatValue<'static> {
        self.context.f32_type().const_float(val as f64)
    }

    // -----------------------------------------------------------------------
    // Block/statement emission
    // -----------------------------------------------------------------------

    fn emit_block(&mut self, block: &Block) -> Result<(), PdcError> {
        for stmt in &block.stmts {
            self.emit_stmt(stmt)?;
            if self.block_terminated { break; }
        }
        Ok(())
    }

    fn emit_stmt(&mut self, stmt: &Spanned<Stmt>) -> Result<(), PdcError> {
        match &stmt.node {
            Stmt::BuiltinDecl { name, ty, mutable } => {
                if *mutable {
                    // Mutable builtins: reads/writes go directly to the builtins array.
                    self.mutable_builtins.insert(name.clone());
                } else {
                    // Immutable builtins: snapshot the value into a local variable.
                    let info = self.builtin_map.get(name).ok_or_else(|| PdcError::Codegen {
                        message: format!("builtin '{name}' not found in layout"),
                    })?;
                    let offset = info.offset;
                    let declared_ty = ty.clone();

                    let builtins_ptr = self.builder.build_load(
                        self.context.ptr_type(AddressSpace::default()),
                        self.ctx_ptr,
                        "builtins_ptr",
                    ).unwrap().into_pointer_value();

                    let f64_ty = self.context.f64_type();
                    let elem_ptr = unsafe {
                        self.builder.build_gep(
                            f64_ty,
                            builtins_ptr,
                            &[self.i32_const(offset as i64).into()],
                            "builtin_gep",
                        ).unwrap()
                    };
                    let f64_val = self.builder.build_load(f64_ty, elem_ptr, "builtin_f64").unwrap();
                    let val = self.convert_value(f64_val, &PdcType::F64, &declared_ty);
                    self.define_variable(name, &declared_ty, val);
                }
            }
            Stmt::ConstDecl { vis: _, name, ty, value } | Stmt::VarDecl { vis: _, name, ty, value } => {
                let is_var = matches!(&stmt.node, Stmt::VarDecl { .. });
                let val = self.emit_expr(value)?;
                let expr_ty = self.node_type(value.id).clone();
                let final_ty = ty.clone().unwrap_or(expr_ty.clone());

                // State block variables: store to state block and skip local registration.
                if is_var && self.is_state_var(name) {
                    let converted = self.convert_value(val, &expr_ty, &final_ty);
                    self.store_state_var(name, converted)?;
                    return Ok(());
                }

                let is_pointer_type = match &final_ty {
                    PdcType::Struct(_) | PdcType::Tuple(_) => true,
                    PdcType::Enum(ename) => {
                        self.enums.get(ename).map_or(false, |info| info.variants.iter().any(|v| !v.field_types.is_empty()))
                    }
                    _ => false,
                };
                if is_pointer_type {
                    // Pointer types are stored directly (no conversion)
                    let alloca = self.builder.build_alloca(
                        self.context.ptr_type(AddressSpace::default()), name
                    ).unwrap();
                    self.builder.build_store(alloca, val).unwrap();
                    self.variables.insert(name.clone(), (alloca, final_ty));
                } else {
                    let converted = self.convert_value(val, &expr_ty, &final_ty);
                    self.define_variable(name, &final_ty, converted);
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
                    let elem_size = pdc_type_byte_size(elem_ty);
                    let store_val = self.float_to_int_if_needed(converted, elem_ty);
                    let set_name = format!("pdc_array_set_{elem_size}");
                    self.emit_runtime_call_raw(&set_name, &[self.ctx_ptr.into(), arr_handle, idx, store_val], None)?;
                }
            }
            Stmt::FieldAssign { object, field, value } => {
                let kernel_handle = self.emit_expr(object)?;
                let val_ty = self.node_type(value.id).clone();

                // Create string handle for the field name
                let field_bytes = field.as_bytes();
                let i8_ty = self.context.i8_type();
                let arr_ty = i8_ty.array_type(field_bytes.len() as u32);
                let arr_val = i8_ty.const_array(
                    &field_bytes.iter().map(|&b| i8_ty.const_int(b as u64, false)).collect::<Vec<_>>(),
                );
                let global = self.module.add_global(arr_ty, None, "field_name");
                global.set_initializer(&arr_val);
                global.set_constant(true);
                let field_ptr = global.as_pointer_value();
                let field_len = self.i32_const(field_bytes.len() as i64);
                let field_handle = self.emit_runtime_call_raw("pdc_string_new",
                    &[self.ctx_ptr.into(), field_ptr.into(), field_len.into()],
                    Some(self.context.i32_type().into()))?;

                if let PdcType::Enum(ref ename) = val_ty {
                    if ename == "Bind" {
                        // Destructure Bind.In(buffer) / Bind.Out(buffer) at compile time
                        if let Expr::MethodCall { method, args, .. } = &value.node {
                            let direction = if method == "In" { 0i64 } else { 1i64 };
                            let buffer_handle = self.emit_expr(&args[0])?;
                            let dir_val = self.i32_const(direction);
                            self.emit_runtime_call_raw("pdc_bind_buffer",
                                &[self.ctx_ptr.into(), kernel_handle, buffer_handle, field_handle, dir_val.into()],
                                None)?;
                        }
                    }
                } else {
                    // Scalar arg: emit set_kernel_arg_f64
                    let val = self.emit_expr(value)?;
                    let converted = self.convert_value(val, &val_ty, &PdcType::F64);
                    self.emit_runtime_call_raw("pdc_set_kernel_arg_f64",
                        &[self.ctx_ptr.into(), kernel_handle, field_handle, converted],
                        None)?;
                }
            }
            Stmt::TupleDestructure { names, value, .. } => {
                let tuple_ptr = self.emit_expr(value)?;
                let val_ty = self.node_type(value.id).clone();
                if let PdcType::Tuple(ref elems) = val_ty {
                    let f64_ty = self.context.f64_type();
                    for (i, name) in names.iter().enumerate() {
                        if name == "_" { continue; }
                        let elem_ty = &elems[i];
                        let field_ptr = unsafe {
                            self.builder.build_gep(f64_ty, tuple_ptr.into_pointer_value(),
                                &[self.i32_const(i as i64).into()], "tuple_gep").unwrap()
                        };
                        let raw = self.builder.build_load(f64_ty, field_ptr, "tuple_field").unwrap();
                        let val = self.narrow_from_f64(raw.into_float_value(), elem_ty);
                        self.define_variable(name, elem_ty, val);
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

                let (_, var_ty) = self.variables.get(name).cloned().ok_or_else(|| PdcError::Codegen {
                    message: format!("undefined variable '{name}'"),
                })?;
                let converted = self.convert_value(val, &expr_ty, &var_ty);
                self.store_variable(name, converted)?;
            }
            Stmt::ExprStmt(expr) => {
                self.emit_expr(expr)?;
            }
            Stmt::If { condition, then_body, elsif_clauses, else_body } => {
                self.emit_if(condition, then_body, elsif_clauses, else_body)?;
            }
            Stmt::While { condition, body } => {
                self.emit_while(condition, body)?;
            }
            Stmt::For { mutable: _, var_name, start, end, inclusive, body } => {
                self.emit_for(var_name, start, end, *inclusive, body)?;
            }
            Stmt::ForEach { mutable: _, var_name, destructure_names, collection, body } => {
                self.emit_foreach(var_name, destructure_names, collection, body)?;
            }
            Stmt::Loop { body } => {
                self.emit_loop(body)?;
            }
            Stmt::Break => {
                let lc = self.loop_stack.last().ok_or_else(|| PdcError::Codegen {
                    message: "break outside of loop".into(),
                })?;
                self.builder.build_unconditional_branch(lc.exit_block).unwrap();
                self.block_terminated = true;
            }
            Stmt::Continue => {
                let lc = self.loop_stack.last().ok_or_else(|| PdcError::Codegen {
                    message: "continue outside of loop".into(),
                })?;
                self.builder.build_unconditional_branch(lc.continue_block).unwrap();
                self.block_terminated = true;
            }
            Stmt::Return(value) => {
                if let Some(expr) = value {
                    let val = self.emit_expr(expr)?;
                    if let Some(out_ptr) = self.sret_ptr {
                        // sret: copy result into caller-provided output pointer
                        if let Some(size) = self.compound_type_size(&self.fn_return_type.clone()) {
                            self.copy_compound_to_ptr(val.into_pointer_value(), out_ptr, size);
                        }
                        self.builder.build_return(None).unwrap();
                    } else {
                        self.builder.build_return(Some(&val)).unwrap();
                    }
                } else {
                    self.builder.build_return(None).unwrap();
                }
                self.block_terminated = true;
            }
            Stmt::Match { scrutinee, arms } => {
                self.emit_match(scrutinee, arms)?;
            }
            Stmt::FnDef(_) | Stmt::Import { .. } | Stmt::StructDef(_) | Stmt::EnumDef(_) | Stmt::TypeAlias { .. } | Stmt::TestDef { .. } => {}
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Control flow
    // -----------------------------------------------------------------------

    fn emit_if(
        &mut self,
        condition: &Spanned<Expr>,
        then_body: &Block,
        elsif_clauses: &[(Spanned<Expr>, Block)],
        else_body: &Option<Block>,
    ) -> Result<(), PdcError> {
        let cond_val = self.emit_expr(condition)?.into_int_value();
        let cond_i1 = self.to_i1(cond_val);

        let then_block = self.context.append_basic_block(self.function, "then");
        let merge_block = self.context.append_basic_block(self.function, "merge");

        let mut next_block = if !elsif_clauses.is_empty() || else_body.is_some() {
            self.context.append_basic_block(self.function, "else")
        } else {
            merge_block
        };

        self.builder.build_conditional_branch(cond_i1, then_block, next_block).unwrap();

        self.builder.position_at_end(then_block);
        self.block_terminated = false;
        self.emit_block(then_body)?;
        if !self.block_terminated {
            self.builder.build_unconditional_branch(merge_block).unwrap();
        }

        for (i, (cond, body)) in elsif_clauses.iter().enumerate() {
            self.builder.position_at_end(next_block);
            let cond_val = self.emit_expr(cond)?.into_int_value();
            let cond_i1 = self.to_i1(cond_val);
            let elsif_block = self.context.append_basic_block(self.function, "elsif");
            let is_last = i == elsif_clauses.len() - 1;
            next_block = if is_last && else_body.is_none() {
                merge_block
            } else {
                self.context.append_basic_block(self.function, "else")
            };
            self.builder.build_conditional_branch(cond_i1, elsif_block, next_block).unwrap();
            self.builder.position_at_end(elsif_block);
            self.block_terminated = false;
            self.emit_block(body)?;
            if !self.block_terminated {
                self.builder.build_unconditional_branch(merge_block).unwrap();
            }
        }

        if let Some(else_b) = else_body {
            self.builder.position_at_end(next_block);
            self.block_terminated = false;
            self.emit_block(else_b)?;
            if !self.block_terminated {
                self.builder.build_unconditional_branch(merge_block).unwrap();
            }
        }

        self.builder.position_at_end(merge_block);
        self.block_terminated = false;
        Ok(())
    }

    fn emit_while(&mut self, condition: &Spanned<Expr>, body: &Block) -> Result<(), PdcError> {
        let header = self.context.append_basic_block(self.function, "while_hdr");
        let body_bb = self.context.append_basic_block(self.function, "while_body");
        let exit = self.context.append_basic_block(self.function, "while_exit");

        self.builder.build_unconditional_branch(header).unwrap();

        self.builder.position_at_end(header);
        let cond_val = self.emit_expr(condition)?.into_int_value();
        let cond_i1 = self.to_i1(cond_val);
        self.builder.build_conditional_branch(cond_i1, body_bb, exit).unwrap();

        self.builder.position_at_end(body_bb);
        self.block_terminated = false;
        self.loop_stack.push(LlvmLoopContext { continue_block: header, exit_block: exit });
        self.emit_block(body)?;
        self.loop_stack.pop();
        if !self.block_terminated {
            self.builder.build_unconditional_branch(header).unwrap();
        }

        self.builder.position_at_end(exit);
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
        let end_val = self.emit_expr(end)?.into_int_value();

        self.define_variable(var_name, &PdcType::I32, start_val);

        let header = self.context.append_basic_block(self.function, "for_hdr");
        let body_bb = self.context.append_basic_block(self.function, "for_body");
        let latch = self.context.append_basic_block(self.function, "for_latch");
        let exit = self.context.append_basic_block(self.function, "for_exit");

        self.builder.build_unconditional_branch(header).unwrap();

        self.builder.position_at_end(header);
        let i_val = self.load_variable(var_name)?.into_int_value();
        let pred = if inclusive { IntPredicate::SLE } else { IntPredicate::SLT };
        let cond = self.builder.build_int_compare(pred, i_val, end_val, "for_cond").unwrap();
        self.builder.build_conditional_branch(cond, body_bb, exit).unwrap();

        self.builder.position_at_end(body_bb);
        self.block_terminated = false;
        self.loop_stack.push(LlvmLoopContext { continue_block: latch, exit_block: exit });
        self.emit_block(body)?;
        self.loop_stack.pop();
        if !self.block_terminated {
            self.builder.build_unconditional_branch(latch).unwrap();
        }

        self.builder.position_at_end(latch);
        let i_val = self.load_variable(var_name)?.into_int_value();
        let next = self.builder.build_int_add(i_val, self.i32_const(1), "for_inc").unwrap();
        self.store_variable(var_name, next.into())?;
        self.builder.build_unconditional_branch(header).unwrap();

        self.builder.position_at_end(exit);
        self.block_terminated = false;
        Ok(())
    }

    fn emit_foreach(
        &mut self,
        var_name: &str,
        destructure_names: &[String],
        collection: &Spanned<Expr>,
        body: &Block,
    ) -> Result<(), PdcError> {
        let coll_ty = self.node_type(collection.id).clone();
        let elem_ty = match &coll_ty {
            PdcType::Array(et) => *et.clone(),
            _ => return Err(PdcError::Codegen { message: "for-each requires an array".into() }),
        };

        let arr_handle = self.emit_expr(collection)?;
        let len_val = self.emit_runtime_call_raw(
            "pdc_array_len", &[self.ctx_ptr.into(), arr_handle], Some(self.context.i32_type().into()),
        )?.into_int_value();

        // Index variable
        let idx_name = "__foreach_idx";
        self.define_variable(idx_name, &PdcType::I32, self.i32_const(0).into());

        let header = self.context.append_basic_block(self.function, "foreach_hdr");
        let body_bb = self.context.append_basic_block(self.function, "foreach_body");
        let latch = self.context.append_basic_block(self.function, "foreach_latch");
        let exit = self.context.append_basic_block(self.function, "foreach_exit");

        self.builder.build_unconditional_branch(header).unwrap();

        self.builder.position_at_end(header);
        let idx_val = self.load_variable(idx_name)?.into_int_value();
        let cond = self.builder.build_int_compare(IntPredicate::SLT, idx_val, len_val, "fe_cond").unwrap();
        self.builder.build_conditional_branch(cond, body_bb, exit).unwrap();

        self.builder.position_at_end(body_bb);
        self.block_terminated = false;

        // Load element
        let elem_size = pdc_type_byte_size(&elem_ty);
        let get_name = format!("pdc_array_get_{elem_size}");
        let int_type = match elem_size { 1 => self.context.i8_type().into(), 2 => self.context.i16_type().into(), 4 => self.context.i32_type().into(), _ => self.context.i64_type().into() };
        let idx_for_get = self.load_variable(idx_name)?;
        let raw = self.emit_runtime_call_raw(&get_name, &[self.ctx_ptr.into(), arr_handle, idx_for_get], Some(int_type))?;
        let elem_val = self.int_to_float_if_needed(raw, &elem_ty);

        if !destructure_names.is_empty() {
            if let PdcType::Tuple(ref tuple_elems) = elem_ty {
                let f64_ty = self.context.f64_type();
                for (i, name) in destructure_names.iter().enumerate() {
                    if name == "_" { continue; }
                    let field_ty = &tuple_elems[i];
                    let field_ptr = unsafe {
                        self.builder.build_gep(f64_ty, elem_val.into_pointer_value(),
                            &[self.i32_const(i as i64).into()], "destr_gep").unwrap()
                    };
                    let raw = self.builder.build_load(f64_ty, field_ptr, "destr_field").unwrap();
                    let val = self.narrow_from_f64(raw.into_float_value(), field_ty);
                    self.define_variable(name, field_ty, val);
                }
            }
        } else {
            self.define_variable(var_name, &elem_ty, elem_val);
        }

        self.loop_stack.push(LlvmLoopContext { continue_block: latch, exit_block: exit });
        self.emit_block(body)?;
        self.loop_stack.pop();
        if !self.block_terminated {
            self.builder.build_unconditional_branch(latch).unwrap();
        }

        self.builder.position_at_end(latch);
        let idx_val = self.load_variable(idx_name)?.into_int_value();
        let next = self.builder.build_int_add(idx_val, self.i32_const(1), "fe_inc").unwrap();
        self.store_variable(idx_name, next.into())?;
        self.builder.build_unconditional_branch(header).unwrap();

        self.builder.position_at_end(exit);
        self.block_terminated = false;
        Ok(())
    }

    fn emit_loop(&mut self, body: &Block) -> Result<(), PdcError> {
        let body_bb = self.context.append_basic_block(self.function, "loop_body");
        let exit = self.context.append_basic_block(self.function, "loop_exit");

        self.builder.build_unconditional_branch(body_bb).unwrap();
        self.builder.position_at_end(body_bb);
        self.block_terminated = false;
        self.loop_stack.push(LlvmLoopContext { continue_block: body_bb, exit_block: exit });
        self.emit_block(body)?;
        self.loop_stack.pop();
        if !self.block_terminated {
            self.builder.build_unconditional_branch(body_bb).unwrap();
        }

        self.builder.position_at_end(exit);
        self.block_terminated = false;
        Ok(())
    }

    fn emit_match(
        &mut self,
        scrutinee: &Spanned<Expr>,
        arms: &[MatchArm],
    ) -> Result<(), PdcError> {
        let scr_val = self.emit_expr(scrutinee)?;
        let scr_ty = self.node_type(scrutinee.id).clone();

        let has_data_variants = if let PdcType::Enum(ref ename) = scr_ty {
            self.enums.get(ename).map_or(false, |info| info.variants.iter().any(|v| !v.field_types.is_empty()))
        } else {
            false
        };

        let tag_val = if has_data_variants {
            // Load tag from pointer
            self.builder.build_load(self.context.i32_type(), scr_val.into_pointer_value(), "tag").unwrap().into_int_value()
        } else {
            scr_val.into_int_value()
        };

        let merge_block = self.context.append_basic_block(self.function, "match_merge");

        for (i, arm) in arms.iter().enumerate() {
            let is_last = i == arms.len() - 1;
            match &arm.pattern {
                MatchPattern::EnumVariant { enum_name: pat_enum, variant, bindings } => {
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
                    let variant_val = self.i32_const(variant_idx as i64);
                    let cmp = self.builder.build_int_compare(IntPredicate::EQ, tag_val, variant_val, "match_cmp").unwrap();

                    let arm_block = self.context.append_basic_block(self.function, "match_arm");
                    let next_block = if is_last { merge_block } else {
                        self.context.append_basic_block(self.function, "match_next")
                    };

                    self.builder.build_conditional_branch(cmp, arm_block, next_block).unwrap();
                    self.builder.position_at_end(arm_block);
                    self.block_terminated = false;

                    if !bindings.is_empty() && has_data_variants {
                        let variant_info = &info.variants[variant_idx];
                        let f64_ty = self.context.f64_type();
                        let i8_ty = self.context.i8_type();
                        for (bi, bname) in bindings.iter().enumerate() {
                            if bname == "_" { continue; }
                            let field_ty = &variant_info.field_types[bi];
                            // Offset: 8 bytes for tag + bi * 8
                            let byte_offset = (8 + bi * 8) as i64;
                            let byte_ptr = unsafe {
                                self.builder.build_gep(i8_ty, scr_val.into_pointer_value(),
                                    &[self.context.i64_type().const_int(byte_offset as u64, false).into()],
                                    "variant_gep").unwrap()
                            };
                            let raw = self.builder.build_load(f64_ty, byte_ptr, "variant_field").unwrap();
                            let val = self.narrow_from_f64(raw.into_float_value(), field_ty);
                            self.define_variable(bname, field_ty, val);
                        }
                    }

                    self.emit_block(&arm.body)?;
                    if !self.block_terminated {
                        self.builder.build_unconditional_branch(merge_block).unwrap();
                    }

                    if !is_last {
                        self.builder.position_at_end(next_block);
                        self.block_terminated = false;
                    }
                }
                MatchPattern::Wildcard => {
                    self.block_terminated = false;
                    self.emit_block(&arm.body)?;
                    if !self.block_terminated {
                        self.builder.build_unconditional_branch(merge_block).unwrap();
                    }
                }
            }
        }

        self.builder.position_at_end(merge_block);
        self.block_terminated = false;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Expression emission
    // -----------------------------------------------------------------------

    fn emit_expr(&mut self, expr: &Spanned<Expr>) -> Result<BasicValueEnum<'static>, PdcError> {
        match &expr.node {
            Expr::Literal(lit) => self.emit_literal(lit, expr.id),
            Expr::Variable(name) => {
                if self.is_mutable_builtin(name) {
                    return self.load_mutable_builtin(name);
                }
                if self.is_state_var(name) {
                    return self.load_state_var(name);
                }
                self.load_variable(name)
            }
            Expr::BinaryOp { op, left, right } => {
                let lt = self.node_type(left.id).clone();
                let rt = self.node_type(right.id).clone();
                let result_ty = self.node_type(expr.id).clone();

                let op_name = binop_to_op_name(*op);
                if let Some(overloads) = self.op_overloads.get(op_name) {
                    let arg_types = [lt.clone(), rt.clone()];
                    if let Some((idx, _)) = overloads.sigs.iter().enumerate().find(|(_, sig)| {
                        sig.params.len() == 2 && sig.params[0] == arg_types[0] && sig.params[1] == arg_types[1]
                    }) {
                        let lval = self.emit_expr(left)?;
                        let rval = self.emit_expr(right)?;
                        return self.emit_operator_call(op_name, idx, &[lval, rval], &result_ty);
                    }
                }

                if let PdcType::Array(_) = &result_ty {
                    return self.emit_array_broadcast(*op, left, right, &lt, &rt, &result_ty);
                }

                let lval = self.emit_expr(left)?;
                let rval = self.emit_expr(right)?;
                self.emit_binary_op(*op, lval, rval, &lt, &rt, &result_ty)
            }
            Expr::UnaryOp { op, operand } => {
                let operand_ty = self.node_type(operand.id).clone();

                let uop_name = unaryop_to_op_name(*op);
                if let Some(overloads) = self.op_overloads.get(uop_name) {
                    if let Some((idx, _)) = overloads.sigs.iter().enumerate().find(|(_, sig)| {
                        sig.params.len() == 1 && sig.params[0] == operand_ty
                    }) {
                        let result_ty = self.node_type(expr.id).clone();
                        let val = self.emit_expr(operand)?;
                        return self.emit_operator_call(uop_name, idx, &[val], &result_ty);
                    }
                }

                let val = self.emit_expr(operand)?;
                match op {
                    UnaryOp::Neg => {
                        if operand_ty.is_float() {
                            Ok(self.builder.build_float_neg(val.into_float_value(), "neg").unwrap().into())
                        } else {
                            let zero = self.i32_const(0);
                            Ok(self.builder.build_int_sub(zero, val.into_int_value(), "neg").unwrap().into())
                        }
                    }
                    UnaryOp::Not => {
                        let one = self.i8_const(1);
                        Ok(self.builder.build_xor(val.into_int_value(), one, "not").unwrap().into())
                    }
                    UnaryOp::BitNot => {
                        let llvm_ty = pdc_type_to_llvm(&operand_ty, self.context);
                        let neg_one = llvm_ty.into_int_type().const_all_ones();
                        Ok(self.builder.build_xor(val.into_int_value(), neg_one, "bitnot").unwrap().into())
                    }
                }
            }
            Expr::Call { name, args, arg_names } => {
                let has_named = arg_names.iter().any(|n| n.is_some());
                if has_named {
                    if let Some(info) = self.structs.get(name).cloned() {
                        return self.emit_struct_construct_from_call(name, args, arg_names, &info);
                    }
                }
                self.emit_call(name, args, expr.id)
            }
            Expr::MethodCall { object, method, args } => {
                let obj_ty = self.node_type(object.id).clone();
                if let PdcType::Module(ref mod_name) = obj_ty {
                    // Buffer factory: Buffer.I32() → pdc_create_buffer(ctx, type_code)
                    if mod_name == "Buffer" {
                        let type_code = match method.as_str() {
                            "F32" => 0, "I32" => 1, "U32" => 2,
                            "Vec2F32" => 3, "Vec3F32" => 4, "Vec4F32" => 5,
                            _ => 0,
                        };
                        let code_val = self.i32_const(type_code);
                        return self.emit_runtime_call_raw("pdc_create_buffer",
                            &[self.ctx_ptr.into(), code_val.into()],
                            Some(self.context.i32_type().into()));
                    }
                    // Kernel factory: Kernel.Sim("name", "path") → pdc_load_kernel(ctx, name, path, kind)
                    if mod_name == "Kernel" {
                        let kind = match method.as_str() {
                            "Pixel" => 0, "Sim" => 1, _ => 0,
                        };
                        let name_val = self.emit_expr(&args[0])?;
                        let path_val = self.emit_expr(&args[1])?;
                        let kind_val = self.i32_const(kind);
                        return self.emit_runtime_call_raw("pdc_load_kernel",
                            &[self.ctx_ptr.into(), name_val, path_val, kind_val.into()],
                            Some(self.context.i32_type().into()));
                    }
                    let qualified = format!("{mod_name}::{method}");
                    return self.emit_call(&qualified, args, expr.id);
                }
                if let PdcType::Tuple(ref elems) = obj_ty {
                    if method == "len" {
                        return Ok(self.i32_const(elems.len() as i64).into());
                    }
                }
                if let PdcType::Slice(ref elem_ty) = obj_ty {
                    let slice_ptr = self.emit_expr(object)?.into_pointer_value();
                    return match method.as_str() {
                        "len" => {
                            // Load length from offset 8 in the slice struct
                            let i32_ty = self.context.i32_type();
                            let i8_ty = self.context.i8_type();
                            let len_ptr = unsafe {
                                self.builder.build_gep(i8_ty, slice_ptr,
                                    &[self.context.i64_type().const_int(8, false).into()], "sl_len_gep").unwrap()
                            };
                            Ok(self.builder.build_load(i32_ty, len_ptr, "sl_len").unwrap())
                        }
                        "get" => {
                            let idx = self.emit_expr(&args[0])?;
                            let i32_ty = self.context.i32_type();
                            let i8_ty = self.context.i8_type();
                            let handle_val = self.builder.build_load(i32_ty, slice_ptr, "sl_handle").unwrap();
                            let start_ptr = unsafe {
                                self.builder.build_gep(i8_ty, slice_ptr,
                                    &[self.context.i64_type().const_int(4, false).into()], "sl_start_gep").unwrap()
                            };
                            let start_val = self.builder.build_load(i32_ty, start_ptr, "sl_start").unwrap();
                            let elem_size = pdc_type_byte_size(elem_ty);
                            let get_name = format!("pdc_slice_get_{elem_size}");
                            let int_type: BasicTypeEnum = match elem_size { 1 => self.context.i8_type().into(), 2 => self.context.i16_type().into(), 4 => self.context.i32_type().into(), _ => self.context.i64_type().into() };
                            let raw = self.emit_runtime_call_raw(&get_name, &[self.ctx_ptr.into(), handle_val, start_val, idx], Some(int_type))?;
                            Ok(self.int_to_float_if_needed(raw, elem_ty))
                        }
                        _ => Err(PdcError::Codegen { message: format!("unknown slice method '{method}'") }),
                    };
                }
                if obj_ty == PdcType::Str {
                    let handle = self.emit_expr(object)?;
                    return match method.as_str() {
                        "len" => self.emit_runtime_call_raw("pdc_string_len", &[self.ctx_ptr.into(), handle], Some(self.context.i32_type().into())),
                        "concat" => {
                            let arg = self.emit_expr(&args[0])?;
                            self.emit_runtime_call_raw("pdc_string_concat", &[self.ctx_ptr.into(), handle, arg], Some(self.context.i32_type().into()))
                        }
                        "slice" => {
                            let start = self.emit_expr(&args[0])?;
                            let end = self.emit_expr(&args[1])?;
                            self.emit_runtime_call_raw("pdc_string_slice", &[self.ctx_ptr.into(), handle, start, end], Some(self.context.i32_type().into()))
                        }
                        "char_at" => {
                            let idx = self.emit_expr(&args[0])?;
                            self.emit_runtime_call_raw("pdc_string_char_at", &[self.ctx_ptr.into(), handle, idx], Some(self.context.i32_type().into()))
                        }
                        _ => Err(PdcError::Codegen { message: format!("unknown string method '{method}'") }),
                    };
                }
                if let PdcType::Array(ref elem_ty) = obj_ty {
                    return self.emit_array_method(object, method, args, elem_ty);
                }
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
                let size = info.fields.len();
                let f64_ty = self.context.f64_type();
                let arr_ty = f64_ty.array_type(size as u32);
                let alloca = self.builder.build_alloca(arr_ty, "struct").unwrap();

                for (fname, fexpr) in fields {
                    let val = self.emit_expr(fexpr)?;
                    let fexpr_ty = self.node_type(fexpr.id).clone();
                    let field_idx = info.fields.iter().position(|(n, _)| n == fname)
                        .ok_or_else(|| PdcError::Codegen {
                            message: format!("struct '{name}' has no field '{fname}'"),
                        })?;
                    let field_ty = &info.fields[field_idx].1;
                    let converted = self.convert_value(val, &fexpr_ty, field_ty);
                    let store_val = self.widen_to_f64(converted, field_ty);
                    let field_ptr = unsafe {
                        self.builder.build_gep(f64_ty, alloca,
                            &[self.i32_const(field_idx as i64).into()], "field_gep").unwrap()
                    };
                    self.builder.build_store(field_ptr, store_val).unwrap();
                }
                Ok(alloca.into())
            }
            Expr::Index { object, index } => {
                let obj_ty = self.node_type(object.id).clone();
                if let PdcType::Array(ref elem_ty) = obj_ty {
                    let arr_handle = self.emit_expr(object)?;
                    let idx = self.emit_expr(index)?;
                    let elem_size = pdc_type_byte_size(elem_ty);
                    let get_name = format!("pdc_array_get_{elem_size}");
                    let int_type: BasicTypeEnum = match elem_size { 1 => self.context.i8_type().into(), 2 => self.context.i16_type().into(), 4 => self.context.i32_type().into(), _ => self.context.i64_type().into() };
                    let raw = self.emit_runtime_call_raw(&get_name, &[self.ctx_ptr.into(), arr_handle, idx], Some(int_type))?;
                    Ok(self.int_to_float_if_needed(raw, elem_ty))
                } else if let PdcType::Slice(ref elem_ty) = obj_ty {
                    let slice_ptr = self.emit_expr(object)?.into_pointer_value();
                    let idx = self.emit_expr(index)?;
                    let i32_ty = self.context.i32_type();
                    let i8_ty = self.context.i8_type();
                    let handle_val = self.builder.build_load(i32_ty, slice_ptr, "sl_handle").unwrap();
                    let start_ptr = unsafe {
                        self.builder.build_gep(i8_ty, slice_ptr,
                            &[self.context.i64_type().const_int(4, false).into()], "sl_start_gep").unwrap()
                    };
                    let start_val = self.builder.build_load(i32_ty, start_ptr, "sl_start").unwrap();
                    let elem_size = pdc_type_byte_size(elem_ty);
                    let get_name = format!("pdc_slice_get_{elem_size}");
                    let int_type: BasicTypeEnum = match elem_size { 1 => self.context.i8_type().into(), 2 => self.context.i16_type().into(), 4 => self.context.i32_type().into(), _ => self.context.i64_type().into() };
                    let raw = self.emit_runtime_call_raw(&get_name, &[self.ctx_ptr.into(), handle_val, start_val, idx], Some(int_type))?;
                    Ok(self.int_to_float_if_needed(raw, elem_ty))
                } else {
                    Err(PdcError::Codegen { message: "cannot index non-array/slice type".into() })
                }
            }
            Expr::TupleConstruct { elements } => {
                let f64_ty = self.context.f64_type();
                let arr_ty = f64_ty.array_type(elements.len() as u32);
                let alloca = self.builder.build_alloca(arr_ty, "tuple").unwrap();
                for (i, elem) in elements.iter().enumerate() {
                    let val = self.emit_expr(elem)?;
                    let elem_ty = self.node_type(elem.id).clone();
                    let store_val = self.widen_to_f64(val, &elem_ty);
                    let field_ptr = unsafe {
                        self.builder.build_gep(f64_ty, alloca,
                            &[self.i32_const(i as i64).into()], "tuple_gep").unwrap()
                    };
                    self.builder.build_store(field_ptr, store_val).unwrap();
                }
                Ok(alloca.into())
            }
            Expr::TupleIndex { object, index } => {
                let obj_val = self.emit_expr(object)?.into_pointer_value();
                let result_ty = self.node_type(expr.id).clone();
                let f64_ty = self.context.f64_type();
                let field_ptr = unsafe {
                    self.builder.build_gep(f64_ty, obj_val,
                        &[self.i32_const(*index as i64).into()], "tuple_idx_gep").unwrap()
                };
                let raw = self.builder.build_load(f64_ty, field_ptr, "tuple_elem").unwrap().into_float_value();
                Ok(self.narrow_from_f64(raw, &result_ty))
            }
            Expr::Ternary { condition, then_expr, else_expr } => {
                let cond_val = self.emit_expr(condition)?.into_int_value();
                let cond_i1 = self.to_i1(cond_val);
                let result_ty = self.node_type(expr.id).clone();
                let llvm_ty = pdc_type_to_llvm(&result_ty, self.context);

                let result_alloca = self.builder.build_alloca(llvm_ty, "ternary").unwrap();
                let default = self.default_value(&result_ty);
                self.builder.build_store(result_alloca, default).unwrap();

                let then_bb = self.context.append_basic_block(self.function, "tern_then");
                let else_bb = self.context.append_basic_block(self.function, "tern_else");
                let merge_bb = self.context.append_basic_block(self.function, "tern_merge");

                self.builder.build_conditional_branch(cond_i1, then_bb, else_bb).unwrap();

                self.builder.position_at_end(then_bb);
                let then_val = self.emit_expr(then_expr)?;
                let then_ty = self.node_type(then_expr.id).clone();
                let then_conv = self.convert_value(then_val, &then_ty, &result_ty);
                self.builder.build_store(result_alloca, then_conv).unwrap();
                self.builder.build_unconditional_branch(merge_bb).unwrap();

                self.builder.position_at_end(else_bb);
                let else_val = self.emit_expr(else_expr)?;
                let else_ty = self.node_type(else_expr.id).clone();
                let else_conv = self.convert_value(else_val, &else_ty, &result_ty);
                self.builder.build_store(result_alloca, else_conv).unwrap();
                self.builder.build_unconditional_branch(merge_bb).unwrap();

                self.builder.position_at_end(merge_bb);
                Ok(self.builder.build_load(llvm_ty, result_alloca, "ternary_val").unwrap())
            }
            Expr::FieldAccess { object, field } => {
                let obj_ty = self.node_type(object.id).clone();

                if let PdcType::Module(ref mod_name) = obj_ty {
                    let qualified = format!("{mod_name}::{field}");
                    let var_name = if self.variables.contains_key(&qualified) {
                        qualified
                    } else {
                        field.clone()
                    };
                    return self.load_variable(&var_name);
                }

                if let PdcType::Enum(ref ename) = obj_ty {
                    let info = self.enums.get(ename).ok_or_else(|| PdcError::Codegen {
                        message: format!("undefined enum '{ename}'"),
                    })?.clone();
                    let idx = info.variants.iter().position(|v| v.name == *field)
                        .ok_or_else(|| PdcError::Codegen {
                            message: format!("enum '{ename}' has no variant '{field}'"),
                        })?;
                    return Ok(self.i32_const(idx as i64).into());
                }

                let obj_val = self.emit_expr(object)?.into_pointer_value();

                if let PdcType::Struct(ref sname) = obj_ty {
                    let info = self.structs.get(sname).ok_or_else(|| PdcError::Codegen {
                        message: format!("undefined struct '{sname}'"),
                    })?.clone();
                    let field_idx = info.fields.iter().position(|(n, _)| n == field)
                        .ok_or_else(|| PdcError::Codegen {
                            message: format!("struct '{sname}' has no field '{field}'"),
                        })?;
                    let field_ty = &info.fields[field_idx].1;
                    let f64_ty = self.context.f64_type();
                    let field_ptr = unsafe {
                        self.builder.build_gep(f64_ty, obj_val,
                            &[self.i32_const(field_idx as i64).into()], "field_gep").unwrap()
                    };
                    let raw = self.builder.build_load(f64_ty, field_ptr, "field_val").unwrap().into_float_value();
                    Ok(self.narrow_from_f64(raw, field_ty))
                } else {
                    Err(PdcError::Codegen {
                        message: format!("cannot access field '{field}' on non-struct type"),
                    })
                }
            }
            Expr::DotShorthand(variant) => {
                let ty = self.node_type(expr.id).clone();
                if let PdcType::Enum(ref ename) = ty {
                    let info = self.enums.get(ename).ok_or_else(|| PdcError::Codegen {
                        message: format!("undefined enum '{ename}'"),
                    })?.clone();
                    let idx = info.variants.iter().position(|v| v.name == *variant)
                        .ok_or_else(|| PdcError::Codegen {
                            message: format!("enum '{ename}' has no variant '{variant}'"),
                        })?;
                    Ok(self.i32_const(idx as i64).into())
                } else {
                    Err(PdcError::Codegen {
                        message: format!("dot-shorthand '.{variant}' was not resolved to an enum type"),
                    })
                }
            }
        }
    }

    fn emit_literal(&mut self, lit: &Literal, id: u32) -> Result<BasicValueEnum<'static>, PdcError> {
        let ty = self.node_type(id);
        match lit {
            Literal::Int(v) => {
                if ty.is_float() {
                    if *ty == PdcType::F32 {
                        Ok(self.f32_const(*v as f32).into())
                    } else {
                        Ok(self.f64_const(*v as f64).into())
                    }
                } else {
                    let llvm_ty = pdc_type_to_llvm(ty, self.context).into_int_type();
                    Ok(llvm_ty.const_int(*v as u64, *v < 0).into())
                }
            }
            Literal::Float(v) => {
                if *ty == PdcType::F32 {
                    Ok(self.f32_const(*v as f32).into())
                } else {
                    Ok(self.f64_const(*v).into())
                }
            }
            Literal::Bool(v) => Ok(self.i8_const(*v as i64).into()),
            Literal::String(s) => {
                let bytes = s.as_bytes();
                let len = bytes.len();
                if len == 0 {
                    let null_ptr = self.context.ptr_type(AddressSpace::default()).const_zero();
                    let len_val = self.i32_const(0);
                    self.emit_runtime_call_raw("pdc_string_new",
                        &[self.ctx_ptr.into(), null_ptr.into(), len_val.into()],
                        Some(self.context.i32_type().into()))
                } else {
                    // Store bytes as a global constant
                    let i8_ty = self.context.i8_type();
                    let arr_ty = i8_ty.array_type(len as u32);
                    let byte_vals: Vec<_> = bytes.iter().map(|&b| i8_ty.const_int(b as u64, false)).collect();
                    let arr_val = i8_ty.const_array(&byte_vals);
                    let global = self.module.add_global(arr_ty, None, "str_data");
                    global.set_initializer(&arr_val);
                    global.set_constant(true);
                    let ptr = global.as_pointer_value();
                    let len_val = self.i32_const(len as i64);
                    self.emit_runtime_call_raw("pdc_string_new",
                        &[self.ctx_ptr.into(), ptr.into(), len_val.into()],
                        Some(self.context.i32_type().into()))
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Function calls
    // -----------------------------------------------------------------------

    fn emit_call(
        &mut self,
        name: &str,
        args: &[Spanned<Expr>],
        call_id: u32,
    ) -> Result<BasicValueEnum<'static>, PdcError> {
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
            return self.emit_runtime_call_raw(runtime_name, &[self.ctx_ptr.into(), a, b], None);
        }
        if name == "assert_near" && args.len() == 3 {
            let a = self.emit_expr(&args[0])?;
            let b = self.emit_expr(&args[1])?;
            let eps = self.emit_expr(&args[2])?;
            return self.emit_runtime_call_raw("pdc_assert_near", &[self.ctx_ptr.into(), a, b, eps], None);
        }
        if name == "assert_true" && args.len() == 1 {
            let cond = self.emit_expr(&args[0])?;
            let val = self.builder.build_int_s_extend(cond.into_int_value(), self.context.i64_type(), "sext").unwrap();
            return self.emit_runtime_call_raw("pdc_assert_true", &[self.ctx_ptr.into(), val.into()], None);
        }

        // Array functions
        if matches!(name, "push" | "get" | "set" | "len") && !args.is_empty() {
            let first_ty = self.node_type(args[0].id).clone();
            if let PdcType::Array(ref elem_ty) = first_ty {
                let elem_ty = *elem_ty.clone();
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

        // User-defined function
        let resolved_name = if let Some(qualified) = self.fn_aliases.get(name) {
            qualified.clone()
        } else if self.user_fns.contains_key(name) {
            name.to_string()
        } else {
            String::new()
        };

        if let Some(overloads) = self.user_fns.get(&resolved_name) {
            let overloads = overloads.clone();
            let arg_types: Vec<PdcType> = args.iter().map(|a| self.node_type(a.id).clone()).collect();

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
            let func = self.user_fn_llvm[&mangled];

            let ret_ty = self.node_type(call_id).clone();
            let sret_alloca = if let Some(size) = self.compound_type_size(&ret_ty) {
                let f64_ty = self.context.f64_type();
                let arr_ty = f64_ty.array_type(size / 8);
                Some(self.builder.build_alloca(arr_ty, "sret_out").unwrap())
            } else {
                None
            };

            let mut arg_vals: Vec<BasicMetadataValueEnum> = vec![self.ctx_ptr.into()];
            if let Some(out_ptr) = sret_alloca {
                arg_vals.push(out_ptr.into());
            }
            for (i, arg) in args.iter().enumerate() {
                let val = self.emit_expr(arg)?;
                let from_ty = self.node_type(arg.id).clone();
                let to_ty = &sig.params[i];
                let converted = self.convert_value(val, &from_ty, to_ty);
                arg_vals.push(converted.into());
            }
            for i in args.len()..sig.params.len() {
                let default_idx = i - sig.required;
                let default_expr = &sig.defaults[default_idx];
                let val = self.emit_expr(default_expr)?;
                let from_ty = self.node_type(default_expr.id).clone();
                let to_ty = &sig.params[i];
                let converted = self.convert_value(val, &from_ty, to_ty);
                arg_vals.push(converted.into());
            }

            if let Some(out_ptr) = sret_alloca {
                self.builder.build_call(func, &arg_vals, "userfn").unwrap();
                Ok(out_ptr.into())
            } else {
                let call = self.builder.build_call(func, &arg_vals, "userfn").unwrap();
                match call.try_as_basic_value().basic() {
                    Some(val) => Ok(val),
                    None => Ok(self.i32_const(0).into()),
                }
            }
        } else if name.starts_with("Array<") {
            // Array constructor
            let result_ty = self.node_type(call_id).clone();
            let elem_size = if let PdcType::Array(ref et) = result_ty {
                pdc_type_byte_size(et)
            } else { 8 };
            let size_val = self.i32_const(elem_size as i64);
            self.emit_runtime_call_raw("pdc_array_new", &[self.ctx_ptr.into(), size_val.into()], Some(self.context.i32_type().into()))
        } else {
            // Runtime function call
            let runtime_name = match name {
                "Path" => "pdc_path".to_string(),
                "display_buffer" => "pdc_display_buffer".to_string(),
                "swap" => "pdc_swap_buffers".to_string(),
                "run" => "pdc_run_kernel".to_string(),
                "push" => "pdc_array_push".to_string(),
                "len" => "pdc_array_len".to_string(),
                "get" => "pdc_array_get".to_string(),
                "set" => "pdc_array_set".to_string(),
                "fill" if args.len() == 3 => "pdc_fill_styled".to_string(),
                "stroke" if args.len() == 5 => "pdc_stroke_styled".to_string(),
                other => format!("pdc_{}", other),
            };

            let takes_ctx = matches!(
                name,
                "Path"
                | "move_to" | "line_to" | "quad_to" | "cubic_to" | "close" | "fill" | "stroke"
                | "fill_styled" | "stroke_styled"
                | "push" | "len" | "get" | "set"
                | "display_buffer" | "swap" | "run"
                | "display" | "load_texture"
                | "load_scene" | "run_scene" | "scene_tiles_x" | "scene_num_paths" | "scene_buffer"
                | "request_redraw"
                | "set_max_samples" | "is_converged" | "accumulate_sample"
                | "display_accumulated" | "reset_accumulation"
            );

            let mut arg_vals: Vec<BasicValueEnum> = Vec::new();
            if takes_ctx {
                arg_vals.push(self.ctx_ptr.into());
            }
            for arg in args {
                let val = self.emit_expr(arg)?;
                let arg_ty = self.node_type(arg.id).clone();
                let converted = self.convert_for_call(val, &arg_ty, name);
                arg_vals.push(converted);
            }

            let ret_type = self.call_return_type(name);
            self.emit_runtime_call_raw(&runtime_name, &arg_vals.iter().map(|v| (*v).into()).collect::<Vec<_>>(), ret_type)
        }
    }

    fn call_return_type(&self, name: &str) -> Option<BasicTypeEnum<'static>> {
        match name {
            "Path" | "len" | "load_texture"
            | "load_scene" | "scene_buffer"
            | "is_converged" => Some(self.context.i32_type().into()),
            "get" | "scene_tiles_x" | "scene_num_paths" => Some(self.context.f64_type().into()),
            "move_to" | "line_to" | "quad_to" | "cubic_to" | "close" | "fill" | "stroke"
            | "fill_styled" | "stroke_styled" | "push" | "set"
            | "bind" | "display_buffer" | "swap" | "run" | "set_arg"
            | "display"
            | "run_scene"
            | "request_redraw" | "set_max_samples" | "accumulate_sample"
            | "display_accumulated" | "reset_accumulation" => None,
            _ => Some(self.context.f64_type().into()),
        }
    }

    fn convert_for_call(&mut self, val: BasicValueEnum<'static>, from: &PdcType, func_name: &str) -> BasicValueEnum<'static> {
        let is_stroke_float = func_name == "stroke" && from.is_float();
        if is_stroke_float && *from != PdcType::F32 {
            return self.builder.build_float_trunc(val.into_float_value(), self.context.f32_type(), "fdemote").unwrap().into();
        }
        val
    }

    // -----------------------------------------------------------------------
    // Runtime calls
    // -----------------------------------------------------------------------

    fn emit_runtime_call_raw(
        &mut self,
        runtime_name: &str,
        args: &[BasicValueEnum<'static>],
        ret_type: Option<BasicTypeEnum<'static>>,
    ) -> Result<BasicValueEnum<'static>, PdcError> {
        // Build function type from arguments
        let param_types: Vec<BasicMetadataTypeEnum> = args.iter()
            .map(|v| v.get_type().into())
            .collect();

        let fn_type = if let Some(rt) = ret_type {
            rt.fn_type(&param_types, false)
        } else {
            self.context.void_type().fn_type(&param_types, false)
        };

        let func = self.module.get_function(runtime_name).unwrap_or_else(|| {
            self.module.add_function(runtime_name, fn_type, Some(inkwell::module::Linkage::External))
        });

        let arg_vals: Vec<BasicMetadataValueEnum> = args.iter().map(|v| (*v).into()).collect();
        let call = self.builder.build_call(func, &arg_vals, runtime_name).unwrap();

        match call.try_as_basic_value().basic() {
            Some(val) => Ok(val),
            None => Ok(self.i32_const(0).into()),
        }
    }

    // -----------------------------------------------------------------------
    // Array methods
    // -----------------------------------------------------------------------

    fn emit_array_method(
        &mut self,
        object: &Spanned<Expr>,
        method: &str,
        args: &[Spanned<Expr>],
        elem_ty: &PdcType,
    ) -> Result<BasicValueEnum<'static>, PdcError> {
        let handle = self.emit_expr(object)?;
        let elem_size = pdc_type_byte_size(elem_ty);

        match method {
            "push" => {
                let val = self.emit_expr(&args[0])?;
                let val_ty = self.node_type(args[0].id).clone();
                let converted = self.convert_value(val, &val_ty, elem_ty);
                let store_val = self.float_to_int_if_needed(converted, elem_ty);
                let push_name = format!("pdc_array_push_{elem_size}");
                self.emit_runtime_call_raw(&push_name, &[self.ctx_ptr.into(), handle, store_val], None)?;
                Ok(self.i32_const(0).into())
            }
            "len" => {
                self.emit_runtime_call_raw("pdc_array_len", &[self.ctx_ptr.into(), handle], Some(self.context.i32_type().into()))
            }
            "get" => {
                let idx = self.emit_expr(&args[0])?;
                let get_name = format!("pdc_array_get_{elem_size}");
                let int_type: BasicTypeEnum = match elem_size { 1 => self.context.i8_type().into(), 2 => self.context.i16_type().into(), 4 => self.context.i32_type().into(), _ => self.context.i64_type().into() };
                let raw = self.emit_runtime_call_raw(&get_name, &[self.ctx_ptr.into(), handle, idx], Some(int_type))?;
                Ok(self.int_to_float_if_needed(raw, elem_ty))
            }
            "set" => {
                let idx = self.emit_expr(&args[0])?;
                let val = self.emit_expr(&args[1])?;
                let val_ty = self.node_type(args[1].id).clone();
                let converted = self.convert_value(val, &val_ty, elem_ty);
                let store_val = self.float_to_int_if_needed(converted, elem_ty);
                let set_name = format!("pdc_array_set_{elem_size}");
                self.emit_runtime_call_raw(&set_name, &[self.ctx_ptr.into(), handle, idx, store_val], None)?;
                Ok(self.i32_const(0).into())
            }
            "map" => {
                self.emit_array_map(object, &args[0], elem_ty)
            }
            "slice" => {
                let start_val = self.emit_expr(&args[0])?;
                let end_val = self.emit_expr(&args[1])?;
                let len_val = self.builder.build_int_sub(end_val.into_int_value(), start_val.into_int_value(), "sl_len").unwrap();

                // Allocate 12-byte struct: (handle: i32, start: i32, len: i32)
                let i32_ty = self.context.i32_type();
                let slice_ty = self.context.struct_type(&[i32_ty.into(), i32_ty.into(), i32_ty.into()], false);
                let alloca = self.builder.build_alloca(slice_ty, "slice").unwrap();

                let i8_ty = self.context.i8_type();
                // Store handle at offset 0
                self.builder.build_store(alloca, handle.into_int_value()).unwrap();
                // Store start at offset 4
                let start_ptr = unsafe {
                    self.builder.build_gep(i8_ty, alloca,
                        &[self.context.i64_type().const_int(4, false).into()], "sl_start_ptr").unwrap()
                };
                self.builder.build_store(start_ptr, start_val.into_int_value()).unwrap();
                // Store len at offset 8
                let len_ptr = unsafe {
                    self.builder.build_gep(i8_ty, alloca,
                        &[self.context.i64_type().const_int(8, false).into()], "sl_len_ptr").unwrap()
                };
                self.builder.build_store(len_ptr, len_val).unwrap();

                Ok(alloca.into())
            }
            _ => Err(PdcError::Codegen {
                message: format!("unknown array method '{method}'"),
            }),
        }
    }

    fn emit_array_broadcast(
        &mut self,
        op: BinOp,
        left: &Spanned<Expr>,
        right: &Spanned<Expr>,
        lt: &PdcType,
        rt: &PdcType,
        result_ty: &PdcType,
    ) -> Result<BasicValueEnum<'static>, PdcError> {
        let result_elem_ty = match result_ty {
            PdcType::Array(et) => *et.clone(),
            _ => return Err(PdcError::Codegen { message: "broadcast result must be array".into() }),
        };
        let result_elem_size = pdc_type_byte_size(&result_elem_ty);

        let l_is_arr = matches!(lt, PdcType::Array(_));
        let r_is_arr = matches!(rt, PdcType::Array(_));

        let lval = self.emit_expr(left)?;
        let rval = self.emit_expr(right)?;

        let arr_handle = if l_is_arr { lval } else { rval };
        let len_val = self.emit_runtime_call_raw("pdc_array_len", &[self.ctx_ptr.into(), arr_handle], Some(self.context.i32_type().into()))?.into_int_value();

        let size_val = self.i32_const(result_elem_size as i64);
        let new_arr = self.emit_runtime_call_raw("pdc_array_new", &[self.ctx_ptr.into(), size_val.into()], Some(self.context.i32_type().into()))?;

        let l_elem_ty = match lt { PdcType::Array(et) => *et.clone(), _ => lt.clone() };
        let r_elem_ty = match rt { PdcType::Array(et) => *et.clone(), _ => rt.clone() };
        let op_result_ty = if result_elem_ty == PdcType::Bool && !matches!(op, BinOp::And | BinOp::Or) {
            if l_elem_ty.is_float() || r_elem_ty.is_float() { PdcType::F64 } else { PdcType::I32 }
        } else {
            result_elem_ty.clone()
        };

        let idx_name = "__broadcast_idx";
        self.define_variable(idx_name, &PdcType::I32, self.i32_const(0).into());

        let header = self.context.append_basic_block(self.function, "bc_hdr");
        let body_bb = self.context.append_basic_block(self.function, "bc_body");
        let exit = self.context.append_basic_block(self.function, "bc_exit");

        self.builder.build_unconditional_branch(header).unwrap();

        self.builder.position_at_end(header);
        let idx_val = self.load_variable(idx_name)?.into_int_value();
        let cond = self.builder.build_int_compare(IntPredicate::SLT, idx_val, len_val, "bc_cond").unwrap();
        self.builder.build_conditional_branch(cond, body_bb, exit).unwrap();

        self.builder.position_at_end(body_bb);

        let l_elem = if l_is_arr {
            let l_elem_size = pdc_type_byte_size(&l_elem_ty);
            let get_name = format!("pdc_array_get_{l_elem_size}");
            let int_type: BasicTypeEnum = match l_elem_size { 1 => self.context.i8_type().into(), 2 => self.context.i16_type().into(), 4 => self.context.i32_type().into(), _ => self.context.i64_type().into() };
            let idx_for_get = self.load_variable(idx_name)?;
            let raw = self.emit_runtime_call_raw(&get_name, &[self.ctx_ptr.into(), lval, idx_for_get], Some(int_type))?;
            self.int_to_float_if_needed(raw, &l_elem_ty)
        } else {
            lval
        };

        let r_elem = if r_is_arr {
            let r_elem_size = pdc_type_byte_size(&r_elem_ty);
            let get_name = format!("pdc_array_get_{r_elem_size}");
            let int_type: BasicTypeEnum = match r_elem_size { 1 => self.context.i8_type().into(), 2 => self.context.i16_type().into(), 4 => self.context.i32_type().into(), _ => self.context.i64_type().into() };
            let idx_for_get = self.load_variable(idx_name)?;
            let raw = self.emit_runtime_call_raw(&get_name, &[self.ctx_ptr.into(), rval, idx_for_get], Some(int_type))?;
            self.int_to_float_if_needed(raw, &r_elem_ty)
        } else {
            rval
        };

        let elem_result = self.emit_binary_op(op, l_elem, r_elem, &l_elem_ty, &r_elem_ty, &op_result_ty)?;

        let final_val = if result_elem_ty == PdcType::Bool {
            elem_result
        } else {
            self.convert_value(elem_result, &op_result_ty, &result_elem_ty)
        };

        let store_val = self.float_to_int_if_needed(final_val, &result_elem_ty);
        let push_name = format!("pdc_array_push_{result_elem_size}");
        self.emit_runtime_call_raw(&push_name, &[self.ctx_ptr.into(), new_arr, store_val], None)?;

        let idx_val = self.load_variable(idx_name)?.into_int_value();
        let next = self.builder.build_int_add(idx_val, self.i32_const(1), "bc_inc").unwrap();
        self.store_variable(idx_name, next.into())?;
        self.builder.build_unconditional_branch(header).unwrap();

        self.builder.position_at_end(exit);

        Ok(new_arr)
    }

    fn emit_array_map(
        &mut self,
        array_expr: &Spanned<Expr>,
        fn_ref_expr: &Spanned<Expr>,
        elem_ty: &PdcType,
    ) -> Result<BasicValueEnum<'static>, PdcError> {
        let arr_handle = self.emit_expr(array_expr)?;

        let fn_name = match &fn_ref_expr.node {
            Expr::Variable(name) => name.clone(),
            _ => return Err(PdcError::Codegen { message: "map() argument must be a function name".into() }),
        };

        let result_ty = self.node_type(fn_ref_expr.id).clone();
        let ret_ty = match &result_ty {
            PdcType::FnRef { ret, .. } => *ret.clone(),
            _ => elem_ty.clone(),
        };
        let ret_size = pdc_type_byte_size(&ret_ty);

        let size_val = self.i32_const(ret_size as i64);
        let new_arr = self.emit_runtime_call_raw("pdc_array_new", &[self.ctx_ptr.into(), size_val.into()], Some(self.context.i32_type().into()))?;
        let len_val = self.emit_runtime_call_raw("pdc_array_len", &[self.ctx_ptr.into(), arr_handle], Some(self.context.i32_type().into()))?.into_int_value();

        let idx_name = "__map_idx";
        self.define_variable(idx_name, &PdcType::I32, self.i32_const(0).into());

        let header = self.context.append_basic_block(self.function, "map_hdr");
        let body_bb = self.context.append_basic_block(self.function, "map_body");
        let exit = self.context.append_basic_block(self.function, "map_exit");

        self.builder.build_unconditional_branch(header).unwrap();

        self.builder.position_at_end(header);
        let idx_val = self.load_variable(idx_name)?.into_int_value();
        let cond = self.builder.build_int_compare(IntPredicate::SLT, idx_val, len_val, "map_cond").unwrap();
        self.builder.build_conditional_branch(cond, body_bb, exit).unwrap();

        self.builder.position_at_end(body_bb);
        let elem_size = pdc_type_byte_size(elem_ty);
        let get_name = format!("pdc_array_get_{elem_size}");
        let int_type: BasicTypeEnum = match elem_size { 1 => self.context.i8_type().into(), 2 => self.context.i16_type().into(), 4 => self.context.i32_type().into(), _ => self.context.i64_type().into() };
        let idx_for_get = self.load_variable(idx_name)?;
        let raw = self.emit_runtime_call_raw(&get_name, &[self.ctx_ptr.into(), arr_handle, idx_for_get], Some(int_type))?;
        let elem_val = self.int_to_float_if_needed(raw, elem_ty);

        let converted = self.convert_value(elem_val, elem_ty, &PdcType::F64);
        let fn_result = self.emit_fn_ref_call(&fn_name, converted, elem_ty, &ret_ty)?;

        let store_val = self.float_to_int_if_needed(fn_result, &ret_ty);
        let push_name = format!("pdc_array_push_{ret_size}");
        self.emit_runtime_call_raw(&push_name, &[self.ctx_ptr.into(), new_arr, store_val], None)?;

        let idx_val = self.load_variable(idx_name)?.into_int_value();
        let next = self.builder.build_int_add(idx_val, self.i32_const(1), "map_inc").unwrap();
        self.store_variable(idx_name, next.into())?;
        self.builder.build_unconditional_branch(header).unwrap();

        self.builder.position_at_end(exit);
        Ok(new_arr)
    }

    fn emit_fn_ref_call(
        &mut self,
        fn_name: &str,
        arg: BasicValueEnum<'static>,
        _arg_ty: &PdcType,
        ret_ty: &PdcType,
    ) -> Result<BasicValueEnum<'static>, PdcError> {
        let resolved = if let Some(qualified) = self.fn_aliases.get(fn_name) {
            qualified.clone()
        } else {
            fn_name.to_string()
        };

        if let Some(overloads) = self.user_fns.get(&resolved) {
            let overloads = overloads.clone();
            let (overload_idx, _) = overloads.sigs.iter().enumerate()
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
            let func = self.user_fn_llvm[&mangled];
            if let Some(size) = self.compound_type_size(ret_ty) {
                let f64_ty = self.context.f64_type();
                let arr_ty = f64_ty.array_type(size / 8);
                let out_ptr = self.builder.build_alloca(arr_ty, "sret_out").unwrap();
                self.builder.build_call(func, &[self.ctx_ptr.into(), out_ptr.into(), arg.into()], "fnref").unwrap();
                return Ok(out_ptr.into());
            }
            let call = self.builder.build_call(func, &[self.ctx_ptr.into(), arg.into()], "fnref").unwrap();
            return match call.try_as_basic_value().basic() {
                Some(val) => Ok(val),
                None => Ok(self.i32_const(0).into()),
            };
        }

        let runtime_name = format!("pdc_{fn_name}");
        let ret_llvm = if *ret_ty != PdcType::Void {
            Some(pdc_type_to_llvm(ret_ty, self.context))
        } else {
            None
        };
        self.emit_runtime_call_raw(&runtime_name, &[arg], ret_llvm)
    }

    // -----------------------------------------------------------------------
    // Struct/enum construction
    // -----------------------------------------------------------------------

    fn emit_struct_construct_from_call(
        &mut self,
        name: &str,
        args: &[Spanned<Expr>],
        arg_names: &[Option<String>],
        info: &StructInfo,
    ) -> Result<BasicValueEnum<'static>, PdcError> {
        let f64_ty = self.context.f64_type();
        let arr_ty = f64_ty.array_type(info.fields.len() as u32);
        let alloca = self.builder.build_alloca(arr_ty, "struct_call").unwrap();

        for (i, arg) in args.iter().enumerate() {
            let val = self.emit_expr(arg)?;
            let arg_ty = self.node_type(arg.id).clone();
            let fname = arg_names[i].as_ref().unwrap();
            let field_idx = info.fields.iter().position(|(n, _)| n == fname)
                .ok_or_else(|| PdcError::Codegen {
                    message: format!("struct '{name}' has no field '{fname}'"),
                })?;
            let field_ty = &info.fields[field_idx].1;
            let converted = self.convert_value(val, &arg_ty, field_ty);
            let store_val = self.widen_to_f64(converted, field_ty);
            let field_ptr = unsafe {
                self.builder.build_gep(f64_ty, alloca,
                    &[self.i32_const(field_idx as i64).into()], "field_gep").unwrap()
            };
            self.builder.build_store(field_ptr, store_val).unwrap();
        }

        Ok(alloca.into())
    }

    fn emit_enum_construct(
        &mut self,
        enum_name: &str,
        variant_name: &str,
        args: &[Spanned<Expr>],
    ) -> Result<BasicValueEnum<'static>, PdcError> {
        let info = self.enums.get(enum_name).ok_or_else(|| PdcError::Codegen {
            message: format!("undefined enum '{enum_name}'"),
        })?.clone();

        let variant_idx = info.variants.iter().position(|v| v.name == variant_name)
            .ok_or_else(|| PdcError::Codegen {
                message: format!("enum '{enum_name}' has no variant '{variant_name}'"),
            })?;

        let max_fields = info.variants.iter().map(|v| v.field_types.len()).max().unwrap_or(0);
        let total_bytes = 8 + max_fields * 8; // 8 for tag+pad, 8 per field

        let i8_ty = self.context.i8_type();
        let arr_ty = i8_ty.array_type(total_bytes as u32);
        let alloca = self.builder.build_alloca(arr_ty, "enum").unwrap();

        // Write tag at offset 0
        let tag_val = self.i32_const(variant_idx as i64);
        self.builder.build_store(alloca, tag_val).unwrap();

        // Write fields at offsets 8, 16, 24, ...
        for (i, arg) in args.iter().enumerate() {
            let val = self.emit_expr(arg)?;
            let arg_ty = self.node_type(arg.id).clone();
            let store_val = self.widen_to_f64(val, &arg_ty);
            let byte_offset = (8 + i * 8) as u64;
            let field_ptr = unsafe {
                self.builder.build_gep(i8_ty, alloca,
                    &[self.context.i64_type().const_int(byte_offset, false).into()], "enum_field_gep").unwrap()
            };
            self.builder.build_store(field_ptr, store_val).unwrap();
        }

        Ok(alloca.into())
    }

    fn emit_operator_call(
        &mut self,
        op_name: &str,
        overload_idx: usize,
        args: &[BasicValueEnum<'static>],
        result_type: &PdcType,
    ) -> Result<BasicValueEnum<'static>, PdcError> {
        let func = *self.op_fn_map.get(&(op_name.to_string(), overload_idx))
            .ok_or_else(|| PdcError::Codegen {
                message: format!("operator function '{}' overload {} not found", op_name, overload_idx),
            })?;

        let sret_alloca = if let Some(size) = self.compound_type_size(result_type) {
            let f64_ty = self.context.f64_type();
            let arr_ty = f64_ty.array_type(size / 8);
            Some(self.builder.build_alloca(arr_ty, "sret_out").unwrap())
        } else {
            None
        };

        let mut call_args: Vec<BasicMetadataValueEnum> = vec![self.ctx_ptr.into()];
        if let Some(out_ptr) = sret_alloca {
            call_args.push(out_ptr.into());
        }
        for arg in args {
            call_args.push((*arg).into());
        }

        if let Some(out_ptr) = sret_alloca {
            self.builder.build_call(func, &call_args, "op").unwrap();
            Ok(out_ptr.into())
        } else {
            let call = self.builder.build_call(func, &call_args, "op").unwrap();
            match call.try_as_basic_value().basic() {
                Some(val) => Ok(val),
                None => Ok(self.i32_const(0).into()),
            }
        }
    }

    // -----------------------------------------------------------------------
    // Binary operations
    // -----------------------------------------------------------------------

    fn emit_binary_op(
        &mut self,
        op: BinOp,
        lval: BasicValueEnum<'static>,
        rval: BasicValueEnum<'static>,
        lt: &PdcType,
        rt: &PdcType,
        result_ty: &PdcType,
    ) -> Result<BasicValueEnum<'static>, PdcError> {
        // String operations
        if *lt == PdcType::Str && *rt == PdcType::Str {
            return match op {
                BinOp::Add => {
                    self.emit_runtime_call_raw("pdc_string_concat", &[self.ctx_ptr.into(), lval, rval], Some(self.context.i32_type().into()))
                }
                BinOp::Eq => {
                    self.emit_runtime_call_raw("pdc_string_eq", &[self.ctx_ptr.into(), lval, rval], Some(self.context.i8_type().into()))
                }
                BinOp::NotEq => {
                    let eq = self.emit_runtime_call_raw("pdc_string_eq", &[self.ctx_ptr.into(), lval, rval], Some(self.context.i8_type().into()))?;
                    let one = self.i8_const(1);
                    Ok(self.builder.build_xor(eq.into_int_value(), one, "neq").unwrap().into())
                }
                _ => Err(PdcError::Codegen { message: "unsupported string operator".into() }),
            };
        }

        // Logical
        if matches!(op, BinOp::And | BinOp::Or) {
            let lval = self.convert_value(lval, lt, result_ty);
            let rval = self.convert_value(rval, rt, result_ty);
            return Ok(match op {
                BinOp::And => self.builder.build_and(lval.into_int_value(), rval.into_int_value(), "and").unwrap().into(),
                BinOp::Or => self.builder.build_or(lval.into_int_value(), rval.into_int_value(), "or").unwrap().into(),
                _ => unreachable!(),
            });
        }

        // Comparison
        if result_ty == &PdcType::Bool {
            let cmp_type = if lt.is_float() || rt.is_float() {
                if lt == &PdcType::F32 && rt == &PdcType::F32 { PdcType::F32 } else { PdcType::F64 }
            } else {
                PdcType::I32
            };
            // Convert operands from their original types to the comparison type,
            // not from result_ty (Bool), to avoid double-conversion.
            let lval = self.convert_value(lval, lt, &cmp_type);
            let rval = self.convert_value(rval, rt, &cmp_type);

            return if cmp_type.is_float() {
                let pred = match op {
                    BinOp::Eq => FloatPredicate::OEQ,
                    BinOp::NotEq => FloatPredicate::ONE,
                    BinOp::Lt => FloatPredicate::OLT,
                    BinOp::LtEq => FloatPredicate::OLE,
                    BinOp::Gt => FloatPredicate::OGT,
                    BinOp::GtEq => FloatPredicate::OGE,
                    _ => unreachable!(),
                };
                let cmp = self.builder.build_float_compare(pred, lval.into_float_value(), rval.into_float_value(), "fcmp").unwrap();
                // i1 -> i8
                Ok(self.builder.build_int_z_extend(cmp, self.context.i8_type(), "zext").unwrap().into())
            } else {
                let pred = match op {
                    BinOp::Eq => IntPredicate::EQ,
                    BinOp::NotEq => IntPredicate::NE,
                    BinOp::Lt => IntPredicate::SLT,
                    BinOp::LtEq => IntPredicate::SLE,
                    BinOp::Gt => IntPredicate::SGT,
                    BinOp::GtEq => IntPredicate::SGE,
                    _ => unreachable!(),
                };
                let cmp = self.builder.build_int_compare(pred, lval.into_int_value(), rval.into_int_value(), "icmp").unwrap();
                Ok(self.builder.build_int_z_extend(cmp, self.context.i8_type(), "zext").unwrap().into())
            };
        }

        // For arithmetic and bitwise operators, convert operands to result type.
        let lval = self.convert_value(lval, lt, result_ty);
        let rval = self.convert_value(rval, rt, result_ty);

        // Bitwise
        match op {
            BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor | BinOp::Shl | BinOp::Shr => {
                return Ok(match op {
                    BinOp::BitAnd => self.builder.build_and(lval.into_int_value(), rval.into_int_value(), "band").unwrap().into(),
                    BinOp::BitOr => self.builder.build_or(lval.into_int_value(), rval.into_int_value(), "bor").unwrap().into(),
                    BinOp::BitXor => self.builder.build_xor(lval.into_int_value(), rval.into_int_value(), "bxor").unwrap().into(),
                    BinOp::Shl => self.builder.build_left_shift(lval.into_int_value(), rval.into_int_value(), "shl").unwrap().into(),
                    BinOp::Shr => self.builder.build_right_shift(lval.into_int_value(), rval.into_int_value(), true, "shr").unwrap().into(),
                    _ => unreachable!(),
                });
            }
            _ => {}
        }

        // Arithmetic
        if result_ty.is_float() {
            Ok(match op {
                BinOp::Add => self.builder.build_float_add(lval.into_float_value(), rval.into_float_value(), "fadd").unwrap().into(),
                BinOp::Sub => self.builder.build_float_sub(lval.into_float_value(), rval.into_float_value(), "fsub").unwrap().into(),
                BinOp::Mul => self.builder.build_float_mul(lval.into_float_value(), rval.into_float_value(), "fmul").unwrap().into(),
                BinOp::Div => self.builder.build_float_div(lval.into_float_value(), rval.into_float_value(), "fdiv").unwrap().into(),
                BinOp::Mod => {
                    // floor(a/b) * b, then a - that
                    let div = self.builder.build_float_div(lval.into_float_value(), rval.into_float_value(), "div").unwrap();
                    let floored = self.call_f64_intrinsic("llvm.floor.f64", div);
                    let prod = self.builder.build_float_mul(floored, rval.into_float_value(), "prod").unwrap();
                    self.builder.build_float_sub(lval.into_float_value(), prod, "fmod").unwrap().into()
                }
                BinOp::Pow => {
                    self.emit_runtime_call_raw("pdc_pow", &[lval, rval], Some(self.context.f64_type().into()))?
                }
                _ => unreachable!(),
            })
        } else {
            Ok(match op {
                BinOp::Add => self.builder.build_int_add(lval.into_int_value(), rval.into_int_value(), "iadd").unwrap().into(),
                BinOp::Sub => self.builder.build_int_sub(lval.into_int_value(), rval.into_int_value(), "isub").unwrap().into(),
                BinOp::Mul => self.builder.build_int_mul(lval.into_int_value(), rval.into_int_value(), "imul").unwrap().into(),
                BinOp::Div => self.builder.build_int_signed_div(lval.into_int_value(), rval.into_int_value(), "sdiv").unwrap().into(),
                BinOp::Mod => self.builder.build_int_signed_rem(lval.into_int_value(), rval.into_int_value(), "srem").unwrap().into(),
                BinOp::Pow => {
                    let lf = self.builder.build_signed_int_to_float(lval.into_int_value(), self.context.f64_type(), "i2f").unwrap();
                    let rf = self.builder.build_signed_int_to_float(rval.into_int_value(), self.context.f64_type(), "i2f").unwrap();
                    let result_f64 = self.emit_runtime_call_raw("pdc_pow", &[lf.into(), rf.into()], Some(self.context.f64_type().into()))?;
                    let target_ty = pdc_type_to_llvm(result_ty, self.context).into_int_type();
                    self.builder.build_float_to_signed_int(result_f64.into_float_value(), target_ty, "f2i").unwrap().into()
                }
                _ => unreachable!(),
            })
        }
    }

    // -----------------------------------------------------------------------
    // Type conversions
    // -----------------------------------------------------------------------

    fn convert_value(&mut self, val: BasicValueEnum<'static>, from: &PdcType, to: &PdcType) -> BasicValueEnum<'static> {
        if from == to { return val; }

        // Bool -> float: zero-extend to i32, then int-to-float
        if *from == PdcType::Bool && to.is_float() {
            let i32_val = self.builder.build_int_z_extend(val.into_int_value(), self.context.i32_type(), "zext").unwrap();
            let f64_val = self.builder.build_signed_int_to_float(i32_val, self.context.f64_type(), "b2f").unwrap();
            return if *to == PdcType::F32 {
                self.builder.build_float_trunc(f64_val, self.context.f32_type(), "fdem").unwrap().into()
            } else {
                f64_val.into()
            };
        }

        // Float -> bool: compare != 0.0
        if from.is_float() && *to == PdcType::Bool {
            let f64_val = if *from == PdcType::F32 {
                self.builder.build_float_ext(val.into_float_value(), self.context.f64_type(), "fprom").unwrap()
            } else {
                val.into_float_value()
            };
            let zero = self.context.f64_type().const_float(0.0);
            let cmp = self.builder.build_float_compare(
                inkwell::FloatPredicate::ONE, f64_val, zero, "f2b",
            ).unwrap();
            return self.builder.build_int_z_extend(cmp, self.context.i8_type(), "zext").unwrap().into();
        }

        // Bool -> int: zero-extend
        if *from == PdcType::Bool && to.is_int() {
            let to_ty = pdc_type_to_llvm(to, self.context).into_int_type();
            return self.builder.build_int_z_extend(val.into_int_value(), to_ty, "zext").unwrap().into();
        }

        // Int -> bool: compare != 0
        if from.is_int() && *to == PdcType::Bool {
            let zero = pdc_type_to_llvm(from, self.context).into_int_type().const_zero();
            let cmp = self.builder.build_int_compare(
                inkwell::IntPredicate::NE, val.into_int_value(), zero, "i2b",
            ).unwrap();
            return self.builder.build_int_z_extend(cmp, self.context.i8_type(), "zext").unwrap().into();
        }

        // Float -> float
        if from.is_float() && to.is_float() {
            return if *to == PdcType::F64 {
                self.builder.build_float_ext(val.into_float_value(), self.context.f64_type(), "fprom").unwrap().into()
            } else {
                self.builder.build_float_trunc(val.into_float_value(), self.context.f32_type(), "fdem").unwrap().into()
            };
        }

        // Int -> float
        if from.is_int() && to.is_float() {
            let from_ty = pdc_type_to_llvm(from, self.context).into_int_type();
            let wide = if from_ty.get_bit_width() < 64 {
                self.builder.build_int_s_extend(val.into_int_value(), self.context.i64_type(), "sext").unwrap()
            } else {
                val.into_int_value()
            };
            let f64_val = self.builder.build_signed_int_to_float(wide, self.context.f64_type(), "i2f").unwrap();
            return if *to == PdcType::F32 {
                self.builder.build_float_trunc(f64_val, self.context.f32_type(), "fdem").unwrap().into()
            } else {
                f64_val.into()
            };
        }

        // Float -> int
        if from.is_float() && to.is_int() {
            let f64_val = if *from == PdcType::F32 {
                self.builder.build_float_ext(val.into_float_value(), self.context.f64_type(), "fprom").unwrap()
            } else {
                val.into_float_value()
            };
            let i64_val = self.builder.build_float_to_signed_int(f64_val, self.context.i64_type(), "f2i").unwrap();
            let to_ty = pdc_type_to_llvm(to, self.context).into_int_type();
            return if to_ty.get_bit_width() < 64 {
                self.builder.build_int_truncate(i64_val, to_ty, "trunc").unwrap().into()
            } else {
                i64_val.into()
            };
        }

        // Int -> int
        if from.is_int() && to.is_int() {
            let from_ty = pdc_type_to_llvm(from, self.context).into_int_type();
            let to_ty = pdc_type_to_llvm(to, self.context).into_int_type();
            return if to_ty.get_bit_width() > from_ty.get_bit_width() {
                self.builder.build_int_s_extend(val.into_int_value(), to_ty, "sext").unwrap().into()
            } else if to_ty.get_bit_width() < from_ty.get_bit_width() {
                self.builder.build_int_truncate(val.into_int_value(), to_ty, "trunc").unwrap().into()
            } else {
                val
            };
        }

        val
    }

    fn widen_to_f64(&mut self, val: BasicValueEnum<'static>, ty: &PdcType) -> FloatValue<'static> {
        match ty {
            PdcType::F64 => val.into_float_value(),
            PdcType::F32 => self.builder.build_float_ext(val.into_float_value(), self.context.f64_type(), "fprom").unwrap(),
            PdcType::I32 | PdcType::U32 | PdcType::PathHandle | PdcType::BufferHandle | PdcType::KernelHandle => {
                self.builder.build_signed_int_to_float(val.into_int_value(), self.context.f64_type(), "i2f").unwrap()
            }
            PdcType::Bool => {
                let i32_val = self.builder.build_int_z_extend(val.into_int_value(), self.context.i32_type(), "zext").unwrap();
                self.builder.build_signed_int_to_float(i32_val, self.context.f64_type(), "i2f").unwrap()
            }
            _ => val.into_float_value(),
        }
    }

    fn narrow_from_f64(&mut self, val: FloatValue<'static>, ty: &PdcType) -> BasicValueEnum<'static> {
        match ty {
            PdcType::F64 => val.into(),
            PdcType::F32 => self.builder.build_float_trunc(val, self.context.f32_type(), "fdem").unwrap().into(),
            PdcType::I32 | PdcType::U32 | PdcType::PathHandle | PdcType::BufferHandle | PdcType::KernelHandle => {
                self.builder.build_float_to_signed_int(val, self.context.i32_type(), "f2i").unwrap().into()
            }
            PdcType::Bool => {
                let i32_val = self.builder.build_float_to_signed_int(val, self.context.i32_type(), "f2i").unwrap();
                self.builder.build_int_truncate(i32_val, self.context.i8_type(), "trunc").unwrap().into()
            }
            _ => val.into(),
        }
    }

    fn float_to_int_if_needed(&self, val: BasicValueEnum<'static>, ty: &PdcType) -> BasicValueEnum<'static> {
        match ty {
            PdcType::F32 => self.builder.build_bit_cast(val, self.context.i32_type(), "f2i_bc").unwrap(),
            PdcType::F64 => self.builder.build_bit_cast(val, self.context.i64_type(), "f2i_bc").unwrap(),
            _ => val,
        }
    }

    fn int_to_float_if_needed(&self, val: BasicValueEnum<'static>, ty: &PdcType) -> BasicValueEnum<'static> {
        match ty {
            PdcType::F32 => self.builder.build_bit_cast(val, self.context.f32_type(), "i2f_bc").unwrap(),
            PdcType::F64 => self.builder.build_bit_cast(val, self.context.f64_type(), "i2f_bc").unwrap(),
            _ => val,
        }
    }

    fn to_i1(&self, val: IntValue<'static>) -> IntValue<'static> {
        let zero = val.get_type().const_zero();
        self.builder.build_int_compare(IntPredicate::NE, val, zero, "to_i1").unwrap()
    }

    /// Copy compound data (struct/tuple) from src to dst pointer, word by word.
    fn copy_compound_to_ptr(&self, src_ptr: PointerValue<'static>, dst_ptr: PointerValue<'static>, size_bytes: u32) {
        let f64_ty = self.context.f64_type();
        let num_words = size_bytes / 8;
        for i in 0..num_words {
            let src_gep = unsafe {
                self.builder.build_gep(f64_ty, src_ptr,
                    &[self.i32_const(i as i64).into()], "copy_src_gep").unwrap()
            };
            let val = self.builder.build_load(f64_ty, src_gep, "copy_val").unwrap();
            let dst_gep = unsafe {
                self.builder.build_gep(f64_ty, dst_ptr,
                    &[self.i32_const(i as i64).into()], "copy_dst_gep").unwrap()
            };
            self.builder.build_store(dst_gep, val).unwrap();
        }
    }

    fn compound_type_size(&self, ty: &PdcType) -> Option<u32> {
        match ty {
            PdcType::Struct(name) => {
                self.structs.get(name).map(|info| (info.fields.len() * 8) as u32)
            }
            PdcType::Tuple(elems) => Some((elems.len() * 8) as u32),
            _ => None,
        }
    }

    fn call_f64_intrinsic(&self, name: &str, val: FloatValue<'static>) -> FloatValue<'static> {
        let f64_ty = self.context.f64_type();
        let intrinsic = inkwell::intrinsics::Intrinsic::find(name)
            .unwrap_or_else(|| panic!("intrinsic {name} not found"));
        let decl = intrinsic.get_declaration(self.module, &[f64_ty.into()]).unwrap();
        self.builder.build_call(decl, &[val.into()], "intrinsic")
            .unwrap()
            .try_as_basic_value()
            .basic()
            .unwrap()
            .into_float_value()
    }
}
