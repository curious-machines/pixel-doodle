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
use super::type_check::{EnumInfo, StructInfo, UserFnSig};

/// Compiled PDC function type.
pub type PdcSceneFn = unsafe extern "C" fn(*mut runtime::PdcContext);

/// JIT-compiled PDC program.
pub struct CompiledProgram {
    pub fn_ptr: PdcSceneFn,
    _module: JITModule,
}

pub struct BuiltinInfo {
    pub offset: usize,
    pub ty: PdcType,
}

pub fn compile(
    program: &Program,
    types: &[PdcType],
    builtins_layout: &[(&str, PdcType)],
    user_fns: &HashMap<String, UserFnSig>,
    structs: &HashMap<String, StructInfo>,
    enums: &HashMap<String, EnumInfo>,
) -> Result<CompiledProgram, PdcError> {
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
    let mut fn_defs: Vec<&FnDef> = Vec::new();
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

    for stmt in &program.stmts {
        match &stmt.node {
            Stmt::FnDef(fndef) => fn_defs.push(fndef),
            Stmt::ConstDecl { value, .. } if is_pure_literal(&value.node) => {
                top_level_consts.push(stmt);
            }
            _ => {}
        }
    }

    for fndef in &fn_defs {
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

        let func_id = jit_module
            .declare_function(&format!("pdc_userfn_{}", fndef.name), Linkage::Local, &sig)
            .map_err(|e| PdcError::Codegen {
                message: format!("declare user fn: {e}"),
            })?;
        user_fn_ids.insert(fndef.name.clone(), func_id);
    }

    // Compile each user function
    for fndef in &fn_defs {
        let func_id = user_fn_ids[&fndef.name];
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
            struct_vars: HashMap::new(),
            block_terminated: false,
        };

        // Define parameters as variables
        for (i, param) in fndef.params.iter().enumerate() {
            let val = cg.builder.block_params(entry)[i + 1]; // +1 for ctx_ptr
            let var = cg.new_variable(&param.name, &param.ty);
            cg.builder.def_var(var, val);
        }

        // Emit top-level constants so they're accessible in this function
        for const_stmt in &top_level_consts {
            cg.emit_stmt(const_stmt)?;
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
            struct_vars: HashMap::new(),
            block_terminated: false,
        };

        for stmt in &program.stmts {
            if matches!(&stmt.node, Stmt::FnDef(_) | Stmt::Import { .. } | Stmt::StructDef(_)) {
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

    jit_module.finalize_definitions().map_err(|e| PdcError::Codegen {
        message: format!("finalize: {e}"),
    })?;

    let code_ptr = jit_module.get_finalized_function(main_id);
    let fn_ptr: PdcSceneFn = unsafe { std::mem::transmute(code_ptr) };

    Ok(CompiledProgram {
        fn_ptr,
        _module: jit_module,
    })
}

fn build_builtin_map(layout: &[(&str, PdcType)]) -> HashMap<String, BuiltinInfo> {
    layout
        .iter()
        .enumerate()
        .map(|(i, (name, ty))| {
            (
                name.to_string(),
                BuiltinInfo {
                    offset: i,
                    ty: ty.clone(),
                },
            )
        })
        .collect()
}

fn pdc_type_to_cl(ty: &PdcType, pointer_type: cranelift_codegen::ir::Type) -> cranelift_codegen::ir::Type {
    match ty {
        PdcType::F32 => F32,
        PdcType::F64 => F64,
        PdcType::I8 | PdcType::U8 | PdcType::Bool => I8,
        PdcType::I16 | PdcType::U16 => I16,
        PdcType::I32 | PdcType::U32 => I32,
        PdcType::I64 | PdcType::U64 => I64,
        PdcType::PathHandle => I32,
        PdcType::Struct(_) => pointer_type, // structs are pointers to stack memory
        PdcType::Enum(_) => I32, // enums are u32 constants
        PdcType::Array(_) => I32, // arrays are handles (u32)
        PdcType::Void => I32,
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
    #[allow(dead_code)]
    user_fns: &'a HashMap<String, UserFnSig>,
    structs: &'a HashMap<String, StructInfo>,
    enums: &'a HashMap<String, EnumInfo>,
    /// Set to true after a terminator instruction (return, break, continue).
    block_terminated: bool,
}

impl<'a, 'b> CodegenCtx<'a, 'b> {
    fn new_variable(&mut self, name: &str, ty: &PdcType) -> Variable {
        let cl_type = pdc_type_to_cl(ty, self.pointer_type);
        let var = self.builder.declare_var(cl_type);
        self.variables.insert(name.to_string(), (var, ty.clone()));
        var
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
            Stmt::BuiltinDecl { name, ty } => {
                let info = self.builtin_map.get(name).ok_or_else(|| PdcError::Codegen {
                    message: format!("builtin '{name}' not found in layout"),
                })?;
                let offset = info.offset;
                let declared_ty = ty.clone();

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
                let val = self.convert_value(f64_val, &PdcType::F64, &declared_ty);
                let var = self.new_variable(name, &declared_ty);
                self.builder.def_var(var, val);
            }
            Stmt::ConstDecl { name, ty, value } | Stmt::VarDecl { name, ty, value } => {
                let val = self.emit_expr(value)?;
                let expr_ty = self.node_type(value.id).clone();
                let final_ty = ty.clone().unwrap_or(expr_ty.clone());
                // Struct and data-variant enum values are pointers to stack slots
                let is_pointer_type = match &final_ty {
                    PdcType::Struct(_) => true,
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
            Stmt::Assign { name, value } => {
                let val = self.emit_expr(value)?;
                let (var, var_ty) = self.variables.get(name).cloned().ok_or_else(|| {
                    PdcError::Codegen {
                        message: format!("undefined variable '{name}'"),
                    }
                })?;
                let expr_ty = self.node_type(value.id).clone();
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
            Stmt::For {
                var_name,
                start,
                end,
                body,
            } => {
                self.emit_for(var_name, start, end, body)?;
            }
            Stmt::Loop { body } => {
                self.emit_loop(body)?;
            }
            Stmt::Break => {
                // Jump to the loop exit block (stored in a side channel — simplified: use trap for now)
                // TODO: proper break/continue with block tracking
                self.builder.ins().trap(cranelift_codegen::ir::TrapCode::unwrap_user(1));
            }
            Stmt::Continue => {
                self.builder.ins().trap(cranelift_codegen::ir::TrapCode::unwrap_user(2));
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
            Stmt::FnDef(_) | Stmt::Import { .. } | Stmt::StructDef(_) | Stmt::EnumDef(_) => {
                // Already handled
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
        self.emit_block(body)?;
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
        body: &Block,
    ) -> Result<(), PdcError> {
        let start_val = self.emit_expr(start)?;
        let end_val = self.emit_expr(end)?;

        let var = self.new_variable(var_name, &PdcType::I32);
        self.builder.def_var(var, start_val);

        let header_block = self.builder.create_block();
        let body_block = self.builder.create_block();
        let exit_block = self.builder.create_block();

        self.builder.ins().jump(header_block, &[]);

        // Header: check i < end
        self.builder.switch_to_block(header_block);
        let i_val = self.builder.use_var(var);
        let cond = self.builder.ins().icmp(IntCC::SignedLessThan, i_val, end_val);
        self.builder.ins().brif(cond, body_block, &[], exit_block, &[]);

        // Body
        self.builder.switch_to_block(body_block);
        self.builder.seal_block(body_block);
        self.block_terminated = false;
        self.emit_block(body)?;
        if !self.block_terminated {
            // Increment
            let i_val = self.builder.use_var(var);
            let one = self.builder.ins().iconst(I32, 1);
            let next = self.builder.ins().iadd(i_val, one);
            self.builder.def_var(var, next);
            self.builder.ins().jump(header_block, &[]);
        }

        self.builder.seal_block(header_block);
        self.builder.switch_to_block(exit_block);
        self.builder.seal_block(exit_block);
        self.block_terminated = false;
        Ok(())
    }

    fn emit_loop(&mut self, body: &Block) -> Result<(), PdcError> {
        let body_block = self.builder.create_block();
        let _exit_block = self.builder.create_block();

        self.builder.ins().jump(body_block, &[]);

        self.builder.switch_to_block(body_block);
        self.block_terminated = false;
        self.emit_block(body)?;
        if !self.block_terminated {
            self.builder.ins().jump(body_block, &[]);
        }

        self.builder.seal_block(body_block);
        // exit_block would be used by break — deferred
        Ok(())
    }

    fn emit_expr(
        &mut self,
        expr: &Spanned<Expr>,
    ) -> Result<cranelift_codegen::ir::Value, PdcError> {
        match &expr.node {
            Expr::Literal(lit) => self.emit_literal(lit, expr.id),
            Expr::Variable(name) => {
                let (var, _) = self.variables.get(name).cloned().ok_or_else(|| {
                    PdcError::Codegen {
                        message: format!("undefined variable '{name}'"),
                    }
                })?;
                Ok(self.builder.use_var(var))
            }
            Expr::BinaryOp { op, left, right } => {
                let lval = self.emit_expr(left)?;
                let rval = self.emit_expr(right)?;
                let lt = self.node_type(left.id).clone();
                let rt = self.node_type(right.id).clone();
                let result_ty = self.node_type(expr.id).clone();
                self.emit_binary_op(*op, lval, rval, &lt, &rt, &result_ty)
            }
            Expr::UnaryOp { op, operand } => {
                let val = self.emit_expr(operand)?;
                let ty = self.node_type(operand.id);
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
                // Check if this is enum variant construction: EnumName.Variant(args)
                let obj_ty = self.node_type(object.id).clone();
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
            Expr::FieldAccess { object, field } => {
                let obj_ty = self.node_type(object.id).clone();

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
        }
    }

    fn emit_call(
        &mut self,
        name: &str,
        args: &[Spanned<Expr>],
        _call_id: u32,
    ) -> Result<cranelift_codegen::ir::Value, PdcError> {
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
        if let Some(&func_id) = self.user_fn_ids.get(name) {
            let sig = self.user_fns.get(name).unwrap().clone();
            let mut arg_vals = vec![self.ctx_ptr];
            for (i, arg) in args.iter().enumerate() {
                let val = self.emit_expr(arg)?;
                let from_ty = self.node_type(arg.id).clone();
                let to_ty = &sig.params[i];
                let converted = self.convert_value(val, &from_ty, to_ty);
                arg_vals.push(converted);
            }

            let func_ref = self.module.declare_func_in_func(func_id, self.builder.func);
            let call = self.builder.ins().call(func_ref, &arg_vals);
            let results = self.builder.inst_results(call);
            if results.is_empty() {
                return Ok(self.builder.ins().iconst(I32, 0));
            }
            return Ok(results[0]);
        }

        // Runtime function call
        let runtime_name = if name.starts_with("Array<") {
            "pdc_array_new".to_string()
        } else {
            match name {
                "Path" => "pdc_path".to_string(),
                "push" => "pdc_array_push".to_string(),
                "len" => "pdc_array_len".to_string(),
                "get" => "pdc_array_get".to_string(),
                "set" => "pdc_array_set".to_string(),
                other => format!("pdc_{}", other),
            }
        };

        let takes_ctx = name.starts_with("Array<") || matches!(
            name,
            "Path" | "move_to" | "line_to" | "quad_to" | "cubic_to" | "close" | "fill" | "stroke"
            | "push" | "len" | "get" | "set"
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
            Ok(self.builder.inst_results(call)[0])
        } else {
            Ok(self.builder.ins().iconst(I32, 0))
        }
    }

    fn call_return_type(&self, name: &str) -> Option<cranelift_codegen::ir::Type> {
        if name.starts_with("Array<") {
            return Some(I32);
        }
        match name {
            "Path" => Some(I32),
            "len" => Some(I32),
            "get" => Some(F64),
            "move_to" | "line_to" | "quad_to" | "cubic_to" | "close" | "fill" | "stroke"
            | "push" | "set" => None,
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

        val
    }

    /// Widen any scalar value to f64 for storage in struct memory (8 bytes per field).
    fn widen_to_f64(&mut self, val: cranelift_codegen::ir::Value, ty: &PdcType) -> cranelift_codegen::ir::Value {
        match ty {
            PdcType::F64 => val,
            PdcType::F32 => self.builder.ins().fpromote(F64, val),
            PdcType::I32 | PdcType::U32 | PdcType::PathHandle => {
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
            PdcType::I32 | PdcType::U32 | PdcType::PathHandle => {
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

    fn emit_binary_op(
        &mut self,
        op: BinOp,
        lval: cranelift_codegen::ir::Value,
        rval: cranelift_codegen::ir::Value,
        lt: &PdcType,
        rt: &PdcType,
        result_ty: &PdcType,
    ) -> Result<cranelift_codegen::ir::Value, PdcError> {
        let lval = self.convert_value(lval, lt, result_ty);
        let rval = self.convert_value(rval, rt, result_ty);

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
            let lval = self.convert_value(lval, result_ty, &cmp_type);
            let rval = self.convert_value(rval, result_ty, &cmp_type);

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
                _ => unreachable!(),
            })
        } else {
            Ok(match op {
                BinOp::Add => self.builder.ins().iadd(lval, rval),
                BinOp::Sub => self.builder.ins().isub(lval, rval),
                BinOp::Mul => self.builder.ins().imul(lval, rval),
                BinOp::Div => self.builder.ins().sdiv(lval, rval),
                BinOp::Mod => self.builder.ins().srem(lval, rval),
                _ => unreachable!(),
            })
        }
    }
}
