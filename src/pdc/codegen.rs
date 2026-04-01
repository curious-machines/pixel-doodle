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

/// Compiled PDC function type.
pub type PdcSceneFn = unsafe extern "C" fn(*mut runtime::PdcContext);

/// JIT-compiled PDC program.
pub struct CompiledProgram {
    pub fn_ptr: PdcSceneFn,
    _module: JITModule, // kept alive so fn_ptr remains valid
}

/// Builtin variable info: offset into builtins array and type.
pub struct BuiltinInfo {
    pub offset: usize,
    pub ty: PdcType,
}

pub fn compile(
    program: &Program,
    types: &[PdcType],
    builtins_layout: &[(&str, PdcType)],
) -> Result<CompiledProgram, PdcError> {
    // Set up Cranelift JIT
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

    // Register runtime symbols
    for (name, ptr) in runtime::runtime_symbols() {
        jit_builder.symbol(name, ptr);
    }

    let mut jit_module = JITModule::new(jit_builder);

    // Build function signature: extern "C" fn(*mut PdcContext)
    let mut sig = jit_module.make_signature();
    sig.params.push(AbiParam::new(pointer_type));
    sig.call_conv = call_conv;

    let func_id = jit_module
        .declare_function("pdc_main", Linkage::Local, &sig)
        .map_err(|e| PdcError::Codegen {
            message: format!("declare function: {e}"),
        })?;

    let mut ctx = jit_module.make_context();
    ctx.func.signature = sig;
    ctx.func.name = UserFuncName::user(0, 0);

    let mut fb_ctx = FunctionBuilderContext::new();
    {
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);
        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);
        builder.seal_block(entry_block);

        let ctx_ptr = builder.block_params(entry_block)[0];

        // Build builtin variable info
        let mut builtin_map: HashMap<String, BuiltinInfo> = HashMap::new();
        for (i, (name, ty)) in builtins_layout.iter().enumerate() {
            builtin_map.insert(name.to_string(), BuiltinInfo {
                offset: i,
                ty: ty.clone(),
            });
        }

        let mut cg = CodegenCtx {
            builder: &mut builder,
            module: &mut jit_module,
            ctx_ptr,
            variables: HashMap::new(),
            builtin_map,
            type_table: types,
            call_conv,
            pointer_type,
        };

        for item in &program.items {
            cg.emit_item(item)?;
        }

        cg.builder.ins().return_(&[]);
        drop(cg);
        builder.finalize();
    }

    jit_module
        .define_function(func_id, &mut ctx)
        .map_err(|e| PdcError::Codegen {
            message: format!("define function: {e}"),
        })?;

    jit_module.finalize_definitions().map_err(|e| PdcError::Codegen {
        message: format!("finalize: {e}"),
    })?;

    let code_ptr = jit_module.get_finalized_function(func_id);
    let fn_ptr: PdcSceneFn = unsafe { std::mem::transmute(code_ptr) };

    Ok(CompiledProgram {
        fn_ptr,
        _module: jit_module,
    })
}

struct CodegenCtx<'a, 'b> {
    builder: &'a mut FunctionBuilder<'b>,
    module: &'a mut JITModule,
    ctx_ptr: cranelift_codegen::ir::Value,
    variables: HashMap<String, (Variable, PdcType)>,
    builtin_map: HashMap<String, BuiltinInfo>,
    type_table: &'a [PdcType],
    call_conv: CallConv,
    pointer_type: cranelift_codegen::ir::Type,
}

impl<'a, 'b> CodegenCtx<'a, 'b> {
    fn pdc_type_to_cl(&self, ty: &PdcType) -> cranelift_codegen::ir::Type {
        match ty {
            PdcType::F32 => F32,
            PdcType::F64 => F64,
            PdcType::I32 | PdcType::U32 => I32,
            PdcType::Bool => I8,
            PdcType::PathHandle => I32,
            PdcType::Void => I32, // shouldn't be used as a value
            PdcType::Unknown => I64,
        }
    }

    fn new_variable(&mut self, name: &str, ty: &PdcType) -> Variable {
        let cl_type = self.pdc_type_to_cl(ty);
        let var = self.builder.declare_var(cl_type);
        self.variables.insert(name.to_string(), (var, ty.clone()));
        var
    }

    fn node_type(&self, id: u32) -> &PdcType {
        &self.type_table[id as usize]
    }

    fn emit_item(&mut self, item: &Spanned<Item>) -> Result<(), PdcError> {
        match &item.node {
            Item::BuiltinDecl { name, ty } => {
                // Load builtin value from context
                let info = self.builtin_map.get(name).ok_or_else(|| PdcError::Codegen {
                    message: format!("builtin '{name}' not found in layout"),
                })?;
                let offset = info.offset;
                let declared_ty = ty.clone();

                // Load builtins pointer from PdcContext (first field)
                let builtins_ptr = self.builder.ins().load(
                    self.pointer_type,
                    MemFlags::trusted(),
                    self.ctx_ptr,
                    0, // builtins is first field
                );

                // Load f64 value at offset
                let f64_val = self.builder.ins().load(
                    F64,
                    MemFlags::trusted(),
                    builtins_ptr,
                    (offset * 8) as i32,
                );

                // Convert to declared type
                let val = self.convert_value(f64_val, &PdcType::F64, &declared_ty);

                let var = self.new_variable(name, &declared_ty);
                self.builder.def_var(var, val);
                Ok(())
            }
            Item::ConstDecl { name, ty, value } | Item::VarDecl { name, ty, value } => {
                let val = self.emit_expr(value)?;
                let expr_ty = self.node_type(value.id).clone();
                let final_ty = ty.clone().unwrap_or(expr_ty.clone());
                let converted = self.convert_value(val, &expr_ty, &final_ty);
                let var = self.new_variable(name, &final_ty);
                self.builder.def_var(var, converted);
                Ok(())
            }
            Item::Assign { name, value } => {
                let val = self.emit_expr(value)?;
                let (var, var_ty) = self.variables.get(name).cloned().ok_or_else(|| {
                    PdcError::Codegen {
                        message: format!("undefined variable '{name}'"),
                    }
                })?;
                let expr_ty = self.node_type(value.id).clone();
                let converted = self.convert_value(val, &expr_ty, &var_ty);
                self.builder.def_var(var, converted);
                Ok(())
            }
            Item::ExprStmt(expr) => {
                self.emit_expr(expr)?;
                Ok(())
            }
        }
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
            Expr::Call { name, args } => self.emit_call(name, args, expr.id),
            Expr::MethodCall {
                object,
                method,
                args,
            } => {
                // UFCS: prepend object to args
                let mut all_args = vec![object.as_ref().clone()];
                all_args.extend(args.iter().cloned());
                self.emit_call(method, &all_args, expr.id)
            }
        }
    }

    fn emit_literal(
        &mut self,
        lit: &Literal,
        id: u32,
    ) -> Result<cranelift_codegen::ir::Value, PdcError> {
        let ty = self.node_type(id);
        match lit {
            Literal::Int(v) => match ty {
                PdcType::U32 | PdcType::I32 | PdcType::PathHandle => {
                    Ok(self.builder.ins().iconst(I32, *v))
                }
                PdcType::F32 => Ok(self.builder.ins().f32const(*v as f32)),
                PdcType::F64 => Ok(self.builder.ins().f64const(*v as f64)),
                _ => Ok(self.builder.ins().iconst(I32, *v)),
            },
            Literal::Float(v) => match ty {
                PdcType::F32 => Ok(self.builder.ins().f32const(*v as f32)),
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
        // Check for type cast
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

        // Runtime function call — map PDC name to runtime symbol name
        let runtime_name = match name {
            "Path" => "pdc_path".to_string(),
            other => format!("pdc_{}", other),
        };

        // Determine if function takes ctx as first arg
        let takes_ctx = matches!(
            name,
            "Path" | "move_to" | "line_to" | "quad_to" | "cubic_to" | "close" | "fill" | "stroke"
        );

        // Emit argument values
        let mut arg_vals = Vec::new();
        if takes_ctx {
            arg_vals.push(self.ctx_ptr);
        }
        for arg in args {
            let val = self.emit_expr(arg)?;
            let arg_ty = self.node_type(arg.id).clone();
            // Convert to the expected parameter type
            let converted = self.convert_for_call(val, &arg_ty, name, args.len(), arg_vals.len() - if takes_ctx { 1 } else { 0 });
            arg_vals.push(converted);
        }

        // Build call signature
        let mut sig = self.module.make_signature();
        sig.call_conv = self.call_conv;
        for val in &arg_vals {
            let ty = self.builder.func.dfg.value_type(*val);
            sig.params.push(AbiParam::new(ty));
        }

        // Determine return type
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
            // Void function — return a dummy value
            Ok(self.builder.ins().iconst(I32, 0))
        }
    }

    fn call_return_type(&self, name: &str) -> Option<cranelift_codegen::ir::Type> {
        match name {
            "Path" => Some(I32),
            "move_to" | "line_to" | "quad_to" | "cubic_to" | "close" | "fill" | "stroke" => None,
            // Math functions return f64
            _ => Some(F64),
        }
    }

    fn convert_for_call(
        &mut self,
        val: cranelift_codegen::ir::Value,
        from: &PdcType,
        func_name: &str,
        _num_args: usize,
        _arg_idx: usize,
    ) -> cranelift_codegen::ir::Value {
        // Path/draw functions expect f32 for coordinates, u32 for handles/colors
        let needs_f32 = matches!(
            func_name,
            "move_to" | "line_to" | "quad_to" | "cubic_to" | "stroke"
        );

        if needs_f32 && from.is_float() && *from != PdcType::F32 {
            return self.builder.ins().fdemote(F32, val);
        }
        if needs_f32 && from.is_int() {
            let f64_val = self.builder.ins().fcvt_from_sint(F64, val);
            return self.builder.ins().fdemote(F32, f64_val);
        }

        val
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
        match (from, to) {
            (PdcType::F64, PdcType::F32) => self.builder.ins().fdemote(F32, val),
            (PdcType::F32, PdcType::F64) => self.builder.ins().fpromote(F64, val),
            (PdcType::I32, PdcType::F64) | (PdcType::U32, PdcType::F64) => {
                self.builder.ins().fcvt_from_sint(F64, val)
            }
            (PdcType::I32, PdcType::F32) | (PdcType::U32, PdcType::F32) => {
                let f64_val = self.builder.ins().fcvt_from_sint(F64, val);
                self.builder.ins().fdemote(F32, f64_val)
            }
            (PdcType::F64, PdcType::I32) | (PdcType::F64, PdcType::U32) => {
                self.builder.ins().fcvt_to_sint_sat(I32, val)
            }
            (PdcType::F32, PdcType::I32) | (PdcType::F32, PdcType::U32) => {
                let f64_val = self.builder.ins().fpromote(F64, val);
                self.builder.ins().fcvt_to_sint_sat(I32, f64_val)
            }
            _ => val, // no-op for compatible types
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
        // Convert operands to the result type
        let lval = self.convert_value(lval, lt, result_ty);
        let rval = self.convert_value(rval, rt, result_ty);

        if result_ty == &PdcType::Bool {
            // Comparison operators — determine operand type for comparison
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
                // Need to re-emit operands with proper type
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

        // Arithmetic operators
        if result_ty.is_float() {
            Ok(match op {
                BinOp::Add => self.builder.ins().fadd(lval, rval),
                BinOp::Sub => self.builder.ins().fsub(lval, rval),
                BinOp::Mul => self.builder.ins().fmul(lval, rval),
                BinOp::Div => self.builder.ins().fdiv(lval, rval),
                BinOp::Mod => {
                    // fmod: a - floor(a/b) * b
                    let div = self.builder.ins().fdiv(lval, rval);
                    let floored = self.builder.ins().floor(div);
                    let prod = self.builder.ins().fmul(floored, rval);
                    self.builder.ins().fsub(lval, prod)
                }
                _ => unreachable!("comparison handled above"),
            })
        } else {
            Ok(match op {
                BinOp::Add => self.builder.ins().iadd(lval, rval),
                BinOp::Sub => self.builder.ins().isub(lval, rval),
                BinOp::Mul => self.builder.ins().imul(lval, rval),
                BinOp::Div => self.builder.ins().sdiv(lval, rval),
                BinOp::Mod => self.builder.ins().srem(lval, rval),
                _ => unreachable!("comparison handled above"),
            })
        }
    }
}
