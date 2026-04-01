use std::collections::HashMap;

use super::ast::*;
use super::error::PdcError;
use super::span::Spanned;

/// A built-in function signature.
#[allow(dead_code)]
struct BuiltinFn {
    params: Vec<PdcType>,
    ret: PdcType,
    takes_ctx: bool,
}

/// A user-defined function signature (discovered during type checking).
#[derive(Clone)]
pub struct UserFnSig {
    pub params: Vec<PdcType>,
    pub ret: PdcType,
}

/// Struct definition info for type checking.
#[derive(Clone)]
pub struct StructInfo {
    pub fields: Vec<(String, PdcType)>,
}

/// Enum variant info for type checking.
#[derive(Clone)]
pub struct EnumVariantInfo {
    pub name: String,
    pub field_names: Vec<String>,
    pub field_types: Vec<PdcType>,
}

/// Enum definition info for type checking.
#[derive(Clone)]
pub struct EnumInfo {
    pub variants: Vec<EnumVariantInfo>,
}

pub struct TypeChecker {
    scopes: Vec<HashMap<String, PdcType>>,
    builtins: HashMap<String, BuiltinFn>,
    /// User-defined function signatures.
    pub user_fns: HashMap<String, UserFnSig>,
    /// User-defined struct definitions.
    pub structs: HashMap<String, StructInfo>,
    /// User-defined enum definitions.
    pub enums: HashMap<String, EnumInfo>,
    /// Type assigned to each AST node, indexed by node ID.
    pub types: Vec<PdcType>,
}

impl TypeChecker {
    pub fn new() -> Self {
        let mut tc = Self {
            scopes: vec![HashMap::new()],
            builtins: HashMap::new(),
            user_fns: HashMap::new(),
            structs: HashMap::new(),
            enums: HashMap::new(),
            types: Vec::new(),
        };
        tc.register_builtins();
        tc
    }

    fn register_builtins(&mut self) {
        self.builtins.insert("Path".into(), BuiltinFn {
            params: vec![],
            ret: PdcType::PathHandle,
            takes_ctx: true,
        });

        for name in &["move_to", "line_to"] {
            self.builtins.insert(name.to_string(), BuiltinFn {
                params: vec![PdcType::PathHandle, PdcType::F64, PdcType::F64],
                ret: PdcType::Void,
                takes_ctx: true,
            });
        }
        self.builtins.insert("quad_to".into(), BuiltinFn {
            params: vec![PdcType::PathHandle, PdcType::F64, PdcType::F64, PdcType::F64, PdcType::F64],
            ret: PdcType::Void,
            takes_ctx: true,
        });
        self.builtins.insert("cubic_to".into(), BuiltinFn {
            params: vec![PdcType::PathHandle, PdcType::F64, PdcType::F64, PdcType::F64, PdcType::F64, PdcType::F64, PdcType::F64],
            ret: PdcType::Void,
            takes_ctx: true,
        });
        self.builtins.insert("close".into(), BuiltinFn {
            params: vec![PdcType::PathHandle],
            ret: PdcType::Void,
            takes_ctx: true,
        });

        self.builtins.insert("fill".into(), BuiltinFn {
            params: vec![PdcType::PathHandle, PdcType::U32],
            ret: PdcType::Void,
            takes_ctx: true,
        });
        self.builtins.insert("stroke".into(), BuiltinFn {
            params: vec![PdcType::PathHandle, PdcType::F32, PdcType::U32],
            ret: PdcType::Void,
            takes_ctx: true,
        });

        // Array methods also registered as builtins so UFCS-as-function works:
        // push(arr, val) as well as arr.push(val)
        // Note: MethodCall path has type-aware handling for generic arrays.
        let arr_ty = PdcType::Array(Box::new(PdcType::F64));
        self.builtins.insert("push".into(), BuiltinFn {
            params: vec![arr_ty.clone(), PdcType::F64],
            ret: PdcType::Void,
            takes_ctx: true,
        });
        self.builtins.insert("len".into(), BuiltinFn {
            params: vec![arr_ty.clone()],
            ret: PdcType::I32,
            takes_ctx: true,
        });
        self.builtins.insert("get".into(), BuiltinFn {
            params: vec![arr_ty.clone(), PdcType::I32],
            ret: PdcType::F64,
            takes_ctx: true,
        });
        self.builtins.insert("set".into(), BuiltinFn {
            params: vec![arr_ty.clone(), PdcType::I32, PdcType::F64],
            ret: PdcType::Void,
            takes_ctx: true,
        });

        for name in &["sin", "cos", "tan", "asin", "acos", "atan", "sqrt", "abs", "floor", "ceil", "round", "exp", "ln", "log2", "log10", "fract"] {
            self.builtins.insert(name.to_string(), BuiltinFn {
                params: vec![PdcType::F64],
                ret: PdcType::F64,
                takes_ctx: false,
            });
        }
        for name in &["min", "max", "atan2", "fmod"] {
            self.builtins.insert(name.to_string(), BuiltinFn {
                params: vec![PdcType::F64, PdcType::F64],
                ret: PdcType::F64,
                takes_ctx: false,
            });
        }
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn define_var(&mut self, name: &str, ty: PdcType) {
        self.scopes.last_mut().unwrap().insert(name.to_string(), ty);
    }

    fn lookup_var(&self, name: &str) -> Option<&PdcType> {
        for scope in self.scopes.iter().rev() {
            if let Some(ty) = scope.get(name) {
                return Some(ty);
            }
        }
        None
    }

    fn set_type(&mut self, id: u32, ty: PdcType) {
        let id = id as usize;
        if id >= self.types.len() {
            self.types.resize(id + 1, PdcType::Unknown);
        }
        self.types[id] = ty;
    }

    pub fn check_program(&mut self, program: &Program) -> Result<(), PdcError> {
        // First pass: register all struct and function signatures
        for stmt in &program.stmts {
            match &stmt.node {
                Stmt::FnDef(fndef) => {
                    self.user_fns.insert(fndef.name.clone(), UserFnSig {
                        params: fndef.params.iter().map(|p| p.ty.clone()).collect(),
                        ret: fndef.return_type.clone(),
                    });
                }
                Stmt::StructDef(sdef) => {
                    self.structs.insert(sdef.name.clone(), StructInfo {
                        fields: sdef.fields.iter().map(|f| (f.name.clone(), f.ty.clone())).collect(),
                    });
                }
                Stmt::EnumDef(edef) => {
                    self.enums.insert(edef.name.clone(), EnumInfo {
                        variants: edef.variants.iter().map(|v| EnumVariantInfo {
                            name: v.name.clone(),
                            field_names: v.fields.iter().map(|f| f.name.clone()).collect(),
                            field_types: v.fields.iter().map(|f| f.ty.clone()).collect(),
                        }).collect(),
                    });
                    // Register enum name as a variable so EnumName.Variant resolves
                    self.define_var(&edef.name, PdcType::Enum(edef.name.clone()));
                }
                _ => {}
            }
        }

        // Second pass: type check everything
        for stmt in &program.stmts {
            self.check_stmt(stmt)?;
        }
        Ok(())
    }

    fn check_stmt(&mut self, stmt: &Spanned<Stmt>) -> Result<(), PdcError> {
        match &stmt.node {
            Stmt::BuiltinDecl { name, ty } => {
                self.define_var(name, ty.clone());
            }
            Stmt::ConstDecl { name, ty, value } | Stmt::VarDecl { name, ty, value } => {
                let val_ty = self.check_expr(value)?;
                let final_ty = if let Some(declared) = ty {
                    self.check_compatible(&val_ty, declared, value.span)?;
                    declared.clone()
                } else {
                    val_ty
                };
                self.define_var(name, final_ty);
            }
            Stmt::Assign { name, value } => {
                let expected = self.lookup_var(name).cloned().ok_or_else(|| PdcError::Type {
                    span: stmt.span,
                    message: format!("undefined variable '{name}'"),
                })?;
                let val_ty = self.check_expr(value)?;
                self.check_compatible(&val_ty, &expected, value.span)?;
            }
            Stmt::ExprStmt(expr) => {
                self.check_expr(expr)?;
            }
            Stmt::If {
                condition,
                then_body,
                elsif_clauses,
                else_body,
            } => {
                let cond_ty = self.check_expr(condition)?;
                if cond_ty != PdcType::Bool {
                    return Err(PdcError::Type {
                        span: condition.span,
                        message: format!("if condition must be bool, got {cond_ty}"),
                    });
                }
                self.check_block(then_body)?;
                for (cond, body) in elsif_clauses {
                    let ct = self.check_expr(cond)?;
                    if ct != PdcType::Bool {
                        return Err(PdcError::Type {
                            span: cond.span,
                            message: format!("elsif condition must be bool, got {ct}"),
                        });
                    }
                    self.check_block(body)?;
                }
                if let Some(else_b) = else_body {
                    self.check_block(else_b)?;
                }
            }
            Stmt::While { condition, body } => {
                let cond_ty = self.check_expr(condition)?;
                if cond_ty != PdcType::Bool {
                    return Err(PdcError::Type {
                        span: condition.span,
                        message: format!("while condition must be bool, got {cond_ty}"),
                    });
                }
                self.check_block(body)?;
            }
            Stmt::For {
                var_name,
                start,
                end,
                body,
            } => {
                let st = self.check_expr(start)?;
                let et = self.check_expr(end)?;
                if !st.is_int() {
                    return Err(PdcError::Type {
                        span: start.span,
                        message: format!("for range start must be integer, got {st}"),
                    });
                }
                if !et.is_int() {
                    return Err(PdcError::Type {
                        span: end.span,
                        message: format!("for range end must be integer, got {et}"),
                    });
                }
                self.push_scope();
                self.define_var(var_name, PdcType::I32);
                for s in &body.stmts {
                    self.check_stmt(s)?;
                }
                self.pop_scope();
            }
            Stmt::Loop { body } => {
                self.check_block(body)?;
            }
            Stmt::Break | Stmt::Continue => {}
            Stmt::Return(value) => {
                if let Some(expr) = value {
                    self.check_expr(expr)?;
                }
            }
            Stmt::Match { scrutinee, arms } => {
                let scr_ty = self.check_expr(scrutinee)?;

                let enum_name = match &scr_ty {
                    PdcType::Enum(name) => name.clone(),
                    _ => return Err(PdcError::Type {
                        span: scrutinee.span,
                        message: format!("cannot match non-enum type {scr_ty}"),
                    }),
                };

                let info = self.enums.get(&enum_name).cloned().ok_or_else(|| PdcError::Type {
                    span: scrutinee.span,
                    message: format!("undefined enum '{enum_name}'"),
                })?;

                let mut covered_variants: Vec<String> = Vec::new();
                let mut has_wildcard = false;

                for arm in arms {
                    self.push_scope();

                    match &arm.pattern {
                        MatchPattern::EnumVariant { enum_name: pat_enum, variant, bindings } => {
                            // Verify enum name matches (or is empty for dot-shorthand)
                            if !pat_enum.is_empty() && *pat_enum != enum_name {
                                return Err(PdcError::Type {
                                    span: scrutinee.span,
                                    message: format!("match pattern {pat_enum}.{variant} doesn't match scrutinee type {enum_name}"),
                                });
                            }

                            let vi = info.variants.iter().find(|v| v.name == *variant)
                                .ok_or_else(|| PdcError::Type {
                                    span: scrutinee.span,
                                    message: format!("enum '{enum_name}' has no variant '{variant}'"),
                                })?;

                            // Define destructured bindings
                            for (bi, bname) in bindings.iter().enumerate() {
                                if bname != "_" && bi < vi.field_types.len() {
                                    self.define_var(bname, vi.field_types[bi].clone());
                                }
                            }

                            covered_variants.push(variant.clone());
                        }
                        MatchPattern::Wildcard => {
                            has_wildcard = true;
                        }
                    }

                    for s in &arm.body.stmts {
                        self.check_stmt(s)?;
                    }
                    self.pop_scope();
                }

                // Exhaustiveness check
                if !has_wildcard {
                    for v in &info.variants {
                        if !covered_variants.contains(&v.name) {
                            return Err(PdcError::Type {
                                span: scrutinee.span,
                                message: format!("non-exhaustive match: missing variant '{}.{}'", enum_name, v.name),
                            });
                        }
                    }
                }
            }
            Stmt::Import { .. } | Stmt::StructDef(_) | Stmt::EnumDef(_) => {
                // Already registered in first pass
            }
            Stmt::FnDef(fndef) => {
                // Function bodies see all outer scope variables (top-level consts, builtins)
                self.push_scope();
                for param in &fndef.params {
                    self.define_var(&param.name, param.ty.clone());
                }
                for s in &fndef.body.stmts {
                    self.check_stmt(s)?;
                }
                self.pop_scope();
            }
        }
        Ok(())
    }

    fn check_block(&mut self, block: &Block) -> Result<(), PdcError> {
        self.push_scope();
        for stmt in &block.stmts {
            self.check_stmt(stmt)?;
        }
        self.pop_scope();
        Ok(())
    }

    fn check_expr(&mut self, expr: &Spanned<Expr>) -> Result<PdcType, PdcError> {
        let ty = match &expr.node {
            Expr::Literal(lit) => match lit {
                Literal::Int(v) => {
                    if *v < 0 || *v <= i32::MAX as i64 {
                        PdcType::I32
                    } else {
                        PdcType::U32
                    }
                }
                Literal::Float(_) => PdcType::F64,
                Literal::Bool(_) => PdcType::Bool,
            },
            Expr::Variable(name) => {
                self.lookup_var(name).cloned().ok_or_else(|| PdcError::Type {
                    span: expr.span,
                    message: format!("undefined variable '{name}'"),
                })?
            }
            Expr::BinaryOp { op, left, right } => {
                let lt = self.check_expr(left)?;
                let rt = self.check_expr(right)?;
                self.check_binary_op(*op, &lt, &rt, expr.span)?
            }
            Expr::UnaryOp { op, operand } => {
                let t = self.check_expr(operand)?;
                match op {
                    UnaryOp::Neg => {
                        if !t.is_numeric() {
                            return Err(PdcError::Type {
                                span: expr.span,
                                message: format!("cannot negate type {t}"),
                            });
                        }
                        t
                    }
                    UnaryOp::Not => {
                        if t != PdcType::Bool {
                            return Err(PdcError::Type {
                                span: expr.span,
                                message: format!("cannot apply ! to type {t}"),
                            });
                        }
                        PdcType::Bool
                    }
                }
            }
            Expr::Call { name, args, arg_names } => {
                // Struct construction: named args + known struct name
                let has_named_args = arg_names.iter().any(|n| n.is_some());
                if has_named_args {
                    if let Some(info) = self.structs.get(name).cloned() {
                        // Struct construction
                        for (i, arg) in args.iter().enumerate() {
                            let val_ty = self.check_expr(arg)?;
                            let fname = arg_names[i].as_ref().unwrap();
                            let expected = info.fields.iter()
                                .find(|(n, _)| n == fname)
                                .map(|(_, t)| t.clone())
                                .ok_or_else(|| PdcError::Type {
                                    span: arg.span,
                                    message: format!("struct '{name}' has no field '{fname}'"),
                                })?;
                            self.check_compatible(&val_ty, &expected, arg.span)?;
                        }
                        self.set_type(expr.id, PdcType::Struct(name.clone()));
                        return Ok(PdcType::Struct(name.clone()));
                    }
                    // Not a struct — check as named function call below
                }

                // Array<type>() constructor
                if name.starts_with("Array<") {
                    let elem_type_str = &name[6..name.len()-1]; // extract type between < >
                    let elem_ty = match elem_type_str {
                        "f32" => PdcType::F32,
                        "f64" => PdcType::F64,
                        "i32" => PdcType::I32,
                        "u32" => PdcType::U32,
                        "bool" => PdcType::Bool,
                        _ => PdcType::F64, // default
                    };
                    let arr_ty = PdcType::Array(Box::new(elem_ty));
                    self.set_type(expr.id, arr_ty.clone());
                    return Ok(arr_ty);
                }

                if let Some(cast_ty) = self.is_type_cast(name) {
                    if args.len() != 1 {
                        return Err(PdcError::Type {
                            span: expr.span,
                            message: format!("type cast {name}() expects 1 argument, got {}", args.len()),
                        });
                    }
                    self.check_expr(&args[0])?;
                    cast_ty
                } else if let Some(builtin) = self.builtins.get(name.as_str()) {
                    let expected_params = builtin.params.clone();
                    let ret = builtin.ret.clone();
                    if args.len() != expected_params.len() {
                        return Err(PdcError::Type {
                            span: expr.span,
                            message: format!(
                                "function '{name}' expects {} arguments, got {}",
                                expected_params.len(),
                                args.len()
                            ),
                        });
                    }
                    for (i, arg) in args.iter().enumerate() {
                        let arg_ty = self.check_expr(arg)?;
                        self.check_compatible(&arg_ty, &expected_params[i], arg.span)?;
                    }
                    ret
                } else if let Some(sig) = self.user_fns.get(name.as_str()).cloned() {
                    if args.len() != sig.params.len() {
                        return Err(PdcError::Type {
                            span: expr.span,
                            message: format!(
                                "function '{name}' expects {} arguments, got {}",
                                sig.params.len(),
                                args.len()
                            ),
                        });
                    }
                    for (i, arg) in args.iter().enumerate() {
                        let arg_ty = self.check_expr(arg)?;
                        self.check_compatible(&arg_ty, &sig.params[i], arg.span)?;
                    }
                    sig.ret
                } else {
                    return Err(PdcError::Type {
                        span: expr.span,
                        message: format!("undefined function '{name}'"),
                    });
                }
            }
            Expr::MethodCall { object, method, args } => {
                let obj_ty = self.check_expr(object)?;

                // Array methods: push, len, get, set
                if let PdcType::Array(ref elem_ty) = obj_ty {
                    match method.as_str() {
                        "push" => {
                            if args.len() == 1 {
                                let arg_ty = self.check_expr(&args[0])?;
                                self.check_compatible(&arg_ty, elem_ty, args[0].span)?;
                            }
                            PdcType::Void
                        }
                        "len" => PdcType::I32,
                        "get" => {
                            if args.len() == 1 {
                                self.check_expr(&args[0])?;
                            }
                            *elem_ty.clone()
                        }
                        "set" => {
                            if args.len() == 2 {
                                self.check_expr(&args[0])?;
                                let val_ty = self.check_expr(&args[1])?;
                                self.check_compatible(&val_ty, elem_ty, args[1].span)?;
                            }
                            PdcType::Void
                        }
                        _ => {
                            return Err(PdcError::Type {
                                span: expr.span,
                                message: format!("array has no method '{method}'"),
                            });
                        }
                    }
                } else
                // Enum variant construction: EnumName.Variant(args)
                if let PdcType::Enum(ref ename) = obj_ty {
                    let info = self.enums.get(ename).cloned().ok_or_else(|| PdcError::Type {
                        span: expr.span,
                        message: format!("undefined enum '{ename}'"),
                    })?;
                    let var_info = info.variants.iter().find(|v| v.name == *method)
                        .ok_or_else(|| PdcError::Type {
                            span: expr.span,
                            message: format!("enum '{ename}' has no variant '{method}'"),
                        })?.clone();
                    if args.len() != var_info.field_types.len() {
                        return Err(PdcError::Type {
                            span: expr.span,
                            message: format!(
                                "variant '{}.{}' expects {} arguments, got {}",
                                ename, method, var_info.field_types.len(), args.len(),
                            ),
                        });
                    }
                    for (i, arg) in args.iter().enumerate() {
                        let arg_ty = self.check_expr(arg)?;
                        self.check_compatible(&arg_ty, &var_info.field_types[i], arg.span)?;
                    }
                    PdcType::Enum(ename.clone())
                } else if let Some(builtin) = self.builtins.get(method.as_str()) {
                    let expected_params = builtin.params.clone();
                    let ret = builtin.ret.clone();
                    let total_args = 1 + args.len();
                    if total_args != expected_params.len() {
                        return Err(PdcError::Type {
                            span: expr.span,
                            message: format!(
                                "method '{method}' expects {} arguments (including self), got {}",
                                expected_params.len(),
                                total_args,
                            ),
                        });
                    }
                    self.check_compatible(&obj_ty, &expected_params[0], object.span)?;
                    for (i, arg) in args.iter().enumerate() {
                        let arg_ty = self.check_expr(arg)?;
                        self.check_compatible(&arg_ty, &expected_params[i + 1], arg.span)?;
                    }
                    ret
                } else if let Some(sig) = self.user_fns.get(method.as_str()).cloned() {
                    let total_args = 1 + args.len();
                    if total_args != sig.params.len() {
                        return Err(PdcError::Type {
                            span: expr.span,
                            message: format!(
                                "method '{method}' expects {} arguments (including self), got {}",
                                sig.params.len(),
                                total_args,
                            ),
                        });
                    }
                    self.check_compatible(&obj_ty, &sig.params[0], object.span)?;
                    for (i, arg) in args.iter().enumerate() {
                        let arg_ty = self.check_expr(arg)?;
                        self.check_compatible(&arg_ty, &sig.params[i + 1], arg.span)?;
                    }
                    sig.ret
                } else {
                    return Err(PdcError::Type {
                        span: expr.span,
                        message: format!("undefined method '{method}'"),
                    });
                }
            }
            Expr::FieldAccess { object, field } => {
                let obj_ty = self.check_expr(object)?;
                match &obj_ty {
                    PdcType::Struct(name) => {
                        let info = self.structs.get(name).cloned().ok_or_else(|| PdcError::Type {
                            span: expr.span,
                            message: format!("undefined struct '{name}'"),
                        })?;
                        info.fields.iter()
                            .find(|(n, _)| n == field)
                            .map(|(_, t)| t.clone())
                            .ok_or_else(|| PdcError::Type {
                                span: expr.span,
                                message: format!("struct '{name}' has no field '{field}'"),
                            })?
                    }
                    PdcType::Enum(name) => {
                        let info = self.enums.get(name).cloned().ok_or_else(|| PdcError::Type {
                            span: expr.span,
                            message: format!("undefined enum '{name}'"),
                        })?;
                        if !info.variants.iter().any(|v| v.name == *field) {
                            return Err(PdcError::Type {
                                span: expr.span,
                                message: format!("enum '{name}' has no variant '{field}'"),
                            });
                        }
                        PdcType::Enum(name.clone())
                    }
                    _ => {
                        return Err(PdcError::Type {
                            span: expr.span,
                            message: format!("cannot access field '{field}' on type {obj_ty}"),
                        });
                    }
                }
            }
            Expr::StructConstruct { name, fields } => {
                let info = self.structs.get(name).cloned().ok_or_else(|| PdcError::Type {
                    span: expr.span,
                    message: format!("undefined struct '{name}'"),
                })?;
                for (fname, fexpr) in fields {
                    let val_ty = self.check_expr(fexpr)?;
                    let expected = info.fields.iter()
                        .find(|(n, _)| n == fname)
                        .map(|(_, t)| t.clone())
                        .ok_or_else(|| PdcError::Type {
                            span: fexpr.span,
                            message: format!("struct '{name}' has no field '{fname}'"),
                        })?;
                    self.check_compatible(&val_ty, &expected, fexpr.span)?;
                }
                PdcType::Struct(name.clone())
            }
        };

        self.set_type(expr.id, ty.clone());
        Ok(ty)
    }

    fn is_type_cast(&self, name: &str) -> Option<PdcType> {
        match name {
            "f32" => Some(PdcType::F32),
            "f64" => Some(PdcType::F64),
            "i32" => Some(PdcType::I32),
            "u32" => Some(PdcType::U32),
            "bool" => Some(PdcType::Bool),
            _ => None,
        }
    }

    fn check_binary_op(
        &self,
        op: BinOp,
        left: &PdcType,
        right: &PdcType,
        span: super::span::Span,
    ) -> Result<PdcType, PdcError> {
        match op {
            BinOp::And | BinOp::Or => {
                if *left != PdcType::Bool || *right != PdcType::Bool {
                    return Err(PdcError::Type {
                        span,
                        message: format!("logical operator requires bool, got {left} and {right}"),
                    });
                }
                Ok(PdcType::Bool)
            }
            BinOp::Eq | BinOp::NotEq | BinOp::Lt | BinOp::LtEq | BinOp::Gt | BinOp::GtEq => {
                // Enum comparison: same enum type
                if let (PdcType::Enum(a), PdcType::Enum(b)) = (left, right) {
                    if a != b {
                        return Err(PdcError::Type {
                            span,
                            message: format!("cannot compare different enum types {a} and {b}"),
                        });
                    }
                    return Ok(PdcType::Bool);
                }
                let _ = self.unify_numeric(left, right, span)?;
                Ok(PdcType::Bool)
            }
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod => {
                self.unify_numeric(left, right, span)
            }
        }
    }

    fn unify_numeric(
        &self,
        a: &PdcType,
        b: &PdcType,
        span: super::span::Span,
    ) -> Result<PdcType, PdcError> {
        if !a.is_numeric() || !b.is_numeric() {
            return Err(PdcError::Type {
                span,
                message: format!("expected numeric types, got {a} and {b}"),
            });
        }
        if a == b {
            return Ok(a.clone());
        }
        match (a, b) {
            (PdcType::F64, _) | (_, PdcType::F64) => Ok(PdcType::F64),
            (PdcType::F32, _) | (_, PdcType::F32) => Ok(PdcType::F32),
            (PdcType::I32, PdcType::U32) | (PdcType::U32, PdcType::I32) => Ok(PdcType::I32),
            _ => Ok(a.clone()),
        }
    }

    fn check_compatible(
        &self,
        from: &PdcType,
        to: &PdcType,
        span: super::span::Span,
    ) -> Result<(), PdcError> {
        if from == to {
            return Ok(());
        }
        if from.is_numeric() && to.is_numeric() {
            return Ok(());
        }
        // Arrays with compatible element types
        if let (PdcType::Array(_), PdcType::Array(_)) = (from, to) {
            return Ok(());
        }
        Err(PdcError::Type {
            span,
            message: format!("type mismatch: expected {to}, got {from}"),
        })
    }
}
