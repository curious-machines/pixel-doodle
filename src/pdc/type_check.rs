use std::collections::{HashMap, HashSet};

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
#[derive(Clone, Debug)]
pub struct UserFnSig {
    pub params: Vec<PdcType>,
    pub ret: PdcType,
}

/// Overload set: one or more signatures for the same function name.
#[derive(Clone, Debug)]
pub struct OverloadSet {
    pub sigs: Vec<UserFnSig>,
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
    /// Variables declared as const — assignments to these are rejected.
    const_vars: HashSet<String>,
    builtins: HashMap<String, BuiltinFn>,
    /// User-defined function signatures (supports overloading).
    pub user_fns: HashMap<String, OverloadSet>,
    /// User-defined struct definitions.
    pub structs: HashMap<String, StructInfo>,
    /// User-defined enum definitions.
    pub enums: HashMap<String, EnumInfo>,
    /// Type aliases: `type Name = ExistingType`
    pub type_aliases: HashMap<String, PdcType>,
    /// Type assigned to each AST node, indexed by node ID.
    pub types: Vec<PdcType>,
}

impl TypeChecker {
    pub fn new() -> Self {
        let mut tc = Self {
            scopes: vec![HashMap::new()],
            const_vars: HashSet::new(),
            builtins: HashMap::new(),
            user_fns: HashMap::new(),
            structs: HashMap::new(),
            enums: HashMap::new(),
            type_aliases: HashMap::new(),
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

        // Styled draw overloads
        self.builtins.insert("fill_styled".into(), BuiltinFn {
            params: vec![PdcType::PathHandle, PdcType::U32, PdcType::I32],
            ret: PdcType::Void,
            takes_ctx: true,
        });
        self.builtins.insert("stroke_styled".into(), BuiltinFn {
            params: vec![PdcType::PathHandle, PdcType::F32, PdcType::U32, PdcType::I32, PdcType::I32],
            ret: PdcType::Void,
            takes_ctx: true,
        });

        // Built-in enums: FillRule, LineCap, LineJoin
        self.enums.insert("FillRule".into(), EnumInfo {
            variants: vec![
                EnumVariantInfo { name: "EvenOdd".into(), field_names: vec![], field_types: vec![] },
                EnumVariantInfo { name: "NonZero".into(), field_names: vec![], field_types: vec![] },
            ],
        });
        self.define_var("FillRule", PdcType::Enum("FillRule".into()));

        self.enums.insert("LineCap".into(), EnumInfo {
            variants: vec![
                EnumVariantInfo { name: "Butt".into(), field_names: vec![], field_types: vec![] },
                EnumVariantInfo { name: "Round".into(), field_names: vec![], field_types: vec![] },
                EnumVariantInfo { name: "Square".into(), field_names: vec![], field_types: vec![] },
            ],
        });
        self.define_var("LineCap", PdcType::Enum("LineCap".into()));

        self.enums.insert("LineJoin".into(), EnumInfo {
            variants: vec![
                EnumVariantInfo { name: "Miter".into(), field_names: vec![], field_types: vec![] },
                EnumVariantInfo { name: "Round".into(), field_names: vec![], field_types: vec![] },
                EnumVariantInfo { name: "Bevel".into(), field_names: vec![], field_types: vec![] },
            ],
        });
        self.define_var("LineJoin", PdcType::Enum("LineJoin".into()));

        for name in &["sin", "cos", "tan", "asin", "acos", "atan", "sqrt", "abs", "floor", "ceil", "round", "exp", "ln", "log2", "log10", "fract", "exp2"] {
            self.builtins.insert(name.to_string(), BuiltinFn {
                params: vec![PdcType::F64],
                ret: PdcType::F64,
                takes_ctx: false,
            });
        }
        for name in &["min", "max", "atan2", "fmod", "pow"] {
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

    /// Resolve type aliases. If the type is a Struct(name) that matches a type alias, return the aliased type.
    fn resolve_type(&self, ty: &PdcType) -> PdcType {
        match ty {
            PdcType::Struct(name) => {
                if let Some(aliased) = self.type_aliases.get(name) {
                    self.resolve_type(aliased)
                } else {
                    ty.clone()
                }
            }
            PdcType::Array(elem) => {
                PdcType::Array(Box::new(self.resolve_type(elem)))
            }
            _ => ty.clone(),
        }
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
                Stmt::TypeAlias { name, ty } => {
                    let resolved = self.resolve_type(ty);
                    self.type_aliases.insert(name.clone(), resolved);
                }
                _ => {}
            }
        }
        // Second sub-pass: register structs and functions (aliases now available)
        for stmt in &program.stmts {
            match &stmt.node {
                Stmt::FnDef(fndef) => {
                    let sig = UserFnSig {
                        params: fndef.params.iter().map(|p| self.resolve_type(&p.ty)).collect(),
                        ret: self.resolve_type(&fndef.return_type),
                    };
                    self.user_fns.entry(fndef.name.clone())
                        .or_insert_with(|| OverloadSet { sigs: Vec::new() })
                        .sigs.push(sig);
                }
                Stmt::StructDef(sdef) => {
                    self.structs.insert(sdef.name.clone(), StructInfo {
                        fields: sdef.fields.iter().map(|f| (f.name.clone(), self.resolve_type(&f.ty))).collect(),
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
                let resolved = self.resolve_type(ty);
                self.define_var(name, resolved);
                self.const_vars.insert(name.clone());
            }
            Stmt::ConstDecl { name, ty, value } => {
                let val_ty = self.check_expr(value)?;
                let final_ty = if let Some(declared) = ty {
                    let resolved = self.resolve_type(declared);
                    self.check_compatible(&val_ty, &resolved, value.span)?;
                    resolved
                } else {
                    val_ty
                };
                self.define_var(name, final_ty);
                self.const_vars.insert(name.clone());
            }
            Stmt::VarDecl { name, ty, value } => {
                let val_ty = self.check_expr(value)?;
                let final_ty = if let Some(declared) = ty {
                    let resolved = self.resolve_type(declared);
                    self.check_compatible(&val_ty, &resolved, value.span)?;
                    resolved
                } else {
                    val_ty
                };
                self.define_var(name, final_ty);
            }
            Stmt::TupleDestructure { names, value, is_const } => {
                let val_ty = self.check_expr(value)?;
                if let PdcType::Tuple(elems) = &val_ty {
                    if names.len() != elems.len() {
                        return Err(PdcError::Type {
                            span: value.span,
                            message: format!(
                                "tuple destructure: expected {} names, got {} (tuple has {} elements)",
                                elems.len(), names.len(), elems.len(),
                            ),
                        });
                    }
                    for (i, name) in names.iter().enumerate() {
                        if name != "_" {
                            self.define_var(name, elems[i].clone());
                            if *is_const {
                                self.const_vars.insert(name.clone());
                            }
                        }
                    }
                } else {
                    return Err(PdcError::Type {
                        span: value.span,
                        message: format!("cannot destructure non-tuple type {val_ty}"),
                    });
                }
            }
            Stmt::IndexAssign { object, index, value } => {
                let obj_ty = self.check_expr(object)?;
                self.check_expr(index)?;
                let val_ty = self.check_expr(value)?;
                if let PdcType::Array(ref elem_ty) = obj_ty {
                    self.check_compatible(&val_ty, elem_ty, value.span)?;
                } else {
                    return Err(PdcError::Type {
                        span: object.span,
                        message: format!("cannot index-assign non-array type {obj_ty}"),
                    });
                }
            }
            Stmt::Assign { name, value } => {
                if self.const_vars.contains(name) {
                    return Err(PdcError::Type {
                        span: stmt.span,
                        message: format!("cannot assign to const variable '{name}'"),
                    });
                }
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
            Stmt::For { mutable,
                var_name,
                start,
                end,
                inclusive: _,
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
                if !mutable {
                    self.const_vars.insert(var_name.clone());
                }
                for s in &body.stmts {
                    self.check_stmt(s)?;
                }
                if !mutable {
                    self.const_vars.remove(var_name);
                }
                self.pop_scope();
            }
            Stmt::ForEach { mutable,
                var_name,
                destructure_names,
                collection,
                body,
            } => {
                let coll_ty = self.check_expr(collection)?;
                let elem_ty = match &coll_ty {
                    PdcType::Array(et) => *et.clone(),
                    _ => {
                        return Err(PdcError::Type {
                            span: collection.span,
                            message: format!("cannot iterate over type {coll_ty}, expected Array"),
                        });
                    }
                };
                self.push_scope();
                if !destructure_names.is_empty() {
                    // Destructuring: element must be a tuple
                    if let PdcType::Tuple(ref elems) = elem_ty {
                        if destructure_names.len() != elems.len() {
                            return Err(PdcError::Type {
                                span: collection.span,
                                message: format!(
                                    "for-each destructure: expected {} names, got {} (tuple has {} elements)",
                                    elems.len(), destructure_names.len(), elems.len(),
                                ),
                            });
                        }
                        for (i, name) in destructure_names.iter().enumerate() {
                            if name != "_" {
                                self.define_var(name, elems[i].clone());
                                if !mutable {
                                    self.const_vars.insert(name.clone());
                                }
                            }
                        }
                    } else {
                        return Err(PdcError::Type {
                            span: collection.span,
                            message: format!("cannot destructure non-tuple element type {elem_ty}"),
                        });
                    }
                } else {
                    self.define_var(var_name, elem_ty);
                    if !mutable {
                        self.const_vars.insert(var_name.clone());
                    }
                }
                for s in &body.stmts {
                    self.check_stmt(s)?;
                }
                if !mutable {
                    if !destructure_names.is_empty() {
                        for name in destructure_names {
                            self.const_vars.remove(name);
                        }
                    } else {
                        self.const_vars.remove(var_name);
                    }
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
            Stmt::Import { module, names } => {
                // Namespaced import: `import math` → define `math` as a module namespace
                if names.is_empty() {
                    self.define_var(module, PdcType::Module(module.clone()));
                }
            }
            Stmt::StructDef(_) | Stmt::EnumDef(_) | Stmt::TypeAlias { .. } => {
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
                    } else if *v <= u32::MAX as i64 {
                        PdcType::U32
                    } else {
                        PdcType::I64
                    }
                }
                Literal::Float(_) => PdcType::F64,
                Literal::Bool(_) => PdcType::Bool,
                Literal::String(_) => PdcType::Str,
            },
            Expr::Variable(name) => {
                if let Some(ty) = self.lookup_var(name).cloned() {
                    ty
                } else if let Some(builtin) = self.builtins.get(name.as_str()) {
                    // Function reference to a builtin
                    PdcType::FnRef {
                        params: builtin.params.clone(),
                        ret: Box::new(builtin.ret.clone()),
                    }
                } else if let Some(overloads) = self.user_fns.get(name.as_str()) {
                    // Function reference to a user function (use first overload)
                    let sig = &overloads.sigs[0];
                    PdcType::FnRef {
                        params: sig.params.clone(),
                        ret: Box::new(sig.ret.clone()),
                    }
                } else {
                    return Err(PdcError::Type {
                        span: expr.span,
                        message: format!("undefined variable '{name}'"),
                    });
                }
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
                    UnaryOp::BitNot => {
                        if !t.is_int() {
                            return Err(PdcError::Type {
                                span: expr.span,
                                message: format!("cannot apply ~ to type {t}, expected integer"),
                            });
                        }
                        t
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

                // Overloaded builtins: fill/stroke with style parameters
                if name == "fill" && args.len() == 3 {
                    // fill(path, color, rule) → fill_styled
                    for arg in args.iter() { self.check_expr(arg)?; }
                    self.set_type(expr.id, PdcType::Void);
                    return Ok(PdcType::Void);
                }
                if name == "stroke" && args.len() == 5 {
                    // stroke(path, width, color, cap, join) → stroke_styled
                    for arg in args.iter() { self.check_expr(arg)?; }
                    self.set_type(expr.id, PdcType::Void);
                    return Ok(PdcType::Void);
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
                } else if let Some(overloads) = self.user_fns.get(name.as_str()).cloned() {
                    // Type-check args first to get their types for overload resolution
                    let mut arg_types = Vec::new();
                    for arg in args.iter() {
                        arg_types.push(self.check_expr(arg)?);
                    }
                    let sig = self.resolve_overload(&overloads, &arg_types)
                        .ok_or_else(|| PdcError::Type {
                            span: expr.span,
                            message: format!(
                                "no matching overload for '{name}' with argument types ({})",
                                arg_types.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(", "),
                            ),
                        })?;
                    for (i, arg) in args.iter().enumerate() {
                        self.check_compatible(&arg_types[i], &sig.params[i], arg.span)?;
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

                // Module namespaced call: math.sin(x) → sin(x)
                if let PdcType::Module(_) = &obj_ty {
                    // Check args
                    let mut arg_types = Vec::new();
                    for arg in args.iter() {
                        arg_types.push(self.check_expr(arg)?);
                    }
                    // Try user function first
                    if let Some(overloads) = self.user_fns.get(method.as_str()).cloned() {
                        let sig = self.resolve_overload(&overloads, &arg_types)
                            .ok_or_else(|| PdcError::Type {
                                span: expr.span,
                                message: format!("no matching overload for '{method}'"),
                            })?;
                        for (i, arg) in args.iter().enumerate() {
                            self.check_compatible(&arg_types[i], &sig.params[i], arg.span)?;
                        }
                        self.set_type(expr.id, sig.ret.clone());
                        return Ok(sig.ret);
                    }
                    // Try builtin
                    if let Some(builtin) = self.builtins.get(method.as_str()) {
                        let ret = builtin.ret.clone();
                        self.set_type(expr.id, ret.clone());
                        return Ok(ret);
                    }
                    return Err(PdcError::Type {
                        span: expr.span,
                        message: format!("module has no function '{method}'"),
                    });
                }

                // Tuple methods: len
                if let PdcType::Tuple(_) = &obj_ty {
                    match method.as_str() {
                        "len" => PdcType::I32,
                        _ => {
                            return Err(PdcError::Type {
                                span: expr.span,
                                message: format!("tuple has no method '{method}'"),
                            });
                        }
                    }
                } else
                // Slice methods: len, get
                if let PdcType::Slice(ref elem_ty) = obj_ty {
                    match method.as_str() {
                        "len" => PdcType::I32,
                        "get" => {
                            if args.len() == 1 {
                                self.check_expr(&args[0])?;
                            }
                            *elem_ty.clone()
                        }
                        _ => {
                            return Err(PdcError::Type {
                                span: expr.span,
                                message: format!("slice has no method '{method}'"),
                            });
                        }
                    }
                } else
                // String methods
                if obj_ty == PdcType::Str {
                    match method.as_str() {
                        "len" => PdcType::I32,
                        "slice" => {
                            if args.len() == 2 {
                                self.check_expr(&args[0])?;
                                self.check_expr(&args[1])?;
                            }
                            PdcType::Str
                        }
                        "concat" => {
                            if args.len() == 1 {
                                let arg_ty = self.check_expr(&args[0])?;
                                if arg_ty != PdcType::Str {
                                    return Err(PdcError::Type {
                                        span: args[0].span,
                                        message: format!("concat expects string, got {arg_ty}"),
                                    });
                                }
                            }
                            PdcType::Str
                        }
                        "char_at" => {
                            if args.len() == 1 {
                                self.check_expr(&args[0])?;
                            }
                            PdcType::Str
                        }
                        _ => {
                            return Err(PdcError::Type {
                                span: expr.span,
                                message: format!("string has no method '{method}'"),
                            });
                        }
                    }
                } else
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
                        "map" => {
                            if args.len() != 1 {
                                return Err(PdcError::Type {
                                    span: expr.span,
                                    message: "map() expects exactly 1 argument (a function reference)".into(),
                                });
                            }
                            let fn_ty = self.check_expr(&args[0])?;
                            match &fn_ty {
                                PdcType::FnRef { params, ret } => {
                                    if params.len() != 1 {
                                        return Err(PdcError::Type {
                                            span: args[0].span,
                                            message: format!("map function must take 1 parameter, got {}", params.len()),
                                        });
                                    }
                                    self.check_compatible(elem_ty, &params[0], args[0].span)?;
                                    PdcType::Array(Box::new(*ret.clone()))
                                }
                                _ => {
                                    return Err(PdcError::Type {
                                        span: args[0].span,
                                        message: format!("map() argument must be a function reference, got {fn_ty}"),
                                    });
                                }
                            }
                        }
                        "slice" => {
                            if args.len() == 2 {
                                self.check_expr(&args[0])?;
                                self.check_expr(&args[1])?;
                            }
                            PdcType::Slice(elem_ty.clone())
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
                } else if let Some(overloads) = self.user_fns.get(method.as_str()).cloned() {
                    // Build full arg types: [obj_ty, arg0_ty, arg1_ty, ...]
                    let mut arg_types = vec![obj_ty.clone()];
                    for arg in args.iter() {
                        arg_types.push(self.check_expr(arg)?);
                    }
                    let sig = self.resolve_overload(&overloads, &arg_types)
                        .ok_or_else(|| PdcError::Type {
                            span: expr.span,
                            message: format!(
                                "no matching overload for method '{method}' with argument types ({})",
                                arg_types.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(", "),
                            ),
                        })?;
                    self.check_compatible(&obj_ty, &sig.params[0], object.span)?;
                    for (i, arg) in args.iter().enumerate() {
                        self.check_compatible(&arg_types[i + 1], &sig.params[i + 1], arg.span)?;
                    }
                    sig.ret
                } else {
                    return Err(PdcError::Type {
                        span: expr.span,
                        message: format!("undefined method '{method}'"),
                    });
                }
            }
            Expr::Index { object, index } => {
                let obj_ty = self.check_expr(object)?;
                self.check_expr(index)?;
                if let PdcType::Array(elem_ty) = &obj_ty {
                    *elem_ty.clone()
                } else if let PdcType::Slice(elem_ty) = &obj_ty {
                    *elem_ty.clone()
                } else {
                    return Err(PdcError::Type {
                        span: expr.span,
                        message: format!("cannot index type {obj_ty}"),
                    });
                }
            }
            Expr::TupleConstruct { elements } => {
                let elem_types: Vec<PdcType> = elements.iter()
                    .map(|e| self.check_expr(e))
                    .collect::<Result<_, _>>()?;
                PdcType::Tuple(elem_types)
            }
            Expr::TupleIndex { object, index } => {
                let obj_ty = self.check_expr(object)?;
                if let PdcType::Tuple(elems) = &obj_ty {
                    if *index < elems.len() {
                        elems[*index].clone()
                    } else {
                        return Err(PdcError::Type {
                            span: expr.span,
                            message: format!("tuple index {} out of range (tuple has {} elements)", index, elems.len()),
                        });
                    }
                } else {
                    return Err(PdcError::Type {
                        span: expr.span,
                        message: format!("cannot index non-tuple type {obj_ty}"),
                    });
                }
            }
            Expr::FieldAccess { object, field } => {
                let obj_ty = self.check_expr(object)?;
                match &obj_ty {
                    PdcType::Module(_) => {
                        // Module namespaced field: math.PI → PI
                        self.lookup_var(field).cloned()
                            .ok_or_else(|| PdcError::Type {
                                span: expr.span,
                                message: format!("module has no member '{field}'"),
                            })?
                    }
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
            Expr::Ternary { condition, then_expr, else_expr } => {
                let cond_ty = self.check_expr(condition)?;
                if cond_ty != PdcType::Bool {
                    return Err(PdcError::Type {
                        span: condition.span,
                        message: format!("ternary condition must be bool, got {cond_ty}"),
                    });
                }
                let then_ty = self.check_expr(then_expr)?;
                let else_ty = self.check_expr(else_expr)?;
                if then_ty == else_ty {
                    then_ty
                } else if then_ty.is_numeric() && else_ty.is_numeric() {
                    self.unify_numeric(&then_ty, &else_ty, expr.span)?
                } else {
                    return Err(PdcError::Type {
                        span: expr.span,
                        message: format!("ternary branches must have compatible types, got {then_ty} and {else_ty}"),
                    });
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
            "i8" => Some(PdcType::I8),
            "i16" => Some(PdcType::I16),
            "i32" => Some(PdcType::I32),
            "i64" => Some(PdcType::I64),
            "u8" => Some(PdcType::U8),
            "u16" => Some(PdcType::U16),
            "u32" => Some(PdcType::U32),
            "u64" => Some(PdcType::U64),
            "bool" => Some(PdcType::Bool),
            _ => {
                // Check type aliases that resolve to numeric types
                if let Some(aliased) = self.type_aliases.get(name) {
                    if aliased.is_numeric() || *aliased == PdcType::Bool {
                        return Some(aliased.clone());
                    }
                }
                None
            }
        }
    }

    fn check_binary_op(
        &self,
        op: BinOp,
        left: &PdcType,
        right: &PdcType,
        span: super::span::Span,
    ) -> Result<PdcType, PdcError> {
        // Array broadcasting: array op array, array op scalar, scalar op array
        let is_arithmetic_or_cmp = matches!(op,
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod | BinOp::Pow |
            BinOp::Eq | BinOp::NotEq | BinOp::Lt | BinOp::LtEq | BinOp::Gt | BinOp::GtEq |
            BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor | BinOp::Shl | BinOp::Shr
        );
        if is_arithmetic_or_cmp {
            match (left, right) {
                (PdcType::Array(elem_l), PdcType::Array(elem_r)) => {
                    // array op array: element-wise
                    let result_elem = self.check_binary_op(op, elem_l, elem_r, span)?;
                    return if matches!(op, BinOp::Eq | BinOp::NotEq | BinOp::Lt | BinOp::LtEq | BinOp::Gt | BinOp::GtEq) {
                        Ok(PdcType::Array(Box::new(PdcType::Bool)))
                    } else {
                        Ok(PdcType::Array(Box::new(result_elem)))
                    };
                }
                (PdcType::Array(elem), scalar) if scalar.is_numeric() => {
                    let result_elem = self.check_binary_op(op, elem, scalar, span)?;
                    return if matches!(op, BinOp::Eq | BinOp::NotEq | BinOp::Lt | BinOp::LtEq | BinOp::Gt | BinOp::GtEq) {
                        Ok(PdcType::Array(Box::new(PdcType::Bool)))
                    } else {
                        Ok(PdcType::Array(Box::new(result_elem)))
                    };
                }
                (scalar, PdcType::Array(elem)) if scalar.is_numeric() => {
                    let result_elem = self.check_binary_op(op, scalar, elem, span)?;
                    return if matches!(op, BinOp::Eq | BinOp::NotEq | BinOp::Lt | BinOp::LtEq | BinOp::Gt | BinOp::GtEq) {
                        Ok(PdcType::Array(Box::new(PdcType::Bool)))
                    } else {
                        Ok(PdcType::Array(Box::new(result_elem)))
                    };
                }
                _ => {} // fall through to normal checking
            }
        }

        // String operations
        if *left == PdcType::Str || *right == PdcType::Str {
            match op {
                BinOp::Add => {
                    if *left == PdcType::Str && *right == PdcType::Str {
                        return Ok(PdcType::Str);
                    }
                    return Err(PdcError::Type {
                        span,
                        message: format!("cannot add {left} and {right}"),
                    });
                }
                BinOp::Eq | BinOp::NotEq => {
                    if *left == PdcType::Str && *right == PdcType::Str {
                        return Ok(PdcType::Bool);
                    }
                    return Err(PdcError::Type {
                        span,
                        message: format!("cannot compare {left} and {right}"),
                    });
                }
                _ => {
                    return Err(PdcError::Type {
                        span,
                        message: format!("operator not supported for string type"),
                    });
                }
            }
        }

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
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod | BinOp::Pow => {
                self.unify_numeric(left, right, span)
            }
            BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor | BinOp::Shl | BinOp::Shr => {
                if !left.is_int() || !right.is_int() {
                    return Err(PdcError::Type {
                        span,
                        message: format!("bitwise operator requires integer types, got {left} and {right}"),
                    });
                }
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
        // Widening: float wins over int, larger wins over smaller
        match (a, b) {
            (PdcType::F64, _) | (_, PdcType::F64) => Ok(PdcType::F64),
            (PdcType::F32, _) | (_, PdcType::F32) => Ok(PdcType::F32),
            (PdcType::I64, _) | (_, PdcType::I64) => Ok(PdcType::I64),
            (PdcType::U64, _) | (_, PdcType::U64) => Ok(PdcType::I64), // mixed sign → signed
            (PdcType::I32, _) | (_, PdcType::I32) => Ok(PdcType::I32),
            (PdcType::U32, _) | (_, PdcType::U32) => Ok(PdcType::I32),
            (PdcType::I16, _) | (_, PdcType::I16) => Ok(PdcType::I16),
            (PdcType::U16, _) | (_, PdcType::U16) => Ok(PdcType::I16),
            _ => Ok(PdcType::I32), // i8/u8 → i32
        }
    }

    /// Resolve the best-matching overload for given argument types.
    fn resolve_overload(&self, overloads: &OverloadSet, arg_types: &[PdcType]) -> Option<UserFnSig> {
        // Exact match first
        for sig in &overloads.sigs {
            if sig.params.len() == arg_types.len() &&
               sig.params.iter().zip(arg_types).all(|(p, a)| p == a) {
                return Some(sig.clone());
            }
        }
        // Compatible match (numeric coercion)
        for sig in &overloads.sigs {
            if sig.params.len() == arg_types.len() &&
               sig.params.iter().zip(arg_types).all(|(p, a)| {
                   p == a || (p.is_numeric() && a.is_numeric()) ||
                   matches!((p, a), (PdcType::Array(_), PdcType::Array(_)))
               }) {
                return Some(sig.clone());
            }
        }
        None
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
