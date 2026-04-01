use std::collections::HashMap;

use super::ast::*;
use super::error::PdcError;
use super::span::Spanned;

/// A built-in function signature.
#[allow(dead_code)]
struct BuiltinFn {
    params: Vec<PdcType>,
    ret: PdcType,
    /// If true, first param is the PdcContext pointer (not passed by PDC code).
    takes_ctx: bool,
}

pub struct TypeChecker {
    scopes: Vec<HashMap<String, PdcType>>,
    builtins: HashMap<String, BuiltinFn>,
    /// Type assigned to each AST node, indexed by node ID.
    pub types: Vec<PdcType>,
}

impl TypeChecker {
    pub fn new() -> Self {
        let mut tc = Self {
            scopes: vec![HashMap::new()],
            builtins: HashMap::new(),
            types: Vec::new(),
        };
        tc.register_builtins();
        tc
    }

    fn register_builtins(&mut self) {
        // Path constructor
        self.builtins.insert("Path".into(), BuiltinFn {
            params: vec![],
            ret: PdcType::PathHandle,
            takes_ctx: true,
        });

        // Path primitives (all take ctx + handle as first visible arg)
        for name in &["move_to", "line_to"] {
            self.builtins.insert(name.to_string(), BuiltinFn {
                params: vec![PdcType::PathHandle, PdcType::F32, PdcType::F32],
                ret: PdcType::Void,
                takes_ctx: true,
            });
        }
        self.builtins.insert("quad_to".into(), BuiltinFn {
            params: vec![PdcType::PathHandle, PdcType::F32, PdcType::F32, PdcType::F32, PdcType::F32],
            ret: PdcType::Void,
            takes_ctx: true,
        });
        self.builtins.insert("cubic_to".into(), BuiltinFn {
            params: vec![PdcType::PathHandle, PdcType::F32, PdcType::F32, PdcType::F32, PdcType::F32, PdcType::F32, PdcType::F32],
            ret: PdcType::Void,
            takes_ctx: true,
        });
        self.builtins.insert("close".into(), BuiltinFn {
            params: vec![PdcType::PathHandle],
            ret: PdcType::Void,
            takes_ctx: true,
        });

        // Draw commands
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

        // Math functions (f64 -> f64, no ctx)
        for name in &["sin", "cos", "tan", "asin", "acos", "atan", "sqrt", "abs", "floor", "ceil", "round", "exp", "ln", "log2", "log10", "fract"] {
            self.builtins.insert(name.to_string(), BuiltinFn {
                params: vec![PdcType::F64],
                ret: PdcType::F64,
                takes_ctx: false,
            });
        }
        // 2-arg math
        for name in &["min", "max", "atan2", "fmod"] {
            self.builtins.insert(name.to_string(), BuiltinFn {
                params: vec![PdcType::F64, PdcType::F64],
                ret: PdcType::F64,
                takes_ctx: false,
            });
        }

        // Type casts (handled specially, not as regular functions)
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
        for item in &program.items {
            self.check_item(item)?;
        }
        Ok(())
    }

    fn check_item(&mut self, item: &Spanned<Item>) -> Result<(), PdcError> {
        match &item.node {
            Item::BuiltinDecl { name, ty } => {
                self.define_var(name, ty.clone());
                Ok(())
            }
            Item::ConstDecl { name, ty, value } => {
                let val_ty = self.check_expr(value)?;
                let final_ty = if let Some(declared) = ty {
                    self.check_compatible(&val_ty, declared, value.span)?;
                    declared.clone()
                } else {
                    val_ty
                };
                self.define_var(name, final_ty);
                Ok(())
            }
            Item::VarDecl { name, ty, value } => {
                let val_ty = self.check_expr(value)?;
                let final_ty = if let Some(declared) = ty {
                    self.check_compatible(&val_ty, declared, value.span)?;
                    declared.clone()
                } else {
                    val_ty
                };
                self.define_var(name, final_ty);
                Ok(())
            }
            Item::Assign { name, value } => {
                let expected = self.lookup_var(name).cloned().ok_or_else(|| PdcError::Type {
                    span: item.span,
                    message: format!("undefined variable '{name}'"),
                })?;
                let val_ty = self.check_expr(value)?;
                self.check_compatible(&val_ty, &expected, value.span)?;
                Ok(())
            }
            Item::ExprStmt(expr) => {
                self.check_expr(expr)?;
                Ok(())
            }
        }
    }

    fn check_expr(&mut self, expr: &Spanned<Expr>) -> Result<PdcType, PdcError> {
        let ty = match &expr.node {
            Expr::Literal(lit) => match lit {
                Literal::Int(v) => {
                    // Hex-range values → u32, others → i32
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
            Expr::Call { name, args } => {
                // Check for type cast: f32(x), i32(x), etc.
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
                } else {
                    return Err(PdcError::Type {
                        span: expr.span,
                        message: format!("undefined function '{name}'"),
                    });
                }
            }
            Expr::MethodCall { object, method, args } => {
                // UFCS: desugar to Call(method, [object] + args)
                let obj_ty = self.check_expr(object)?;

                // Look up as a regular function with object as first arg
                if let Some(builtin) = self.builtins.get(method.as_str()) {
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
                } else {
                    return Err(PdcError::Type {
                        span: expr.span,
                        message: format!("undefined method '{method}'"),
                    });
                }
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
                let _ = self.unify_numeric(left, right, span)?;
                Ok(PdcType::Bool)
            }
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod => {
                self.unify_numeric(left, right, span)
            }
        }
    }

    /// Find the common type for two numeric types (implicit widening).
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
        // Widening rules: i32 → f64, u32 → f64, f32 → f64
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
        // Allow implicit widening
        if from.is_numeric() && to.is_numeric() {
            return Ok(());
        }
        Err(PdcError::Type {
            span,
            message: format!("type mismatch: expected {to}, got {from}"),
        })
    }
}
