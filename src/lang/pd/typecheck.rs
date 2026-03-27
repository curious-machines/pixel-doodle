use crate::kernel_ir::{ScalarType, ValType};
use super::ast::*;
use std::collections::HashMap;

#[derive(Debug)]
pub struct TypeError {
    pub line: usize,
    pub col: usize,
    pub message: String,
}

impl std::fmt::Display for TypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}: {}", self.line, self.col, self.message)
    }
}

fn err(span: &Span, msg: String) -> TypeError {
    TypeError { line: span.line, col: span.col, message: msg }
}

pub type TypeResult<T> = Result<T, TypeError>;

/// Typed expression — every node annotated with its resolved type.
#[derive(Debug, Clone)]
pub enum TExpr {
    FloatLit(f64),
    F32Lit(f32),
    IntLit(u64, ValType),
    I8Lit(i8),
    U8Lit(u8),
    I16Lit(i16),
    U16Lit(u16),
    I32Lit(i32),
    U32Lit(u32),
    I64Lit(i64),
    U64Lit(u64),
    BoolLit(bool),
    Ident(String, ValType),
    BinOp {
        op: BinOpKind,
        lhs: Box<TExpr>,
        rhs: Box<TExpr>,
        ty: ValType,
    },
    UnaryOp {
        op: UnaryOpKind,
        expr: Box<TExpr>,
        ty: ValType,
    },
    Call {
        name: String,
        args: Vec<TExpr>,
        ty: ValType,
    },
    Cast {
        expr: Box<TExpr>,
        from: ValType,
        to: ValType,
    },
    IfElse {
        cond: Box<TExpr>,
        then_expr: Box<TExpr>,
        else_expr: Box<TExpr>,
        ty: ValType,
    },
    FieldAccess {
        expr: Box<TExpr>,
        field: String,
        ty: ValType,
    },
}

impl TExpr {
    pub fn ty(&self) -> ValType {
        match self {
            TExpr::FloatLit(_) => ValType::F64,
            TExpr::F32Lit(_) => ValType::F32,
            TExpr::IntLit(_, ty) => *ty,
            TExpr::I8Lit(_) => ValType::I8,
            TExpr::U8Lit(_) => ValType::U8,
            TExpr::I16Lit(_) => ValType::I16,
            TExpr::U16Lit(_) => ValType::U16,
            TExpr::I32Lit(_) => ValType::I32,
            TExpr::U32Lit(_) => ValType::U32,
            TExpr::I64Lit(_) => ValType::I64,
            TExpr::U64Lit(_) => ValType::U64,
            TExpr::BoolLit(_) => ValType::BOOL,
            TExpr::Ident(_, ty) => *ty,
            TExpr::BinOp { ty, .. } => *ty,
            TExpr::UnaryOp { ty, .. } => *ty,
            TExpr::Call { ty, .. } => *ty,
            TExpr::Cast { to, .. } => *to,
            TExpr::IfElse { ty, .. } => *ty,
            TExpr::FieldAccess { ty, .. } => *ty,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TCarryDef {
    pub name: String,
    pub ty: ValType,
    pub init: TExpr,
}

#[derive(Debug, Clone)]
pub enum TStmt {
    Let {
        name: String,
        ty: ValType,
        expr: TExpr,
    },
    While {
        carry: Vec<TCarryDef>,
        body: Vec<TStmt>,
    },
    BreakIf {
        cond: TExpr,
    },
    Yield {
        values: Vec<TExpr>,
    },
    Emit {
        expr: TExpr,
    },
    BufStore {
        buf_name: String,
        x: TExpr,
        y: TExpr,
        val: TExpr,
    },
}

#[derive(Debug, Clone)]
pub struct TFnDef {
    pub name: String,
    pub params: Vec<Param>,
    pub return_ty: ValType,
    pub body: Vec<TStmt>,
}

#[derive(Debug, Clone)]
pub struct TKernelDef {
    pub name: String,
    pub params: Vec<Param>,
    pub return_ty: ValType,
    pub buffers: Vec<BufferParam>,
    pub body: Vec<TStmt>,
}

#[derive(Debug, Clone)]
pub struct TProgram {
    pub fns: Vec<TFnDef>,
    pub kernel: TKernelDef,
}

struct Checker {
    /// Stack of scopes. Each scope maps names to types.
    scopes: Vec<HashMap<String, ValType>>,
    /// Inline function definitions (not yet type-checked body, just signatures).
    fn_sigs: HashMap<String, (Vec<Param>, ValType)>,
    /// Buffer declarations (name → is_output).
    buffers: HashMap<String, bool>,
}

impl Checker {
    fn new() -> Self {
        Self {
            scopes: vec![HashMap::new()],
            fn_sigs: HashMap::new(),
            buffers: HashMap::new(),
        }
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn define(&mut self, name: &str, ty: ValType, span: &Span) -> TypeResult<()> {
        // Check for redefinition in current scope only
        if self.scopes.last().unwrap().contains_key(name) {
            return Err(err(span, format!("variable '{}' already defined in this scope", name)));
        }
        self.scopes.last_mut().unwrap().insert(name.to_string(), ty);
        Ok(())
    }

    fn lookup(&self, name: &str) -> Option<ValType> {
        for scope in self.scopes.iter().rev() {
            if let Some(ty) = scope.get(name) {
                return Some(*ty);
            }
        }
        None
    }

    fn check_expr(&self, expr: &Expr, expected: Option<ValType>) -> TypeResult<TExpr> {
        match expr {
            Expr::FloatLit(v, _) => Ok(TExpr::FloatLit(*v)),
            Expr::F32Lit(v, _) => Ok(TExpr::F32Lit(*v)),
            Expr::I8Lit(v, _) => Ok(TExpr::I8Lit(*v)),
            Expr::U8Lit(v, _) => Ok(TExpr::U8Lit(*v)),
            Expr::I16Lit(v, _) => Ok(TExpr::I16Lit(*v)),
            Expr::U16Lit(v, _) => Ok(TExpr::U16Lit(*v)),
            Expr::U32Lit(v, _) => Ok(TExpr::U32Lit(*v)),
            Expr::I32Lit(v, _) => Ok(TExpr::I32Lit(*v)),
            Expr::I64Lit(v, _) => Ok(TExpr::I64Lit(*v)),
            Expr::U64Lit(v, _) => Ok(TExpr::U64Lit(*v)),
            Expr::BoolLit(v, _) => Ok(TExpr::BoolLit(*v)),

            Expr::IntLit(v, span) => {
                // Resolve bare integer from context
                let ty = expected.unwrap_or_else(|| {
                    // Default: if it fits u32, use u32; else... it's ambiguous
                    ValType::U32
                });
                match ty {
                    ValType::Scalar(ScalarType::F64) => Ok(TExpr::FloatLit(*v as f64)),
                    ValType::Scalar(ScalarType::F32) => Ok(TExpr::F32Lit(*v as f32)),
                    ValType::Scalar(ScalarType::I8) => {
                        if *v > i8::MAX as u64 {
                            return Err(err(span, format!("integer {} too large for i8", v)));
                        }
                        Ok(TExpr::I8Lit(*v as i8))
                    }
                    ValType::Scalar(ScalarType::U8) => {
                        if *v > u8::MAX as u64 {
                            return Err(err(span, format!("integer {} too large for u8", v)));
                        }
                        Ok(TExpr::U8Lit(*v as u8))
                    }
                    ValType::Scalar(ScalarType::I16) => {
                        if *v > i16::MAX as u64 {
                            return Err(err(span, format!("integer {} too large for i16", v)));
                        }
                        Ok(TExpr::I16Lit(*v as i16))
                    }
                    ValType::Scalar(ScalarType::U16) => {
                        if *v > u16::MAX as u64 {
                            return Err(err(span, format!("integer {} too large for u16", v)));
                        }
                        Ok(TExpr::U16Lit(*v as u16))
                    }
                    ValType::Scalar(ScalarType::U32) => {
                        if *v > u32::MAX as u64 {
                            return Err(err(span, format!("integer {} too large for u32", v)));
                        }
                        Ok(TExpr::U32Lit(*v as u32))
                    }
                    ValType::Scalar(ScalarType::I32) => {
                        if *v > i32::MAX as u64 {
                            return Err(err(span, format!("integer {} too large for i32", v)));
                        }
                        Ok(TExpr::I32Lit(*v as i32))
                    }
                    ValType::Scalar(ScalarType::I64) => {
                        if *v > i64::MAX as u64 {
                            return Err(err(span, format!("integer {} too large for i64", v)));
                        }
                        Ok(TExpr::I64Lit(*v as i64))
                    }
                    ValType::Scalar(ScalarType::U64) => {
                        Ok(TExpr::U64Lit(*v))
                    }
                    ValType::Scalar(ScalarType::Bool) => Err(err(span, "cannot use integer literal as bool".into())),
                    ValType::Vec { .. } => Err(err(span, "cannot use integer literal as vec type".into())),
                    ValType::Mat { .. } => Err(err(span, "cannot use integer literal as mat type".into())),
                }
            }

            Expr::Ident(name, span) => {
                let ty = self.lookup(name)
                    .ok_or_else(|| err(span, format!("undefined variable '{}'", name)))?;
                Ok(TExpr::Ident(name.clone(), ty))
            }

            Expr::BinOp { op, lhs, rhs, span } => {
                self.check_binop(*op, lhs, rhs, span)
            }

            Expr::UnaryOp { op, expr: inner, span } => {
                match op {
                    UnaryOpKind::Neg => {
                        let inner = self.check_expr(inner, Some(ValType::F64))?;
                        let ty = inner.ty();
                        let negatable = match ty {
                            ValType::Scalar(s) => s.is_float() || s.is_integer(),
                            ValType::Vec { .. } => true,
                            ValType::Mat { .. } => false,
                        };
                        if !negatable {
                            return Err(err(span, format!("cannot negate {}", ty)));
                        }
                        Ok(TExpr::UnaryOp { op: *op, expr: Box::new(inner), ty })
                    }
                    UnaryOpKind::Not => {
                        let inner = self.check_expr(inner, Some(ValType::BOOL))?;
                        if inner.ty() != ValType::BOOL {
                            return Err(err(span, format!("'!' requires bool, got {}", inner.ty())));
                        }
                        Ok(TExpr::UnaryOp { op: *op, expr: Box::new(inner), ty: ValType::BOOL })
                    }
                }
            }

            Expr::Call { name, args, span } => {
                self.check_call(name, args, span)
            }

            Expr::Cast { expr: inner, ty, span } => {
                let inner = self.check_expr(inner, None)?;
                let from = inner.ty();
                // Validate the conversion exists
                match (&from, ty) {
                    // Allow any scalar-to-scalar cast except involving Bool
                    (ValType::Scalar(f), ValType::Scalar(t))
                        if *f != ScalarType::Bool && *t != ScalarType::Bool && f != t => {}
                    _ => return Err(err(span, format!("cannot cast {} to {}", from, ty))),
                }
                Ok(TExpr::Cast { expr: Box::new(inner), from, to: *ty })
            }

            Expr::FieldAccess { expr: inner, field, span } => {
                let inner = self.check_expr(inner, None)?;
                let inner_ty = inner.ty();
                match (inner_ty, field.as_str()) {
                    (ValType::Vec { len: 2, .. }, "x") | (ValType::Vec { len: 2, .. }, "y") |
                    (ValType::Vec { len: 3, .. }, "x") | (ValType::Vec { len: 3, .. }, "y") | (ValType::Vec { len: 3, .. }, "z") |
                    (ValType::Vec { len: 4, .. }, "x") | (ValType::Vec { len: 4, .. }, "y") | (ValType::Vec { len: 4, .. }, "z") | (ValType::Vec { len: 4, .. }, "w") => {
                        let elem = inner_ty.element_scalar();
                        Ok(TExpr::FieldAccess {
                            expr: Box::new(inner),
                            field: field.clone(),
                            ty: ValType::Scalar(elem),
                        })
                    }
                    (ValType::Vec { len: 2, .. }, "z") | (ValType::Vec { len: 2, .. }, "w") => {
                        Err(err(span, format!("vec2 has no '{}' component", field)))
                    }
                    (ValType::Vec { len: 3, .. }, "w") => {
                        Err(err(span, "vec3 has no 'w' component".into()))
                    }
                    _ => {
                        Err(err(span, format!("type {} has no field '{}'", inner_ty, field)))
                    }
                }
            }

            Expr::IfElse { cond, then_expr, else_expr, span } => {
                let cond = self.check_expr(cond, Some(ValType::BOOL))?;
                if cond.ty() != ValType::BOOL {
                    return Err(err(span, format!("if condition must be bool, got {}", cond.ty())));
                }
                let then_val = self.check_expr(then_expr, expected)?;
                let else_val = self.check_expr(else_expr, Some(then_val.ty()))?;
                if then_val.ty() != else_val.ty() {
                    return Err(err(span, format!(
                        "if branches have different types: {} vs {}",
                        then_val.ty(), else_val.ty()
                    )));
                }
                let ty = then_val.ty();
                Ok(TExpr::IfElse {
                    cond: Box::new(cond),
                    then_expr: Box::new(then_val),
                    else_expr: Box::new(else_val),
                    ty,
                })
            }
        }
    }

    fn check_binop(&self, op: BinOpKind, lhs: &Expr, rhs: &Expr, span: &Span) -> TypeResult<TExpr> {
        use BinOpKind::*;

        match op {
            // Logical: bool × bool → bool
            And | Or => {
                let l = self.check_expr(lhs, Some(ValType::BOOL))?;
                let r = self.check_expr(rhs, Some(ValType::BOOL))?;
                if l.ty() != ValType::BOOL || r.ty() != ValType::BOOL {
                    return Err(err(span, format!("logical op requires bool operands, got {} and {}", l.ty(), r.ty())));
                }
                Ok(TExpr::BinOp { op, lhs: Box::new(l), rhs: Box::new(r), ty: ValType::BOOL })
            }

            // Comparison: T × T → bool
            Eq | Ne | Lt | Le | Gt | Ge => {
                // Try to infer one from the other
                let l = self.check_expr(lhs, None)?;
                let r = self.check_expr(rhs, Some(l.ty()))?;
                if l.ty() != r.ty() {
                    // Re-check lhs with rhs context
                    let r2 = self.check_expr(rhs, None)?;
                    let l2 = self.check_expr(lhs, Some(r2.ty()))?;
                    if l2.ty() != r2.ty() {
                        return Err(err(span, format!("comparison operands must match: {} vs {}", l2.ty(), r2.ty())));
                    }
                    return Ok(TExpr::BinOp { op, lhs: Box::new(l2), rhs: Box::new(r2), ty: ValType::BOOL });
                }
                Ok(TExpr::BinOp { op, lhs: Box::new(l), rhs: Box::new(r), ty: ValType::BOOL })
            }

            // Bitwise: u32 × u32 → u32 or i32 × i32 → i32
            BitAnd | BitOr | BitXor | Shl | Shr => {
                let l = self.check_expr(lhs, Some(ValType::U32))?;
                let r = self.check_expr(rhs, Some(l.ty()))?;
                let ty = l.ty();
                let is_int = matches!(ty, ValType::Scalar(s) if s.is_integer());
                if !is_int || r.ty() != ty {
                    return Err(err(span, format!("bitwise op requires matching integer operands, got {} and {}", l.ty(), r.ty())));
                }
                Ok(TExpr::BinOp { op, lhs: Box::new(l), rhs: Box::new(r), ty })
            }

            // Arithmetic: T × T → T (f64, u32, or vec)
            Add | Sub | Mul | Div | Rem => {
                let l = self.check_expr(lhs, expected_for_arith(rhs, &|e| self.check_expr(e, None)))?;
                let r = self.check_expr(rhs, Some(l.ty()))?;

                // Handle matrix multiply: mat * vec -> vec, mat * mat -> mat
                if op == Mul {
                    if let ValType::Mat { size, elem } = l.ty() {
                        let col_ty = ValType::Vec { len: size, elem };
                        if r.ty() == col_ty {
                            // mat * vec -> vec
                            return Ok(TExpr::BinOp { op, lhs: Box::new(l), rhs: Box::new(r), ty: col_ty });
                        }
                        let mat_ty = ValType::Mat { size, elem };
                        if r.ty() == mat_ty {
                            // mat * mat -> mat
                            return Ok(TExpr::BinOp { op, lhs: Box::new(l), rhs: Box::new(r), ty: mat_ty });
                        }
                    }
                }

                // Handle scalar-vec multiply: f64 * vec or vec * f64
                if op == Mul {
                    if l.ty() == ValType::F64 && r.ty().is_vec() {
                        let ty = r.ty();
                        return Ok(TExpr::BinOp { op, lhs: Box::new(l), rhs: Box::new(r), ty });
                    }
                    if l.ty().is_vec() && r.ty() == ValType::F64 {
                        let ty = l.ty();
                        return Ok(TExpr::BinOp { op, lhs: Box::new(l), rhs: Box::new(r), ty });
                    }
                }
                // Handle vec / f64
                if op == Div && l.ty().is_vec() && r.ty() == ValType::F64 {
                    let ty = l.ty();
                    return Ok(TExpr::BinOp { op, lhs: Box::new(l), rhs: Box::new(r), ty });
                }

                if l.ty() != r.ty() {
                    // Re-check with rhs context
                    let r2 = self.check_expr(rhs, None)?;
                    let l2 = self.check_expr(lhs, Some(r2.ty()))?;

                    // Check mixed mat*vec / mat*mat again with re-resolved types
                    if op == Mul {
                        if let ValType::Mat { size, elem } = l2.ty() {
                            let col_ty = ValType::Vec { len: size, elem };
                            if r2.ty() == col_ty {
                                return Ok(TExpr::BinOp { op, lhs: Box::new(l2), rhs: Box::new(r2), ty: col_ty });
                            }
                            let mat_ty = ValType::Mat { size, elem };
                            if r2.ty() == mat_ty {
                                return Ok(TExpr::BinOp { op, lhs: Box::new(l2), rhs: Box::new(r2), ty: mat_ty });
                            }
                        }
                    }

                    // Check mixed scalar-vec again with re-resolved types
                    if op == Mul {
                        if l2.ty() == ValType::F64 && r2.ty().is_vec() {
                            let ty = r2.ty();
                            return Ok(TExpr::BinOp { op, lhs: Box::new(l2), rhs: Box::new(r2), ty });
                        }
                        if l2.ty().is_vec() && r2.ty() == ValType::F64 {
                            let ty = l2.ty();
                            return Ok(TExpr::BinOp { op, lhs: Box::new(l2), rhs: Box::new(r2), ty });
                        }
                    }
                    if op == Div && l2.ty().is_vec() && r2.ty() == ValType::F64 {
                        let ty = l2.ty();
                        return Ok(TExpr::BinOp { op, lhs: Box::new(l2), rhs: Box::new(r2), ty });
                    }

                    if l2.ty() != r2.ty() {
                        return Err(err(span, format!("arithmetic operands must match: {} vs {}", l2.ty(), r2.ty())));
                    }
                    let ty = l2.ty();
                    return Ok(TExpr::BinOp { op, lhs: Box::new(l2), rhs: Box::new(r2), ty });
                }
                let ty = l.ty();
                if ty == ValType::BOOL {
                    return Err(err(span, "cannot do arithmetic on bool".into()));
                }
                if ty.is_mat() && op != Mul {
                    return Err(err(span, format!("only '*' is supported for mat types, got '{:?}'", op)));
                }
                if ty.is_vec() && op == Rem {
                    return Err(err(span, "remainder not supported for vec types".into()));
                }
                Ok(TExpr::BinOp { op, lhs: Box::new(l), rhs: Box::new(r), ty })
            }
        }
    }

    fn check_call(&self, name: &str, args: &[Expr], span: &Span) -> TypeResult<TExpr> {
        // Check built-in functions first, then user-defined
        if let Some(result) = self.check_builtin_call(name, args, span)? {
            return Ok(result);
        }

        // User-defined function
        let (params, ret_ty) = self.fn_sigs.get(name)
            .ok_or_else(|| err(span, format!("undefined function '{}'", name)))?;

        if args.len() != params.len() {
            return Err(err(span, format!(
                "function '{}' expects {} args, got {}",
                name, params.len(), args.len()
            )));
        }

        let mut targs = Vec::new();
        for (arg, param) in args.iter().zip(params.iter()) {
            let ta = self.check_expr(arg, Some(param.ty))?;
            if ta.ty() != param.ty {
                return Err(err(arg.span(), format!(
                    "argument type mismatch for '{}': expected {}, got {}",
                    param.name, param.ty, ta.ty()
                )));
            }
            targs.push(ta);
        }

        Ok(TExpr::Call { name: name.to_string(), args: targs, ty: *ret_ty })
    }

    fn check_builtin_call(&self, name: &str, args: &[Expr], span: &Span) -> TypeResult<Option<TExpr>> {
        // Unary f64 → f64 functions
        let unary_f64 = [
            "sqrt", "floor", "ceil", "sin", "cos", "tan",
            "asin", "acos", "atan", "exp", "exp2", "log", "log2", "log10",
            "round", "trunc", "fract",
        ];
        if unary_f64.contains(&name) {
            if args.len() != 1 {
                return Err(err(span, format!("'{}' expects 1 argument", name)));
            }
            let arg = self.check_expr(&args[0], Some(ValType::F64))?;
            if arg.ty() != ValType::F64 {
                return Err(err(span, format!("'{}' requires f64 argument, got {}", name, arg.ty())));
            }
            return Ok(Some(TExpr::Call { name: name.to_string(), args: vec![arg], ty: ValType::F64 }));
        }

        // Binary f64 × f64 → f64
        let binary_f64 = ["atan2", "pow"];
        if binary_f64.contains(&name) {
            if args.len() != 2 {
                return Err(err(span, format!("'{}' expects 2 arguments", name)));
            }
            let a = self.check_expr(&args[0], Some(ValType::F64))?;
            let b = self.check_expr(&args[1], Some(ValType::F64))?;
            if a.ty() != ValType::F64 || b.ty() != ValType::F64 {
                return Err(err(span, format!("'{}' requires f64 arguments", name)));
            }
            return Ok(Some(TExpr::Call { name: name.to_string(), args: vec![a, b], ty: ValType::F64 }));
        }

        // hash(u32, u32) → u32
        if name == "hash" {
            if args.len() != 2 {
                return Err(err(span, "hash expects 2 arguments".into()));
            }
            let a = self.check_expr(&args[0], Some(ValType::U32))?;
            let b = self.check_expr(&args[1], Some(ValType::U32))?;
            if a.ty() != ValType::U32 || b.ty() != ValType::U32 {
                return Err(err(span, "hash requires u32 arguments".into()));
            }
            return Ok(Some(TExpr::Call { name: "hash".into(), args: vec![a, b], ty: ValType::U32 }));
        }

        // Convenience builtins
        match name {
            "abs" => {
                if args.len() != 1 {
                    return Err(err(span, "abs expects 1 argument".into()));
                }
                let arg = self.check_expr(&args[0], None)?;
                match arg.ty() {
                    ValType::Scalar(ScalarType::F64) => return Ok(Some(TExpr::Call { name: "abs".into(), args: vec![arg], ty: ValType::F64 })),
                    ValType::Vec { .. } => {
                        let ty = arg.ty();
                        return Ok(Some(TExpr::Call { name: "abs".into(), args: vec![arg], ty }));
                    }
                    _ => return Err(err(span, format!("abs requires f64 or vec argument, got {}", arg.ty()))),
                }
            }
            "min" | "max" => {
                if args.len() != 2 {
                    return Err(err(span, format!("'{}' expects 2 arguments", name)));
                }
                let a = self.check_expr(&args[0], None)?;
                let b = self.check_expr(&args[1], Some(a.ty()))?;
                if a.ty() != b.ty() {
                    return Err(err(span, format!("'{}' arguments must match: {} vs {}", name, a.ty(), b.ty())));
                }
                match a.ty() {
                    ValType::Scalar(ScalarType::F64) | ValType::Scalar(ScalarType::F32) | ValType::Scalar(ScalarType::U32) | ValType::Scalar(ScalarType::I32) | ValType::Vec { .. } => {
                        let ty = a.ty();
                        return Ok(Some(TExpr::Call { name: name.to_string(), args: vec![a, b], ty }));
                    }
                    _ => return Err(err(span, format!("'{}' not supported for {}", name, a.ty()))),
                }
            }
            "vec2" | "vec3" | "vec4" => {
                let expected_len: usize = match name {
                    "vec2" => 2, "vec3" => 3, "vec4" => 4, _ => unreachable!(),
                };
                if args.len() != expected_len {
                    return Err(err(span, format!("{} expects {} arguments", name, expected_len)));
                }
                // Check first arg without type hint to infer element type
                let first = self.check_expr(&args[0], None)?;
                let elem = match first.ty() {
                    ValType::Scalar(s) if s != ScalarType::Bool => s,
                    other => return Err(err(span, format!("{} arguments must be numeric scalars, got {}", name, other))),
                };
                let expected_ty = ValType::Scalar(elem);
                let mut checked = vec![first];
                for arg in &args[1..] {
                    let c = self.check_expr(arg, Some(expected_ty))?;
                    if c.ty() != expected_ty {
                        return Err(err(span, format!("{} arguments must all be {}, got {}", name, elem, c.ty())));
                    }
                    checked.push(c);
                }
                return Ok(Some(TExpr::Call {
                    name: name.to_string(),
                    args: checked,
                    ty: ValType::Vec { len: expected_len as u8, elem },
                }));
            }
            "mat2" | "mat3" | "mat4" => {
                let expected_cols: usize = match name {
                    "mat2" => 2, "mat3" => 3, "mat4" => 4, _ => unreachable!(),
                };
                if args.len() != expected_cols {
                    return Err(err(span, format!("{} expects {} column arguments", name, expected_cols)));
                }
                // Check first arg to infer element type
                let first = self.check_expr(&args[0], None)?;
                let elem = match first.ty() {
                    ValType::Vec { len, elem } if len == expected_cols as u8 => elem,
                    other => return Err(err(span, format!(
                        "{} arguments must be vec{}<T>, got {}", name, expected_cols, other
                    ))),
                };
                let col_ty = ValType::Vec { len: expected_cols as u8, elem };
                let mut checked = vec![first];
                for arg in &args[1..] {
                    let c = self.check_expr(arg, Some(col_ty))?;
                    if c.ty() != col_ty {
                        return Err(err(span, format!(
                            "{} arguments must all be {}, got {}", name, col_ty, c.ty()
                        )));
                    }
                    checked.push(c);
                }
                return Ok(Some(TExpr::Call {
                    name: name.to_string(),
                    args: checked,
                    ty: ValType::Mat { size: expected_cols as u8, elem },
                }));
            }
            "transpose" => {
                if args.len() != 1 {
                    return Err(err(span, "transpose expects 1 argument".into()));
                }
                let arg = self.check_expr(&args[0], None)?;
                if !arg.ty().is_mat() {
                    return Err(err(span, format!("transpose requires mat argument, got {}", arg.ty())));
                }
                let ty = arg.ty();
                return Ok(Some(TExpr::Call { name: "transpose".into(), args: vec![arg], ty }));
            }
            "col" => {
                if args.len() != 2 {
                    return Err(err(span, "col expects 2 arguments: col(mat, index)".into()));
                }
                let mat = self.check_expr(&args[0], None)?;
                let mat_ty = mat.ty();
                if !mat_ty.is_mat() {
                    return Err(err(span, format!("col: first argument must be a matrix, got {}", mat_ty)));
                }
                let idx = self.check_expr(&args[1], Some(ValType::U32))?;
                // Validate index is a literal
                match &idx {
                    TExpr::U32Lit(i) => {
                        let size = match mat_ty { ValType::Mat { size, .. } => size, _ => unreachable!() };
                        if *i >= size as u32 {
                            return Err(err(span, format!("col index {} out of range for mat{}", i, size)));
                        }
                    }
                    TExpr::IntLit(i, _) => {
                        let size = match mat_ty { ValType::Mat { size, .. } => size, _ => unreachable!() };
                        if *i >= size as u64 {
                            return Err(err(span, format!("col index {} out of range for mat{}", i, size)));
                        }
                    }
                    _ => return Err(err(span, "col: index must be an integer literal".into())),
                }
                let col_ty = mat_ty.mat_col_type();
                return Ok(Some(TExpr::Call { name: "col".into(), args: vec![mat, idx], ty: col_ty }));
            }
            "dot" => {
                if args.len() != 2 {
                    return Err(err(span, "dot expects 2 arguments".into()));
                }
                let a = self.check_expr(&args[0], None)?;
                let b = self.check_expr(&args[1], Some(a.ty()))?;
                if !a.ty().is_vec() || a.ty() != b.ty() {
                    return Err(err(span, format!("dot requires matching vec types, got {} and {}", a.ty(), b.ty())));
                }
                return Ok(Some(TExpr::Call { name: "dot".into(), args: vec![a, b], ty: ValType::F64 }));
            }
            "normalize" => {
                if args.len() != 1 {
                    return Err(err(span, "normalize expects 1 argument".into()));
                }
                let arg = self.check_expr(&args[0], None)?;
                if !arg.ty().is_vec() {
                    return Err(err(span, format!("normalize requires vec argument, got {}", arg.ty())));
                }
                let ty = arg.ty();
                return Ok(Some(TExpr::Call { name: "normalize".into(), args: vec![arg], ty }));
            }
            "cross" => {
                if args.len() != 2 {
                    return Err(err(span, "cross expects 2 arguments".into()));
                }
                let vec3_ty = ValType::Vec { len: 3, elem: ScalarType::F64 };
                let a = self.check_expr(&args[0], Some(vec3_ty))?;
                let b = self.check_expr(&args[1], Some(vec3_ty))?;
                if !matches!(a.ty(), ValType::Vec { len: 3, .. }) || !matches!(b.ty(), ValType::Vec { len: 3, .. }) {
                    return Err(err(span, format!("cross requires vec3 arguments, got {} and {}", a.ty(), b.ty())));
                }
                return Ok(Some(TExpr::Call { name: "cross".into(), args: vec![a, b], ty: vec3_ty }));
            }
            "clamp" => {
                if args.len() != 3 {
                    return Err(err(span, "clamp expects 3 arguments".into()));
                }
                let x = self.check_expr(&args[0], Some(ValType::F64))?;
                let lo = self.check_expr(&args[1], Some(ValType::F64))?;
                let hi = self.check_expr(&args[2], Some(ValType::F64))?;
                return Ok(Some(TExpr::Call { name: "clamp".into(), args: vec![x, lo, hi], ty: ValType::F64 }));
            }
            "saturate" => {
                if args.len() != 1 {
                    return Err(err(span, "saturate expects 1 argument".into()));
                }
                let x = self.check_expr(&args[0], Some(ValType::F64))?;
                return Ok(Some(TExpr::Call { name: "saturate".into(), args: vec![x], ty: ValType::F64 }));
            }
            "length" => {
                if args.len() == 1 {
                    let arg = self.check_expr(&args[0], None)?;
                    if arg.ty().is_vec() {
                        return Ok(Some(TExpr::Call { name: "length".into(), args: vec![arg], ty: ValType::F64 }));
                    }
                    return Err(err(span, format!("length with 1 arg requires vec, got {}", arg.ty())));
                }
                if args.len() == 2 {
                    let x = self.check_expr(&args[0], Some(ValType::F64))?;
                    let y = self.check_expr(&args[1], Some(ValType::F64))?;
                    return Ok(Some(TExpr::Call { name: "length".into(), args: vec![x, y], ty: ValType::F64 }));
                }
                return Err(err(span, "length expects 1 or 2 arguments".into()));
            }
            "distance" => {
                if args.len() == 2 {
                    let a = self.check_expr(&args[0], None)?;
                    let b = self.check_expr(&args[1], Some(a.ty()))?;
                    if a.ty().is_vec() && a.ty() == b.ty() {
                        return Ok(Some(TExpr::Call { name: "distance".into(), args: vec![a, b], ty: ValType::F64 }));
                    }
                    // fall through to error
                }
                if args.len() == 4 {
                    let args_t: Vec<TExpr> = args.iter()
                        .map(|a| self.check_expr(a, Some(ValType::F64)))
                        .collect::<Result<_, _>>()?;
                    return Ok(Some(TExpr::Call { name: "distance".into(), args: args_t, ty: ValType::F64 }));
                }
                return Err(err(span, "distance expects 2 (vec) or 4 (f64) arguments".into()));
            }
            "mix" => {
                if args.len() != 3 {
                    return Err(err(span, "mix expects 3 arguments".into()));
                }
                let a = self.check_expr(&args[0], None)?;
                let b = self.check_expr(&args[1], Some(a.ty()))?;
                let t = self.check_expr(&args[2], Some(ValType::F64))?;
                if a.ty().is_vec() && a.ty() == b.ty() && t.ty() == ValType::F64 {
                    let ty = a.ty();
                    return Ok(Some(TExpr::Call { name: "mix".into(), args: vec![a, b, t], ty }));
                }
                if a.ty() == ValType::F64 && b.ty() == ValType::F64 && t.ty() == ValType::F64 {
                    return Ok(Some(TExpr::Call { name: "mix".into(), args: vec![a, b, t], ty: ValType::F64 }));
                }
                return Err(err(span, "mix: args must be (f64,f64,f64) or (vec,vec,f64)".into()));
            }
            "smoothstep" => {
                if args.len() != 3 {
                    return Err(err(span, "smoothstep expects 3 arguments".into()));
                }
                let e0 = self.check_expr(&args[0], Some(ValType::F64))?;
                let e1 = self.check_expr(&args[1], Some(ValType::F64))?;
                let x = self.check_expr(&args[2], Some(ValType::F64))?;
                return Ok(Some(TExpr::Call { name: "smoothstep".into(), args: vec![e0, e1, x], ty: ValType::F64 }));
            }
            "step" => {
                if args.len() != 2 {
                    return Err(err(span, "step expects 2 arguments".into()));
                }
                let edge = self.check_expr(&args[0], Some(ValType::F64))?;
                let x = self.check_expr(&args[1], Some(ValType::F64))?;
                return Ok(Some(TExpr::Call { name: "step".into(), args: vec![edge, x], ty: ValType::F64 }));
            }
            "rgb" => {
                if args.len() != 3 {
                    return Err(err(span, "rgb expects 3 arguments".into()));
                }
                let r = self.check_expr(&args[0], Some(ValType::F64))?;
                let g = self.check_expr(&args[1], Some(ValType::F64))?;
                let b = self.check_expr(&args[2], Some(ValType::F64))?;
                return Ok(Some(TExpr::Call { name: "rgb".into(), args: vec![r, g, b], ty: ValType::U32 }));
            }
            "rgb255" => {
                if args.len() != 3 {
                    return Err(err(span, "rgb255 expects 3 arguments".into()));
                }
                let r = self.check_expr(&args[0], Some(ValType::F64))?;
                let g = self.check_expr(&args[1], Some(ValType::F64))?;
                let b = self.check_expr(&args[2], Some(ValType::F64))?;
                return Ok(Some(TExpr::Call { name: "rgb255".into(), args: vec![r, g, b], ty: ValType::U32 }));
            }
            "gray" => {
                if args.len() != 1 {
                    return Err(err(span, "gray expects 1 argument".into()));
                }
                let v = self.check_expr(&args[0], Some(ValType::F64))?;
                return Ok(Some(TExpr::Call { name: "gray".into(), args: vec![v], ty: ValType::U32 }));
            }
            "gray255" => {
                if args.len() != 1 {
                    return Err(err(span, "gray255 expects 1 argument".into()));
                }
                let v = self.check_expr(&args[0], Some(ValType::F64))?;
                return Ok(Some(TExpr::Call { name: "gray255".into(), args: vec![v], ty: ValType::U32 }));
            }
            "norm" => {
                if args.len() != 1 {
                    return Err(err(span, "norm expects 1 argument".into()));
                }
                let v = self.check_expr(&args[0], Some(ValType::U32))?;
                if v.ty() != ValType::U32 {
                    return Err(err(span, format!("norm requires u32 argument, got {}", v.ty())));
                }
                return Ok(Some(TExpr::Call { name: "norm".into(), args: vec![v], ty: ValType::F64 }));
            }
            "pack_argb" => {
                if args.len() != 3 {
                    return Err(err(span, "pack_argb expects 3 arguments".into()));
                }
                let r = self.check_expr(&args[0], Some(ValType::U32))?;
                let g = self.check_expr(&args[1], Some(ValType::U32))?;
                let b = self.check_expr(&args[2], Some(ValType::U32))?;
                return Ok(Some(TExpr::Call { name: "pack_argb".into(), args: vec![r, g, b], ty: ValType::U32 }));
            }
            "f64_to_u32" => {
                if args.len() != 1 {
                    return Err(err(span, "f64_to_u32 expects 1 argument".into()));
                }
                let v = self.check_expr(&args[0], Some(ValType::F64))?;
                return Ok(Some(TExpr::Call { name: "f64_to_u32".into(), args: vec![v], ty: ValType::U32 }));
            }
            "u32_to_f64" => {
                if args.len() != 1 {
                    return Err(err(span, "u32_to_f64 expects 1 argument".into()));
                }
                let v = self.check_expr(&args[0], Some(ValType::U32))?;
                return Ok(Some(TExpr::Call { name: "u32_to_f64".into(), args: vec![v], ty: ValType::F64 }));
            }
            "buf_load" => {
                if args.len() != 3 {
                    return Err(err(span, "buf_load expects 3 arguments: buf_load(buffer, x, y)".into()));
                }
                // First arg must be an identifier naming a declared read buffer
                let buf_name = match &args[0] {
                    Expr::Ident(name, _) => name.clone(),
                    _ => return Err(err(span, "buf_load: first argument must be a buffer name".into())),
                };
                match self.buffers.get(&buf_name) {
                    Some(true) => return Err(err(span, format!("buf_load: '{}' is a write buffer, cannot read", buf_name))),
                    None => return Err(err(span, format!("buf_load: unknown buffer '{}'", buf_name))),
                    _ => {}
                }
                let x = self.check_expr(&args[1], Some(ValType::U32))?;
                let y = self.check_expr(&args[2], Some(ValType::U32))?;
                return Ok(Some(TExpr::Call {
                    name: format!("buf_load:{}", buf_name),
                    args: vec![x, y],
                    ty: ValType::F64,
                }));
            }
            "select" => {
                if args.len() != 3 {
                    return Err(err(span, "select expects 3 arguments".into()));
                }
                let c = self.check_expr(&args[0], Some(ValType::BOOL))?;
                let t = self.check_expr(&args[1], None)?;
                let e = self.check_expr(&args[2], Some(t.ty()))?;
                let ty = t.ty();
                return Ok(Some(TExpr::Call { name: "select".into(), args: vec![c, t, e], ty }));
            }
            _ => {}
        }

        Ok(None)
    }

    fn check_stmts(&mut self, stmts: &[Stmt]) -> TypeResult<Vec<TStmt>> {
        stmts.iter().map(|s| self.check_stmt(s)).collect()
    }

    fn check_stmt(&mut self, stmt: &Stmt) -> TypeResult<TStmt> {
        match stmt {
            Stmt::Let { name, ty, expr, span } => {
                let texpr = self.check_expr(expr, *ty)?;
                let resolved_ty = if let Some(ann) = ty {
                    if texpr.ty() != *ann {
                        return Err(err(span, format!(
                            "type annotation {} doesn't match expression type {}",
                            ann, texpr.ty()
                        )));
                    }
                    *ann
                } else {
                    texpr.ty()
                };
                self.define(name, resolved_ty, span)?;
                Ok(TStmt::Let { name: name.clone(), ty: resolved_ty, expr: texpr })
            }

            Stmt::While { carry, body, span } => {
                // Type-check carry init expressions in outer scope
                let mut tcarry = Vec::new();
                for c in carry {
                    let init = self.check_expr(&c.init, c.ty)?;
                    let ty = if let Some(ann) = c.ty {
                        if init.ty() != ann {
                            return Err(err(&c.span, format!(
                                "carry var '{}': annotation {} doesn't match init type {}",
                                c.name, ann, init.ty()
                            )));
                        }
                        ann
                    } else {
                        init.ty()
                    };
                    tcarry.push(TCarryDef { name: c.name.clone(), ty, init });
                }

                // Push scope for loop body, define carry vars
                self.push_scope();
                for c in &tcarry {
                    self.define(&c.name, c.ty, span)?;
                }

                let tbody = self.check_stmts(body)?;
                self.pop_scope();

                // Carry vars are live after the loop — define in outer scope
                for c in &tcarry {
                    self.define(&c.name, c.ty, span)?;
                }

                Ok(TStmt::While { carry: tcarry, body: tbody })
            }

            Stmt::BreakIf { cond, span } => {
                let tcond = self.check_expr(cond, Some(ValType::BOOL))?;
                if tcond.ty() != ValType::BOOL {
                    return Err(err(span, format!("break_if condition must be bool, got {}", tcond.ty())));
                }
                Ok(TStmt::BreakIf { cond: tcond })
            }

            Stmt::Yield { values, span: _ } => {
                let tvals: Vec<TExpr> = values.iter()
                    .map(|v| self.check_expr(v, None))
                    .collect::<Result<_, _>>()?;
                Ok(TStmt::Yield { values: tvals })
            }

            Stmt::Emit { expr, span: _ } => {
                let texpr = self.check_expr(expr, None)?;
                Ok(TStmt::Emit { expr: texpr })
            }

            Stmt::BufStore { buf_name, x, y, val, span } => {
                match self.buffers.get(buf_name) {
                    Some(false) => return Err(err(span, format!("buf_store: '{}' is a read buffer, cannot write", buf_name))),
                    None => return Err(err(span, format!("buf_store: unknown buffer '{}'", buf_name))),
                    _ => {}
                }
                let tx = self.check_expr(x, Some(ValType::U32))?;
                let ty_expr = self.check_expr(y, Some(ValType::U32))?;
                let tval = self.check_expr(val, Some(ValType::F64))?;
                Ok(TStmt::BufStore { buf_name: buf_name.clone(), x: tx, y: ty_expr, val: tval })
            }
        }
    }
}

/// Try to get a type hint from an expression that isn't a bare integer.
fn expected_for_arith(expr: &Expr, check: &dyn Fn(&Expr) -> TypeResult<TExpr>) -> Option<ValType> {
    match expr {
        Expr::FloatLit(_, _) => Some(ValType::F64),
        Expr::F32Lit(_, _) => Some(ValType::F32),
        Expr::I8Lit(_, _) => Some(ValType::I8),
        Expr::U8Lit(_, _) => Some(ValType::U8),
        Expr::I16Lit(_, _) => Some(ValType::I16),
        Expr::U16Lit(_, _) => Some(ValType::U16),
        Expr::I32Lit(_, _) => Some(ValType::I32),
        Expr::U32Lit(_, _) => Some(ValType::U32),
        Expr::I64Lit(_, _) => Some(ValType::I64),
        Expr::U64Lit(_, _) => Some(ValType::U64),
        Expr::BoolLit(_, _) => None,
        Expr::IntLit(_, _) => None,
        Expr::Ident(_, _) | Expr::Call { .. } | Expr::Cast { .. } | Expr::IfElse { .. } | Expr::FieldAccess { .. } => {
            check(expr).ok().map(|t| t.ty())
        }
        Expr::BinOp { .. } | Expr::UnaryOp { .. } => {
            check(expr).ok().map(|t| t.ty())
        }
    }
}

pub fn typecheck(program: &Program) -> TypeResult<TProgram> {
    let mut checker = Checker::new();

    // Register fn signatures
    for f in &program.fns {
        checker.fn_sigs.insert(f.name.clone(), (f.params.clone(), f.return_ty));
    }

    // Type-check fn bodies
    let mut tfns = Vec::new();
    for f in &program.fns {
        checker.push_scope();
        for p in &f.params {
            checker.define(&p.name, p.ty, &f.span)?;
        }
        let tbody = checker.check_stmts(&f.body)?;
        checker.pop_scope();
        tfns.push(TFnDef {
            name: f.name.clone(),
            params: f.params.clone(),
            return_ty: f.return_ty,
            body: tbody,
        });
    }

    // Type-check kernel
    checker.scopes = vec![HashMap::new()];
    for p in &program.kernel.params {
        checker.define(&p.name, p.ty, &program.kernel.span)?;
    }
    // Register buffers
    checker.buffers.clear();
    for b in &program.kernel.buffers {
        checker.buffers.insert(b.name.clone(), b.is_output);
    }
    let tbody = checker.check_stmts(&program.kernel.body)?;

    Ok(TProgram {
        fns: tfns,
        kernel: TKernelDef {
            name: program.kernel.name.clone(),
            params: program.kernel.params.clone(),
            return_ty: program.kernel.return_ty,
            buffers: program.kernel.buffers.clone(),
            body: tbody,
        },
    })
}
