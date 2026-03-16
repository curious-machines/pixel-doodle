use crate::kernel_ir::ValType;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOpKind {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
    And,
    Or,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOpKind {
    Neg,
    Not,
}

#[derive(Debug, Clone)]
pub struct Span {
    pub line: usize,
    pub col: usize,
}

#[derive(Debug, Clone)]
pub enum Expr {
    FloatLit(f64, Span),
    IntLit(u64, Span),
    U32Lit(u32, Span),
    BoolLit(bool, Span),
    Ident(String, Span),
    BinOp {
        op: BinOpKind,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
        span: Span,
    },
    UnaryOp {
        op: UnaryOpKind,
        expr: Box<Expr>,
        span: Span,
    },
    Call {
        name: String,
        args: Vec<Expr>,
        span: Span,
    },
    Cast {
        expr: Box<Expr>,
        ty: ValType,
        span: Span,
    },
    IfElse {
        cond: Box<Expr>,
        then_expr: Box<Expr>,
        else_expr: Box<Expr>,
        span: Span,
    },
    FieldAccess {
        expr: Box<Expr>,
        field: String,
        span: Span,
    },
}

impl Expr {
    pub fn span(&self) -> &Span {
        match self {
            Expr::FloatLit(_, s) => s,
            Expr::IntLit(_, s) => s,
            Expr::U32Lit(_, s) => s,
            Expr::BoolLit(_, s) => s,
            Expr::Ident(_, s) => s,
            Expr::BinOp { span, .. } => span,
            Expr::UnaryOp { span, .. } => span,
            Expr::Call { span, .. } => span,
            Expr::Cast { span, .. } => span,
            Expr::IfElse { span, .. } => span,
            Expr::FieldAccess { span, .. } => span,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CarryDef {
    pub name: String,
    pub ty: Option<ValType>,
    pub init: Expr,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum Stmt {
    Let {
        name: String,
        ty: Option<ValType>,
        expr: Expr,
        span: Span,
    },
    While {
        carry: Vec<CarryDef>,
        body: Vec<Stmt>,
        span: Span,
    },
    BreakIf {
        cond: Expr,
        span: Span,
    },
    Yield {
        values: Vec<Expr>,
        span: Span,
    },
    Emit {
        expr: Expr,
        span: Span,
    },
}

#[derive(Debug, Clone)]
pub struct Param {
    pub name: String,
    pub ty: ValType,
}

#[derive(Debug, Clone)]
pub struct FnDef {
    pub name: String,
    pub params: Vec<Param>,
    pub return_ty: ValType,
    pub body: Vec<Stmt>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct KernelDef {
    pub name: String,
    pub params: Vec<Param>,
    pub return_ty: ValType,
    pub body: Vec<Stmt>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct Program {
    pub fns: Vec<FnDef>,
    pub kernel: KernelDef,
}
