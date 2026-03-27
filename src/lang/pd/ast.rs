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
    F32Lit(f32, Span),
    IntLit(u64, Span),
    I8Lit(i8, Span),
    U8Lit(u8, Span),
    I16Lit(i16, Span),
    U16Lit(u16, Span),
    I32Lit(i32, Span),
    U32Lit(u32, Span),
    I64Lit(i64, Span),
    U64Lit(u64, Span),
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
    StructLit {
        name: String,
        fields: Vec<(String, Expr)>,
        span: Span,
    },
}

impl Expr {
    pub fn span(&self) -> &Span {
        match self {
            Expr::FloatLit(_, s) => s,
            Expr::F32Lit(_, s) => s,
            Expr::IntLit(_, s) => s,
            Expr::I8Lit(_, s) => s,
            Expr::U8Lit(_, s) => s,
            Expr::I16Lit(_, s) => s,
            Expr::U16Lit(_, s) => s,
            Expr::I32Lit(_, s) => s,
            Expr::U32Lit(_, s) => s,
            Expr::I64Lit(_, s) => s,
            Expr::U64Lit(_, s) => s,
            Expr::BoolLit(_, s) => s,
            Expr::Ident(_, s) => s,
            Expr::BinOp { span, .. } => span,
            Expr::UnaryOp { span, .. } => span,
            Expr::Call { span, .. } => span,
            Expr::Cast { span, .. } => span,
            Expr::IfElse { span, .. } => span,
            Expr::FieldAccess { span, .. } => span,
            Expr::StructLit { span, .. } => span,
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
    BufStore {
        buf_name: String,
        x: Expr,
        y: Expr,
        val: Expr,
        span: Span,
    },
}

#[derive(Debug, Clone)]
pub struct Param {
    pub name: String,
    pub ty: ValType,
}

#[derive(Debug, Clone)]
pub struct BufferParam {
    pub name: String,
    pub is_output: bool,
}

#[derive(Debug, Clone)]
pub struct TextureParam {
    pub name: String,
}

#[derive(Debug, Clone)]
pub struct AstStructDef {
    pub name: String,
    pub fields: Vec<(String, ValType)>,
    pub span: Span,
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
    pub buffers: Vec<BufferParam>,
    pub textures: Vec<TextureParam>,
    pub body: Vec<Stmt>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct Program {
    pub struct_defs: Vec<AstStructDef>,
    pub fns: Vec<FnDef>,
    pub kernel: KernelDef,
}
