use super::span::Spanned;

/// Visibility of a definition within a module.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Visibility {
    Public,
    Private,
}

/// PDC types known at compile time.
#[derive(Debug, Clone, PartialEq)]
pub enum PdcType {
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    Bool,
    /// Opaque path handle (u32 internally). Becomes a struct in later phases.
    PathHandle,
    /// UTF-8 string type. Runtime-managed, handle-based.
    Str,
    /// User-defined struct type, referenced by name.
    Struct(String),
    /// User-defined enum type, referenced by name.
    Enum(String),
    /// Array type with element type. Runtime stores as f64 internally.
    Array(Box<PdcType>),
    /// Tuple type: (T, U, ...)
    Tuple(Vec<PdcType>),
    /// Slice type: view into array without copying.
    Slice(Box<PdcType>),
    /// Function reference type: fn(params) -> ret
    FnRef {
        params: Vec<PdcType>,
        ret: Box<PdcType>,
    },
    /// Module namespace for namespaced imports (e.g., `import math` → `math` has type Module).
    Module(String),
    /// Type not yet determined (for inference).
    Unknown,
    /// No return value.
    Void,
}

impl PdcType {
    pub fn is_numeric(&self) -> bool {
        self.is_float() || self.is_int()
    }

    pub fn is_float(&self) -> bool {
        matches!(self, PdcType::F32 | PdcType::F64)
    }

    pub fn is_int(&self) -> bool {
        matches!(self, PdcType::I8 | PdcType::I16 | PdcType::I32 | PdcType::I64
            | PdcType::U8 | PdcType::U16 | PdcType::U32 | PdcType::U64)
    }
}

impl std::fmt::Display for PdcType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PdcType::F32 => write!(f, "f32"),
            PdcType::F64 => write!(f, "f64"),
            PdcType::I8 => write!(f, "i8"),
            PdcType::I16 => write!(f, "i16"),
            PdcType::I32 => write!(f, "i32"),
            PdcType::I64 => write!(f, "i64"),
            PdcType::U8 => write!(f, "u8"),
            PdcType::U16 => write!(f, "u16"),
            PdcType::U32 => write!(f, "u32"),
            PdcType::U64 => write!(f, "u64"),
            PdcType::Bool => write!(f, "bool"),
            PdcType::PathHandle => write!(f, "Path"),
            PdcType::Str => write!(f, "string"),
            PdcType::Struct(name) | PdcType::Enum(name) => write!(f, "{name}"),
            PdcType::Array(elem) => write!(f, "Array<{elem}>"),
            PdcType::Tuple(elems) => {
                write!(f, "(")?;
                for (i, e) in elems.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{e}")?;
                }
                write!(f, ")")
            }
            PdcType::Slice(elem) => write!(f, "slice<{elem}>"),
            PdcType::FnRef { params, ret } => {
                write!(f, "fn(")?;
                for (i, p) in params.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{p}")?;
                }
                write!(f, ") -> {ret}")
            }
            PdcType::Module(name) => write!(f, "module({name})"),
            PdcType::Unknown => write!(f, "unknown"),
            PdcType::Void => write!(f, "void"),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Literal {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    And,
    Or,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UnaryOp {
    Neg,
    Not,
    BitNot,
}

use super::token::TokenKind;

pub fn token_to_op_name(token: &TokenKind) -> Option<&'static str> {
    match token {
        TokenKind::Plus => Some("__op_add__"),
        TokenKind::Minus => Some("__op_sub__"),
        TokenKind::Star => Some("__op_mul__"),
        TokenKind::Slash => Some("__op_div__"),
        TokenKind::Percent => Some("__op_mod__"),
        TokenKind::StarStar => Some("__op_pow__"),
        TokenKind::EqEq => Some("__op_eq__"),
        TokenKind::BangEq => Some("__op_neq__"),
        TokenKind::Lt => Some("__op_lt__"),
        TokenKind::LtEq => Some("__op_lteq__"),
        TokenKind::Gt => Some("__op_gt__"),
        TokenKind::GtEq => Some("__op_gteq__"),
        TokenKind::Amp => Some("__op_bitand__"),
        TokenKind::Pipe => Some("__op_bitor__"),
        TokenKind::Caret => Some("__op_bitxor__"),
        TokenKind::LtLt => Some("__op_shl__"),
        TokenKind::GtGt => Some("__op_shr__"),
        TokenKind::AmpAmp => Some("__op_and__"),
        TokenKind::PipePipe => Some("__op_or__"),
        TokenKind::Bang => Some("__op_not__"),
        TokenKind::Tilde => Some("__op_bitnot__"),
        _ => None,
    }
}

pub fn binop_to_op_name(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "__op_add__",
        BinOp::Sub => "__op_sub__",
        BinOp::Mul => "__op_mul__",
        BinOp::Div => "__op_div__",
        BinOp::Mod => "__op_mod__",
        BinOp::Pow => "__op_pow__",
        BinOp::Eq => "__op_eq__",
        BinOp::NotEq => "__op_neq__",
        BinOp::Lt => "__op_lt__",
        BinOp::LtEq => "__op_lteq__",
        BinOp::Gt => "__op_gt__",
        BinOp::GtEq => "__op_gteq__",
        BinOp::And => "__op_and__",
        BinOp::Or => "__op_or__",
        BinOp::BitAnd => "__op_bitand__",
        BinOp::BitOr => "__op_bitor__",
        BinOp::BitXor => "__op_bitxor__",
        BinOp::Shl => "__op_shl__",
        BinOp::Shr => "__op_shr__",
    }
}

pub fn unaryop_to_op_name(op: UnaryOp) -> &'static str {
    match op {
        UnaryOp::Neg => "__op_neg__",
        UnaryOp::Not => "__op_not__",
        UnaryOp::BitNot => "__op_bitnot__",
    }
}

#[derive(Debug, Clone)]
pub enum Expr {
    Literal(Literal),
    Variable(String),
    BinaryOp {
        op: BinOp,
        left: Box<Spanned<Expr>>,
        right: Box<Spanned<Expr>>,
    },
    UnaryOp {
        op: UnaryOp,
        operand: Box<Spanned<Expr>>,
    },
    /// Function or constructor call: `name(args)` or `name(arg_name: expr, ...)`
    Call {
        name: String,
        args: Vec<Spanned<Expr>>,
        /// Optional argument names (empty = positional, non-empty = named).
        arg_names: Vec<Option<String>>,
    },
    /// UFCS method call: `expr.name(args)` — desugared to Call during type checking
    MethodCall {
        object: Box<Spanned<Expr>>,
        method: String,
        args: Vec<Spanned<Expr>>,
    },
    /// Field access: `expr.field` (resolved during type checking)
    FieldAccess {
        object: Box<Spanned<Expr>>,
        field: String,
    },
    /// Struct constructor with named args: `TypeName(field: value, ...)`
    StructConstruct {
        name: String,
        fields: Vec<(String, Spanned<Expr>)>,
    },
    /// Tuple construction: `(expr, expr, ...)`
    TupleConstruct {
        elements: Vec<Spanned<Expr>>,
    },
    /// Tuple element access: `expr.0`, `expr.1` etc.
    TupleIndex {
        object: Box<Spanned<Expr>>,
        index: usize,
    },
    /// Array index access: `expr[index]`
    Index {
        object: Box<Spanned<Expr>>,
        index: Box<Spanned<Expr>>,
    },
    /// Ternary: `cond ? then_expr : else_expr`
    Ternary {
        condition: Box<Spanned<Expr>>,
        then_expr: Box<Spanned<Expr>>,
        else_expr: Box<Spanned<Expr>>,
    },
}

/// A statement within a block or at top level.
#[derive(Debug, Clone)]
pub enum Stmt {
    /// `builtin const name: type`
    BuiltinDecl { name: String, ty: PdcType },
    /// `[pub] const name [: type] = expr`
    ConstDecl {
        vis: Visibility,
        name: String,
        ty: Option<PdcType>,
        value: Spanned<Expr>,
    },
    /// `[pub] var name [: type] = expr`
    VarDecl {
        vis: Visibility,
        name: String,
        ty: Option<PdcType>,
        value: Spanned<Expr>,
    },
    /// `name = expr`
    Assign { name: String, value: Spanned<Expr> },
    /// `expr[index] = value`
    IndexAssign {
        object: Spanned<Expr>,
        index: Spanned<Expr>,
        value: Spanned<Expr>,
    },
    /// `const (a, b, c) = expr` or `var (a, b, c) = expr`
    TupleDestructure {
        names: Vec<String>,
        value: Spanned<Expr>,
        is_const: bool,
    },
    /// Expression statement (function call, method call)
    ExprStmt(Spanned<Expr>),
    /// `if cond { body } [elsif cond { body }]* [else { body }]`
    If {
        condition: Spanned<Expr>,
        then_body: Block,
        elsif_clauses: Vec<(Spanned<Expr>, Block)>,
        else_body: Option<Block>,
    },
    /// `while cond { body }`
    While {
        condition: Spanned<Expr>,
        body: Block,
    },
    /// `for [const|var] name in start..end { body }` (exclusive)
    /// `for [const|var] name in start..=end { body }` (inclusive)
    For {
        var_name: String,
        mutable: bool,
        start: Spanned<Expr>,
        end: Spanned<Expr>,
        inclusive: bool,
        body: Block,
    },
    /// `for [const|var] name in collection { body }` (iterate over array)
    /// `for [const|var] (a, b) in collection { body }` (destructuring)
    ForEach {
        var_name: String,
        /// Tuple destructuring names. If non-empty, var_name is ignored.
        destructure_names: Vec<String>,
        mutable: bool,
        collection: Spanned<Expr>,
        body: Block,
    },
    /// `loop { body }` — infinite loop, exit with break
    Loop { body: Block },
    /// `match expr { pattern => { body }, ... }`
    Match {
        scrutinee: Spanned<Expr>,
        arms: Vec<MatchArm>,
    },
    /// `break`
    Break,
    /// `continue`
    Continue,
    /// `return [expr]`
    Return(Option<Spanned<Expr>>),
    /// `[pub] fn name(params) [-> type] { body }`
    FnDef(FnDef),
    /// `[pub] struct Name { field: type, ... }`
    StructDef(StructDef),
    /// `[pub] enum Name { Variant1, Variant2, ... }`
    EnumDef(EnumDef),
    /// `[pub] type Name = ExistingType`
    TypeAlias {
        vis: Visibility,
        name: String,
        ty: PdcType,
    },
    /// `test "name" { body }` — PDC-level unit test
    TestDef {
        name: String,
        body: Block,
    },
    /// `import module_name` or `import { names } from module_name`
    Import {
        module: String,
        /// If empty, import the whole module (namespaced access).
        /// If non-empty, import specific names (direct access).
        names: Vec<String>,
    },
}

/// A block of statements.
#[derive(Debug, Clone)]
pub struct Block {
    pub stmts: Vec<Spanned<Stmt>>,
}

/// Function parameter.
#[derive(Debug, Clone)]
pub struct Param {
    pub name: String,
    pub ty: PdcType,
    /// Optional default value expression (evaluated fresh at each call site).
    pub default: Option<Spanned<Expr>>,
}

/// Function definition.
#[derive(Debug, Clone)]
pub struct FnDef {
    pub vis: Visibility,
    pub name: String,
    pub params: Vec<Param>,
    pub return_type: PdcType,
    pub body: Block,
}

/// Struct field definition.
#[derive(Debug, Clone)]
pub struct StructField {
    pub name: String,
    pub ty: PdcType,
}

/// Struct definition.
#[derive(Debug, Clone)]
pub struct StructDef {
    pub vis: Visibility,
    pub name: String,
    pub fields: Vec<StructField>,
}

/// A match arm: pattern => { body }
#[derive(Debug, Clone)]
pub struct MatchArm {
    pub pattern: MatchPattern,
    pub body: Block,
}

/// A pattern in a match arm.
#[derive(Debug, Clone)]
pub enum MatchPattern {
    /// Enum variant, optionally with destructuring bindings:
    /// `EnumName.Variant` or `EnumName.Variant(a, b, c)`
    EnumVariant {
        enum_name: String,
        variant: String,
        bindings: Vec<String>,
    },
    /// Catch-all wildcard (future)
    Wildcard,
}

/// Enum variant field.
#[derive(Debug, Clone)]
pub struct EnumVariantField {
    pub name: String,
    pub ty: PdcType,
}

/// Enum variant definition.
#[derive(Debug, Clone)]
pub struct EnumVariant {
    pub name: String,
    /// Payload fields (empty for C-style variants).
    pub fields: Vec<EnumVariantField>,
}

/// Enum definition with optional data variants.
#[derive(Debug, Clone)]
pub struct EnumDef {
    pub vis: Visibility,
    pub name: String,
    pub variants: Vec<EnumVariant>,
}

/// A parsed module unit.
#[derive(Debug)]
pub struct ModuleUnit {
    pub name: String,
    pub stmts: Vec<Spanned<Stmt>>,
}

/// A complete PDC program: imported modules + main program statements.
#[derive(Debug)]
pub struct Program {
    pub modules: Vec<ModuleUnit>,
    pub stmts: Vec<Spanned<Stmt>>,
}
