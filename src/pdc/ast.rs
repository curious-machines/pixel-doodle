use super::span::Spanned;

/// PDC types known at compile time.
#[derive(Debug, Clone, PartialEq)]
pub enum PdcType {
    F32,
    F64,
    I32,
    U32,
    Bool,
    /// Opaque path handle (u32 internally). Becomes a struct in later phases.
    PathHandle,
    /// User-defined struct type, referenced by name.
    Struct(String),
    /// User-defined enum type, referenced by name.
    Enum(String),
    /// Array of f64 values (handle-based, runtime-managed).
    ArrayF64,
    /// Type not yet determined (for inference).
    Unknown,
    /// No return value.
    Void,
}

impl PdcType {
    pub fn is_numeric(&self) -> bool {
        matches!(self, PdcType::F32 | PdcType::F64 | PdcType::I32 | PdcType::U32)
    }

    pub fn is_float(&self) -> bool {
        matches!(self, PdcType::F32 | PdcType::F64)
    }

    pub fn is_int(&self) -> bool {
        matches!(self, PdcType::I32 | PdcType::U32)
    }
}

impl std::fmt::Display for PdcType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PdcType::F32 => write!(f, "f32"),
            PdcType::F64 => write!(f, "f64"),
            PdcType::I32 => write!(f, "i32"),
            PdcType::U32 => write!(f, "u32"),
            PdcType::Bool => write!(f, "bool"),
            PdcType::PathHandle => write!(f, "Path"),
            PdcType::Struct(name) | PdcType::Enum(name) => write!(f, "{name}"),
            PdcType::ArrayF64 => write!(f, "array<f64>"),
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
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    And,
    Or,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UnaryOp {
    Neg,
    Not,
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
}

/// A statement within a block or at top level.
#[derive(Debug, Clone)]
pub enum Stmt {
    /// `builtin const name: type`
    BuiltinDecl { name: String, ty: PdcType },
    /// `const name [: type] = expr`
    ConstDecl {
        name: String,
        ty: Option<PdcType>,
        value: Spanned<Expr>,
    },
    /// `var name [: type] = expr`
    VarDecl {
        name: String,
        ty: Option<PdcType>,
        value: Spanned<Expr>,
    },
    /// `name = expr`
    Assign { name: String, value: Spanned<Expr> },
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
    /// `for name in start..end { body }` (exclusive range)
    For {
        var_name: String,
        start: Spanned<Expr>,
        end: Spanned<Expr>,
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
    /// `fn name(params) [-> type] { body }`
    FnDef(FnDef),
    /// `struct Name { field: type, ... }`
    StructDef(StructDef),
    /// `enum Name { Variant1, Variant2, ... }`
    EnumDef(EnumDef),
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
}

/// Function definition.
#[derive(Debug, Clone)]
pub struct FnDef {
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
    pub name: String,
    pub variants: Vec<EnumVariant>,
}

/// A complete PDC program (list of top-level statements).
#[derive(Debug)]
pub struct Program {
    pub stmts: Vec<Spanned<Stmt>>,
}
