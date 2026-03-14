/// SSA-based kernel IR for per-pixel computation.
///
/// A kernel body describes per-pixel computation. Backends wrap it in
/// the tile loop (row/col iteration, coordinate math, pixel store).
/// The kernel receives implicit `x: f64` (Var(0)) and `y: f64` (Var(1))
/// and produces a `u32` ARGB color via `emit`.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarType {
    F64,
    U32,
    Bool,
}

impl std::fmt::Display for ScalarType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScalarType::F64 => write!(f, "f64"),
            ScalarType::U32 => write!(f, "u32"),
            ScalarType::Bool => write!(f, "bool"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Var(pub u32);

#[derive(Debug, Clone)]
pub struct Binding {
    pub var: Var,
    pub name: String,
    pub ty: ScalarType,
}

#[derive(Debug, Clone, Copy)]
pub enum Const {
    F64(f64),
    U32(u32),
    Bool(bool),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Not,
    Abs,
    Sqrt,
    Floor,
    Ceil,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvOp {
    F64ToU32,
    U32ToF64,
}

#[derive(Debug, Clone)]
pub enum Inst {
    Const(Const),
    Binary { op: BinOp, lhs: Var, rhs: Var },
    Unary { op: UnaryOp, arg: Var },
    Cmp { op: CmpOp, lhs: Var, rhs: Var },
    Conv { op: ConvOp, arg: Var },
    Select { cond: Var, then_val: Var, else_val: Var },
    PackArgb { r: Var, g: Var, b: Var },
}

#[derive(Debug, Clone)]
pub struct Statement {
    pub binding: Binding,
    pub inst: Inst,
}

/// V1: body is Vec<Statement> (no loops).
/// V2 will change body to Vec<BodyItem> to include While.
#[derive(Debug, Clone)]
pub struct Kernel {
    pub name: String,
    pub body: Vec<Statement>,
    pub emit: Var,
}

impl Kernel {
    /// Look up a binding by Var index.
    pub fn binding(&self, var: Var) -> Option<&Binding> {
        // Check implicit vars
        if var.0 < 2 {
            return None; // x and y are implicit, no binding
        }
        self.body.iter().find(|s| s.binding.var == var).map(|s| &s.binding)
    }

    /// Get the type of a Var.
    pub fn var_type(&self, var: Var) -> Option<ScalarType> {
        match var.0 {
            0 | 1 => Some(ScalarType::F64), // x, y
            _ => self.binding(var).map(|b| b.ty),
        }
    }

    /// Get the name of a Var.
    pub fn var_name(&self, var: Var) -> Option<&str> {
        match var.0 {
            0 => Some("x"),
            1 => Some("y"),
            _ => self.binding(var).map(|b| b.name.as_str()),
        }
    }
}
