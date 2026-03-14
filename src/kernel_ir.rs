/// SSA-based kernel IR for per-pixel computation.
///
/// A kernel body describes per-pixel computation. Backends wrap it in
/// the tile loop (row/col iteration, coordinate math, pixel store).
/// The kernel declares explicit parameters (e.g. `x: f64, y: f64`)
/// which are assigned `Var` indices in declaration order, and produces
/// a `u32` ARGB color via `emit`.

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

/// A loop-carried variable: bound at loop entry, updated each iteration via yield.
#[derive(Debug, Clone)]
pub struct CarryVar {
    pub binding: Binding,
    pub init: Var,
}

/// Structured while loop with loop-carried values.
///
/// Semantics: on each iteration, execute `cond_body`, check `cond`.
/// If true, execute `body`, update carry vars with `yields`, repeat.
/// If false, exit; carry vars hold their current values.
#[derive(Debug, Clone)]
pub struct While {
    pub carry: Vec<CarryVar>,
    pub cond_body: Vec<Statement>,
    pub cond: Var,
    pub body: Vec<Statement>,
    pub yields: Vec<Var>,
}

#[derive(Debug, Clone)]
pub enum BodyItem {
    Stmt(Statement),
    While(While),
}

#[derive(Debug, Clone)]
pub struct Kernel {
    pub name: String,
    pub params: Vec<Binding>,
    pub body: Vec<BodyItem>,
    pub emit: Var,
}

impl Kernel {
    /// Look up a binding by Var index, searching params and body items.
    pub fn binding(&self, var: Var) -> Option<&Binding> {
        if let Some(b) = self.params.iter().find(|b| b.var == var) {
            return Some(b);
        }
        find_binding_in_body(&self.body, var)
    }

    /// Get the type of a Var.
    pub fn var_type(&self, var: Var) -> Option<ScalarType> {
        self.binding(var).map(|b| b.ty)
    }

    /// Get the name of a Var.
    pub fn var_name(&self, var: Var) -> Option<&str> {
        self.binding(var).map(|b| b.name.as_str())
    }
}

fn find_binding_in_stmts(stmts: &[Statement], var: Var) -> Option<&Binding> {
    stmts.iter().find(|s| s.binding.var == var).map(|s| &s.binding)
}

fn find_binding_in_body(body: &[BodyItem], var: Var) -> Option<&Binding> {
    for item in body {
        match item {
            BodyItem::Stmt(stmt) => {
                if stmt.binding.var == var {
                    return Some(&stmt.binding);
                }
            }
            BodyItem::While(w) => {
                for cv in &w.carry {
                    if cv.binding.var == var {
                        return Some(&cv.binding);
                    }
                }
                if let Some(b) = find_binding_in_stmts(&w.cond_body, var) {
                    return Some(b);
                }
                if let Some(b) = find_binding_in_stmts(&w.body, var) {
                    return Some(b);
                }
            }
        }
    }
    None
}
