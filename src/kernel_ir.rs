/// SSA-based kernel IR for per-pixel computation.
///
/// A kernel body describes per-pixel computation. Backends wrap it in
/// the tile loop (row/col iteration, coordinate math, pixel store).
/// The kernel declares explicit parameters (e.g. `x: f64, y: f64`)
/// which are assigned `Var` indices in declaration order, and produces
/// a `u32` ARGB color via `emit`.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValType {
    F64,
    U32,
    Bool,
    Vec2, // 2x f64
    Vec3, // 3x f64
}

impl ValType {
    /// Returns true if this is a vector type.
    pub fn is_vec(self) -> bool {
        matches!(self, ValType::Vec2 | ValType::Vec3)
    }

    /// Number of scalar components (1 for scalars, 2 for Vec2, 3 for Vec3).
    pub fn component_count(self) -> usize {
        match self {
            ValType::Vec2 => 2,
            ValType::Vec3 => 3,
            _ => 1,
        }
    }

    /// The element type of a vector (F64), or self for scalars.
    pub fn element_type(self) -> ValType {
        match self {
            ValType::Vec2 | ValType::Vec3 => ValType::F64,
            other => other,
        }
    }
}

impl std::fmt::Display for ValType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValType::F64 => write!(f, "f64"),
            ValType::U32 => write!(f, "u32"),
            ValType::Bool => write!(f, "bool"),
            ValType::Vec2 => write!(f, "vec2"),
            ValType::Vec3 => write!(f, "vec3"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Var(pub u32);

#[derive(Debug, Clone)]
pub struct Binding {
    pub var: Var,
    pub name: String,
    pub ty: ValType,
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
    Min,
    Max,
    Atan2,
    Pow,
    Hash,
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
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Exp,
    Exp2,
    Log,
    Log2,
    Log10,
    Round,
    Trunc,
    Fract,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvOp {
    F64ToU32,
    U32ToF64,
    U32ToF64Norm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VecBinOp {
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VecUnaryOp {
    Neg,
    Abs,
    Normalize,
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

    // Vector construction
    MakeVec2 { x: Var, y: Var },
    MakeVec3 { x: Var, y: Var, z: Var },

    // Component extraction (vec -> f64)
    VecExtract { vec: Var, index: u8 }, // 0=x, 1=y, 2=z

    // Component-wise binary (vec op vec -> vec)
    VecBinary { op: VecBinOp, lhs: Var, rhs: Var },

    // Scalar-vector multiply (f64 * vec -> vec)
    VecScale { scalar: Var, vec: Var },

    // Unary (vec -> vec)
    VecUnary { op: VecUnaryOp, arg: Var },

    // Reductions (vec -> f64)
    VecDot { lhs: Var, rhs: Var },
    VecLength { arg: Var },

    // Cross product (vec3 x vec3 -> vec3)
    VecCross { lhs: Var, rhs: Var },
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
    pub cond_body: Vec<BodyItem>,
    pub cond: Var,
    pub body: Vec<BodyItem>,
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
    pub return_ty: ValType,
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
    pub fn var_type(&self, var: Var) -> Option<ValType> {
        self.binding(var).map(|b| b.ty)
    }

    /// Get the name of a Var.
    pub fn var_name(&self, var: Var) -> Option<&str> {
        self.binding(var).map(|b| b.name.as_str())
    }
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
                if let Some(b) = find_binding_in_body(&w.cond_body, var) {
                    return Some(b);
                }
                if let Some(b) = find_binding_in_body(&w.body, var) {
                    return Some(b);
                }
            }
        }
    }
    None
}
