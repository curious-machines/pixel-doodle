/// SSA-based kernel IR for per-pixel computation.
///
/// A kernel body describes per-pixel computation. Backends wrap it in
/// the tile loop (row/col iteration, coordinate math, pixel store).
/// The kernel declares explicit parameters (e.g. `x: f64, y: f64`)
/// which are assigned `Var` indices in declaration order, and produces
/// a `u32` ARGB color via `emit`.

/// Scalar (non-compound) types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarType {
    F32,
    F64,
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    Bool,
}

impl ScalarType {
    /// Returns true if this is a floating-point type.
    pub fn is_float(self) -> bool {
        matches!(self, ScalarType::F32 | ScalarType::F64)
    }

    /// Returns true if this is an integer type (signed or unsigned).
    pub fn is_integer(self) -> bool {
        matches!(self, ScalarType::I8 | ScalarType::U8 | ScalarType::I16 | ScalarType::U16
            | ScalarType::I32 | ScalarType::U32 | ScalarType::I64 | ScalarType::U64)
    }

    /// Returns true if this is a signed integer type.
    pub fn is_signed(self) -> bool {
        matches!(self, ScalarType::I8 | ScalarType::I16 | ScalarType::I32 | ScalarType::I64)
    }

    /// Returns true if this is an unsigned integer type.
    pub fn is_unsigned(self) -> bool {
        matches!(self, ScalarType::U8 | ScalarType::U16 | ScalarType::U32 | ScalarType::U64)
    }

    /// Size in bytes of this scalar type.
    pub fn byte_size(self) -> usize {
        match self {
            ScalarType::Bool | ScalarType::I8 | ScalarType::U8 => 1,
            ScalarType::I16 | ScalarType::U16 => 2,
            ScalarType::F32 | ScalarType::I32 | ScalarType::U32 => 4,
            ScalarType::F64 | ScalarType::I64 | ScalarType::U64 => 8,
        }
    }
}

impl std::fmt::Display for ScalarType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScalarType::F32 => write!(f, "f32"),
            ScalarType::F64 => write!(f, "f64"),
            ScalarType::I8 => write!(f, "i8"),
            ScalarType::U8 => write!(f, "u8"),
            ScalarType::I16 => write!(f, "i16"),
            ScalarType::U16 => write!(f, "u16"),
            ScalarType::I32 => write!(f, "i32"),
            ScalarType::U32 => write!(f, "u32"),
            ScalarType::I64 => write!(f, "i64"),
            ScalarType::U64 => write!(f, "u64"),
            ScalarType::Bool => write!(f, "bool"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValType {
    Scalar(ScalarType),
    Vec { len: u8, elem: ScalarType },  // len in {2, 3, 4}
    Mat { size: u8, elem: ScalarType }, // square matrix, size in {2, 3, 4}; column-major
    Array { elem: Box<ValType>, size: u32 }, // fixed-size array
}

impl ValType {
    // Convenience constants for scalar types.
    pub const F32: ValType = ValType::Scalar(ScalarType::F32);
    pub const F64: ValType = ValType::Scalar(ScalarType::F64);
    pub const I8: ValType = ValType::Scalar(ScalarType::I8);
    pub const U8: ValType = ValType::Scalar(ScalarType::U8);
    pub const I16: ValType = ValType::Scalar(ScalarType::I16);
    pub const U16: ValType = ValType::Scalar(ScalarType::U16);
    pub const I32: ValType = ValType::Scalar(ScalarType::I32);
    pub const U32: ValType = ValType::Scalar(ScalarType::U32);
    pub const I64: ValType = ValType::Scalar(ScalarType::I64);
    pub const U64: ValType = ValType::Scalar(ScalarType::U64);
    pub const BOOL: ValType = ValType::Scalar(ScalarType::Bool);

    /// Returns true if this is a vector type.
    pub fn is_vec(&self) -> bool {
        matches!(self, ValType::Vec { .. })
    }

    /// Returns true if this is a matrix type.
    pub fn is_mat(&self) -> bool {
        matches!(self, ValType::Mat { .. })
    }

    /// Returns true if this is an array type.
    pub fn is_array(&self) -> bool {
        matches!(self, ValType::Array { .. })
    }

    /// Number of scalar components (1 for scalars, 2..4 for vectors, N*N for matrices).
    pub fn component_count(&self) -> usize {
        match self {
            ValType::Vec { len, .. } => *len as usize,
            ValType::Mat { size, .. } => (*size as usize) * (*size as usize),
            ValType::Array { size, .. } => *size as usize,
            ValType::Scalar(_) => 1,
        }
    }

    /// The element type of a vector/matrix, or the scalar type itself.
    pub fn element_scalar(&self) -> ScalarType {
        match self {
            ValType::Vec { elem, .. } | ValType::Mat { elem, .. } => *elem,
            ValType::Scalar(s) => *s,
            ValType::Array { elem, .. } => elem.element_scalar(),
        }
    }

    /// For a matrix, the column vector type. Panics if not a matrix.
    pub fn mat_col_type(&self) -> ValType {
        match self {
            ValType::Mat { size, elem } => ValType::Vec { len: *size, elem: *elem },
            _ => panic!("mat_col_type called on non-matrix type"),
        }
    }

    /// For an array, the element type. Panics if not an array.
    pub fn array_elem_type(&self) -> &ValType {
        match self {
            ValType::Array { elem, .. } => elem,
            _ => panic!("array_elem_type called on non-array type"),
        }
    }

    /// For an array, the size. Panics if not an array.
    pub fn array_size(&self) -> u32 {
        match self {
            ValType::Array { size, .. } => *size,
            _ => panic!("array_size called on non-array type"),
        }
    }

    /// Returns true if this is a scalar type (not a compound type).
    pub fn is_scalar(&self) -> bool {
        matches!(self, ValType::Scalar(_))
    }

    /// Returns true if this is a float type (f64 scalar or vec of f64).
    pub fn is_float(&self) -> bool {
        self.element_scalar() == ScalarType::F64
    }
}

impl std::fmt::Display for ValType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValType::Scalar(s) => write!(f, "{}", s),
            ValType::Vec { len, elem } => write!(f, "vec{}<{}>", len, elem),
            ValType::Mat { size, elem } => write!(f, "mat{}<{}>", size, elem),
            ValType::Array { elem, size } => write!(f, "array<{}; {}>", elem, size),
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
    F32(f32),
    F64(f64),
    I8(i8),
    U8(u8),
    I16(i16),
    U16(u16),
    I32(i32),
    U32(u32),
    I64(i64),
    U64(u64),
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

/// Type conversion operation. `norm` indicates a normalizing conversion
/// (e.g. u32 → f64 by dividing by 2^32).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvOp {
    pub from: ScalarType,
    pub to: ScalarType,
    pub norm: bool,
}

impl ConvOp {
    pub const F64_TO_U32: ConvOp = ConvOp { from: ScalarType::F64, to: ScalarType::U32, norm: false };
    pub const U32_TO_F64: ConvOp = ConvOp { from: ScalarType::U32, to: ScalarType::F64, norm: false };
    pub const U32_TO_F64_NORM: ConvOp = ConvOp { from: ScalarType::U32, to: ScalarType::F64, norm: true };
    pub const F32_TO_F64: ConvOp = ConvOp { from: ScalarType::F32, to: ScalarType::F64, norm: false };
    pub const F64_TO_F32: ConvOp = ConvOp { from: ScalarType::F64, to: ScalarType::F32, norm: false };
    pub const I32_TO_F64: ConvOp = ConvOp { from: ScalarType::I32, to: ScalarType::F64, norm: false };
    pub const F64_TO_I32: ConvOp = ConvOp { from: ScalarType::F64, to: ScalarType::I32, norm: false };
    pub const I32_TO_U32: ConvOp = ConvOp { from: ScalarType::I32, to: ScalarType::U32, norm: false };
    pub const U32_TO_I32: ConvOp = ConvOp { from: ScalarType::U32, to: ScalarType::I32, norm: false };
    pub const I32_TO_F32: ConvOp = ConvOp { from: ScalarType::I32, to: ScalarType::F32, norm: false };
    pub const F32_TO_I32: ConvOp = ConvOp { from: ScalarType::F32, to: ScalarType::I32, norm: false };
    pub const F32_TO_U32: ConvOp = ConvOp { from: ScalarType::F32, to: ScalarType::U32, norm: false };
    pub const U32_TO_F32: ConvOp = ConvOp { from: ScalarType::U32, to: ScalarType::F32, norm: false };
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

    // Vector construction (length determines vec2/vec3/vec4)
    MakeVec(Vec<Var>),

    // Component extraction (vec -> scalar)
    VecExtract { vec: Var, index: u8 }, // 0=x, 1=y, 2=z, 3=w

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

    // Matrix construction from column vectors (N columns of vecN)
    MakeMat(Vec<Var>),

    // Matrix-vector multiply (matN * vecN -> vecN)
    MatMulVec { mat: Var, vec: Var },

    // Matrix-matrix multiply (matN * matN -> matN)
    MatMul { lhs: Var, rhs: Var },

    // Matrix transpose (matN -> matN)
    MatTranspose { arg: Var },

    // Column extraction (matN -> vecN, by column index)
    MatCol { mat: Var, index: u8 },

    // Array operations
    /// Create a fixed-size array from elements.
    ArrayNew(Vec<Var>),
    /// Get element at index from array. Index is a Var of integer type.
    ArrayGet { array: Var, index: Var },
    /// Set element at index in array, producing a new array.
    ArraySet { array: Var, index: Var, val: Var },

    // Buffer operations (for simulation kernels)
    /// Load f64 from buffer `buf` at position (x, y). Width/height for wrapping
    /// come from the kernel's implicit width/height parameters.
    BufLoad { buf: u32, x: Var, y: Var },
    /// Store f64 value to buffer `buf` at position (x, y).
    BufStore { buf: u32, x: Var, y: Var, val: Var },
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

/// Declaration of a buffer accessible from the kernel.
#[derive(Debug, Clone)]
pub struct BufDecl {
    pub name: String,
    pub is_output: bool,
}

#[derive(Debug, Clone)]
pub struct Kernel {
    pub name: String,
    pub params: Vec<Binding>,
    pub return_ty: ValType,
    pub body: Vec<BodyItem>,
    pub emit: Var,
    /// Buffer declarations for simulation kernels. Empty for standard pixel kernels.
    pub buffers: Vec<BufDecl>,
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
    pub fn var_type(&self, var: Var) -> Option<&ValType> {
        self.binding(var).map(|b| &b.ty)
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
