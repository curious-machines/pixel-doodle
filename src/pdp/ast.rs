/// Span in source for error reporting.
#[derive(Debug, Clone, Copy)]
pub struct Span {
    pub line: usize,
    pub col: usize,
}

/// A literal value in the config.
#[derive(Debug, Clone)]
pub enum Literal {
    Float(f64),
    Int(i64),
    Bool(bool),
    Str(String),
    /// Reference to a pdp variable by name.
    VarRef(String),
}

/// A named argument: `name: value`.
#[derive(Debug, Clone)]
pub struct NamedArg {
    pub name: String,
    pub value: Literal,
    pub span: Span,
}

// ── Kernel declarations ──

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelKind {
    Pixel,
    Standard,
}

impl KernelKind {
    /// Parameter names provided by the tile loop for this kernel kind.
    /// These are not supplied by the user — the runtime injects them.
    pub fn tile_loop_params(&self) -> &'static [&'static str] {
        match self {
            KernelKind::Pixel => &["x", "y", "px", "py", "sample_index", "time"],
            KernelKind::Standard => &["px", "py", "width", "height"],
        }
    }
}

#[derive(Debug, Clone)]
pub struct KernelDecl {
    pub kind: KernelKind,
    pub name: String,
    pub path: String,
    pub span: Span,
}

// ── Buffer declarations ──

/// GPU buffer element type for storage buffer allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuElementType {
    F32,       // 4 bytes
    Vec2F32,   // 8 bytes — vec2<f32>
    Vec3F32,   // 12 bytes (padded to 16 in practice) — vec3<f32>
    Vec4F32,   // 16 bytes — vec4<f32>
    I32,       // 4 bytes
    U32,       // 4 bytes
}

impl GpuElementType {
    /// Size in bytes per element.
    pub fn byte_size(self) -> u32 {
        match self {
            GpuElementType::F32 => 4,
            GpuElementType::Vec2F32 => 8,
            GpuElementType::Vec3F32 => 16, // padded to 16 for alignment
            GpuElementType::Vec4F32 => 16,
            GpuElementType::I32 => 4,
            GpuElementType::U32 => 4,
        }
    }
}

#[derive(Debug, Clone)]
pub enum BufferInit {
    /// `constant(value)` — built-in fill.
    Constant(f64),
}

#[derive(Debug, Clone)]
pub struct BufferDecl {
    pub name: String,
    /// GPU element type annotation, e.g. `buffer field: gpu(vec2<f32>) = ...`.
    /// None for CPU buffers (f64 arrays).
    pub gpu_type: Option<GpuElementType>,
    pub init: BufferInit,
    pub span: Span,
}

// ── Builtin declarations ──

/// Type annotation on a `builtin` declaration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinType {
    U32,
    U64,
    F64,
    Bool,
}

impl std::fmt::Display for BuiltinType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BuiltinType::U32 => write!(f, "u32"),
            BuiltinType::U64 => write!(f, "u64"),
            BuiltinType::F64 => write!(f, "f64"),
            BuiltinType::Bool => write!(f, "bool"),
        }
    }
}

/// `builtin const name: type` or `builtin var name: type`.
#[derive(Debug, Clone)]
pub struct BuiltinDecl {
    pub name: String,
    pub ty: BuiltinType,
    /// `true` for `builtin var`, `false` for `builtin const`.
    pub mutable: bool,
    pub span: Span,
}

// ── Variables ──

/// Whether a user-declared variable is mutable or not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mutability {
    Var,
    Const,
}

#[derive(Debug, Clone)]
pub struct RangeSpec {
    pub ty: BuiltinType,
    pub min: f64,
    pub max: f64,
    pub wrap: bool,
}

#[derive(Debug, Clone)]
pub struct VarDecl {
    pub name: String,
    pub mutability: Mutability,
    /// Optional type annotation (e.g., `var speed: f64 = 1.0`).
    pub ty: Option<BuiltinType>,
    pub range: Option<RangeSpec>,
    pub default: Literal,
    pub span: Span,
}

// ── Settings ──

#[derive(Debug, Clone, Default)]
pub struct Settings {
    pub entries: Vec<SettingsEntry>,
}

#[derive(Debug, Clone)]
pub struct SettingsEntry {
    pub key: String,
    pub value: Literal,
    pub span: Span,
}

// ── Key bindings ──

/// A value expression in key bindings: `0.02`, `0.02 / zoom`, `0.02 * speed`.
#[derive(Debug, Clone)]
pub enum ValueExpr {
    Literal(f64),
    BinOp {
        left: f64,
        op: CompoundOp,
        right: String,
    },
}

/// The action performed when a key is pressed.
#[derive(Debug, Clone)]
pub enum Action {
    /// `variable = !variable`
    Toggle(String),
    /// `variable += expr` (or -= *= /=)
    CompoundAssign {
        target: String,
        op: CompoundOp,
        value: ValueExpr,
    },
    /// `variable = variable + expr` (expanded form)
    BinAssign {
        target: String,
        op: CompoundOp,
        value: ValueExpr,
    },
    /// `variable = literal` (direct assignment)
    Assign {
        target: String,
        value: f64,
    },
    /// `quit` — request application exit
    Quit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompoundOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventKind {
    Keydown,
    Keypress,
    Keyup,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MouseEventKind {
    Mousedown,
    Click,
    Mouseup,
}

#[derive(Debug, Clone)]
pub struct EventBinding {
    pub kind: EventKind,
    pub key_name: String,
    pub actions: Vec<Action>,
    pub span: Span,
}

// ── Pipeline ──

/// A buffer binding inside a `run` or `display` step: `{ param: buffer, ... }`.
#[derive(Debug, Clone)]
pub struct BufferBinding {
    pub param_name: String,
    pub buffer_name: String,
    /// Marks this binding as an output (e.g. `pixels: out pixels` in GPU display steps).
    pub is_output: bool,
    pub span: Span,
}

/// How many iterations for a `loop` step.
#[derive(Debug, Clone)]
pub enum IterCount {
    /// A fixed integer literal.
    Fixed(u32),
    /// A variable name (looked up at runtime).
    Variable(String),
}

#[derive(Debug, Clone)]
pub enum PipelineStep {
    /// `run kernel_name(args...) { bindings }` — execute a kernel.
    /// Output buffers use `out` qualifier in bindings: `{ param: out buffer }`.
    Run {
        kernel_name: String,
        args: Vec<NamedArg>,
        input_bindings: Vec<BufferBinding>,
        span: Span,
    },
    /// `display` (CPU) or `display buffer_name` (GPU) — present pixels to screen.
    Display {
        buffer_name: Option<String>,
        span: Span,
    },
    /// `init { steps }` — run once at startup.
    Init {
        body: Vec<PipelineStep>,
        span: Span,
    },
    /// `swap a, b`
    Swap {
        a: String,
        b: String,
        span: Span,
    },
    /// `loop(iterations: N) { steps }`
    Loop {
        iterations: IterCount,
        body: Vec<PipelineStep>,
        span: Span,
    },
    /// `accumulate(samples: N) { steps }`
    Accumulate {
        samples: u32,
        body: Vec<PipelineStep>,
        span: Span,
    },
    /// `on mousedown { steps }` or `on mouseup { steps }`
    OnMouse {
        kind: MouseEventKind,
        body: Vec<PipelineStep>,
        span: Span,
    },
}

#[derive(Debug, Clone)]
pub struct Pipeline {
    /// Pipeline name: None for unnamed, Some("pd"), Some("pdir"), Some("gpu"), etc.
    pub name: Option<String>,
    /// Builtin declarations scoped to this pipeline.
    pub builtins: Vec<BuiltinDecl>,
    /// Kernels scoped to this pipeline.
    pub kernels: Vec<KernelDecl>,
    /// Buffers scoped to this pipeline.
    pub buffers: Vec<BufferDecl>,
    /// Pipeline execution steps.
    pub steps: Vec<PipelineStep>,
    pub span: Span,
}

// ── Top-level config ──

#[derive(Debug, Clone)]
pub struct Config {
    pub title: Option<String>,
    /// Top-level builtin declarations (shared across all pipelines).
    pub builtins: Vec<BuiltinDecl>,
    pub variables: Vec<VarDecl>,
    pub settings: Settings,
    pub event_bindings: Vec<EventBinding>,
    /// One or more named pipelines. First pipeline is the default.
    pub pipelines: Vec<Pipeline>,
}
