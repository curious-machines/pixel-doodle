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
    Sim,
    Init,
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
    F32,    // 4 bytes
    Vec2f,  // 8 bytes
    Vec3f,  // 12 bytes (padded to 16 in practice)
    Vec4f,  // 16 bytes
    I32,    // 4 bytes
    U32,    // 4 bytes
}

impl GpuElementType {
    /// Size in bytes per element.
    pub fn byte_size(self) -> u32 {
        match self {
            GpuElementType::F32 => 4,
            GpuElementType::Vec2f => 8,
            GpuElementType::Vec3f => 16, // padded to 16 for alignment
            GpuElementType::Vec4f => 16,
            GpuElementType::I32 => 4,
            GpuElementType::U32 => 4,
        }
    }
}

#[derive(Debug, Clone)]
pub enum BufferInit {
    /// `constant(value)` — built-in fill.
    Constant(f64),
    /// `init_name(arg: val, ...)` — call an init kernel.
    InitKernel {
        kernel_name: String,
        args: Vec<NamedArg>,
    },
}

#[derive(Debug, Clone)]
pub struct BufferDecl {
    pub name: String,
    /// GPU element type annotation, e.g. `buffer field: gpu(vec2f) = ...`.
    /// None for CPU buffers (f64 arrays).
    pub gpu_type: Option<GpuElementType>,
    pub init: BufferInit,
    pub span: Span,
}

// ── Variables ──

#[derive(Debug, Clone)]
pub struct RangeSpec {
    pub min: f64,
    pub max: f64,
    pub wrap: bool,
}

#[derive(Debug, Clone)]
pub struct VarDecl {
    pub name: String,
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

/// The action performed when a key is pressed.
#[derive(Debug, Clone)]
pub enum Action {
    /// `variable = !variable`
    Toggle(String),
    /// `variable += literal` (or -= *= /=)
    CompoundAssign {
        target: String,
        op: CompoundOp,
        value: f64,
    },
    /// `variable = variable + literal` (expanded form)
    BinAssign {
        target: String,
        op: CompoundOp,
        value: f64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompoundOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Clone)]
pub struct KeyBinding {
    pub key_name: String,
    pub action: Action,
    pub span: Span,
}

// ── Pipeline ──

/// A buffer binding inside a `run` or `display` step: `{ param: buffer, ... }`.
#[derive(Debug, Clone)]
pub struct BufferBinding {
    pub param_name: String,
    pub buffer_name: String,
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
    /// `outputs = run kernel_name(args...) { bindings }` or just `run kernel_name`
    Run {
        outputs: Vec<String>,
        kernel_name: String,
        args: Vec<NamedArg>,
        input_bindings: Vec<BufferBinding>,
        span: Span,
    },
    /// `outputs = display kernel_name { bindings }` or just `display kernel_name`
    Display {
        outputs: Vec<String>,
        kernel_name: String,
        args: Vec<NamedArg>,
        input_bindings: Vec<BufferBinding>,
        span: Span,
    },
    /// `swap a <-> b, c <-> d`
    Swap {
        pairs: Vec<(String, String)>,
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
    /// `on click(continuous: true) { steps }`
    OnClick {
        continuous: bool,
        body: Vec<PipelineStep>,
        span: Span,
    },
}

#[derive(Debug, Clone)]
pub struct Pipeline {
    /// Pipeline name: None for unnamed, Some("cpu") or Some("gpu") for named.
    pub name: Option<String>,
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
    pub variables: Vec<VarDecl>,
    pub settings: Settings,
    pub key_bindings: Vec<KeyBinding>,
    /// One or more pipelines. Unnamed (single) or named (cpu/gpu).
    pub pipelines: Vec<Pipeline>,
}
