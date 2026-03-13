/// Minimal kernel IR — just enough to represent what to JIT compile.
/// This will grow into the full custom language later.
#[derive(Debug, Clone)]
pub enum KernelIr {
    Mandelbrot { max_iter: u32 },
}
