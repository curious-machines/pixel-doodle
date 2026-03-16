use crate::kernel_ir::Kernel;

/// JIT'd function: writes ARGB pixels for rows [row_start, row_end).
/// `output` points to the start of this tile's chunk, not the full buffer.
///
/// `sample_index`: which sample pass this is (0, 1, 2, ...).
/// When `0xFFFFFFFF`, no jitter is applied (non-progressive mode).
///
/// `time`: elapsed time in seconds since the window opened. Kernels that
/// declare a `time: f64` parameter receive this value for animation.
pub type TileKernelFn = unsafe extern "C" fn(
    output: *mut u32,
    width: u32,
    height: u32,
    x_min: f64,
    y_min: f64,
    x_step: f64,
    y_step: f64,
    row_start: u32,
    row_end: u32,
    sample_index: u32,
    time: f64,
);

pub trait JitBackend {
    fn compile(&self, kernel: &Kernel) -> Box<dyn CompiledKernel>;
}

/// A compiled kernel that can be called from multiple threads.
/// The implementor must ensure the function pointer remains valid
/// for the lifetime of this object.
pub trait CompiledKernel: Send + Sync {
    fn function_ptr(&self) -> TileKernelFn;
}

#[cfg(feature = "cranelift-backend")]
pub mod cranelift;

#[cfg(feature = "llvm-backend")]
pub mod llvm;
