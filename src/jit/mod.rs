use crate::kernel_ir::{Kernel, ValType};

// ── Texture ABI ─────────────────────────────────────────────────────────

/// Per-texture descriptor passed to JIT'd kernels.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TextureSlot {
    pub data: *const u8,
    pub width: u32,
    pub height: u32,
}

// SAFETY: TextureSlot contains a raw pointer to immutable data that is
// shared read-only across threads. The pointed-to TextureData lives for
// the duration of the Runtime, and kernel execution borrows it immutably.
unsafe impl Send for TextureSlot {}
unsafe impl Sync for TextureSlot {}

/// Helper called from JIT'd code: load RGBA at integer coords with repeat wrapping.
/// Writes 4 f32 values (r, g, b, a) into `out`.
pub extern "C" fn pd_tex_load_repeat(
    slots: *const TextureSlot, tex: u32, x: i32, y: i32, out: *mut f32,
) {
    unsafe {
        let slot = &*slots.add(tex as usize);
        let w = slot.width as i32;
        let h = slot.height as i32;
        let wx = ((x % w) + w) % w;
        let wy = ((y % h) + h) % h;
        read_pixel(slot, wx as u32, wy as u32, out);
    }
}

/// Helper called from JIT'd code: load RGBA at integer coords with clamp-to-edge.
pub extern "C" fn pd_tex_load_clamp(
    slots: *const TextureSlot, tex: u32, x: i32, y: i32, out: *mut f32,
) {
    unsafe {
        let slot = &*slots.add(tex as usize);
        let wx = x.clamp(0, slot.width as i32 - 1) as u32;
        let wy = y.clamp(0, slot.height as i32 - 1) as u32;
        read_pixel(slot, wx, wy, out);
    }
}

/// Helper: nearest-neighbor sample at normalized UV with repeat wrapping.
pub extern "C" fn pd_tex_sample_nearest_repeat(
    slots: *const TextureSlot, tex: u32, u: f64, v: f64, out: *mut f32,
) {
    unsafe {
        let slot = &*slots.add(tex as usize);
        let x = (u as f32 * slot.width as f32).floor() as i32;
        let y = (v as f32 * slot.height as f32).floor() as i32;
        let w = slot.width as i32;
        let h = slot.height as i32;
        let wx = ((x % w) + w) % w;
        let wy = ((y % h) + h) % h;
        read_pixel(slot, wx as u32, wy as u32, out);
    }
}

/// Helper: nearest-neighbor sample at normalized UV with clamp.
pub extern "C" fn pd_tex_sample_nearest_clamp(
    slots: *const TextureSlot, tex: u32, u: f64, v: f64, out: *mut f32,
) {
    unsafe {
        let slot = &*slots.add(tex as usize);
        let x = (u as f32 * slot.width as f32).floor() as i32;
        let y = (v as f32 * slot.height as f32).floor() as i32;
        let wx = x.clamp(0, slot.width as i32 - 1) as u32;
        let wy = y.clamp(0, slot.height as i32 - 1) as u32;
        read_pixel(slot, wx, wy, out);
    }
}

/// Helper: bilinear sample at normalized UV with repeat wrapping.
pub extern "C" fn pd_tex_sample_bilinear_repeat(
    slots: *const TextureSlot, tex: u32, u: f64, v: f64, out: *mut f32,
) {
    unsafe {
        let slot = &*slots.add(tex as usize);
        let fx = u as f32 * slot.width as f32 - 0.5;
        let fy = v as f32 * slot.height as f32 - 0.5;
        let x0 = fx.floor() as i32;
        let y0 = fy.floor() as i32;
        let frac_x = fx - x0 as f32;
        let frac_y = fy - y0 as f32;
        let w = slot.width as i32;
        let h = slot.height as i32;

        let mut c00 = [0.0f32; 4]; let mut c10 = [0.0f32; 4];
        let mut c01 = [0.0f32; 4]; let mut c11 = [0.0f32; 4];
        read_pixel(slot, (((x0 % w) + w) % w) as u32, (((y0 % h) + h) % h) as u32, c00.as_mut_ptr());
        read_pixel(slot, ((((x0+1) % w) + w) % w) as u32, (((y0 % h) + h) % h) as u32, c10.as_mut_ptr());
        read_pixel(slot, (((x0 % w) + w) % w) as u32, ((((y0+1) % h) + h) % h) as u32, c01.as_mut_ptr());
        read_pixel(slot, ((((x0+1) % w) + w) % w) as u32, ((((y0+1) % h) + h) % h) as u32, c11.as_mut_ptr());

        for i in 0..4 {
            let top = c00[i] + (c10[i] - c00[i]) * frac_x;
            let bot = c01[i] + (c11[i] - c01[i]) * frac_x;
            *out.add(i) = top + (bot - top) * frac_y;
        }
    }
}

/// Helper: bilinear sample at normalized UV with clamp.
pub extern "C" fn pd_tex_sample_bilinear_clamp(
    slots: *const TextureSlot, tex: u32, u: f64, v: f64, out: *mut f32,
) {
    unsafe {
        let slot = &*slots.add(tex as usize);
        let fx = u as f32 * slot.width as f32 - 0.5;
        let fy = v as f32 * slot.height as f32 - 0.5;
        let x0 = fx.floor() as i32;
        let y0 = fy.floor() as i32;
        let frac_x = fx - x0 as f32;
        let frac_y = fy - y0 as f32;
        let max_x = slot.width as i32 - 1;
        let max_y = slot.height as i32 - 1;

        let mut c00 = [0.0f32; 4]; let mut c10 = [0.0f32; 4];
        let mut c01 = [0.0f32; 4]; let mut c11 = [0.0f32; 4];
        read_pixel(slot, x0.clamp(0, max_x) as u32, y0.clamp(0, max_y) as u32, c00.as_mut_ptr());
        read_pixel(slot, (x0+1).clamp(0, max_x) as u32, y0.clamp(0, max_y) as u32, c10.as_mut_ptr());
        read_pixel(slot, x0.clamp(0, max_x) as u32, (y0+1).clamp(0, max_y) as u32, c01.as_mut_ptr());
        read_pixel(slot, (x0+1).clamp(0, max_x) as u32, (y0+1).clamp(0, max_y) as u32, c11.as_mut_ptr());

        for i in 0..4 {
            let top = c00[i] + (c10[i] - c00[i]) * frac_x;
            let bot = c01[i] + (c11[i] - c01[i]) * frac_x;
            *out.add(i) = top + (bot - top) * frac_y;
        }
    }
}

/// Read a single pixel from texture data, writing 4 f32 components to `out`.
#[inline]
unsafe fn read_pixel(slot: &TextureSlot, x: u32, y: u32, out: *mut f32) {
    unsafe {
        let idx = (y * slot.width + x) as usize * 4;
        let data = slot.data;
        *out = *data.add(idx) as f32 / 255.0;
        *out.add(1) = *data.add(idx + 1) as f32 / 255.0;
        *out.add(2) = *data.add(idx + 2) as f32 / 255.0;
        *out.add(3) = *data.add(idx + 3) as f32 / 255.0;
    }
}

// ── User-argument layout ────────────────────────────────────────────────

/// Describes one user-defined kernel argument and its position in the
/// packed `user_args` byte buffer.
#[derive(Debug, Clone)]
pub struct UserArgSlot {
    pub name: String,
    pub offset: usize, // byte offset into the buffer
    pub ty: ValType,
}

/// Compute the user-arg layout for a kernel given a set of built-in names.
/// Returns the slots and the total buffer size in bytes.
pub fn compute_user_arg_layout(kernel: &Kernel, builtins: &[&str]) -> (Vec<UserArgSlot>, usize) {
    let mut offset: usize = 0;
    let mut slots = Vec::new();
    for param in &kernel.params {
        if builtins.contains(&param.name.as_str()) {
            continue;
        }
        let size = match &param.ty {
            ValType::Scalar(s) => s.byte_size(),
            _ => panic!("unsupported user-arg type {:?} for param '{}'", param.ty, param.name),
        };
        // Natural alignment
        let align = size;
        offset = (offset + align - 1) & !(align - 1);
        slots.push(UserArgSlot {
            name: param.name.clone(),
            offset,
            ty: param.ty.clone(),
        });
        offset += size;
    }
    // Align total size to 8 bytes for safety
    let total = (offset + 7) & !7;
    (slots, total)
}

// ── Kernel function signatures ──────────────────────────────────────────

/// JIT'd function: writes ARGB pixels for rows [row_start, row_end).
/// `output` points to the start of this tile's chunk, not the full buffer.
///
/// `sample_index`: which sample pass this is (0, 1, 2, ...).
/// When `0xFFFFFFFF`, no jitter is applied (non-progressive mode).
///
/// `time`: elapsed time in seconds since the window opened. Kernels that
/// declare a `time: f64` parameter receive this value for animation.
///
/// `user_args`: pointer to a packed byte buffer of user-defined argument
/// values. May be null when the kernel has no user args.
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
    user_args: *const u8,
    tex_slots: *const TextureSlot,
);

pub trait JitBackend {
    fn compile(&self, kernel: &Kernel, user_args: &[UserArgSlot]) -> Box<dyn CompiledKernel>;
    fn compile_sim(&self, kernel: &Kernel, user_args: &[UserArgSlot]) -> Box<dyn CompiledSimKernel>;
}

/// A compiled kernel that can be called from multiple threads.
/// The implementor must ensure the function pointer remains valid
/// for the lifetime of this object.
pub trait CompiledKernel: Send + Sync {
    fn function_ptr(&self) -> TileKernelFn;
}

/// JIT'd simulation kernel: iterates over a tile of rows, reads/writes
/// f64 buffer arrays, and produces ARGB pixels.
///
/// Buffer layout: each buffer is a contiguous `f64` array of `width * height`
/// elements in row-major order. The kernel computes wrapping indices internally.
///
/// `buf_ptrs[0..num_read]` are read-only input buffers.
/// `buf_out_ptrs[0..num_write]` are write-only output buffers.
///
/// `user_args`: pointer to a packed byte buffer of user-defined argument
/// values. May be null when the kernel has no user args.
pub type SimTileKernelFn = unsafe extern "C" fn(
    output: *mut u32,
    width: u32,
    height: u32,
    row_start: u32,
    row_end: u32,
    buf_ptrs: *const *const f64,
    buf_out_ptrs: *const *mut f64,
    user_args: *const u8,
    tex_slots: *const TextureSlot,
);

pub trait CompiledSimKernel: Send + Sync {
    fn function_ptr(&self) -> SimTileKernelFn;
}

#[cfg(feature = "cranelift-backend")]
pub mod cranelift;

#[cfg(feature = "llvm-backend")]
pub mod llvm;

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;

    #[test]
    fn texture_slot_layout() {
        // JIT codegen hardcodes these offsets for TexWidth/TexHeight
        assert_eq!(mem::size_of::<TextureSlot>(), 16);
        assert_eq!(mem::offset_of!(TextureSlot, width), 8);
        assert_eq!(mem::offset_of!(TextureSlot, height), 12);
    }
}
