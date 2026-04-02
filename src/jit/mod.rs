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

// ── Texture sampling helpers (called from WGSL CPU backends) ────────────
// These are registered as symbols with the JIT and called from generated code,
// so they appear unused to the compiler when backends are feature-gated.

/// Load RGBA at integer coords with repeat wrapping.
#[allow(dead_code)]
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

/// Load RGBA at integer coords with clamp-to-edge.
#[allow(dead_code)]
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

/// Nearest-neighbor sample at normalized UV with repeat wrapping.
#[allow(dead_code)]
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

/// Nearest-neighbor sample at normalized UV with clamp.
#[allow(dead_code)]
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

/// Bilinear sample at normalized UV with repeat wrapping.
#[allow(dead_code)]
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

/// Bilinear sample at normalized UV with clamp.
#[allow(dead_code)]
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

// ── WGSL GPU-on-CPU shared types ────────────────────────────────────────

/// JIT'd WGSL compute kernel function signature.
///
/// `buffers` is an array of pointers, one per storage buffer binding (in binding
/// order). For pixel shaders: buffers[0] = output (u32), buffers[1] = accum (f32).
/// For sim shaders: buffers[N] corresponds to @binding(N+1).
/// `tex_slots` is a pointer to an array of TextureSlot structs (16 bytes each),
/// one per texture binding in binding order. May be null if no textures.
///
/// The kernel processes rows `[row_start, row_end)`. The `params` buffer still
/// contains the full image width/height for view mapping — row_start/row_end
/// only control which rows this call computes.
#[cfg(any(feature = "cranelift-backend", feature = "llvm-backend"))]
pub type WgslKernelFn = unsafe extern "C" fn(
    params: *const u8,
    buffers: *const *mut u8,
    tex_slots: *const u8,
    width: u32,
    height: u32,
    stride: u32,
    row_start: u32,
    row_end: u32,
);

/// Info about a Params struct member.
#[cfg(any(feature = "cranelift-backend", feature = "llvm-backend"))]
#[derive(Clone)]
pub struct ParamMember {
    pub name: String,
    pub offset: u32,
}

/// Compiled WGSL kernel — holds the JIT handle (to keep code alive) and the
/// function pointer. Used by both Cranelift and LLVM CPU render backends.
#[cfg(any(feature = "cranelift-backend", feature = "llvm-backend"))]
pub struct CompiledWgslKernel {
    _jit_handle: Box<dyn Send + Sync>,
    pub fn_ptr: WgslKernelFn,
    /// WGSL variable name → index into the buffers array, for each storage buffer.
    pub binding_map: std::collections::HashMap<String, usize>,
    /// Number of storage buffers.
    pub num_storage_buffers: usize,
    /// Params struct members with names and byte offsets.
    pub params_members: Vec<ParamMember>,
    /// Byte size per element for each storage buffer, indexed by buffer position.
    pub buffer_elem_bytes: Vec<u32>,
}

// SAFETY: fn_ptr points to JIT'd code kept alive by _jit_handle.
// The function is pure (reads params/buffers passed by caller).
#[cfg(any(feature = "cranelift-backend", feature = "llvm-backend"))]
unsafe impl Send for CompiledWgslKernel {}
#[cfg(any(feature = "cranelift-backend", feature = "llvm-backend"))]
unsafe impl Sync for CompiledWgslKernel {}

#[cfg(feature = "cranelift-backend")]
pub mod wgsl_cranelift;

#[cfg(feature = "llvm-backend")]
pub mod wgsl_llvm;

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
