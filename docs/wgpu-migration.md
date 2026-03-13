# wgpu Migration & GPU Backend

## Motivation

Replace minifb with wgpu to:
1. Enable a GPU compute backend (`--backend gpu`) for speed comparison against CPU backends
2. Hardware-accelerated presentation for all backends (CPU and GPU)
3. When running the GPU backend, pixel data stays on-GPU — no readback or upload needed

The GPU backend is a **separate backend**, not a unified CPU/GPU abstraction. CPU backends continue to JIT or compile native code and write to a `Vec<u32>`; the GPU backend runs a handwritten WGSL compute shader.

## Current Architecture

```
main.rs          Parse --backend, compile kernel → TileKernelFn, run window loop
render.rs        Split buffer into 16-row tiles, dispatch via Rayon par_chunks_mut
jit/mod.rs       TileKernelFn type alias, JitBackend + CompiledKernel traits
jit/cranelift.rs Cranelift JIT compiler
jit/llvm.rs      LLVM JIT compiler
native_kernel.rs Native Rust reference kernel
kernel_ir.rs     KernelIr enum (currently just Mandelbrot)
```

Display: minifb `Window::update_with_buffer(&[u32], w, h)`
Parallelism: Rayon `par_chunks_mut` over 16-row tiles

## Target Architecture

```
main.rs          Parse --backend, dispatch to CPU or GPU path
render.rs        (unchanged) tile-based Rayon dispatch for CPU backends
display.rs       wgpu surface + winit event loop, texture upload for CPU path
gpu/mod.rs       GPU backend: wgpu compute pipeline, WGSL shader dispatch
gpu/mandelbrot.wgsl  Compute shader
```

### Two display paths

1. **CPU backends** (native, cranelift, llvm): render to `Vec<u32>`, upload to wgpu texture via `queue.write_texture()`, draw fullscreen quad
2. **GPU backend**: compute shader writes directly to a storage buffer, copy to texture, present — no CPU-side pixel buffer

## Implementation Plan

### Phase 1: Replace minifb with wgpu + winit (CPU path only)

All existing backends keep working. No GPU compute yet.

1. **Add dependencies**: `wgpu`, `winit`, `pollster` (for blocking async). Remove `minifb`.
2. **Create `display.rs`** module:
   - Initialize wgpu instance, adapter, device, queue
   - Create winit window (1200x900)
   - Create a `Rgba8Unorm` texture (1200x900) + bind group + render pipeline with a fullscreen triangle/quad + sampler
   - Provide `fn upload_and_present(&[u32])` that writes CPU buffer to texture and presents
3. **Adapt `main.rs`**:
   - Replace minifb window creation with `display::Display`
   - winit event loop replaces `while window.is_open()`
   - Keyboard handling via winit `WindowEvent::KeyboardInput`
   - On each frame: call `render::render()` (unchanged), then `display.upload_and_present(&buffer)`
4. **Verify**: all three CPU backends produce identical output to current minifb version

### Phase 2: GPU compute backend

1. **Create `gpu/mandelbrot.wgsl`**:
   - Compute shader with `@workgroup_size(16, 16)` (256 threads per workgroup)
   - Uniform buffer: `width`, `height`, `x_min`, `y_min`, `x_step`, `y_step`, `max_iter`
   - Storage buffer: `array<u32>` for output pixels
   - Same Mandelbrot + coloring logic as `native_kernel.rs`
2. **Create `gpu/mod.rs`**:
   - `GpuBackend` struct holding wgpu device, queue, compute pipeline, bind groups, buffers
   - `fn render(&self, width, height, center_x, center_y, zoom)` that:
     - Writes uniform buffer
     - Dispatches compute shader: `ceil(width/16) × ceil(height/16)` workgroups
     - Copies storage buffer → texture (or use storage texture directly if format allows)
   - The GPU backend owns its own wgpu resources but shares the device/queue with the display module
3. **Wire into `main.rs`**:
   - `--backend gpu` skips JIT compilation, creates `GpuBackend`
   - In the event loop: calls `gpu_backend.render()` instead of `render::render()`
   - Present the GPU-rendered texture directly (no CPU buffer involved)
4. **Timing**: measure GPU render time via wgpu timestamp queries or CPU-side fence timing, display in title bar alongside CPU backend times

### Phase 3: Polish

- Feature-gate wgpu behind a `gpu-backend` feature flag (or just make it always-on since it replaces minifb)
- Ensure clean shutdown (wgpu resource cleanup)
- Update `docs/backends.md` with GPU backend instructions

## Key Design Decisions

**wgpu is always the display layer**, not just for the GPU backend. This avoids maintaining two windowing systems.

**GPU backend does not use the `JitBackend`/`CompiledKernel` traits.** Those traits return a `TileKernelFn` (CPU function pointer) which doesn't apply. The GPU backend has its own interface.

**Shared device/queue.** The display module and GPU backend share one wgpu device. The display module creates it; the GPU backend receives references.

**winit event loop.** winit uses a callback-based event loop (`run()` or `run_app()`). The main loop structure changes from a `while` loop to event handlers. This is the biggest structural change in Phase 1.

## Dependencies

```toml
wgpu = "25"
winit = "0.30"
pollster = "0.4"
```

Remove: `minifb = "0.28"`

## Risk / Open Questions

- **winit event loop model**: winit 0.30 uses `ApplicationHandler` trait. More boilerplate than minifb but well-documented. The pan/zoom keyboard handling maps directly.
- **wgpu texture format**: surface preferred format varies by platform. Need to handle BGRA vs RGBA swizzle (or just use `Rgba8UnormSrgb` and convert).
- **GPU timer precision**: timestamp queries aren't available on all backends. CPU-side `queue.on_submitted_work_done()` is a fallback for timing.
