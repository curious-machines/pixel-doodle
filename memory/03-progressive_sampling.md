---
name: Progressive stochastic sampling design
description: Three-level progressive rendering — stochastic AA, 2D ray marching, 2D light transport. Accumulation architecture, hash-based jitter, GPU on-device design.
type: project
---

# Progressive Stochastic Sampling

## Overview

Render 2D SDF scenes with progressive refinement: each frame adds one stochastic sample per pixel, accumulating toward a converged image. The image starts noisy and smooths over time.

This is a three-level vision, implemented incrementally:

| Level | Name | What it does |
|-------|------|-------------|
| 1 | Stochastic AA | Jitter sample position within each pixel, accumulate average. Smooth edges on SDF boundaries. |
| 2 | 2D Ray Marching | Cast random-direction rays from each pixel, march using SDF distance. Soft shadows, occlusion. |
| 3 | 2D Light Transport | Full path tracing in 2D with emissive shapes, bounces, global illumination. |

Level 1 is the foundation — accumulation buffer, hash-based jitter, progressive event loop. Levels 2 and 3 build on the same infrastructure, adding kernel-side RNG and more complex evaluation.

## Architecture

### Per-Pixel Jitter (Level 1)

The JIT tile loop (not the host, not the PDIR kernel) computes jitter per pixel:

1. Host passes `sample_index: u32` — one value per frame, identifying which sample this is
2. Tile loop hashes `(col, row, sample_index)` to get two pseudo-random values in [0, 1)
3. Jitter is applied: `cx += jitter_x * x_step`, `cy += jitter_y * y_step`
4. Kernel body evaluates at the jittered position — it doesn't know about the jitter

This keeps PDIR kernels simple. For Level 2/3, kernel-side RNG ops will be added so the kernel can generate random ray directions internally.

### Hash Function

Pure integer arithmetic (no function calls, works in Cranelift IR, LLVM IR, and WGSL):

```
h = col * 0x45d9f3b + row
h = h * 0x45d9f3b + sample_index
h ^= h >> 16
h *= 0x45d9f3b
h ^= h >> 16
jitter_x = (h & 0xFFFF) / 65536.0
jitter_y = (h >> 16) / 65536.0
```

16 bits per axis is sufficient for sub-pixel jitter (1/65536 of a pixel). Level 2/3 will use two separate hashes for full 32-bit precision per axis.

### Accumulation Buffer

**CPU path**: Host-side `Vec<f32>` with 3 channels per pixel (R, G, B sums). After each sample pass:
1. Render to scratch buffer (u32 ARGB per pixel)
2. Unpack ARGB, add to accumulation sums
3. Divide by sample count, pack back to u32 for display

**GPU path**: On-device `array<vec4<f32>>` storage buffer. The compute shader:
1. Evaluates kernel with jittered coords → u32 color
2. Unpacks to vec3<f32>, adds to accumulation buffer
3. Divides by sample count, packs to u32 output buffer

All accumulation stays on GPU — no per-sample CPU readback. The host just dispatches compute passes and presents.

### Progressive Event Loop

```
on redraw:
    if view changed (pan/zoom):
        reset accumulation (zero buffer, sample_count = 0)
    if sample_count < max_samples:
        render one sample pass (sample_index = sample_count)
        accumulate
        display resolved image
        update title: "backend | Xms | sample N/M | zoom"
        request_redraw()  // keep refining
    else:
        display last resolved image (no re-render)
```

### Backward Compatibility

Without `--samples`, behavior is pixel-identical to current:
- `sample_index = 0xFFFFFFFF` sentinel skips jitter in tile loops
- No accumulation buffer allocated
- Single render per view change

## Scene Description

Initially: SDF scenes are hardcoded in PDIR kernels. The kernel receives `(x, y)` and evaluates its built-in scene.

Future: Pass scene data to the kernel (e.g., as a read-only buffer of shape descriptors). This enables animation and dynamic scene changes without recompiling the kernel. The TileKernelFn ABI would gain a `scene_data: *const u8` parameter.

## Convergence Control

Initially: Fixed sample count via `--samples N`. Rendering stops after N samples.

Future: Time-based convergence — render for T seconds, then stop. Could also add variance-based convergence (stop when pixel variance drops below threshold).

## GPU Efficiency Notes

The GPU backend is the reference design and should run at full speed:
- Accumulation buffer is a persistent `storage` buffer on GPU, never read back per-sample
- The compute shader does accumulate + resolve in one dispatch (no second pass needed)
- On reset (pan/zoom), host writes zeroes to accum buffer via `queue.write_buffer`
- Future: could batch multiple sample dispatches per frame for higher throughput
