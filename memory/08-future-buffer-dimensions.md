---
name: Future — configurable buffer dimensions
description: Design notes for allowing non-screen-sized buffers in PDP (lookup tables, downsampled passes, particle storage, etc.)
type: project
---

# Configurable Buffer Dimensions (Future)

## Current State

All PDP buffers are exactly `width x height` (screen resolution). Size is hardcoded in `runtime.rs:init_buffers()` as `vec![val; width * height]`. Buffer access in kernels uses `y * width + x` indexing. No mechanism for custom dimensions.

## Why Change

Every GPU shading language (WGSL, GLSL, Metal, HLSL) allows buffers/textures with arbitrary dimensions independent of screen size. This enables:

### Concrete Use Cases

1. **Lookup tables / palettes**: A 256-element 1D color ramp or transfer function. Tiny, constant, doesn't need a full screen-sized buffer.

2. **Downsampled buffers**: Half-res or quarter-res for blur, bloom, ambient occlusion. Huge performance win — e.g., a blur pass at 1/4 resolution is 16x fewer pixels.

3. **Tiled noise**: A small 64x64 noise texture tiled across the screen. Saves memory vs generating full-res noise.

4. **Particle/agent storage**: A buffer of N particles — not grid-shaped at all. E.g., `buffer particles: dims(1024, 1) = constant(0.0)` for 1024 particles.

5. **Histogram/reduction buffers**: Small accumulation buffers (e.g., 256 bins for a histogram).

6. **Multi-resolution simulation**: Coarse grid for pressure solve (faster convergence), fine grid for rendering.

## Possible Syntax

```pdp
# Explicit dimensions
buffer palette: dims(256, 1) = constant(0.0)
buffer blur_buf: dims(width/2, height/2) = constant(0.0)
buffer noise: dims(64, 64) = constant(0.0)

# Current behavior (unchanged) — defaults to width x height
buffer state = constant(0.0)
```

Dimensions could be:
- Integer literals: `dims(256, 1)`
- Expressions involving intrinsics: `dims(width/2, height/2)`
- Variable references: `dims(grid_w, grid_h)` where grid_w/grid_h are PDP variables

## Implementation Considerations

- `BufferDecl` in `pdp/ast.rs` needs optional dimension fields
- `runtime.rs:init_buffers()` must use per-buffer dimensions instead of global width/height
- Kernel BufLoad/BufStore need per-buffer width/height (currently they use the global width/height from the tile loop)
- The `width`/`height` params passed to sim kernels are screen dimensions — buffer-specific dimensions would need a different mechanism (additional kernel params, or buffer metadata accessible in the kernel)
- GPU buffers already have stride alignment — custom dimensions would work similarly
- Dynamic resizing (e.g., buffer tracks window size / 2) would need resize-on-window-change logic

## Reference: How Other Systems Handle This

- **WGSL/WebGPU**: Buffers are arbitrary byte arrays; textures have explicit dimensions at creation
- **GLSL/Vulkan**: Textures and storage buffers independently dimensioned
- **ShaderToy**: "Buffers" are screen-sized by default, but texture inputs can be any resolution
- **Metal/HLSL**: All buffer/texture dimensions explicitly specified
