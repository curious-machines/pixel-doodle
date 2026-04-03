# Opaque Handle Types

## Overview

PDC uses opaque handle types for resources managed by the Rust runtime. These are distinct types at the PDC level but compile down to plain `i32` values — the type system provides safety while the codegen has zero overhead.

Each handle type is registered as a `PdcType` variant in `ast.rs`, with its constructor and methods registered as builtins in `type_check.rs:register_builtins()`.

## Implemented

### Path

Opaque handle for 2D vector paths. Created via constructor, manipulated via methods.

```
var p = Path()
p.move_to(0.0, 0.0)
p.line_to(100.0, 0.0)
p.quad_to(cx, cy, x, y)
p.cubic_to(c1x, c1y, c2x, c2y, x, y)
p.close()
p.fill(color)
p.fill(color, FillRule.EvenOdd)           // styled overload
p.stroke(width, color)
p.stroke(width, color, LineCap.Round, LineJoin.Miter)  // styled overload
```

- Type: `PdcType::PathHandle`
- Codegen: `i32`
- Runtime: `pdc_path`, `pdc_move_to`, `pdc_line_to`, etc.

### Buffer

Opaque handle for GPU/CPU data buffers. Sized to `width × height × element_size`.

```
var field = Buffer(.Vec4F32, 0.0)

field.bind("field_in", 0)       // bind to kernel parameter
field.display_buffer()          // display as pixel output
swap(field, field_next)         // swap two buffers (free function reads more naturally)
```

- Type: `PdcType::BufferHandle`
- Codegen: `i32`
- Constructor: `Buffer(BufferType) -> Buffer` maps to `pdc_create_buffer`, always zero-initialized
- BufferType enum: `F32`, `I32`, `U32`, `Vec2F32`, `Vec3F32`, `Vec4F32`
- Non-zero initialization: use an init kernel (several examples already do this)

### Kernel

Opaque handle for compiled WGSL compute kernels.

```
var kern = Kernel("advect", "smoke_advect.wgsl", .Sim)

kern.set_arg("radius", 15.0)   // set parameter for next run
kern.run()                      // dispatch kernel
```

- Type: `PdcType::KernelHandle`
- Codegen: `i32`
- Constructor: `Kernel(string, string, KernelType) -> Kernel` maps to `pdc_load_kernel`
- KernelType enum: `Pixel` (type 0, writes pixel output), `Sim` (type 1, general compute)
- Note: `set_arg` accumulates args for the next `run()` call. The kernel handle is passed through but not yet validated against the pending args.

## Future

### Scene

Scene kernels currently use free functions with bare `i32` handles (`load_scene`, `run_scene`, `scene_buffer`, `scene_tiles_x`, `scene_num_paths`). These should become a `Scene` opaque type:

```
var scene = Scene("stress", "stress_scene.pdc")

scene.run()
scene.buffer("segments").bind("segments", 0)
scene.tiles_x()
scene.num_paths()
```

### Texture

Textures currently use `load_texture()` returning bare `i32`. Could become:

```
var tex = Texture("photo", "image.png")
```

### Generic Buffer constructor syntax

Replace `Buffer(.Vec4F32)` with WGSL type syntax: `Buffer<vec4<f32>>()`. This would:
- Use the actual WGSL type names directly, reducing the conceptual mapping
- Eliminate the `BufferType` enum entirely
- Follow the existing `Array<f64>()` pattern in PDC

```
var field = Buffer<vec4<f32>>()
var grid = Buffer<i32>()
var pixels = Buffer<u32>()
```

The parser would handle `Buffer<type>()` like it handles `Array<type>()`, encoding the WGSL type in the call name. The codegen maps WGSL type names to type codes.

Parsing challenge: WGSL types like `vec4<f32>` have nested angle brackets. The parser would need to handle the full set: `f32`, `i32`, `u32`, `vec2<f32>`, `vec3<f32>`, `vec4<f32>`.

## Open Questions

### Builtin Enum Discoverability

Builtin enums (`BufferType`, `KernelType`, `FillRule`, etc.) are registered only in Rust (`type_check.rs:register_builtins`). There's no way to discover valid variants from PDC source. Options:

1. **Better error messages** — list valid variants when a variant doesn't match
2. **VS Code autocomplete** — suggest variants when typing `.` inside a typed call
3. **`builtin enum` declarations** — require `builtin enum BufferType` in `.pdc` files, making the type visible in source while the compiler verifies it matches the Rust-side definition
