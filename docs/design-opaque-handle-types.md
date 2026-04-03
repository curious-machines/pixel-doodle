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
var field = Buffer.Vec4F32()
var grid = Buffer.I32()
var pixels = Buffer.U32()

field.display_buffer()          // display as pixel output
swap(field, field_next)         // swap two buffers (free function)
```

- Type: `PdcType::BufferHandle`
- Codegen: `i32`
- Constructor: `Buffer.Variant()` — variant is the element type. Maps to `pdc_create_buffer`, always zero-initialized
- Variants: `F32`, `I32`, `U32`, `Vec2F32`, `Vec3F32`, `Vec4F32`
- Non-zero initialization: use an init kernel (several examples already do this)

### Kernel

Opaque handle for compiled WGSL compute kernels. Supports virtual properties for buffer bindings and scalar arguments.

```
var kern = Kernel.Sim("advect", "smoke_advect.wgsl")
var pixel_kern = Kernel.Pixel("render", "render.wgsl")

// Buffer bindings via virtual properties
kern.field_in = Bind.In(field)
kern.field_out = Bind.Out(field_next)

// Scalar arguments via virtual properties
kern.radius = 15.0
kern.inject_x = mouse_x

kern.run()                      // dispatch kernel
```

- Type: `PdcType::KernelHandle`
- Codegen: `i32`
- Constructor: `Kernel.Variant(name, path)` — variant is the kernel kind. Maps to `pdc_load_kernel`
- Variants: `Pixel` (type 0, writes pixel output), `Sim` (type 1, general compute)

### Kernel Virtual Properties

Kernel handles support write-only virtual properties. Assigning to a property on a kernel sets either a buffer binding or a scalar argument, depending on the RHS type.

**Scope:** Kernel handles only. No other handle type gets virtual properties.

**Rule:** Virtual properties have no declared type. The type is inferred from the RHS. Because there's no expected type context, dot-shorthand cannot resolve — the RHS must be fully qualified (`Bind.In(buffer)`, not `.In(buffer)`).

**Codegen lowering:** The compiler inspects the RHS type to decide which runtime call to emit:

- RHS is `Bind.In(buffer)` → `pdc_bind_buffer(ctx, kernel, buffer, "field_name", 0)`
- RHS is `Bind.Out(buffer)` → `pdc_bind_buffer(ctx, kernel, buffer, "field_name", 1)`
- RHS is scalar (f64, f32, i32, bool) → `pdc_set_kernel_arg_f64(ctx, kernel, "field_name", value)`

**Performance:** Identical to the old `set_arg` / `bind` method calls. The property syntax is purely compile-time sugar.

### Bind Enum

`Bind` is an enum with payload variants that carry a `BufferHandle`:

```
Bind.In(buffer)    // input binding (direction 0)
Bind.Out(buffer)   // output binding (direction 1)
```

This is a compile-time-only construct — codegen destructures it into the buffer handle and direction constant. No `Bind` value is ever materialized at runtime.

This is the first use of enum variant constructors as expressions in PDC. See `design-enum-variant-constructors.md` for the general feature.

### Per-Kernel Persistent State

Buffer bindings and scalar arguments are stored per-kernel and persist until overwritten. `run()` reads the kernel's state without clearing it.

This enables setting binds once before a loop:

```
step_kern.grid_in = Bind.In(grid)
step_kern.grid_out = Bind.Out(grid_next)
for _i in 0..40 {
    step_kern.run()          // reuses binds from above
    swap(grid, grid_next)    // swaps data, handles stable
}
```

`swap()` exchanges buffer data, not handle IDs, so persistent binds remain correct across ping-pong iterations.

### Texture

Opaque handle for loaded image textures. Created via constructor, accessible in WGSL shaders by the declared name.

```
var tex = Texture("img", "images/lake.jpg")
```

- Type: `PdcType::TextureHandle`
- Codegen: `i32`
- Constructor: `Texture(name, path)` — loads image, returns handle. Maps to `pdc_load_texture`
- The `name` parameter determines the WGSL binding name; the shader declares `var img: texture_2d<f32>` to access it

#### Future Texture Operations

These are not yet implemented but would be useful additions:

- **Query dimensions:** `tex.width()`, `tex.height()` — read texture size from PDC for layout/coordinate calculations
- **Unload/replace at runtime:** `tex.replace("new_image.png")` — swap the underlying image without restarting
- **Error checking:** `tex.is_valid()` — check if the texture loaded successfully (constructor currently returns -1 on failure)
- **Dynamic texture selection:** pass texture handles to kernels or use them to choose which texture to bind at runtime
- **Pixel readback in PDC:** `tex.sample(u, v)` or `tex.pixel(x, y)` — read pixel data from PDC code, not just from WGSL shaders

## Future

### Scene

Scene kernels currently use free functions with bare `i32` handles (`load_scene`, `run_scene`, `scene_buffer`, `scene_tiles_x`, `scene_num_paths`). These should become a `Scene` opaque type:

```
var scene = Scene("stress", "stress_scene.pdc")

scene.run()
scene.buffer("segments")
scene.tiles_x()
scene.num_paths()
```

### Buffer with explicit dimensions

Currently `Buffer.I32()` always creates a buffer sized to `width × height`. A future extension could allow explicit dimensions:

```
var grid = Buffer.I32(width: 100, height: 1)
```

This would be useful for 1D buffers, non-square grids, or buffers that don't match the display dimensions.

## Open Questions

### Wildcard `_` in for-in loops

Currently `_` is only recognized as a discard in match pattern bindings (`.Variant(_, y)`). For-in loops use `_i` as a naming convention, but the compiler doesn't treat it specially — it's a regular variable.

We should support `_` as a proper wildcard in for-in loops:

```
for _ in 0..40 {
    // loop body doesn't use the index
}
```

This would mean the compiler doesn't allocate or bind the loop variable. If we later add unused-variable warnings, `_`-prefixed names (like `_i`) should suppress them, following the Rust convention.

### Builtin Enum Discoverability

Builtin enums (`BufferType`, `KernelType`, `FillRule`, etc.) are registered only in Rust (`type_check.rs:register_builtins`). There's no way to discover valid variants from PDC source. Options:

1. **Better error messages** — list valid variants when a variant doesn't match
2. **VS Code autocomplete** — suggest variants when typing `.` inside a typed call
3. **`builtin enum` declarations** — require `builtin enum BufferType` in `.pdc` files, making the type visible in source while the compiler verifies it matches the Rust-side definition
