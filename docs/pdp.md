# PDP -- Pixel Doodle Pipeline Reference

PDP is a declarative configuration language that describes how to run pixel-doodle examples. A `.pdp` file specifies what kernels to load, how to initialize buffers, what pipeline of steps to execute each frame, and how user input maps to actions.

## Running a PDP File

```bash
cargo run --release -- examples/basic/gradient/gradient.pdp
cargo run --release -- examples/sim/smoke/smoke.pdp --output frame.ppm
cargo run --release -- examples/sim/gray_scott/gray_scott.pdp --bench
cargo run --release -- examples/sim/gray_scott/gray_scott.pdp --set render=gpu
```

Additional CLI flags:

| Flag | Description |
|------|-------------|
| `--output <file.ppm>` | Render one frame and save as PPM (no window) |
| `--bench` | Headless benchmark (5 warmup + 100 timed frames) |
| `--bench-frames <N>` | Number of benchmark frames (default: 100) |
| `--settings <file.pds>` | Override settings from a `.pds` file |
| `--set key=value` | Override a single setting or variable |

## File Structure

A `.pdp` file has top-level directives (title, variables, settings, key bindings) and one or more `pipeline` blocks. Kernel and buffer declarations go inside pipeline blocks. Lines starting with `#` are comments.

```
# This is a comment
title = "My Example"

on key(plus) zoom *= 1.1

pipeline {
  pixel kernel "my_kernel.wgsl"
  run my_kernel
  display
}
```

## Pipeline Blocks

Every `.pdp` file must have at least one `pipeline` block. Pipelines contain kernel declarations, buffer declarations, and execution steps.

### Single pipeline (unnamed)

```
pipeline {
  pixel kernel "gradient.wgsl"
  run gradient
  display
}
```

### Named pipelines

A single file can define multiple named pipelines for different variants:

```
pipeline cpu {
  pixel kernel "mandelbrot.wgsl"
  accumulate(samples: 256) {
    run mandelbrot
    display
  }
}

pipeline gpu {
  pixel kernel "mandelbrot.wgsl"
  run mandelbrot with(pixels: out pixels)
  display pixels
}
```

Pipeline names are arbitrary identifiers. Common conventions: `cpu`, `gpu`.

Pipeline selection:
- **One pipeline**: always used
- **Multiple pipelines**: the first pipeline is the default
- **Explicit selection**: `--set pipeline=gpu` or `settings { pipeline = "gpu" }`

## Kernel Declarations

Kernel declarations must be inside a `pipeline` block. Three types:

```
pipeline {
  pixel kernel "gradient.wgsl"                  # pixel shader -- per-pixel computation
  kernel "gray_scott.wgsl"                      # general kernel -- reads/writes buffers
  scene kernel "basic.pdc"                      # PDC scene -- JIT-compiled vector geometry
}
```

### Unnamed kernels

When no name is given, the kernel name is derived from the filename:

```
pixel kernel "gradient.wgsl"      # name: gradient
kernel "smoke/advect.wgsl"        # name: advect
```

### Named kernels

Use `name = "path"` to give a kernel an explicit name:

```
kernel advect = "smoke/advect.wgsl"
kernel divergence = "smoke/divergence.wgsl"
```

### Kernel types

| Type | Description | ABI |
|------|-------------|-----|
| `pixel` | Per-pixel shader. Receives view coordinates (x, y), produces ARGB pixel. | `TileKernelFn` |
| *(bare)* `kernel` | General kernel. Reads/writes f64 buffers, produces ARGB pixels. Used for simulation steps. | `SimTileKernelFn` |
| `scene` | PDC scene kernel (`.pdc` file). JIT-compiled via Cranelift, produces vector geometry that is flattened, tiled, and uploaded as GPU buffers for a WGSL rasterizer. | `PdcSceneFn` |

Initialization kernels are no longer a separate declaration type. Instead, use a bare `kernel` and run it inside an `init { }` block (see [Pipeline Steps](#pipeline-steps)).

### Scene kernels

Scene kernels reference `.pdc` files that describe vector geometry using the PDC language. When a scene kernel executes, the runtime:

1. Runs the compiled PDC program to produce draw commands (paths, fills, strokes)
2. Flattens curves into line segments and bins them into tiles
3. Uploads scene data buffers to the GPU: `segments`, `seg_path_ids`, `tile_offsets`, `tile_counts`, `tile_indices`, `path_colors`, `path_fill_rules`
4. Sets runtime variables: `__scene_tiles_x`, `__scene_tiles_y`, `__scene_num_paths`

These buffers and variables are available to subsequent WGSL rasterizer kernels without explicit buffer declarations. A typical scene pipeline pairs a `scene kernel` with a tile rasterizer `kernel`:

```
pipeline gpu {
  scene kernel basic = "basic.pdc"
  kernel rasterize = "tile_raster.wgsl"

  buffer pixels: gpu(u32) = constant(0)

  run basic
  run rasterize(
    tile_size: 16,
    tiles_x: __scene_tiles_x,
    num_paths: __scene_num_paths
  ) with(
    segments: segments,
    seg_path_ids: seg_path_ids,
    tile_offsets: tile_offsets,
    tile_counts: tile_counts,
    tile_indices: tile_indices,
    path_colors: path_colors,
    path_fill_rules: path_fill_rules,
    pixels: out pixels
  )
  display pixels
}
```

### Path resolution

Paths are relative to the config file's directory by default. Use `@root/` to reference from the project root:

```
pixel kernel "gradient.wgsl"                       # relative to config file
pixel kernel "@root/examples/shared/util.wgsl"     # relative to project root
```

## Buffer Declarations

Buffer declarations must be inside a `pipeline` block. CPU simulation buffers are `width x height` arrays of `f64`. GPU buffers require a type annotation:

```
pipeline {
  # CPU buffers
  buffer u = constant(1.0)                         # fill with 1.0
  buffer state = constant(0.0)                     # fill with 0.0 (use init block to populate)

  # GPU buffers (with type annotation)
  buffer field: gpu(vec2<f32>) = constant(0.0)
  buffer pixels: gpu(u32) = constant(0.0)
}
```

### GPU element types

| Type | Size | Description |
|------|------|-------------|
| `f32` | 4 bytes | Single float |
| `vec2<f32>` | 8 bytes | 2-component float vector |
| `vec3<f32>` | 16 bytes | 3-component float vector (padded) |
| `vec4<f32>` | 16 bytes | 4-component float vector |
| `i32` | 4 bytes | Signed integer |
| `u32` | 4 bytes | Unsigned integer |

## Texture Declarations

Texture declarations must be inside a `pipeline` block. They load read-only image files (PNG, JPEG) for use by kernels:

```
pipeline {
  texture noise = file("noise.png")
  texture photo = file("../../images/photo.jpg")
}
```

### Syntax

```
texture name = file("path/to/image.png")
```

- Paths are relative to the config file's directory
- Supported formats: PNG, JPEG (via the `image` crate)
- Textures are loaded as RGBA8 data
- Texture names must match the WGSL binding names in the kernel

### Example

```
pipeline {
  pixel kernel "textured.wgsl"
  texture albedo = file("albedo.png")
  texture normal = file("normal.png")
  run textured
  display
}
```

## Variables

Mutable values that can be modified by key bindings and referenced by pipeline constructs:

```
gravity = 9.8                              # simple variable, no bounds
iterations: range<u32>(1..10) = 8              # clamped to [1, 10]
color_mode: range<u32>(0..3, wrap: true) = 0   # wraps around at boundaries
paused = false                             # boolean (stored as 0.0/1.0)
```

### Range syntax

```
name: range<type>(min..max) = default
name: range<type>(min..max, wrap: true) = default
```

- Without `wrap`: values are clamped to `[min, max]`
- With `wrap: true`: values wrap around when exceeding bounds

## Intrinsic Globals

These variables exist without declaration. They are maintained by the runtime and can be modified by key bindings:

| Name | Type | Description |
|------|------|-------------|
| `width` | u32 | Canvas width in pixels |
| `height` | u32 | Canvas height in pixels |
| `center_x` | f64 | Viewport center X coordinate |
| `center_y` | f64 | Viewport center Y coordinate |
| `zoom` | f64 | Viewport zoom level |
| `mouse_x` | f64 | Mouse X position in pixels |
| `mouse_y` | f64 | Mouse Y position in pixels |
| `time` | f64 | Elapsed seconds since start |
| `paused` | bool | Whether `frame` auto-increments |
| `frame` | u64 | Current frame number |

When `paused` is true, `frame` stops auto-incrementing. Manually setting `frame += 1` enables single-stepping.

## Builtin Declarations

Builtin declarations expose intrinsic globals to the PDP file, optionally with a default value. Use `builtin const` for read-only access and `builtin var` for read-write:

```
builtin const width: u32
builtin const mouse_x: f64
builtin var paused: bool
builtin var frame: u64
```

### Default values

Builtin vars can specify a default value that overrides the runtime default:

```
builtin var paused: bool = true     # start paused -- render one frame then stop
```

This is useful for static scenes (e.g. PDC vector rendering) that should render once without continuous animation. When `paused` starts true, the pipeline executes one frame then stops the event loop from polling.

## Title

```
title = "Smoke Simulation"
```

Sets the window title. Default: kernel filename without extension.

## Settings Block

Execution settings not visible to kernels:

```
settings {
  threads = 4              # worker thread count
  render = "gpu"            # render mode: "gpu" or "cpu"
  codegen = "cranelift"    # JIT backend (for cpu render): "cranelift" or "llvm"
  tile_height = 8          # rows per tile for parallel dispatch
}
```

## Key Bindings

Bind keyboard keys to variable modifications:

```
on key(space) paused = !paused
on key(period) frame += 1
on key(bracket_right) iterations += 1
on key(bracket_left) iterations -= 1
on key(up) gravity += 0.1
on key(left) center_x -= 0.1
on key(plus) zoom *= 1.1
on key(minus) zoom /= 1.1
on key(0) zoom = 1.0
```

### Block syntax

A single key can trigger multiple actions using a block:

```
on key(0) {
  center_x = 0.0
  center_y = 0.0
  zoom = 1.0
}
```

### Action expressions

| Form | Description |
|------|-------------|
| `var = !var` | Toggle boolean |
| `var = literal` | Direct assignment |
| `var += literal` | Add to variable |
| `var -= literal` | Subtract from variable |
| `var *= literal` | Multiply variable |
| `var /= literal` | Divide variable |
| `var = var + literal` | Expanded add form |
| `var = var - literal` | Expanded subtract form |
| `var = var * literal` | Expanded multiply form |
| `var = var / literal` | Expanded divide form |
| `quit` | Exit the application |

Variables with `range<type>()` are automatically clamped or wrapped after modification.

### Key names

| PDP name | Key |
|----------|-----|
| `space` | Space bar |
| `period` | Period (.) |
| `comma` | Comma (,) |
| `left` | Left arrow |
| `right` | Right arrow |
| `up` | Up arrow |
| `down` | Down arrow |
| `plus` | = / Numpad + |
| `minus` | - / Numpad - |
| `bracket_left` | [ |
| `bracket_right` | ] |
| `0` .. `9` | Number keys |
| `r` | R key |
| `escape` | Escape |
| `q` | Q key |

## Include

Reuse shared configuration across multiple `.pdp` files:

```
include "../../shared/pan_zoom.pdp"
```

- Paths are resolved relative to the including file's directory
- Included files may contain: variables, key bindings, settings, title, and nested includes
- Included files must **not** contain pipeline blocks
- Circular includes are detected and silently deduplicated

Included content is merged into the including file as if written inline.

## Pipeline Steps

### `run` -- Execute a kernel

```
run kernel_name                                              # no I/O (pixel kernel)
run mandelbrot(max_iter: 256)                                # user-defined argument
run mandelbrot(max_iter: max_iters)                          # argument from a variable
run inject(value: 1.0, radius: 5) with(target: out state)     # built-in inject
run advect with(vx_in: vx0, den_in: density0, vx_out: out vx) # sim kernel with bindings
```

- **Arguments**: `(name: value, ...)` passed to user-defined kernel parameters (see below)
- **Buffer bindings**: `with(param: buffer, ...)` maps kernel buffer slots to pipeline buffers
- **Output bindings**: use the `out` qualifier to mark write buffers: `with(slot: out buffer)`

#### User-defined kernel arguments

Kernels can declare parameters beyond the built-in names (`x`, `y`, `px`, `py`, `sample_index`, `time`, `width`, `height` for pixel kernels; `px`, `py`, `width`, `height` for simulation kernels). Any parameter whose name is not a built-in is a **user-defined argument** that must be supplied in the `run` statement.

Must be called with that argument in the .pdp file:

```
run mandelbrot(max_iter: 256)
```

Argument values can be:
- **Literals**: `256`, `0.037`, `true`
- **Variable references**: bare identifiers that resolve to a declared variable or intrinsic

```
max_iters: range<u32>(10..500) = 256
on key(up) max_iters += 10
on key(down) max_iters -= 10

pipeline {
  pixel kernel "mandelbrot.wgsl"
  run mandelbrot(max_iter: max_iters)    # resolved each frame from the variable
  display
}
```

Variable references are resolved at runtime before each kernel dispatch, so changes via key bindings or `--set` take effect immediately.

Supported argument types: all scalar types (`f32`, `f64`, `i8`, `u8`, `i16`, `u16`, `i32`, `u32`, `i64`, `u64`). All user args must be explicitly provided -- there are no default values.

### `display` -- Show pixels on screen

Presents pixel data to the screen. Takes no arguments (CPU) or a buffer name (GPU):

```
display                    # CPU -- display the pixel buffer written by the most recent run step
display pixels             # GPU -- display the named gpu(u32) buffer
```

Every pipeline must have at least one `display` step.

**CPU pipelines**: The runtime maintains a hidden pixel buffer. Kernels executed via `run` write ARGB pixels into it as a side effect. `display` (with no arguments) presents that buffer to the screen.

**GPU pipelines**: Compute shaders run entirely on the GPU and the runtime cannot inject a hidden buffer. Use `display buffer_name` to identify which `gpu(u32)` buffer contains the pixel output:

```
run vis with(grid_in: grid, pixels: out pixels)
display pixels
```

### `init` -- Run steps once at startup

The `init { }` block contains steps that execute exactly once before the first frame. Use it to populate buffers with initial values:

```
init {
  run init_state with(out: out state)
  run init_age with(age: out age)
}
```

Steps inside `init` follow the same syntax as regular pipeline steps (`run`, `swap`, etc.) but `display` is not allowed inside `init`.

### `swap` -- Swap buffers

```
swap u, u_next
swap vx, vx0
swap vy, vy0
```

Swaps buffer contents by pointer (O(1), no copying). One pair per statement.

### `loop` -- Repeat within a frame

```
loop(iterations: 8) {
  # steps repeated 8 times per frame
}

loop(iterations: iterations) {
  # iterations controlled by a variable (adjustable at runtime)
}
```

### `accumulate` -- Progressive sampling across frames

```
accumulate(samples: 256) {
  run mandelbrot
  display
}
```

Each frame renders one sample. Results are accumulated and averaged for display. Resets when viewport changes (center/zoom modification).

### `on click` -- Mouse event handler

```
on click(continuous: true) {
  run inject(value: 1.0, radius: 3) with(target: out state)
  run inject(value: 0.0, radius: 3) with(target: out age)
}
```

- `continuous: true` -- fires every frame while the mouse button is held
- `continuous: false` -- fires once per click (default)
- Position in the pipeline determines execution order relative to other steps

### Built-in `inject` kernel

Writes a value into a buffer around the current mouse position:

```
run inject(value: 0.5, radius: 5) with(target: out buf)
run inject(value: -3.0, radius: 15, falloff: "quadratic") with(target: out buf)
```

| Parameter | Description |
|-----------|-------------|
| `value` | Value to write (f64) |
| `radius` | Injection radius in pixels |
| `falloff` | `"flat"` (default, sets value) or `"quadratic"` (scales by 1-d^2/r^2, adds to existing) |

## Override Layering

Settings and variables can be overridden from multiple sources. Precedence (highest to lowest):

1. `--set key=value` (CLI)
2. `--settings file.pds` (external file)
3. PDP file defaults

### `.pds` file format

A flat key-value file (one per line, `#` comments):

```
# my_machine.pds
threads = 4
render = "gpu"
codegen = "cranelift"
width = 1920
height = 1080
```

## Complete Examples

### Gradient (simplest)

```
pipeline {
  pixel kernel "gradient.wgsl"
  run gradient
  display
}
```

### Mandelbrot (progressive + pan/zoom, multiple pipelines)

```
include "../../shared/pan_zoom.pdp"

pipeline cpu {
  pixel kernel "mandelbrot.wgsl"
  accumulate(samples: 256) {
    run mandelbrot
    display
  }
}

pipeline gpu {
  pixel kernel "mandelbrot.wgsl"
  buffer pixels: gpu(u32) = constant(0)
  run mandelbrot with(pixels: out pixels)
  display pixels
}
```

### Gray-Scott (simulation with multiple pipelines)

```
title = "Gray-Scott Reaction Diffusion"

on key(space) paused = !paused
on key(period) frame += 1

pipeline cpu {
  kernel "gray_scott.wgsl"
  kernel init_u = "init/gray_scott_u.wgsl"
  kernel init_v = "init/gray_scott_v.wgsl"

  buffer u = constant(0.0)
  buffer v = constant(0.0)
  buffer u_next = constant(0.0)
  buffer v_next = constant(0.0)

  init {
    run init_u with(out: out u)
    run init_v with(out: out v)
  }
  on click(continuous: true) {
    run inject(value: 0.5, radius: 5) with(target: out v)
  }
  loop(iterations: 8) {
    run gray_scott with(u_in: u, v_in: v, u_out: out u_next, v_out: out v_next)
    display
    swap u, u_next
    swap v, v_next
  }
}

pipeline gpu {
  kernel step = "gray_scott_step.wgsl"
  kernel vis = "gray_scott_vis.wgsl"

  buffer field: gpu(vec2<f32>) = constant(1.0)
  buffer field_next: gpu(vec2<f32>) = constant(1.0)
  buffer pixels: gpu(u32) = constant(0.0)

  on click(continuous: true) {
    run inject(value: 0.5, radius: 5) with(target: out field)
  }
  loop(iterations: 8) {
    run step with(field_in: field, field_out: out field_next)
    swap field, field_next
  }
  run vis with(field_in: field, pixels: out pixels)
  display pixels
}
```

### Smoke (multi-kernel pipeline)

```
title = "Smoke Simulation"

on key(space) paused = !paused
on key(period) frame += 1

pipeline {
  kernel advect = "advect.wgsl"
  kernel divergence = "divergence.wgsl"
  kernel jacobi = "jacobi.wgsl"
  kernel project = "project.wgsl"

  buffer vx = constant(0.0)
  buffer vy = constant(0.0)
  buffer density = constant(0.0)
  buffer vx0 = constant(0.0)
  buffer vy0 = constant(0.0)
  buffer density0 = constant(0.0)
  buffer pressure = constant(0.0)
  buffer pressure_tmp = constant(0.0)
  buffer divergence = constant(0.0)

  on click(continuous: true) {
    run inject(value: -3.0, radius: 15, falloff: "quadratic") with(target: out vy)
    run inject(value: 0.5, radius: 15, falloff: "quadratic") with(target: out density)
  }
  swap vx, vx0
  swap vy, vy0
  swap density, density0
  run advect with(vx_in: vx0, vy_in: vy0, den_in: density0, vx_out: out vx, vy_out: out vy, den_out: out density)
  run divergence with(vx_in: vx, vy_in: vy, div_out: out divergence)
  loop(iterations: 40) {
    run jacobi with(div_in: divergence, press_in: pressure, press_out: out pressure_tmp)
    swap pressure, pressure_tmp
  }
  run project with(press_in: pressure, vx_in: vx, vy_in: vy, den_in: density, vx_out: out vx0, vy_out: out vy0)
  display
  swap vx, vx0
  swap vy, vy0
}
```

### Game of Life (interactive with speed control)

```
title = "Game of Life"

iterations: range<u32>(1..10) = 1

on key(space) paused = !paused
on key(period) frame += 1
on key(bracket_right) iterations += 1
on key(bracket_left) iterations -= 1

pipeline {
  kernel "game_of_life.wgsl"
  kernel init_state = "init/random_binary.wgsl"

  buffer state = constant(0.0)
  buffer age = constant(0.0)
  buffer state_next = constant(0.0)
  buffer age_next = constant(0.0)

  init {
    run init_state with(out: out state)
  }
  on click(continuous: true) {
    run inject(value: 1.0, radius: 3) with(target: out state)
    run inject(value: 0.0, radius: 3) with(target: out age)
  }
  loop(iterations: iterations) {
    run game_of_life with(state_in: state, age_in: age, state_out: out state_next, age_out: out age_next)
    display
    swap state, state_next
    swap age, age_next
  }
}
```

### PDC Vector Scene (scene kernel + tile rasterizer)

```
title = "PDC Basic"

builtin var paused: bool = true

pipeline gpu {
  scene kernel basic = "basic.pdc"
  kernel rasterize = "tile_raster.wgsl"

  buffer pixels: gpu(u32) = constant(0)

  run basic
  run rasterize(
    tile_size: 16,
    tiles_x: __scene_tiles_x,
    num_paths: __scene_num_paths
  ) with(
    segments: segments,
    seg_path_ids: seg_path_ids,
    tile_offsets: tile_offsets,
    tile_counts: tile_counts,
    tile_indices: tile_indices,
    path_colors: path_colors,
    path_fill_rules: path_fill_rules,
    pixels: out pixels
  )
  display pixels
}
```
