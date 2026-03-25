# PDP — Pixel Doodle Pipeline Reference

PDP is a declarative configuration language that describes how to run pixel-doodle examples. A `.pdp` file specifies what kernels to load, how to initialize buffers, what pipeline of steps to execute each frame, and how user input maps to actions.

## Running a PDP File

```bash
cargo run --release -- examples/basic/gradient/gradient.pdp
cargo run --release -- examples/sim/smoke/smoke.pdp --output frame.ppm
cargo run --release -- examples/sim/gray_scott/gray_scott.pdp --bench
cargo run --release -- examples/sim/gray_scott/gray_scott.pdp --set backend=gpu
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
  pixel kernel "my_kernel.pd"
  display my_kernel
}
```

## Pipeline Blocks

Every `.pdp` file must have at least one `pipeline` block. Pipelines contain kernel declarations, buffer declarations, and execution steps.

### Single pipeline (unnamed)

```
pipeline {
  pixel kernel "gradient.pd"
  display gradient
}
```

### Named pipelines (CPU + GPU)

A single file can define multiple named pipelines for different backends:

```
pipeline cpu {
  pixel kernel "mandelbrot.pd"
  accumulate(samples: 256) {
    display mandelbrot
  }
}

pipeline gpu {
  pixel kernel "mandelbrot.wgsl"
  display mandelbrot
}
```

Pipeline selection:
- **One pipeline**: used regardless of backend setting
- **Multiple pipelines**: `--set backend=gpu` selects `pipeline gpu`, otherwise `pipeline cpu`
- **Default backend**: cranelift (for CPU pipelines)

## Kernel Declarations

Kernel declarations must be inside a `pipeline` block. Three types:

```
pipeline {
  pixel kernel "gradient.pd"                  # pixel shader — per-pixel computation
  sim kernel "gray_scott.pd"                  # simulation — reads/writes buffers
  init kernel init_spots = "init/spots.pd"    # initialization — fills a buffer once
}
```

### Unnamed kernels

When no name is given, the kernel name is derived from the filename:

```
pixel kernel "gradient.pd"      # name: gradient
sim kernel "smoke/advect.pd"    # name: advect
```

### Named kernels

Use `name = "path"` to give a kernel an explicit name:

```
sim kernel advect = "smoke/advect.pd"
sim kernel divergence = "smoke/divergence.pd"
```

### Kernel types

| Type | Description | ABI |
|------|-------------|-----|
| `pixel` | Per-pixel shader. Receives view coordinates (x, y), produces ARGB pixel. | `TileKernelFn` |
| `sim` | Simulation step. Reads/writes f64 buffers, produces ARGB pixels. | `SimTileKernelFn` |
| `init` | Buffer initializer. Writes one f64 per cell. Runs once at startup. | `SimTileKernelFn` |

### Path resolution

Paths are relative to the config file's directory by default. Use `@root/` to reference from the project root:

```
pixel kernel "gradient.pd"                       # relative to config file
pixel kernel "@root/examples/shared/util.pd"     # relative to project root
```

## Buffer Declarations

Buffer declarations must be inside a `pipeline` block. CPU simulation buffers are `width × height` arrays of `f64`. GPU buffers require a type annotation:

```
pipeline {
  # CPU buffers
  buffer u = constant(1.0)                         # fill with 1.0
  buffer state = init_state(density: 0.3, seed: 42) # run init kernel

  # GPU buffers (with type annotation)
  buffer field: gpu(vec2f) = constant(0.0)
  buffer pixels: gpu(u32) = constant(0.0)
}
```

### GPU element types

| Type | Size | Description |
|------|------|-------------|
| `f32` | 4 bytes | Single float |
| `vec2f` | 8 bytes | 2-component float vector |
| `vec3f` | 16 bytes | 3-component float vector (padded) |
| `vec4f` | 16 bytes | 4-component float vector |
| `i32` | 4 bytes | Signed integer |
| `u32` | 4 bytes | Unsigned integer |

## Variables

Mutable values that can be modified by key bindings and referenced by pipeline constructs:

```
gravity = 9.8                              # simple variable, no bounds
iterations: range(1..10) = 8              # clamped to [1, 10]
color_mode: range(0..3, wrap: true) = 0   # wraps around at boundaries
paused = false                             # boolean (stored as 0.0/1.0)
```

### Range syntax

```
name: range(min..max) = default
name: range(min..max, wrap: true) = default
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
  backend = "cranelift"    # JIT backend: "cranelift", "llvm", or "gpu"
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

Variables with `range()` are automatically clamped or wrapped after modification.

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

### `run` — Execute a kernel

```
run kernel_name                                         # no I/O
buf = run inject(value: 1.0, radius: 5)                # built-in, one output
u_next, v_next = run gray_scott { u_in: u, v_in: v }  # input bindings, outputs
```

- **Output assignment**: buffer names on the left of `=` receive the kernel's write buffers in declaration order
- **Parameters**: `(name: value, ...)` passed to the kernel
- **Input bindings**: `{ param: buffer, ... }` maps the kernel's read buffer slots to config buffers

### `display` — Execute and show pixels

Same syntax as `run`, but the kernel's ARGB pixel output is sent to the screen:

```
display gradient                                        # simple pixel kernel
vx0, vy0 = display project { press_in: pressure, ... } # sim kernel with outputs
```

Every pipeline must have at least one `display` step.

### `swap` — Swap buffers

```
swap u <-> u_next
swap vx <-> vx0, vy <-> vy0, density <-> density0
```

Swaps buffer contents by pointer (O(1), no copying). Multiple pairs can be comma-separated.

### `loop` — Repeat within a frame

```
loop(iterations: 8) {
  # steps repeated 8 times per frame
}

loop(iterations: iterations) {
  # iterations controlled by a variable (adjustable at runtime)
}
```

### `accumulate` — Progressive sampling across frames

```
accumulate(samples: 256) {
  display mandelbrot
}
```

Each frame renders one sample. Results are accumulated and averaged for display. Resets when viewport changes (center/zoom modification).

### `on click` — Mouse event handler

```
on click(continuous: true) {
  state = run inject(value: 1.0, radius: 3)
  age = run inject(value: 0.0, radius: 3)
}
```

- `continuous: true` — fires every frame while the mouse button is held
- `continuous: false` — fires once per click (default)
- Position in the pipeline determines execution order relative to other steps

### Built-in `inject` kernel

Writes a value into a buffer around the current mouse position:

```
buf = run inject(value: 0.5, radius: 5)
buf = run inject(value: -3.0, radius: 15, falloff: "quadratic")
```

| Parameter | Description |
|-----------|-------------|
| `value` | Value to write (f64) |
| `radius` | Injection radius in pixels |
| `falloff` | `"flat"` (default, sets value) or `"quadratic"` (scales by 1-d²/r², adds to existing) |

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
backend = "cranelift"
width = 1920
height = 1080
```

## Complete Examples

### Gradient (simplest)

```
pipeline {
  pixel kernel "gradient.pd"
  display gradient
}
```

### Mandelbrot (progressive + pan/zoom, CPU and GPU)

```
include "../../shared/pan_zoom.pdp"

pipeline cpu {
  pixel kernel "mandelbrot.pd"
  accumulate(samples: 256) {
    display mandelbrot
  }
}

pipeline gpu {
  pixel kernel "mandelbrot.wgsl"
  display mandelbrot
}
```

### Gray-Scott (simulation with CPU and GPU pipelines)

```
title = "Gray-Scott Reaction Diffusion"

on key(space) paused = !paused
on key(period) frame += 1

pipeline cpu {
  sim kernel "gray_scott.pd"
  init kernel init_u = "init/gray_scott_u.pd"
  init kernel init_v = "init/gray_scott_v.pd"

  buffer u = init_u()
  buffer v = init_v()
  buffer u_next = constant(0.0)
  buffer v_next = constant(0.0)

  on click(continuous: true) {
    v = run inject(value: 0.5, radius: 5)
  }
  loop(iterations: 8) {
    u_next, v_next = display gray_scott { u_in: u, v_in: v }
    swap u <-> u_next
    swap v <-> v_next
  }
}

pipeline gpu {
  sim kernel step = "gray_scott_step.wgsl"
  sim kernel vis = "gray_scott_vis.wgsl"

  buffer field: gpu(vec2f) = constant(1.0)
  buffer field_next: gpu(vec2f) = constant(1.0)
  buffer pixels: gpu(u32) = constant(0.0)

  on click(continuous: true) {
    field = run inject(value: 0.5, radius: 5)
  }
  loop(iterations: 8) {
    field_next = run step { field_in: field, field_out: field_next }
    swap field <-> field_next
  }
  display vis { field_in: field, pixels: pixels }
}
```

### Smoke (multi-kernel CPU pipeline)

```
title = "Smoke Simulation"

on key(space) paused = !paused
on key(period) frame += 1

pipeline cpu {
  sim kernel advect = "advect.pd"
  sim kernel divergence = "divergence.pd"
  sim kernel jacobi = "jacobi.pd"
  sim kernel project = "project.pd"

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
    vy = run inject(value: -3.0, radius: 15, falloff: "quadratic")
    density = run inject(value: 0.5, radius: 15, falloff: "quadratic")
  }
  swap vx <-> vx0, vy <-> vy0, density <-> density0
  vx, vy, density = run advect { vx_in: vx0, vy_in: vy0, den_in: density0 }
  divergence = run divergence { vx_in: vx, vy_in: vy }
  loop(iterations: 40) {
    pressure_tmp = run jacobi { div_in: divergence, press_in: pressure }
    swap pressure <-> pressure_tmp
  }
  vx0, vy0 = display project { press_in: pressure, vx_in: vx, vy_in: vy, den_in: density }
  swap vx <-> vx0, vy <-> vy0
}
```

### Game of Life (interactive with speed control)

```
title = "Game of Life"

iterations: range(1..10) = 1

on key(space) paused = !paused
on key(period) frame += 1
on key(bracket_right) iterations += 1
on key(bracket_left) iterations -= 1

pipeline cpu {
  sim kernel "game_of_life.pd"
  init kernel init_state = "init/random_binary.pd"

  buffer state = init_state()
  buffer age = constant(0.0)
  buffer state_next = constant(0.0)
  buffer age_next = constant(0.0)

  on click(continuous: true) {
    state = run inject(value: 1.0, radius: 3)
    age = run inject(value: 0.0, radius: 3)
  }
  loop(iterations: iterations) {
    state_next, age_next = display game_of_life { state_in: state, age_in: age }
    swap state <-> state_next
    swap age <-> age_next
  }
}
```
