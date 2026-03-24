# PDP Configuration DSL Design

## Problem

`main.rs` has ~1650 lines with a 13-variant `Backend` enum. Every example requires custom Rust code for buffer allocation, initialization, execution sequencing, buffer swaps, mouse injection, and keyboard controls. Adding a new simulation means touching multiple match arms across setup, rendering, and input handling.

## Goal

A custom DSL (`.pdp` — Pixel Doodle Config) that declaratively describes everything needed to run any kernel example, eliminating per-example Rust code. The app should be able to define and execute kernels at runtime with runtime-adjustable settings (threads, backend, tile height, etc.) rather than CLI-only arguments.

## Design Principles

- **Explicit over magic** — no auto-detection of kernel types, no hidden behavior
- **Everything is a kernel** — init, injection, and constant fill are all kernels under the hood; `constant()` is a built-in convenience
- **Pipeline is the central concept** — all execution (loops, progressive sampling, event handling) is expressed as pipeline steps
- **Assignment syntax (`=`)** for setting values; declaration syntax for resources (kernels, buffers)
- **Boolean settings use `= true/false`** — forward-compatible with UI controls

## Design Decisions

- **GPU**: Include in format if it doesn't overly complicate CPU/JIT config
- **Init strategy**: Use .pd kernels for all buffer initialization (init kernel returns f64 per cell); `constant()` as built-in convenience
- **Pipeline**: Ordered step list (explicit, declarative); encompasses loops, accumulation, and event handlers
- **Native backends**: Defer external plugin loading to later; keep in-tree for now
- **Paths**: Relative to config file by default; `@root/` prefix for project root
- **Format**: Custom DSL (not TOML), purpose-built for this project; hand-written recursive descent parser
- **Runtime configurability**: Settings like thread count, backend, tile height configurable at runtime
- **Minimize CLI args**: Only `--settings file.pds` and `--set key=value` beyond the config file path

---

## PDP Language Specification

### Syntax Overview

Line-oriented, `#` comments, blocks with `{ }`. Minimal punctuation. Parseable with simple recursive descent (same approach as PD parser).

### Kernel Declarations

Three kernel types, explicitly declared:

```
pixel kernel "gradient.pd"
sim kernel "gray_scott.pd"
init kernel init_random = "init/random.pd"
```

- Type is required (`pixel`, `sim`, `init`) — no auto-detection
- Runtime validates kernel signature matches declared type
- Backend inferred from file extension: `.pd`/`.pdir` → CPU/JIT, `.wgsl` → GPU
- Unnamed kernels get a default name from the file's base name (must be valid identifier)
- Named kernels use `name = "path"` syntax

#### Named multi-kernel declarations
```
sim kernel advect = "smoke/advect.pd"
sim kernel divergence = "smoke/divergence.pd"
sim kernel jacobi = "smoke/jacobi.pd"
sim kernel project = "smoke/project.pd"
```

#### Path resolution
```
pixel kernel "gradient.pd"                    # relative to config file
pixel kernel "@root/examples/shared/util.pd"  # relative to project root
```

### Buffer Declarations

```
buffer u = constant(1.0)
buffer v = constant(0.0)
buffer state = init_random(density: 0.3, seed: 42)
```

- All buffers are `width * height` arrays of `f64` (type from init kernel return type — noted for future consideration)
- `constant(value)` — built-in convenience kernel that fills with a value
- `init_name(params...)` — calls a declared init kernel with named parameters
- Init kernels have signature: `init_name(px: u32, py: u32, width: u32, height: u32) -> f64` plus any user-defined parameters

### Variables

Mutable values that can be modified by key bindings and flow into pipeline constructs and kernel parameters:

```
# Simple variable (no clamping)
gravity = 9.8

# Variable with range (clamped on modification)
iterations: range(1..10) = 8

# Variable with wrapping range
color_mode: range(0..3, wrap: true) = 0

# Boolean
paused = false
```

### Intrinsic Globals

Runtime-provided variables, always available, modifiable by both the controller and the PDP:

- `width`, `height` — canvas dimensions
- `center_x`, `center_y`, `zoom` — viewport state
- `mouse_x`, `mouse_y` — mouse position
- `time` — elapsed seconds
- `paused` — whether `frame` auto-increments
- `frame` — current frame number, controller-maintained; auto-increments each display refresh when not paused; PDP can modify (e.g., `frame += 1` for single-step)

These exist without declaration. The config can set initial values and modify them via key bindings.

### Title

```
title = "Smoke Simulation"
```

Top-level assignment. Default: kernel filename without extension. Overridable via `.pds` or `--set`.

### Settings Block

Execution concerns not visible to kernels:

```
settings {
  threads = 4
  backend = "cranelift"
  tile_height = 8
}
```

### Key Bindings and Event Handlers

#### Key bindings (top-level)

```
on key(space) paused = !paused
on key(period) frame += 1
on key(bracket_right) iterations += 1
on key(bracket_left) iterations -= 1
on key(up) gravity += 0.1
on key(down) gravity -= 0.1
on key(left) center_x -= 0.1
on key(right) center_x += 0.1
on key(plus) zoom *= 1.1
on key(minus) zoom /= 1.1
```

Supported expression forms:
- `variable += literal` / `variable = variable + literal`
- `variable -= literal` / `variable = variable - literal`
- `variable *= literal` / `variable = variable * literal`
- `variable /= literal` / `variable = variable / literal`
- `variable = !variable` (boolean toggle)

Variables with `range()` are clamped (or wrapped) automatically.

#### Event handlers (inside pipeline)

```
on click(continuous: true) {
  state = run inject(value: 1.0, radius: 3)
  age = run inject(value: 0.0, radius: 3)
}
```

Event handler body contains regular pipeline operations. Position in pipeline determines when injection happens relative to other steps. `inject` is a built-in convenience kernel; users can write custom injection kernels.

Future event types: scroll, drag, mouse buttons — noted for later.

### Pipeline

The central execution concept. Describes what happens each frame.

#### Pipeline primitives

- `run` — execute a kernel
- `display` — execute a kernel and send pixel output to screen
- `swap` — swap two buffers (pointer swap, O(1))
- `loop` — repeat a block N times within a frame
- `accumulate` — progressive accumulation across frames
- `on <event>` — conditional block triggered by input

#### Run / Display

Tuple assignment for output bindings. Input bindings in `{ }` body:

```
# No inputs, no outputs
run kernel_name

# Parameters, no buffer I/O
buf = run inject(value: -3.0, radius: 15)

# Input bindings, output assignment
u_next, v_next = run gray_scott { u_in: u, v_in: v }

# Display — run + send pixels to screen
display mandelbrot

# Display with buffer outputs
vx0, vy0 = display project { p_in: pressure, vx_in: vx, vy_in: vy, den_in: density }
```

#### Swap

```
swap u <-> u_next
swap vx <-> vx0, vy <-> vy0, density <-> density0
```

#### Loop

```
loop(iterations: 8) {
  u_next, v_next = display gray_scott { u_in: u, v_in: v }
  swap u <-> u_next
  swap v <-> v_next
}
```

`iterations` can reference a variable for runtime control: `loop(iterations: iterations)`.

#### Accumulate

Progressive sampling across frames:

```
accumulate(samples: 256) {
  display mandelbrot
}
```

Spreads iterations across frames with averaging. Resets on viewport change.

#### Event Handlers in Pipeline

```
on click(continuous: true) {
  state = run inject(value: 1.0, radius: 3)
}
```

Position in pipeline is significant — determines execution order relative to other steps.

### Override Layering

Precedence (highest to lowest):
1. `--set key=value` — CLI one-off overrides
2. `--settings file.pds` — external settings/variable overrides
3. PDP file — `settings { }` block and variable declarations

The `.pds` file is a flat key-value file parsed by `parse_settings_body`. Can override both settings and intrinsic globals:

```
# my_machine.pds
threads = 4
backend = "cranelift"
width = 1920
height = 1080
```

---

## Complete Examples

### 1. Gradient (simplest pixel kernel)
```
pixel kernel "gradient.pd"

pipeline {
  display gradient
}
```

### 2. Mandelbrot (progressive sampling with pan/zoom)
```
pixel kernel "mandelbrot.pd"

on key(left) center_x -= 0.1
on key(right) center_x += 0.1
on key(up) center_y -= 0.1
on key(down) center_y += 0.1
on key(plus) zoom *= 1.1
on key(minus) zoom /= 1.1

pipeline {
  accumulate(samples: 256) {
    display mandelbrot
  }
}
```

### 3. Gray-Scott (simple simulation)
```
title = "Gray-Scott Reaction Diffusion"

sim kernel "gray_scott.pd"
init kernel init_u = "init/gray_scott_u.pd"
init kernel init_v = "init/gray_scott_v.pd"

buffer u = init_u()
buffer v = init_v()
buffer u_next = constant(0.0)
buffer v_next = constant(0.0)

on key(space) paused = !paused
on key(period) frame += 1

pipeline {
  on click(continuous: true) {
    v = run inject(value: 0.5, radius: 5)
  }
  loop(iterations: 8) {
    u_next, v_next = display gray_scott { u_in: u, v_in: v }
    swap u <-> u_next
    swap v <-> v_next
  }
}
```

### 4. Smoke (complex multi-kernel pipeline)
```
title = "Smoke Simulation"

sim kernel advect = "smoke/advect.pd"
sim kernel divergence = "smoke/divergence.pd"
sim kernel jacobi = "smoke/jacobi.pd"
sim kernel project = "smoke/project.pd"

buffer vx = constant(0.0)
buffer vy = constant(0.0)
buffer density = constant(0.0)
buffer vx0 = constant(0.0)
buffer vy0 = constant(0.0)
buffer density0 = constant(0.0)
buffer pressure = constant(0.0)
buffer pressure_tmp = constant(0.0)
buffer divergence = constant(0.0)

on key(space) paused = !paused
on key(period) frame += 1

pipeline {
  on click(continuous: true) {
    vy = run inject(value: -3.0, radius: 15, falloff: quadratic)
    density = run inject(value: 0.5, radius: 15, falloff: quadratic)
  }
  swap vx <-> vx0, vy <-> vy0, density <-> density0
  vx, vy, density = run advect { vx_in: vx0, vy_in: vy0, den_in: density0 }
  divergence = run divergence { vx_in: vx, vy_in: vy }
  loop(iterations: 40) {
    pressure_tmp = run jacobi { div_in: divergence, p_in: pressure }
    swap pressure <-> pressure_tmp
  }
  vx0, vy0 = display project { p_in: pressure, vx_in: vx, vy_in: vy, den_in: density }
  swap vx <-> vx0, vy <-> vy0
}
```

### 5. Game of Life
```
title = "Game of Life"

sim kernel "game_of_life.pd"
init kernel init_state = "init/random_binary.pd"

buffer state = init_state(density: 0.3, seed: 42)
buffer age = constant(0.0)
buffer state_next = constant(0.0)
buffer age_next = constant(0.0)

iterations: range(1..10) = 1

on key(space) paused = !paused
on key(period) frame += 1
on key(bracket_right) iterations += 1
on key(bracket_left) iterations -= 1

pipeline {
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

### 6. GPU Mandelbrot
```
pixel kernel "mandelbrot.wgsl"

on key(left) center_x -= 0.1
on key(right) center_x += 0.1
on key(up) center_y -= 0.1
on key(down) center_y += 0.1
on key(plus) zoom *= 1.1
on key(minus) zoom /= 1.1

pipeline {
  accumulate(samples: 256) {
    display mandelbrot
  }
}
```

---

## What This Eliminates

- The 13-variant `Backend` enum in main.rs
- All per-example setup code (buffer allocation, seeding, kernel loading)
- All per-example render loop branches (substep loops, swap patterns, pipeline orchestration)
- All per-example mouse injection and keyboard handling code
- The `--sim` CLI argument and its match arms
- Most CLI arguments (replaced by settings block + .pds files + --set)
- ~1200 lines of per-example code in main.rs

## What Remains in main.rs

- PDP parser invocation and config loading
- Generic `App` struct with one `ExecutionPlan` field
- One `RedrawRequested` handler dispatching through the plan
- Controller logic: auto-increment `frame` when not paused, run pipeline when frame advances
- `compile_cpu_kernel()` and `compile_sim_kernel()` (unchanged)
- CLI: config file path, `--settings file.pds`, `--set key=value`, `--bench`, `--output`

## Open Items (Future Work)

- Additional event types: scroll, drag, mouse buttons
- Buffer element types beyond f64
- External native backend plugin loading (.so/.dylib/.dll)
- UI for runtime parameter adjustment
- `@root/` vs `@project/` path prefix naming
