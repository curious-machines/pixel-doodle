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
- **Init strategy**: Use WGSL kernels for all buffer initialization (init kernel returns f64 per cell); `constant()` as built-in convenience
- **Pipeline**: Ordered step list (explicit, declarative); encompasses loops, accumulation, and event handlers
- **Native backends**: Defer external plugin loading to later; keep in-tree for now
- **Paths**: Relative to config file by default; `@root/` prefix for project root
- **Format**: Custom DSL (not TOML), purpose-built for this project; hand-written recursive descent parser
- **Runtime configurability**: Settings like thread count, backend, tile height configurable at runtime
- **Minimize CLI args**: Only `--settings file.pds` and `--set key=value` beyond the config file path

---

## PDP Language Specification

### Syntax Overview

Line-oriented, `#` comments, blocks with `{ }`. Minimal punctuation. Parseable with simple recursive descent.

### Kernel Declarations

Three kernel types, explicitly declared:

```
pixel kernel "gradient.wgsl"
sim kernel "gray_scott.wgsl"
init kernel init_random = "init/random.wgsl"
```

- Type is required (`pixel`, `sim`, `init`) — no auto-detection
- Runtime validates kernel signature matches declared type
- All kernels use WGSL (`.wgsl` files)
- Unnamed kernels get a default name from the file's base name (must be valid identifier)
- Named kernels use `name = "path"` syntax

#### Named multi-kernel declarations
```
sim kernel advect = "smoke/advect.wgsl"
sim kernel divergence = "smoke/divergence.wgsl"
sim kernel jacobi = "smoke/jacobi.wgsl"
sim kernel project = "smoke/project.wgsl"
```

#### Path resolution
```
pixel kernel "gradient.wgsl"                    # relative to config file
pixel kernel "@root/examples/shared/util.wgsl"  # relative to project root
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

### Builtin Declarations

Runtime-provided variables must be explicitly declared before use:

```
# Read-only builtins
builtin const width: u32
builtin const height: u32
builtin const time: f64
builtin const mouse_x: f64
builtin const mouse_y: f64

# Mutable builtins (modifiable by key handlers)
builtin var center_x: f64
builtin var center_y: f64
builtin var zoom: f64
builtin var paused: bool
builtin var frame: u64
```

Using an undeclared builtin is a hard error. Types must match the intrinsic registry. Declaring a const intrinsic as `var` is an error; declaring a var intrinsic as `const` is allowed (opts into less power). Duplicate declarations across includes are allowed if types match; var/const can differ (most permissive wins).

A shared `builtins.pdp` file declares all builtins — include it to get everything at once.

### Variables

Mutable or immutable values declared with `var` or `const`:

```
# Mutable variable (no clamping)
var gravity = 9.8

# Mutable variable with range (clamped on modification)
var iterations: range(1..10) = 8

# Mutable variable with wrapping range
var color_mode: range(0..3, wrap: true) = 0

# Immutable constant
const max_detail = 256
```

Assigning to a `const` in a key handler is a hard error.

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

### Event Handlers

Six event types, split between top-level config and pipeline steps:

#### Key events (top-level, action syntax)

- **`on keydown(name)`** — fires every frame while key is held (continuous)
- **`on keypress(name)`** — fires once on initial key press (single-fire)
- **`on keyup(name)`** — fires once on key release

```
# Continuous (fires every frame while held) — pan/zoom
on keydown(left) center_x -= 0.1
on keydown(right) center_x += 0.1
on keydown(up) gravity += 0.1
on keydown(down) gravity -= 0.1
on keydown(plus) zoom *= 1.1
on keydown(minus) zoom /= 1.1

# Single-fire (fires once on press) — toggles, discrete actions
on keypress(space) paused = !paused
on keypress(period) frame += 1
on keypress(bracket_right) iterations += 1
on keypress(bracket_left) iterations -= 1
```

Supported expression forms:
- `variable += literal` / `variable = variable + literal`
- `variable -= literal` / `variable = variable - literal`
- `variable *= literal` / `variable = variable * literal`
- `variable /= literal` / `variable = variable / literal`
- `variable = !variable` (boolean toggle)

Variables with `range()` are clamped (or wrapped) automatically.

#### Mouse events (inside pipeline, pipeline step syntax)

- **`on mousedown { steps }`** — fires every frame while mouse button is held (continuous)
- **`on click { steps }`** — fires once on initial mouse press (single-fire)
- **`on mouseup { steps }`** — fires once on mouse button release

```
on mousedown {
  run inject(value: 1.0, radius: 3) with(target: out state)
  run inject(value: 0.0, radius: 3) with(target: out age)
}
```

Event handler body contains regular pipeline operations. Position in pipeline determines when injection happens relative to other steps. `inject` is a built-in convenience kernel; users can write custom injection kernels.

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
on mousedown {
  run inject(value: 1.0, radius: 3) with(target: out state)
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

> **Note:** These examples reflect the original design vision. The implementation evolved:
> kernel and buffer declarations moved inside pipeline blocks, `init_kernel()` buffer
> initializers became `constant()`, and `sim kernel` became plain `kernel`. See the
> actual `.pdp` files in `examples/` for current syntax.

### 1. Gradient (simplest pixel kernel)
```
pipeline {
  pixel kernel "gradient.wgsl"
  run gradient
  display
}
```

### 2. Mandelbrot (progressive sampling with pan/zoom)
```
include "shared/builtins.pdp"
include "shared/pan_zoom.pdp"

pipeline {
  pixel kernel "mandelbrot.wgsl"
  accumulate(samples: 256) {
    run mandelbrot
    display
  }
}
```

### 3. Game of Life (simulation with variables)
```
title = "Game of Life"

builtin var paused: bool
builtin var frame: u64
builtin const mouse_x: f64
builtin const mouse_y: f64

var iterations: range(1..10) = 1

on keypress(space) paused = !paused
on keypress(period) frame += 1
on keypress(bracket_right) iterations += 1
on keypress(bracket_left) iterations -= 1

pipeline {
  kernel "game_of_life.wgsl"
  kernel init_state = "init/random_binary.wgsl"
  kernel inject = "shared/inject.wgsl"

  buffer state = constant(0.0)
  buffer age = constant(0.0)
  buffer state_next = constant(0.0)
  buffer age_next = constant(0.0)

  init {
    run init_state { out: out state }
  }
  on mousedown {
    run inject(inject_x: mouse_x, inject_y: mouse_y, radius: 3.0, value: 1.0) { target: state, target_out: out state }
    run inject(inject_x: mouse_x, inject_y: mouse_y, radius: 3.0, value: 0.0) { target: age, target_out: out age }
  }
  loop(iterations: iterations) {
    run game_of_life { state_in: state, age_in: age, state_out: out state_next, age_out: out age_next }
    display
    swap state <-> state_next
    swap age <-> age_next
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

- Additional event types: scroll, drag
- Buffer element types beyond f64
- External native backend plugin loading (.so/.dylib/.dll)
- UI for runtime parameter adjustment
- `@root/` vs `@project/` path prefix naming
