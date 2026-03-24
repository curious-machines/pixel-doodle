# Usage Guide

pixel-doodle renders pixel kernels and runs simulations in a window. All examples are driven by `.pdc` configuration files that declaratively describe what kernels to load, how to initialize buffers, and what pipeline to execute each frame.

See [pdc.md](pdc.md) for the complete PDC language reference.

## Quick Start

```bash
# Simple pixel kernel
cargo run --release -- examples/basic/gradient/gradient.pdc

# Mandelbrot with progressive sampling
cargo run --release -- examples/basic/mandelbrot/mandelbrot.pdc

# Gray-Scott reaction-diffusion simulation
cargo run --release -- examples/sim/gray_scott/gray_scott.pdc

# Smoke simulation
cargo run --release -- examples/sim/smoke/smoke.pdc

# Game of Life
cargo run --release -- examples/sim/game_of_life/game_of_life.pdc
```

## Command-Line Options

```
pixel-doodle <config.pdc> [options]
```

| Flag | Value | Description |
|------|-------|-------------|
| `--output` | path | Render one frame and save as `.ppm` (no window) |
| `--bench` | — | Headless benchmark mode |
| `--bench-frames` | integer | Frames to time in bench mode (default: 100) |
| `--threads` | integer | CPU thread count |
| `--settings` | path | Override settings from a `.pds` file |
| `--set` | key=value | Override a single setting or variable |
| `--help` | — | Show usage |

## Examples by Category

### Basic Pixel Kernels

Simple per-pixel computation with pan/zoom controls.

| Example | Description |
|---------|-------------|
| `basic/gradient/gradient.pdc` | Color gradient from coordinates |
| `basic/mandelbrot/mandelbrot.pdc` | Mandelbrot fractal with progressive AA |
| `basic/circle/circle.pdc` | Circle distance field |
| `basic/checkerboard/checkerboard.pdc` | Repeating checkerboard pattern |
| `basic/solid_red/solid_red.pdc` | Constant red (simplest possible kernel) |

### SDF (Signed Distance Fields)

Geometric shapes rendered via distance functions.

| Example | Description |
|---------|-------------|
| `sdf/sdf/sdf.pdc` | Circle + box union with glow |
| `sdf/sdf_flower/sdf_flower.pdc` | Six-petal flower with smooth blending |
| `sdf/sdf_rings/sdf_rings.pdc` | Concentric rings (AA test) |
| `sdf/gears/gears.pdc` | Three interlocking gears |
| `sdf/animated_circle/animated_circle.pdc` | Pulsing circle (uses time) |
| `sdf/cylinder_3d/cylinder_3d.pdc` | 3D raymarched cylinder |
| `sdf/sdf2d_demo/sdf2d_demo.pdc` | 2D SDF library demo |
| `sdf/sdf3d_demo/sdf3d_demo.pdc` | 3D SDF library demo |

### Lighting & Ray Marching

Progressive sampling-based illumination.

| Example | Description |
|---------|-------------|
| `lighting/starfield/starfield.pdc` | Procedural starfield |
| `lighting/ray_march_ao/ray_march_ao.pdc` | Ambient occlusion on gears |
| `lighting/light_transport_2d/light_transport_2d.pdc` | 2D path tracing with bounces |
| `lighting/cornell_2d/cornell_2d.pdc` | Soft shadows via light sampling |

### Simulations

Stateful simulations with buffer-based state, mouse interaction, and pause/step controls.

| Example | Description |
|---------|-------------|
| `sim/gray_scott/gray_scott.pdc` | Reaction-diffusion (click to inject) |
| `sim/shallow_water/shallow_water.pdc` | Wave propagation (click for waves) |
| `sim/smoke/smoke.pdc` | Eulerian smoke with 4-kernel pipeline |
| `sim/game_of_life/game_of_life.pdc` | Conway's Game of Life with age coloring |

## Controls

Controls are defined per-example in the `.pdc` file via `on key(...)` bindings. Common patterns:

### Pixel Kernels

| Key | Action |
|-----|--------|
| Arrow keys | Pan |
| `+` / `-` | Zoom |
| Escape / Q | Quit |

### Simulations

| Key | Action |
|-----|--------|
| Space | Pause / resume |
| `.` | Single step (advance one frame) |
| `[` / `]` | Decrease / increase speed (Game of Life) |
| Click / drag | Inject (chemical, waves, smoke, or cells) |
| Escape / Q | Quit |

## Settings

Execution settings can be specified in the `.pdc` file, overridden by a `.pds` file, or set via CLI:

```bash
# Use a machine-specific settings file
cargo run --release -- examples/sim/smoke/smoke.pdc --settings my_machine.pds

# Override individual settings
cargo run --release -- examples/sim/smoke/smoke.pdc --set threads=8 --set tile_height=4
```

Example `.pds` file:
```
threads = 8
backend = "cranelift"
tile_height = 4
```

Override precedence (highest to lowest):
1. `--set key=value`
2. `--settings file.pds`
3. `.pdc` file `settings { }` block

## Benchmarking

```bash
# Benchmark a pixel kernel (100 frames)
cargo run --release -- examples/basic/mandelbrot/mandelbrot.pdc --bench

# Custom frame count
cargo run --release -- examples/basic/mandelbrot/mandelbrot.pdc --bench --bench-frames 500

# Save a frame as PPM
cargo run --release -- examples/basic/gradient/gradient.pdc --output frame.ppm

# Benchmark with specific thread count
cargo run --release -- examples/sim/smoke/smoke.pdc --bench --threads 4
```

## Building

```bash
# Default (Cranelift JIT backend)
cargo build --release

# With LLVM support (requires LLVM 20 dev libraries)
cargo build --release --features llvm-backend

# Without JIT (won't be able to run .pdc files)
cargo build --release --no-default-features
```

### LLVM Setup

Ubuntu/Debian: `sudo apt-get install llvm-20-dev libpolly-20-dev`
macOS: `brew install llvm@20`

If `llvm-config-20` is not on PATH:
```bash
LLVM_SYS_201_PREFIX=/usr/lib/llvm-20 cargo build --release --features llvm-backend
```

To use the LLVM backend, set it in the `.pdc` settings block:
```
settings {
  backend = "llvm"
}
```
Or via CLI: `--set backend=llvm`
