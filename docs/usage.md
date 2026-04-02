# Usage Guide

pixel-doodle renders pixel kernels and runs simulations in a window. All examples are driven by `.pdp` configuration files that declaratively describe what kernels to load, how to initialize buffers, and what pipeline to execute each frame.

See [pdp.md](pdp.md) for the complete PDP language reference.

## Quick Start

```bash
# Simple pixel kernel
cargo run --release -- examples/basic/gradient/gradient.pdp

# Mandelbrot with progressive sampling
cargo run --release -- examples/basic/mandelbrot/mandelbrot.pdp

# Gray-Scott reaction-diffusion simulation
cargo run --release -- examples/sim/gray_scott/gray_scott.pdp

# Smoke simulation
cargo run --release -- examples/sim/smoke/smoke.pdp

# Game of Life
cargo run --release -- examples/sim/game_of_life/game_of_life.pdp
```

## Command-Line Options

```
pixel-doodle <config.pdp> [options]
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
| `basic/gradient/gradient.pdp` | Color gradient from coordinates |
| `basic/mandelbrot/mandelbrot.pdp` | Mandelbrot fractal with progressive AA |
| `basic/circle/circle.pdp` | Circle distance field |
| `basic/checkerboard/checkerboard.pdp` | Repeating checkerboard pattern |
| `basic/solid_red/solid_red.pdp` | Constant red (simplest possible kernel) |

### SDF (Signed Distance Fields)

Geometric shapes rendered via distance functions.

| Example | Description |
|---------|-------------|
| `sdf/sdf/sdf.pdp` | Circle + box union with glow |
| `sdf/sdf_flower/sdf_flower.pdp` | Six-petal flower with smooth blending |
| `sdf/sdf_rings/sdf_rings.pdp` | Concentric rings (AA test) |
| `sdf/gears/gears.pdp` | Three interlocking gears |
| `sdf/animated_circle/animated_circle.pdp` | Pulsing circle (uses time) |
| `sdf/cylinder_3d/cylinder_3d.pdp` | 3D raymarched cylinder |
| `sdf/sdf2d_demo/sdf2d_demo.pdp` | 2D SDF library demo |
| `sdf/sdf3d_demo/sdf3d_demo.pdp` | 3D SDF library demo |

### Lighting & Ray Marching

Progressive sampling-based illumination.

| Example | Description |
|---------|-------------|
| `lighting/starfield/starfield.pdp` | Procedural starfield |
| `lighting/ray_march_ao/ray_march_ao.pdp` | Ambient occlusion on gears |
| `lighting/light_transport_2d/light_transport_2d.pdp` | 2D path tracing with bounces |
| `lighting/cornell_2d/cornell_2d.pdp` | Soft shadows via light sampling |

### Simulations

Stateful simulations with buffer-based state, mouse interaction, and pause/step controls.

| Example | Description |
|---------|-------------|
| `sim/gray_scott/gray_scott.pdp` | Reaction-diffusion (click to inject) |
| `sim/shallow_water/shallow_water.pdp` | Wave propagation (click for waves) |
| `sim/smoke/smoke.pdp` | Eulerian smoke with 4-kernel pipeline |
| `sim/game_of_life/game_of_life.pdp` | Conway's Game of Life with age coloring |

## Controls

Controls are defined per-example in the `.pdp` file via `on key(...)` bindings. Common patterns:

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

Execution settings can be specified in the `.pdp` file, overridden by a `.pds` file, or set via CLI:

```bash
# Use a machine-specific settings file
cargo run --release -- examples/sim/smoke/smoke.pdp --settings my_machine.pds

# Override individual settings
cargo run --release -- examples/sim/smoke/smoke.pdp --set threads=8 --set tile_height=4
```

Example `.pds` file:
```
threads = 8
render = "gpu"
codegen = "cranelift"
tile_height = 4
```

Override precedence (highest to lowest):
1. `--set key=value`
2. `--settings file.pds`
3. `.pdp` file `settings { }` block

## Benchmarking

```bash
# Benchmark a pixel kernel (100 frames)
cargo run --release -- examples/basic/mandelbrot/mandelbrot.pdp --bench

# Custom frame count
cargo run --release -- examples/basic/mandelbrot/mandelbrot.pdp --bench --bench-frames 500

# Save a frame as PPM
cargo run --release -- examples/basic/gradient/gradient.pdp --output frame.ppm

# Benchmark with specific thread count
cargo run --release -- examples/sim/smoke/smoke.pdp --bench --threads 4
```

## Building

```bash
# Default (Cranelift-based CPU backend for WGSL)
cargo build --release

# With LLVM support (requires LLVM 20 dev libraries)
cargo build --release --features llvm-backend

# Without JIT (won't be able to run .pdp files)
cargo build --release --no-default-features
```

### LLVM Setup

Ubuntu/Debian: `sudo apt-get install llvm-20-dev libpolly-20-dev`
macOS: `brew install llvm@20`

If `llvm-config-20` is not on PATH:
```bash
LLVM_SYS_201_PREFIX=/usr/lib/llvm-20 cargo build --release --features llvm-backend
```

To use the LLVM-based CPU backend:
```
settings {
  render = "cpu"
  codegen = "llvm"
}
```
Or via CLI: `--set render=cpu,codegen=llvm`
