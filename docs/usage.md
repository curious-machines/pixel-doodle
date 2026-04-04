# Usage Guide

pixel-doodle renders pixel kernels and runs simulations in a window. All examples are driven by `.pdc` scripts that declaratively describe what kernels to load, how to initialize buffers, and what to execute each frame.

## Quick Start

```bash
# Simple pixel kernel
cargo run --release -- examples/basic/gradient

# Mandelbrot with progressive sampling
cargo run --release -- examples/basic/mandelbrot

# Gray-Scott reaction-diffusion simulation
cargo run --release -- examples/sim/gray_scott

# Smoke simulation
cargo run --release -- examples/sim/smoke

# Game of Life
cargo run --release -- examples/sim/game_of_life
```

## Command-Line Options

```
pixel-doodle <config.pdc | directory> [options]
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
| `basic/gradient` | Color gradient from coordinates |
| `basic/mandelbrot` | Mandelbrot fractal with progressive AA |
| `basic/circle` | Circle distance field |
| `basic/checkerboard` | Repeating checkerboard pattern |
| `basic/solid_red` | Constant red (simplest possible kernel) |

### SDF (Signed Distance Fields)

Geometric shapes rendered via distance functions.

| Example | Description |
|---------|-------------|
| `sdf/sdf` | Circle + box union with glow |
| `sdf/sdf_flower` | Six-petal flower with smooth blending |
| `sdf/sdf_rings` | Concentric rings (AA test) |
| `sdf/gears` | Three interlocking gears |
| `sdf/animated_circle` | Pulsing circle (uses time) |
| `sdf/cylinder_3d` | 3D raymarched cylinder |
| `sdf/sdf2d_demo` | 2D SDF library demo |
| `sdf/sdf3d_demo` | 3D SDF library demo |

### Lighting & Ray Marching

Progressive sampling-based illumination.

| Example | Description |
|---------|-------------|
| `lighting/starfield` | Procedural starfield |
| `lighting/ray_march_ao` | Ambient occlusion on gears |
| `lighting/light_transport_2d` | 2D path tracing with bounces |
| `lighting/cornell_2d` | Soft shadows via light sampling |

### Simulations

Stateful simulations with buffer-based state, mouse interaction, and pause/step controls.

| Example | Description |
|---------|-------------|
| `sim/gray_scott` | Reaction-diffusion (click to inject) |
| `sim/shallow_water` | Wave propagation (click for waves) |
| `sim/smoke` | Eulerian smoke with 4-kernel pipeline |
| `sim/game_of_life` | Conway's Game of Life with age coloring |

## Controls

Controls are defined per-example in the `.pdc` file via event handlers. Common patterns:

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

Settings can be overridden by a `.pds` file or set via CLI:

```bash
# Use a machine-specific settings file
cargo run --release -- examples/sim/smoke --settings my_machine.pds

# Override individual settings
cargo run --release -- examples/sim/smoke --set threads=8
```

Example `.pds` file:
```
threads = 8
render = "gpu"
codegen = "cranelift"
```

Override precedence (highest to lowest):
1. `--set key=value`
2. `--settings file.pds`
3. PDC script defaults

## Benchmarking

```bash
# Benchmark a pixel kernel (100 frames)
cargo run --release -- examples/basic/mandelbrot --bench

# Custom frame count
cargo run --release -- examples/basic/mandelbrot --bench --bench-frames 500

# Save a frame as PPM
cargo run --release -- examples/basic/gradient --output frame.ppm

# Benchmark with specific thread count
cargo run --release -- examples/sim/smoke --bench --threads 4
```

## Building

```bash
# Default (Cranelift-based CPU backend for WGSL)
cargo build --release

# With LLVM support (requires LLVM 20 dev libraries)
cargo build --release --features llvm-backend
```

### LLVM Setup

Ubuntu/Debian: `sudo apt-get install llvm-20-dev libpolly-20-dev`
macOS: `brew install llvm@20`

If `llvm-config-20` is not on PATH:
```bash
LLVM_SYS_201_PREFIX=/usr/lib/llvm-20 cargo build --release --features llvm-backend
```

To use the LLVM-based CPU backend:
```bash
--set render=cpu,codegen=llvm
```
