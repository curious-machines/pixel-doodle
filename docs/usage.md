# Usage Guide

pixel-doodle renders pixel kernels and runs fluid simulations in a 1200x900 window. There are two main modes: **kernel rendering** (the default) and **simulation mode**.

## Command-Line Options

| Flag | Value | Description |
|------|-------|-------------|
| `--backend` | `native`, `cranelift`, `llvm`, `gpu` | Execution backend (default: `native`) |
| `--kernel` | path | Kernel source file (`.pd`, `.pdl`, or `.wgsl`) |
| `--sim` | `gray-scott`, `shallow-water`, `smoke`, `game-of-life` | Run a simulation |
| `--density` | float (0.0–1.0) | Initial cell density for game-of-life (default: 0.3) |
| `--samples` | integer | Progressive sampling passes (kernel mode only) |
| `--threads` | integer | CPU thread count (CPU backends only) |
| `--tile-height` | integer | Rows per rayon tile (default: 1) |
| `--dump-ir` | — | Print lowered IR to stderr and exit |
| `--bench` | — | Headless benchmark mode |
| `--bench-frames` | integer | Frames to time in bench mode (default: 100) |
| `--output` | path | Write a single frame to `.ppm` file |

## Kernel Rendering

Renders a per-pixel kernel function. The kernel receives normalized coordinates and produces an ARGB pixel. Without `--kernel`, a built-in Mandelbrot is used.

```bash
# Built-in Mandelbrot (native CPU)
cargo run --release

# Built-in Mandelbrot (GPU compute shader)
cargo run --release -- --backend gpu

# Custom PD kernel
cargo run --release -- --kernel examples/basic/mandelbrot.pd

# Custom PDL kernel (SSA format)
cargo run --release -- --backend cranelift --kernel examples/basic/mandelbrot.pd

# Custom WGSL shader (GPU only)
cargo run --release -- --backend gpu --kernel examples/basic/mandelbrot.wgsl

# LLVM backend (requires --features llvm-backend)
cargo run --release --features llvm-backend -- --backend llvm --kernel examples/basic/mandelbrot.pd
```

### Progressive Sampling

For stochastic kernels (those using a `time` parameter), `--samples N` accumulates N jittered samples per pixel for antialiasing:

```bash
cargo run --release -- --backend gpu --samples 256
```

### Controls (kernel mode)

| Key | Action |
|-----|--------|
| Arrow keys | Pan |
| `+` / `-` | Zoom |
| Escape | Quit |

## Simulations

Simulations run continuously and are controlled via mouse interaction.

### Gray-Scott Reaction-Diffusion

Two chemicals diffuse and react, producing organic patterns (mazes, worms, cell division). Click/drag to inject chemical V.

```bash
# Native CPU
cargo run --release -- --sim gray-scott

# GPU compute shader
cargo run --release -- --sim gray-scott --backend gpu

# Cranelift JIT with PD kernel
cargo run --release -- --sim gray-scott --backend cranelift --kernel examples/sim/gray_scott.pd

# Cranelift JIT with PDL kernel (SSA format)
cargo run --release -- --sim gray-scott --backend cranelift --kernel examples/sim/gray_scott.pdl

# LLVM JIT
cargo run --release --features llvm-backend -- --sim gray-scott --backend llvm --kernel examples/sim/gray_scott.pd
```

### Shallow Water Waves

Height-field wave propagation using Lax-Friedrichs scheme. Click/drag to create wave bumps.

```bash
# Native CPU
cargo run --release -- --sim shallow-water

# GPU compute shader
cargo run --release -- --sim shallow-water --backend gpu

# Cranelift JIT
cargo run --release -- --sim shallow-water --backend cranelift --kernel examples/sim/shallow_water.pd

# LLVM JIT
cargo run --release --features llvm-backend -- --sim shallow-water --backend llvm --kernel examples/sim/shallow_water.pd
```

### Smoke (Stable Fluids)

Eulerian smoke simulation with semi-Lagrangian advection and Jacobi pressure projection. Click/drag to inject smoke (white on black). Smoke rises via buoyancy and fades via dissipation.

```bash
# Native CPU
cargo run --release -- --sim smoke --backend native

# GPU compute shader
cargo run --release -- --sim smoke --backend gpu

# Cranelift JIT (uses embedded PD kernels — no --kernel needed)
cargo run --release -- --sim smoke --backend cranelift

# LLVM JIT
cargo run --release --features llvm-backend -- --sim smoke --backend llvm
```

### Game of Life

Conway's Game of Life (B3/S23) with age-based coloring. Alive cells shift green to cyan to blue over time; dead cells fade red to black. Click/drag to draw live cells.

```bash
# GPU compute shader (supports zoom/pan)
cargo run --release -- --sim game-of-life --backend gpu

# Cranelift JIT (uses embedded PD kernel — no --kernel needed)
cargo run --release -- --sim game-of-life --backend cranelift

# LLVM JIT
cargo run --release --features llvm-backend -- --sim game-of-life --backend llvm

# Custom initial density
cargo run --release -- --sim game-of-life --backend gpu --density 0.5
```

### Simulation Controls

| Input | Action |
|-------|--------|
| Click / drag | Inject (chemical, wave bump, smoke, or live cells depending on sim) |
| Space | Pause / resume simulation |
| `.` | Single step (when paused) |
| `]` | Increase speed (up to 10x generations per frame) |
| `[` | Decrease speed |
| Arrow keys | Pan (game-of-life GPU only) |
| `+` / `-` | Zoom (game-of-life GPU only) |
| `0` | Reset pan/zoom |
| Escape | Quit |

### Simulation Backends Summary

| Simulation | native | gpu | cranelift | llvm |
|------------|--------|-----|-----------|------|
| gray-scott | built-in | built-in | `--kernel .pd/.pdl` | `--kernel .pd/.pdl` |
| shallow-water | built-in | built-in | `--kernel .pd/.pdl` | `--kernel .pd/.pdl` |
| smoke | built-in | built-in | embedded PD kernels | embedded PD kernels |
| game-of-life | — | built-in | embedded PD kernel | embedded PD kernel |

## Benchmarking

Run headless timing without opening a window:

```bash
# CPU benchmark (100 frames)
cargo run --release -- --backend cranelift --bench

# GPU benchmark
cargo run --release -- --backend gpu --bench

# Custom frame count
cargo run --release -- --backend cranelift --bench --bench-frames 500

# Save a frame as PPM image
cargo run --release -- --backend gpu --output frame.ppm
```

## Inspecting IR

Dump the lowered SSA IR for any kernel to stderr:

```bash
# From PD source
cargo run --release -- --kernel examples/sim/gray_scott.pd --dump-ir

# From PDL source
cargo run --release -- --kernel examples/sim/gray_scott.pdl --dump-ir
```

## Building

```bash
# Default (native + cranelift backends)
cargo build --release

# With LLVM support (requires LLVM 20 dev libraries)
cargo build --release --features llvm-backend

# Native only (no JIT)
cargo build --release --no-default-features
```

### LLVM Setup

Ubuntu/Debian: `sudo apt-get install llvm-20-dev libpolly-20-dev`
macOS: `brew install llvm@20`

If `llvm-config-20` is not on PATH:
```bash
LLVM_SYS_201_PREFIX=/usr/lib/llvm-20 cargo build --release --features llvm-backend
```
