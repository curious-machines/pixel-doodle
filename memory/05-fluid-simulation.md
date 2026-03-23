# Fluid Simulations

Three grid-based simulations share the same infrastructure (double-buffered fields, rayon tile parallelism, GPU ping-pong compute, mouse injection).

---

# Gray-Scott Reaction-Diffusion

## Overview

Gray-Scott reaction-diffusion simulation implemented across four backends for comparison. The simulation models two chemicals (U and V) that diffuse and react, producing organic patterns (mazes, worms, cell division).

## Algorithm

Per-cell update each timestep:

```
laplacian_u = u[x-1,y] + u[x+1,y] + u[x,y-1] + u[x,y+1] - 4*u[x,y]
laplacian_v = (same for v)
uvv = u * v * v
u_new = u + dt * (Du * laplacian_u - uvv + F * (1 - u))
v_new = v + dt * (Dv * laplacian_v + uvv - (F + k) * v)
```

Parameters: F=0.037, k=0.06, Du=0.21, Dv=0.105, dt=1.0. Toroidal (wrapping) boundary conditions. 8 substeps per frame. 20 scattered seed spots initialized with a deterministic LCG.

## Running

```bash
# Native CPU (rayon parallelism)
cargo run --release -- --sim gray-scott

# GPU compute shader (wgpu/WGSL)
cargo run --release -- --sim gray-scott --backend gpu

# Cranelift JIT (PDL kernel)
cargo run --release -- --sim gray-scott --backend cranelift --kernel examples/sim/gray_scott.pdl

# LLVM JIT (PDL kernel, requires llvm-backend feature)
cargo run --release --features llvm-backend -- --sim gray-scott --backend llvm --kernel examples/sim/gray_scott.pdl
```

Mouse click/drag injects chemical V at the cursor position (native and GPU backends).

## Architecture

### Shared infrastructure

- `--sim <name>` CLI flag selects simulation mode (separate path from kernel rendering)
- `--backend` selects which implementation to use
- Simulation mode always uses `ControlFlow::Poll` (continuous animation)

### Backend 1: Native Rust (`src/simulation.rs`)

- `FluidState` struct with double-buffered `Vec<f32>` fields (u/v and u_next/v_next)
- `step()` runs N substeps, each using `rayon::par_chunks_mut` on output rows
- `to_pixels()` maps V concentration to a color ramp (black -> blue -> white -> orange)
- `inject()` writes chemical V at a position for mouse interaction

### Backend 2: GPU compute shader

- `src/gpu/gray_scott.wgsl` — WGSL compute shader with two entry points:
  - `step`: reads 5-point stencil from `field_in`, writes to `field_out`
  - `visualize`: converts (u,v) to ARGB pixels
- `src/gpu/fluid.rs` — `GpuFluidBackend` manages ping-pong `vec2<f32>` storage buffers
- Multiple substeps = multiple compute dispatches in a single command encoder
- Pixel output buffer uses stride alignment for `copy_buffer_to_texture`
- `inject()` writes directly to the current field buffer via `queue.write_buffer`

### Backend 3: PDL/JIT (`examples/sim/gray_scott.pdl`)

New IR and language features added to support buffer operations:

**IR extensions** (`src/kernel_ir.rs`):
- `BufDecl { name, is_output }` — buffer declaration
- `Kernel::buffers: Vec<BufDecl>` — buffer list on kernel struct
- `Inst::BufLoad { buf, x, y }` — load f64 from buffer at (x,y)
- `Inst::BufStore { buf, x, y, val }` — store f64 to buffer at (x,y)

**Parser extensions** (`src/lang/parser.rs`):
- `buffers(name: read, name: write, ...)` declaration after return type
- `buf_load <buffer> <x> <y>` instruction (produces f64)
- `buf_store <buffer> <x> <y> <val>` instruction (produces dummy u32)

**New JIT ABI** (`src/jit/mod.rs`):
```rust
pub type SimTileKernelFn = unsafe extern "C" fn(
    output: *mut u32,
    width: u32, height: u32,
    row_start: u32, row_end: u32,
    buf_ptrs: *const *const f64,
    buf_out_ptrs: *const *mut f64,
);
```

**Cranelift backend** (`src/jit/cranelift.rs`):
- `compile_sim_kernel()` generates tile loop with integer coordinates and buffer pointer params
- `BufContext` struct threads width/buffer pointers through `lower_inst`
- `BufLoad` emits: load buffer pointer from array, compute flat index, load f64
- `BufStore` emits: same address math, store f64, return dummy 0

**LLVM backend** (`src/jit/llvm.rs`):
- `compile_sim_kernel()` and `build_sim_tile_loop()` mirror the Cranelift approach
- `LlvmBufContext` threads params through lowering functions
- Uses `build_gep` for pointer arithmetic

**Render path** (`src/render.rs`):
- `render_sim()` — rayon tile parallelism for sim kernels
- Takes raw buffer pointer slices, stashes as `usize` for `Send+Sync`

**PDL kernel** (`examples/sim/gray_scott.pdl`):
- Declares `buffers(u_in: read, v_in: read, u_out: write, v_out: write)`
- Kernel params: `px: u32, py: u32, width: u32, height: u32`
- Computes wrapping neighbor coordinates, 5-point Laplacian, Gray-Scott update
- Stores results via `buf_store`, visualizes via color ramp and `pack_argb`
- Uses f64 buffers (promoted from f32 in native; doubles memory but keeps IR simple)

### Main integration (`src/main.rs`)

Three `Backend` variants for simulation:
- `Backend::Simulation` — native CPU
- `Backend::GpuSimulation` — GPU (initialized in `resumed()`)
- `Backend::JitSimulation` — Cranelift/LLVM with double-buffered `Vec<f64>` fields

## Key files

| File | Role |
|------|------|
| `src/simulation.rs` | Native CPU fluid state and step function |
| `src/gpu/fluid.rs` | GPU fluid backend (buffer management, dispatch) |
| `src/gpu/gray_scott.wgsl` | WGSL compute shader (step + visualize) |
| `examples/sim/gray_scott.pdl` | Gray-Scott kernel in PDL |
| `src/kernel_ir.rs` | IR with BufLoad/BufStore instructions |
| `src/lang/parser.rs` | Parser with buffer declarations |
| `src/jit/mod.rs` | SimTileKernelFn ABI definition |
| `src/jit/cranelift.rs` | Cranelift sim kernel compilation |
| `src/jit/llvm.rs` | LLVM sim kernel compilation |
| `src/render.rs` | `render_sim()` rayon tile dispatch |
| `src/main.rs` | CLI integration, backend selection |

---

# Shallow Water Waves

## Overview

2D shallow water equations modeling wave propagation on a height field. Three fields per cell: water height (h), x-velocity (vx), y-velocity (vy). Implemented with native CPU and GPU compute backends.

## Algorithm

Uses the **Lax-Friedrichs scheme** — the center cell value is replaced with the average of its 4 neighbors before applying the update. This adds numerical diffusion that prevents the checkerboard instability inherent in central differences on hyperbolic PDEs.

Per-cell update each timestep:

```
# Lax-Friedrichs averaging (stability)
h_avg  = (h[x-1,y] + h[x+1,y] + h[x,y-1] + h[x,y+1]) / 4
vx_avg = (same for vx neighbors)
vy_avg = (same for vy neighbors)

# Central differences for pressure gradient
dh_dx = (h[x+1,y] - h[x-1,y]) / 2
dh_dy = (h[x,y+1] - h[x,y-1]) / 2

# Velocity update
vx_new = vx_avg - dt * g * dh_dx - dt * damping * vx_avg
vy_new = vy_avg - dt * g * dh_dy - dt * damping * vy_avg

# Height update (flux divergence)
flux_x = (h[x+1,y]*vx[x+1,y] - h[x-1,y]*vx[x-1,y]) / 2
flux_y = (h[x,y+1]*vy[x,y+1] - h[x,y-1]*vy[x,y-1]) / 2
h_new = h_avg - dt * (flux_x + flux_y)
```

Parameters: g=9.8, damping=0.001, dt=0.02. Toroidal boundary conditions. 4 substeps per frame. 5 Gaussian-ish bumps (radius 20) seeded with a deterministic LCG.

## Running

```bash
# Native CPU (rayon parallelism)
cargo run --release -- --sim shallow-water

# GPU compute shader (wgpu/WGSL)
cargo run --release -- --sim shallow-water --backend gpu

# Cranelift JIT (PDL kernel)
cargo run --release -- --sim shallow-water --backend cranelift --kernel examples/sim/shallow_water.pdl

# LLVM JIT (PDL kernel, requires llvm-backend feature)
cargo run --release --features llvm-backend -- --sim shallow-water --backend llvm --kernel examples/sim/shallow_water.pdl
```

Mouse click/drag injects height bumps at the cursor position (all backends).

## Architecture

### Backend 1: Native Rust (`src/simulation.rs`)

- `ShallowWaterState` struct with double-buffered `Vec<f32>` fields (h/vx/vy and h_next/vx_next/vy_next)
- `step()` runs N substeps, each using `rayon::par_chunks_mut` zipped across all three output fields
- `to_pixels()` maps height deviation from rest level (h=1.0) to a color ramp (dark blue → mid blue → light blue → white)
- `inject()` adds a smooth height bump (Gaussian-ish profile)

### Backend 2: GPU compute shader

- `src/gpu/shallow_water.wgsl` — WGSL compute shader with two entry points:
  - `step`: reads 5-point stencil from `field_in` (vec4: h, vx, vy, _), writes to `field_out`
  - `visualize`: converts height deviation to ARGB pixels
- `src/gpu/shallow_water_gpu.rs` — `GpuShallowWaterBackend` manages ping-pong `vec4<f32>` storage buffers
- Same dispatch pattern as Gray-Scott: multiple substeps + visualization in one command encoder
- `inject()` writes height bumps directly via `queue.write_buffer`

### Backend 3: PDL/JIT (`examples/sim/shallow_water.pdl`)

- PDL kernel with 6 buffers: `h_in`, `vx_in`, `vy_in` (read) + `h_out`, `vx_out`, `vy_out` (write)
- Same Lax-Friedrichs scheme as native — neighbor averaging, central differences, flux divergence
- Visualization color ramp computed inline via nested `select` instructions
- 141 SSA statements; compiles via existing `compile_sim_kernel()` (Cranelift or LLVM)
- No changes to parser, IR, or JIT backends — the existing buffer infrastructure handled 6 buffers without modification

### Main integration (`src/main.rs`)

Three `Backend` variants:
- `Backend::ShallowWater` — native CPU
- `Backend::GpuShallowWater` — GPU (initialized in `resumed()`)
- `Backend::JitShallowWater` — Cranelift/LLVM with double-buffered `Vec<f64>` fields (h/vx/vy × 2)

## Key files

| File | Role |
|------|------|
| `src/simulation.rs` | `ShallowWaterState` — CPU step, inject, visualization |
| `src/gpu/shallow_water_gpu.rs` | GPU backend (buffer management, dispatch) |
| `src/gpu/shallow_water.wgsl` | WGSL compute shader (step + visualize) |
| `examples/sim/shallow_water.pdl` | Shallow water kernel in PDL |
| `src/main.rs` | CLI integration (`--sim shallow-water`) |

---

# Smoke (Stable Fluids)

## Overview

2D Eulerian smoke simulation based on Jos Stam's "Stable Fluids" (1999). Solves incompressible Navier-Stokes via semi-Lagrangian advection and pressure projection. More complex than Gray-Scott or shallow water — requires an iterative Jacobi pressure solver (40 iterations per frame) and bilinear interpolation for advection. Implemented with native CPU and GPU compute backends.

## Algorithm

Five steps per frame:

```
1. Advect — Semi-Lagrangian: trace each cell backward through velocity field,
   bilinear interpolate all fields at the source position.
   Apply buoyancy (density pushes upward) and density dissipation.

2. Divergence — Central-difference divergence of advected velocity:
   div = (vx[x+1,y] - vx[x-1,y] + vy[x,y+1] - vy[x,y-1]) * 0.5

3. Jacobi pressure solve — 40 iterations, each:
   p_new = (p_left + p_right + p_up + p_down - divergence) / 4

4. Project — Subtract pressure gradient from velocity:
   vx -= 0.5 * (p[x+1,y] - p[x-1,y])
   vy -= 0.5 * (p[x,y+1] - p[x,y-1])

5. Visualize — Density mapped to grayscale (white smoke on black background)
```

Parameters: dt=4.0, dissipation=0.998, buoyancy=0.08. Open boundary conditions (zero velocity at edges, density exits freely). No initial perturbations — starts blank, user injects smoke via mouse.

## Running

```bash
# GPU compute shader (wgpu/WGSL)
cargo run --release -- --sim smoke --backend gpu

# Native CPU (rayon parallelism)
cargo run --release -- --sim smoke --backend native

# Cranelift JIT (multi-pass PDL kernels)
cargo run --release -- --sim smoke --backend cranelift

# LLVM JIT (multi-pass PDL kernels, requires llvm-backend feature)
cargo run --release --features llvm-backend -- --sim smoke --backend llvm
```

Mouse click/drag injects smoke density + upward velocity impulse at the cursor position (radius 15).

## Architecture

### Field layout

Three logical fields packed into `vec4<f32>` per cell: (vx, vy, density, 0). Separate scalar (`f32`) buffers for pressure (ping-pong pair) and divergence.

### Backend 1: Native Rust (`src/simulation.rs`)

- `SmokeState` struct with separate `Vec<f64>` fields: vx, vy, density (current), vx0, vy0, density0 (previous), pressure, pressure_tmp, divergence
- `step()` performs advect → divergence → 40× Jacobi → project, each parallelized with `rayon::par_chunks_mut` per row
- `sample_bilinear()` helper for semi-Lagrangian advection (clamped, not wrapping)
- `to_pixels()` maps density to grayscale ARGB
- `inject()` adds Gaussian density bump + upward velocity impulse

### Backend 2: GPU compute shader

- `src/gpu/smoke.wgsl` — WGSL compute shader with **5 entry points**:
  - `advect`: semi-Lagrangian with inlined bilinear interpolation + buoyancy + dissipation
  - `divergence`: central-difference velocity divergence
  - `jacobi`: one Jacobi pressure iteration (called 40× per frame)
  - `project`: subtract pressure gradient from velocity
  - `visualize`: density to grayscale ARGB pixels
- `src/gpu/smoke_gpu.rs` — `GpuSmokeBackend` with:
  - 5 compute pipelines, each with its own bind group layout
  - 7 GPU buffers: field_a/field_b (vec4 ping-pong), pressure_a/pressure_b (f32 ping-pong), div_buf (f32), pixel_buffer (u32 stride-aligned)
  - `step_and_render()` encodes all passes in a single command encoder: advect → divergence → 40× jacobi → project → visualize → copy to texture
  - `inject()` writes density + velocity via `queue.write_buffer` to current field buffer
  - `SmokeParams` struct with defaults (dt=4.0, dissipation=0.998, buoyancy=0.08)

### Per-frame GPU dispatch sequence

```
advect:     field_a → field_b       (flip ping)
divergence: field_b → div_buf
jacobi ×40: pressure_a/pressure_b   (ping-pong, using div_buf)
project:    field_b + pressure → field_a   (flip ping back)
visualize:  field_a → pixel_buffer
copy_buffer_to_texture
```

Total: 44 compute dispatches per frame (1 + 1 + 40 + 1 + 1).

### Backend 3: PDL/JIT (multi-pass kernel orchestration)

First simulation to use **multi-pass kernel orchestration** — 4 separate PDL kernels compiled independently, called in sequence by the host with buffer swaps between passes. This works around the single-pass `SimTileKernelFn` ABI limitation.

- `examples/sim/smoke/advect.pdl` (91 stmts) — Semi-Lagrangian advection with inlined bilinear interpolation (4 BufLoads per field × 3 fields = 12 loads), buoyancy, dissipation, boundary zeroing
- `examples/sim/smoke/divergence.pdl` (32 stmts) — Central-difference velocity divergence
- `examples/sim/smoke/jacobi.pdl` (34 stmts) — One Jacobi pressure iteration (called 40× by host)
- `examples/sim/smoke/project.pdl` (45 stmts) — Pressure gradient subtraction + density visualization

Host orchestration per frame:
1. Swap vx/vy/density with vx0/vy0/density0
2. `render_sim(advect_fn, [vx0,vy0,den0], [vx,vy,den])`
3. `render_sim(div_fn, [vx,vy], [divergence])`
4. 40× `render_sim(jacobi_fn, [divergence,pressure], [pressure_tmp])` + swap
5. `render_sim(project_fn, [pressure,vx,vy,density], [vx0,vy0])` + swap back

Neighbor coordinates use `rem` wrapping (not raw subtraction) to avoid u32 underflow segfaults on BufLoad. Boundary results are discarded via `select`.

### Main integration (`src/main.rs`)

Three `Backend` variants:
- `Backend::Smoke` — native CPU
- `Backend::GpuSmoke` — GPU (initialized in `resumed()`)
- `Backend::JitSmoke` — Cranelift/LLVM with 4 compiled kernel functions + 9 `Vec<f64>` buffers

## Key files

| File | Role |
|------|------|
| `src/simulation.rs` | `SmokeState` — CPU advect, pressure solve, project, visualization |
| `src/gpu/smoke_gpu.rs` | GPU backend (5 pipelines, 7 buffers, dispatch orchestration) |
| `src/gpu/smoke.wgsl` | WGSL compute shader (5 entry points) |
| `examples/sim/smoke/advect.pdl` | Semi-Lagrangian advection kernel |
| `examples/sim/smoke/divergence.pdl` | Velocity divergence kernel |
| `examples/sim/smoke/jacobi.pdl` | Jacobi pressure iteration kernel |
| `examples/sim/smoke/project.pdl` | Pressure projection + visualization kernel |
| `src/main.rs` | CLI integration, multi-pass host orchestration |
