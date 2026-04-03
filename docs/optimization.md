# CPU Optimization Analysis

## Hardware: Cortex-X925 (10 big @ 3.9 GHz) + Cortex-A725 (10 little @ 2.8 GHz)

Benchmark data collected 2026-03-21 using gpu-llvm backend, Mandelbrot kernel (256 max iterations), 1200x900 resolution.

## Summary of Findings

### 1. Tile Height

Reducing tile height from the original default of 16 rows to 1 row (per-row work units) improved Mandelbrot performance by **31% at 20 threads** (10.5ms -> 7.2ms avg), with no measurable downside for uniform workloads like the gradient kernel.

| Tile Height | Tiles (900 rows) | Avg ms (20t) | Min ms | Max ms |
|-------------|-------------------|--------------|--------|--------|
| 1           | 900               | 7.24         | 6.58   | 8.43   |
| 2           | 450               | 7.45         | 6.71   | 8.86   |
| 4           | 225               | 7.90         | 7.27   | 8.71   |
| 16 (old)    | 56                | 10.49        | 8.44   | 13.17  |
| 64          | 15                | 27.72        | 18.34  | 30.75  |

**Why:** Mandelbrot has wildly uneven per-pixel cost. With 56 tiles across 20 threads, a single expensive boundary tile becomes the bottleneck. 900 rows lets rayon's work-stealing balance load much more evenly.

**Decision:** Default changed to 1. Configurable via `--tile-height N`.

### 2. Thread Scaling and big.LITTLE Scheduling

| Threads | Speedup | Efficiency | Avg ms |
|---------|---------|------------|--------|
| 1       | 1.0x    | 100%       | 97.5   |
| 2       | 1.97x   | 99%        | 49.5   |
| 4       | 3.63x   | 91%        | 26.9   |
| 8       | 6.55x   | 82%        | 14.9   |
| 10      | 7.59x   | 76%        | 12.9   |
| 20      | 9.33x   | 47%        | 10.5   |

Scaling is near-linear up to 4 threads, good at 8, then diminishing returns. Two factors:

**big.LITTLE scheduling:** `perf stat` showed that at 1-8 threads, 97-99% of execution happened on the A725 (little) cores. Big cores only meaningfully engaged at 10+ threads. However, `taskset` pinning tests showed 10 threads on big cores (12.6ms) was essentially identical to 10 threads unpinned (12.6ms), and worse than 20 threads default (10.6ms). The perf PMU sampling percentages may overstate the little-core bias -- the scheduler appears to handle things reasonably.

**Coordination overhead:** Total CPU-seconds grew from 5.4s (1t) to 7.0s (20t) -- 29% more total work for the same output. This is rayon work-stealing, thread wake/sleep, and synchronization costs.

### 3. What Is NOT a Bottleneck

- **Cache misses:** Under 2% across all configurations. The working set fits comfortably in cache.
- **Branch misprediction:** Under 0.4% for Mandelbrot. The branch predictor handles the escape-check well.
- **Memory bandwidth:** Gradient kernel (memory-bound, trivial compute) scales well even at 20 threads, confirming bandwidth is not saturated.

### 4. GPU Comparison (historical)

These numbers were collected when a GPU compute backend existed (since removed). They remain useful as a reference point.

| Backend | Avg ms | Mpix/s | vs 1-thread CPU |
|---------|--------|--------|-----------------|
| gpu-llvm 1t | 97.5   | 11.1   | 1x              |
| gpu-llvm 20t| 10.5   | 103    | 9.3x            |
| GPU     | 0.74   | 1464   | 132x            |

The GPU was ~14x faster than the best CPU configuration. GPUs run thousands of pixels simultaneously with dedicated hardware scheduling for divergent workloads.

## Vectorization (SIMD) -- Not Yet Implemented

### Current State

The gpu-llvm backend emits **scalar code** for the kernel body. LLVM O3 performs some minor vectorization in the color-packing path (using `<2 x double>` for R/G channel math) but does **not** auto-vectorize the column loop or the Mandelbrot iteration loop.

### Why LLVM Can't Auto-Vectorize

1. **Data-dependent loop exit:** The Mandelbrot iteration loop escapes at different counts for different pixels. LLVM's loop vectorizer cannot handle this -- it requires all vector lanes to exit at the same iteration.

2. **Complex loop body:** The column loop contains the full kernel body including the while loop, which is too complex for the loop vectorizer to analyze and widen.

### What Manual Vectorization Would Require

Emit explicit `<N x double>` vector IR in the LLVM backend to process N pixels (columns) simultaneously:

- Widen coordinate calculations to operate on N x-coordinates at once
- Run the Mandelbrot iteration loop with N lanes, tracking per-lane escape via mask
- All lanes run until the **last** lane escapes (or hits max_iter)
- Pack N results into N pixel stores

This is the same approach GPU compute shaders use -- and is exactly why they're fast at divergent workloads.

### Trade-offs to Consider

**Potential gains:**
- NEON: 128-bit vectors = 2x f64 = process 2 pixels per cycle
- SVE (currently disabled due to legalization bugs): scalable vectors, potentially 4-8x f64 on hardware that supports wider implementations
- Even 2x throughput on the hot loop would be significant

**Potential costs:**
- **Divergent workloads waste work:** For Mandelbrot, pixels inside the set run 256 iterations while nearby exterior pixels might run 10. With SIMD, all lanes run to the maximum of the group. If one pixel in a group of 4 is inside the set, all 4 run 256 iterations -- a 4x waste on the other 3.
- **Register pressure:** Wider vectors mean more live values. On complex kernels, this could cause register spills.
- **Implementation complexity:** The kernel IR lowering would need a "vectorized" mode that emits vector types and handles per-lane masking for control flow.
- **Kernel compatibility:** Only kernels with compatible control flow can be vectorized. Simple kernels (gradient, circles) would benefit most. Divergent kernels (Mandelbrot) benefit less.

### SVE Status

SVE and SVE2 are **explicitly disabled** in the LLVM backend (feature flags `-sve,-sve2`) because LLVM's O3 auto-vectorizer generated scalable vector types that hit legalization bugs on simple kernels. Re-enabling SVE would require:

1. Fixing or working around the legalization bugs
2. Testing across a range of kernels
3. Potentially emitting explicit SVE intrinsics rather than relying on auto-vectorization

### Recommendation

Vectorization should be opt-in via a flag (e.g., `--simd-width 2`) so it can be benchmarked against scalar on a per-kernel basis. Start with NEON (2x f64) before attempting SVE. The gradient kernel is the simplest test case -- no divergent control flow, pure arithmetic.

## Benchmark CLI Flags

| Flag | Description |
|------|-------------|
| `--bench` | Headless benchmark mode (no window) |
| `--bench-frames N` | Number of frames to time (default 100) |
| `--threads N` | Limit CPU thread count |
| `--set tile_height=N` | Rows per work unit (default 1) |
| `--output path.ppm` | Save rendered frame as PPM image |

### Thread scaling sweep example

```bash
for t in 1 2 4 8 10 12 16 20; do
  cargo run --release -- examples/basic/mandelbrot/mandelbrot.pdp --bench --threads $t
done
```

### Tile height sweep example

```bash
for th in 1 2 4 8 16; do
  cargo run --release -- examples/basic/mandelbrot/mandelbrot.pdp --bench --set tile_height=$th
done
```

## Data Files

Raw benchmark data is stored in the repo root:

- `bench_scaling_data.json` -- Thread scaling results (gradient + mandelbrot)
- `bench_perf_data.json` -- `perf stat` hardware counter data
- `bench_tile_data.csv` -- Tile height sweep results
- `bench_scaling_plot.png` -- Thread scaling plot
- `bench_perf_plot.png` -- perf stat analysis plot
- `bench_tile_plot.png` -- Tile height sweep plot

## PDC vs PDP Bridge Overhead Analysis

Analysis date: 2026-04-02

### Background

PDC pipelines are slower than their PDP equivalents despite dispatching the same WGSL kernels. The difference is architectural: PDP orchestrates kernels entirely in Rust with zero per-frame bridge crossings, while PDC's JIT'd `frame()` function calls back into Rust for every pipeline operation.

### Per-Frame Bridge Crossings

Each PDC host function call crosses the JIT→Rust boundary via an `extern "C"` wrapper that goes through `get_host()` (double-deref through `Box<dyn PipelineHost>` vtable dispatch).

**Game of Life example** (`iterations=1`, typical frame):

| Operation | Bridge Crossings | String Allocations |
|---|---|---|
| `bind_buffer` × 4 | 4 | 4 (`.to_string()` each) |
| `run_kernel` × 2 | 2 | 0 |
| `swap_buffers` × 1 | 1 | 0 |
| `display_buffer` × 1 | 1 | 0 |
| `request_redraw` × 1 | 1 | 0 |
| **Total** | **9** | **4** |

With `on_mousedown` active, add 4 more `set_kernel_arg` calls (4 crossings, 4 string allocs) plus `bind_buffer` × 2 + `run_kernel` + `swap_buffers` = 11 additional crossings, 6 more string allocs.

**PDP equivalent: 0 bridge crossings.** Rust reads pre-parsed `PipelineStep` structs and calls kernel function pointers directly.

### Bottleneck Inventory

#### 1. String allocations on every bind/arg call (high impact)

`pipeline_runtime.rs` — `bind_buffer()` (line ~270) and `set_kernel_arg_*()` (lines ~277, ~281) each call `.to_string()` to create an owned `String` from the `&str` handle. These are used once in `run_kernel` then discarded when `pending_bindings.clear()` runs.

In PDP: bindings are parsed once at startup — no per-frame string allocation.

#### 2. Linear search for kernel args by name (medium impact)

`pipeline_runtime.rs` line ~432 — inside `run_kernel`, for each WGSL param member:
```rust
self.pending_args.iter().find(|a| a.name == member.name)
```
Linear scan with string comparison, O(n×m) where n=params, m=pending_args.

In PDP: args matched positionally (`args.get(user_arg_index)`) — O(1).

#### 3. Trait object dispatch overhead (low-medium impact)

`runtime.rs` lines 701-706 — every host call goes through:
```rust
let host_ptr = (*ctx).host as *mut Box<dyn PipelineHost>;
&mut **host_ptr  // double deref + vtable lookup
```
Not huge per-call, but multiplied across 9+ calls per frame.

#### 4. Auto-buffer allocation leak (correctness + perf)

`pipeline_runtime.rs` lines ~387-394 — if any buffer slot is unbound, `run_kernel` pushes a **new** `NamedBuffer` with `format!()` name and fresh `Vec<u8>` every frame. These never get freed.

PDP caches these in `gpu_cpu_auto_buffers` HashMap.

#### 5. Per-tile Vec allocation inside rayon loop (low impact, both paths)

Both PDP (`pdp/runtime.rs` ~953) and PDC (`pipeline_runtime.rs` ~473) create a `Vec<*mut u8>` inside each tile closure. At 1080p with tile_height=1, that's ~1080 small Vec allocations per kernel dispatch. Affects both paths equally.

### Improvement Ideas

#### Quick wins (reduce overhead, no architecture change)

1. **Pre-intern binding/arg names** — PDC compiler knows strings at compile time. Emit buffer slot indices directly instead of string handles that get converted to owned Strings.

2. **Cache bindings across frames** — For kernels where bindings don't change frame-to-frame (common pattern), skip re-binding. Add a "bindings dirty" flag.

3. **Fix auto-buffer leak** — Cache auto-allocated buffers keyed by `(handle, slot)` instead of pushing new ones every frame.

4. **Hoist per-tile Vec** — Pre-build `*mut u8` pointer array once, share read-only across tiles.

#### Medium-term (reduce crossing count)

5. **Batch kernel dispatch** — Single `dispatch_kernel(handle, bindings[], args[])` host function instead of separate `bind_buffer` + `set_kernel_arg` + `run_kernel` calls.

6. **Move kernel arg packing into PDC** — PDC compiler knows the WGSL Params layout at compile time. PDC could write the 256-byte params buffer directly and pass it to a simpler `run_kernel_with_params(handle, params_ptr)`.

#### Longer-term (move more into PDC)

7. **Move buffer pointer resolution into PDC** — If PDC held raw pointers to buffer data (updated once per frame), it could build `buffer_ptrs` without crossing the bridge for each binding.

8. **Move tiled dispatch into PDC** — If PDC could call WGSL kernel function pointers directly, the entire `run_kernel` could happen in JIT code. Only rayon dispatch truly needs Rust.

### Status

None of the above have been implemented yet. Items 3-4 are bug fixes that should be done regardless. Items 1-2 and 5-6 are the most promising for measurable speedup. Items 7-8 align with the longer-term goal of consolidating PDP functionality into PDC.
