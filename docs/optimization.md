# CPU Optimization Analysis

## Hardware: Cortex-X925 (10 big @ 3.9 GHz) + Cortex-A725 (10 little @ 2.8 GHz)

Benchmark data collected 2026-03-21 using LLVM backend, Mandelbrot kernel (256 max iterations), 1200x900 resolution.

## Summary of Findings

### 1. Tile Height

Reducing tile height from the original default of 16 rows to 1 row (per-row work units) improved Mandelbrot performance by **31% at 20 threads** (10.5ms → 7.2ms avg), with no measurable downside for uniform workloads like the gradient kernel.

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

**big.LITTLE scheduling:** `perf stat` showed that at 1-8 threads, 97-99% of execution happened on the A725 (little) cores. Big cores only meaningfully engaged at 10+ threads. However, `taskset` pinning tests showed 10 threads on big cores (12.6ms) was essentially identical to 10 threads unpinned (12.6ms), and worse than 20 threads default (10.6ms). The perf PMU sampling percentages may overstate the little-core bias — the scheduler appears to handle things reasonably.

**Coordination overhead:** Total CPU-seconds grew from 5.4s (1t) to 7.0s (20t) — 29% more total work for the same output. This is rayon work-stealing, thread wake/sleep, and synchronization costs.

### 3. What Is NOT a Bottleneck

- **Cache misses:** Under 2% across all configurations. The working set fits comfortably in cache.
- **Branch misprediction:** Under 0.4% for Mandelbrot. The branch predictor handles the escape-check well.
- **Memory bandwidth:** Gradient kernel (memory-bound, trivial compute) scales well even at 20 threads, confirming bandwidth is not saturated.

### 4. GPU Comparison (historical)

These numbers were collected when a GPU compute backend existed (since removed). They remain useful as a reference point.

| Backend | Avg ms | Mpix/s | vs 1-thread CPU |
|---------|--------|--------|-----------------|
| LLVM 1t | 97.5   | 11.1   | 1x              |
| LLVM 20t| 10.5   | 103    | 9.3x            |
| GPU     | 0.74   | 1464   | 132x            |

The GPU was ~14x faster than the best CPU configuration. GPUs run thousands of pixels simultaneously with dedicated hardware scheduling for divergent workloads.

## Vectorization (SIMD) — Not Yet Implemented

### Current State

The LLVM backend emits **scalar code** for the kernel body. LLVM O3 performs some minor vectorization in the color-packing path (using `<2 x double>` for R/G channel math) but does **not** auto-vectorize the column loop or the Mandelbrot iteration loop.

### Why LLVM Can't Auto-Vectorize

1. **Data-dependent loop exit:** The Mandelbrot iteration loop escapes at different counts for different pixels. LLVM's loop vectorizer cannot handle this — it requires all vector lanes to exit at the same iteration.

2. **Complex loop body:** The column loop contains the full kernel body including the while loop, which is too complex for the loop vectorizer to analyze and widen.

### What Manual Vectorization Would Require

Emit explicit `<N x double>` vector IR in the LLVM backend to process N pixels (columns) simultaneously:

- Widen coordinate calculations to operate on N x-coordinates at once
- Run the Mandelbrot iteration loop with N lanes, tracking per-lane escape via mask
- All lanes run until the **last** lane escapes (or hits max_iter)
- Pack N results into N pixel stores

This is the same approach GPU compute shaders use — and is exactly why they're fast at divergent workloads.

### Trade-offs to Consider

**Potential gains:**
- NEON: 128-bit vectors = 2x f64 = process 2 pixels per cycle
- SVE (currently disabled due to legalization bugs): scalable vectors, potentially 4-8x f64 on hardware that supports wider implementations
- Even 2x throughput on the hot loop would be significant

**Potential costs:**
- **Divergent workloads waste work:** For Mandelbrot, pixels inside the set run 256 iterations while nearby exterior pixels might run 10. With SIMD, all lanes run to the maximum of the group. If one pixel in a group of 4 is inside the set, all 4 run 256 iterations — a 4x waste on the other 3.
- **Register pressure:** Wider vectors mean more live values. On complex kernels, this could cause register spills.
- **Implementation complexity:** The kernel IR lowering would need a "vectorized" mode that emits vector types and handles per-lane masking for control flow.
- **Kernel compatibility:** Only kernels with compatible control flow can be vectorized. Simple kernels (gradient, circles) would benefit most. Divergent kernels (Mandelbrot) benefit less.

### SVE Status

SVE and SVE2 are **explicitly disabled** in the LLVM backend (feature flags `-sve,-sve2`) because LLVM's O3 auto-vectorizer generated scalable vector types that hit legalization bugs on simple kernels. Re-enabling SVE would require:

1. Fixing or working around the legalization bugs
2. Testing across a range of kernels
3. Potentially emitting explicit SVE intrinsics rather than relying on auto-vectorization

### Recommendation

Vectorization should be opt-in via a flag (e.g., `--simd-width 2`) so it can be benchmarked against scalar on a per-kernel basis. Start with NEON (2x f64) before attempting SVE. The gradient kernel is the simplest test case — no divergent control flow, pure arithmetic.

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
  cargo run --release -- examples/basic/mandelbrot/mandelbrot.pdc --bench --threads $t
done
```

### Tile height sweep example

```bash
for th in 1 2 4 8 16; do
  cargo run --release -- examples/basic/mandelbrot/mandelbrot.pdc --bench --set tile_height=$th
done
```

## Data Files

Raw benchmark data is stored in the repo root:

- `bench_scaling_data.json` — Thread scaling results (gradient + mandelbrot)
- `bench_perf_data.json` — `perf stat` hardware counter data
- `bench_tile_data.csv` — Tile height sweep results
- `bench_scaling_plot.png` — Thread scaling plot
- `bench_perf_plot.png` — perf stat analysis plot
- `bench_tile_plot.png` — Tile height sweep plot
