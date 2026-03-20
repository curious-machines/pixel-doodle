# Thread Control & Benchmarking

## Motivation

CPU backends (native/cranelift/llvm) use rayon with all available cores by default. There is no way to limit parallelism or run headless performance measurements. To compare processing speeds across core counts — and against the GPU — we need:

1. A `--threads N` flag to control CPU core usage
2. A `--bench` mode for headless timing without a window
3. GPU headless benchmarking for direct CPU-vs-GPU comparison

## Thread Control (`--threads N`)

### Approach

Rayon's `par_chunks_mut` in `render::render()` dispatches tiles across its global thread pool. Rather than changing the render function, wrap its invocation in a custom `ThreadPool` using `pool.install(|| ...)`. This scopes parallelism to N threads without touching the render internals.

### Implementation

**`src/main.rs`:**

- Add `threads: Option<usize>` to `CliArgs` and `parse_args()`.
- After kernel compilation, conditionally build a rayon `ThreadPool`:
  ```rust
  let thread_pool = args.threads.map(|n| {
      rayon::ThreadPoolBuilder::new()
          .num_threads(n)
          .build()
          .expect("failed to create thread pool")
  });
  ```
- Store the pool in `App` and add a helper to wrap render calls:
  ```rust
  fn with_pool<F: FnOnce() + Send>(&self, f: F) {
      match &self.thread_pool {
          Some(pool) => pool.install(f),
          None => f(),
      }
  }
  ```
- Wrap existing `render::render()` call sites in the event loop.
- Warn on stderr if `--threads` is used with `--backend gpu` (GPU ignores it).

### Design notes

- `render.rs` stays unchanged — the pool scoping is invisible to the render function.
- When `--threads` is omitted, rayon's default global pool is used (all cores), preserving current behavior.

## Bench Mode (`--bench`)

### Approach

When `--bench` is passed, skip the winit event loop entirely. Allocate a pixel buffer, run the kernel for N frames, collect per-frame wall-clock times, print statistics, and exit.

### Implementation

**New file: `src/bench.rs`** (~50 lines):

```rust
pub struct BenchResult {
    pub frame_times: Vec<f64>, // milliseconds
}

impl BenchResult {
    pub fn report(&self, label: &str, width: u32, height: u32) {
        // Compute min, max, avg, median
        // Compute Mpix/s = (width * height) / (avg_ms / 1000) / 1e6
        // Print to stderr
    }
}
```

**Output format** (plain text to stderr):

```
── bench: cranelift (4t) | 100 frames | 1200x900 ──
  min:      3.21 ms
  max:      4.87 ms
  avg:      3.45 ms
  median:   3.38 ms
  total:    345.2 ms
  throughput: 313.0 Mpix/s
  fps:      289.9
```

**`src/main.rs`:**

- Add `bench: bool` and `bench_frames: Option<u32>` (default 100) to `CliArgs`.
- Before the event loop, if `args.bench` and CPU backend:
  1. Allocate `Vec<u32>` buffer at `WIDTH × HEIGHT`.
  2. Run 5 warmup frames (cold JIT, cache priming).
  3. Time `bench_frames` iterations of `render::render()`, wrapped in thread pool if `--threads`.
  4. Print `BenchResult::report()` and return.

### Thread scaling comparison

Thread sweep is handled via a shell loop rather than built into the binary:

```bash
for t in 1 2 4 8; do
  ./run_pd llvm mandelbrot --bench --threads $t
done
```

This keeps the implementation simple. An auto-sweep flag (`--bench-sweep 1,2,4,8`) could be added later if warranted.

## GPU Headless Benchmarking

### Challenge

`GpuBackend::new()` takes `&Display`, and `GpuBackend::render()` calls `display.present_with_commands()` which requires a wgpu surface. Bench mode has no window.

### Approach

Add two new methods to `GpuBackend` without modifying the existing ones:

**`src/gpu/mod.rs`:**

1. **`new_headless(width, height, max_iter, wgsl_source)`** — creates a wgpu `Instance`, requests an adapter with `compatible_surface: None` and `PowerPreference::HighPerformance`, creates device/queue, builds the compute pipeline and buffers. Returns `(Self, Device, Queue)`.

2. **`dispatch_compute(&self, device, queue, ...params...)`** — identical to `render()` up through the compute pass dispatch, but omits the `copy_buffer_to_texture` and `present_with_commands`. Calls `device.poll(wgpu::Maintain::Wait)` to block until the GPU finishes.

**`src/main.rs`:**

- If `args.bench` and `--backend gpu`:
  1. Create headless context via `GpuBackend::new_headless()`.
  2. Run 5 warmup dispatches.
  3. Time `bench_frames` dispatches.
  4. Print report and exit.

### Timing accuracy

Wall-clock timing around `dispatch_compute()` + `device.poll(Wait)` measures end-to-end frame time including CPU-GPU synchronization overhead. This is the right metric for comparing against CPU backends (which also measure end-to-end). For isolated GPU kernel timing, wgpu timestamp queries could be added later.

## Usage Examples

```bash
# Interactive mode with limited threads
./run_pd cranelift mandelbrot --threads 4

# Bench CPU backend
./run_pd llvm mandelbrot --bench
./run_pd cranelift mandelbrot --bench --threads 4

# Bench GPU
./run_pd gpu mandelbrot_wgsl --bench

# Thread scaling sweep
for t in 1 2 4 8; do
  ./run_pd llvm mandelbrot --bench --threads $t --bench-frames 200
done
```

## Files to Modify

| File | Change |
|------|--------|
| `src/main.rs` | CLI args, thread pool creation, bench mode entry points |
| `src/gpu/mod.rs` | Add `new_headless()` and `dispatch_compute()` |
| `src/bench.rs` | New — `BenchResult` struct and reporting |
| `src/render.rs` | Unchanged |
