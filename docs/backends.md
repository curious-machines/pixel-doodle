# Running pixel-doodle with Different Backends

pixel-doodle supports four backends for computing the Mandelbrot fractal: a native Rust implementation, two JIT compilers (Cranelift and LLVM), and a GPU compute backend via wgpu. This lets you compare compile speed, code quality, and CPU vs GPU performance.

## Backends

| Backend    | Description                          | Cargo Feature        | Compile Time | Code Quality |
|------------|--------------------------------------|----------------------|--------------|--------------|
| `native`   | Plain Rust, compiled ahead of time   | (always available)   | 0ms          | Best (rustc) |
| `cranelift` | JIT compiled at startup via Cranelift | `cranelift-backend`  | ~1ms         | Good         |
| `llvm`     | JIT compiled at startup via LLVM     | `llvm-backend`       | ~8ms         | Very good    |
| `gpu`      | WGSL compute shader via wgpu         | (always available)   | 0ms          | GPU-native   |

## Quick Start

```bash
# Native backend (default, no JIT)
cargo run --release

# Cranelift backend (included by default)
cargo run --release -- --backend cranelift

# LLVM backend (requires feature flag + system LLVM)
cargo run --release --features llvm-backend -- --backend llvm

# GPU compute backend
cargo run --release -- --backend gpu
```

## Building

### Native + Cranelift (default)

The `cranelift-backend` feature is enabled by default. A plain `cargo build` gives you both `native` and `cranelift`:

```bash
cargo build --release
```

### Adding LLVM Support

The LLVM backend requires LLVM 18 development libraries installed on your system.

#### Ubuntu / Debian

```bash
sudo apt-get install llvm-18-dev libpolly-18-dev
```

#### macOS (Homebrew)

```bash
brew install llvm@18
```

Then build with the feature enabled:

```bash
cargo build --release --features llvm-backend
```

If `llvm-config-18` is not on your `PATH`, point the build to it:

```bash
LLVM_SYS_180_PREFIX=/usr/lib/llvm-18 cargo build --release --features llvm-backend
```

### Building Without Cranelift

If you only want the native backend:

```bash
cargo build --release --no-default-features
```

## Selecting a Backend at Runtime

Use the `--backend` flag:

```bash
# Explicit native
cargo run --release -- --backend native

# Cranelift JIT
cargo run --release -- --backend cranelift

# LLVM JIT (must be compiled with --features llvm-backend)
cargo run --release --features llvm-backend -- --backend llvm

# GPU compute
cargo run --release -- --backend gpu
```

If no `--backend` is specified, `native` is used.

If you request a backend that wasn't compiled in, the program prints the available backends and exits:

```
Unknown backend 'llvm'. Available: native, gpu, cranelift
```

## GPU Backend Notes

The GPU backend runs a WGSL compute shader (`src/gpu/mandelbrot.wgsl`) via wgpu. It uses whatever GPU is available (Vulkan, Metal, or DX12 depending on platform).

Key differences from the CPU backends:

- **Precision**: Uses f32 instead of f64. At deep zoom levels you will see precision artifacts sooner than with the CPU backends.
- **No CPU pixel buffer**: The compute shader writes directly to a GPU storage buffer, which is copied to the display texture. No data crosses the CPU-GPU boundary.
- **No JIT**: The shader is a handwritten WGSL file, not generated from the custom language IR. It exists for performance comparison, not as a compilation target.

## Window Title

While running, the window title shows:

```
cranelift | 4.2ms | 1.0x | compile 1.2ms
```

- The backend name
- Render time for the current frame
- Current zoom level
- One-time JIT compile time (0ms for native; omitted for GPU)

## Controls

| Key          | Action    |
|--------------|-----------|
| Arrow keys   | Pan       |
| `+` / `-`   | Zoom      |
| `Escape`     | Quit      |
