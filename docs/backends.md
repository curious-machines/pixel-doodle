# Running pixel-doodle with Different Backends

pixel-doodle supports three backends for computing the Mandelbrot fractal: a native Rust implementation and two JIT compilers (Cranelift and LLVM). This lets you compare compile speed versus code quality across approaches.

## Backends

| Backend    | Description                          | Cargo Feature        | Compile Time | Code Quality |
|------------|--------------------------------------|----------------------|--------------|--------------|
| `native`   | Plain Rust, compiled ahead of time   | (always available)   | 0ms          | Best (rustc) |
| `cranelift` | JIT compiled at startup via Cranelift | `cranelift-backend`  | ~1ms         | Good         |
| `llvm`     | JIT compiled at startup via LLVM     | `llvm-backend`       | ~8ms         | Very good    |

## Quick Start

```bash
# Native backend (default, no JIT)
cargo run --release

# Cranelift backend (included by default)
cargo run --release -- --backend cranelift

# LLVM backend (requires feature flag + system LLVM)
cargo run --release --features llvm-backend -- --backend llvm
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
```

If no `--backend` is specified, `native` is used.

If you request a backend that wasn't compiled in, the program prints the available backends and exits:

```
Unknown backend 'llvm'. Available: native, cranelift
```

## Window Title

While running, the window title shows:

```
cranelift | 4.2ms | 1.0x | compile 1.2ms
```

- The backend name
- Render time for the current frame
- Current zoom level
- One-time JIT compile time (0ms for native)

## Controls

| Key          | Action    |
|--------------|-----------|
| Arrow keys   | Pan       |
| `+` / `-`   | Zoom      |
| `Escape`     | Quit      |
