# JIT Backends

pixel-doodle compiles PD/PDIR kernel source into machine code at startup via a JIT backend. The backend is selected in the `.pdp` settings block or via `--set backend=<name>`.

## Available Backends

| Backend     | Description                          | Cargo Feature        | Compile Time | Code Quality |
|-------------|--------------------------------------|----------------------|--------------|--------------|
| `cranelift` | JIT compiled at startup via Cranelift | `cranelift-backend`  | ~1ms         | Good         |
| `llvm`      | JIT compiled at startup via LLVM     | `llvm-backend`       | ~8ms         | Very good    |

Cranelift is the default. It compiles fast and produces good code. LLVM compiles slower but can apply more optimizations (particularly beneficial for complex kernels).

## Selecting a Backend

### In the .pdp file

```
settings {
  backend = "cranelift"
}
```

### Via CLI override

```bash
cargo run --release -- example.pdp --set backend=llvm
```

### Via .pds settings file

```
# my_machine.pds
backend = "llvm"
```

```bash
cargo run --release -- example.pdp --settings my_machine.pds
```

## Building

### Cranelift (default)

The `cranelift-backend` feature is enabled by default:

```bash
cargo build --release
```

### Adding LLVM Support

The LLVM backend requires LLVM 20 development libraries.

#### Ubuntu / Debian

```bash
sudo apt-get install llvm-20-dev libpolly-20-dev
```

#### macOS (Homebrew)

```bash
brew install llvm@20
```

Then build with the feature enabled:

```bash
cargo build --release --features llvm-backend
```

If `llvm-config-20` is not on your `PATH`:

```bash
LLVM_SYS_201_PREFIX=/usr/lib/llvm-20 cargo build --release --features llvm-backend
```

### Building Without JIT

```bash
cargo build --release --no-default-features
```

Without a JIT backend, `.pdp` files that reference `.pd` or `.pdir` kernels will fail at compile time.

## Compilation Pipeline

```
.pd file    →  PD parser    →  Kernel IR (SSA)  →  JIT backend  →  machine code
.pdir file  →  PDIR parser  →  Kernel IR (SSA)  →  JIT backend  →  machine code
```

Both PD and PDIR produce the same Kernel IR. The JIT backend lowers this IR to native machine code. The compiled function is called per-tile by the parallel render loop.

### Two Kernel ABIs

| ABI | Kernel Type | Used For |
|-----|-------------|----------|
| `TileKernelFn` | Pixel kernels | Per-pixel shaders (gradient, mandelbrot, SDF) |
| `SimTileKernelFn` | Simulation kernels | Buffer-based simulations (gray-scott, smoke) |

Both are `unsafe extern "C"` function pointers called from Rust via rayon parallel tile dispatch.

Both ABIs include a `user_args: *const u8` parameter — a pointer to a packed byte buffer of user-defined argument values. Built-in parameters (`x`, `y`, `px`, `py`, `sample_index`, `time` for pixel; `px`, `py`, `width`, `height` for sim) are provided by the tile loop. Non-built-in kernel parameters are loaded from the `user_args` buffer at compile-time-determined byte offsets with typed loads (`f64` = 8 bytes, `u32` = 4 bytes, naturally aligned). When a kernel has no user arguments, `null` is passed.
