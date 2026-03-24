# JIT Backends

pixel-doodle compiles PD/PDL kernel source into machine code at startup via a JIT backend. The backend is selected in the `.pdc` settings block or via `--set backend=<name>`.

## Available Backends

| Backend     | Description                          | Cargo Feature        | Compile Time | Code Quality |
|-------------|--------------------------------------|----------------------|--------------|--------------|
| `cranelift` | JIT compiled at startup via Cranelift | `cranelift-backend`  | ~1ms         | Good         |
| `llvm`      | JIT compiled at startup via LLVM     | `llvm-backend`       | ~8ms         | Very good    |

Cranelift is the default. It compiles fast and produces good code. LLVM compiles slower but can apply more optimizations (particularly beneficial for complex kernels).

## Selecting a Backend

### In the .pdc file

```
settings {
  backend = "cranelift"
}
```

### Via CLI override

```bash
cargo run --release -- example.pdc --set backend=llvm
```

### Via .pds settings file

```
# my_machine.pds
backend = "llvm"
```

```bash
cargo run --release -- example.pdc --settings my_machine.pds
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

Without a JIT backend, `.pdc` files that reference `.pd` or `.pdl` kernels will fail at compile time.

## Compilation Pipeline

```
.pd file  →  PD parser  →  Kernel IR (SSA)  →  JIT backend  →  machine code
.pdl file →  PDL parser →  Kernel IR (SSA)  →  JIT backend  →  machine code
```

Both PD and PDL produce the same Kernel IR. The JIT backend lowers this IR to native machine code. The compiled function is called per-tile by the parallel render loop.

### Two Kernel ABIs

| ABI | Kernel Type | Used For |
|-----|-------------|----------|
| `TileKernelFn` | Pixel kernels | Per-pixel shaders (gradient, mandelbrot, SDF) |
| `SimTileKernelFn` | Simulation kernels | Buffer-based simulations (gray-scott, smoke) |

Both are `unsafe extern "C"` function pointers called from Rust via rayon parallel tile dispatch.
