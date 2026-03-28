# Backends

pixel-doodle compiles WGSL kernel source into executable code at startup. The backend is selected in the `.pdp` settings block or via `--set backend=<name>`.

## Available Backends

| Backend        | Description                                    | Cargo Feature        | Compile Time | Code Quality |
|----------------|------------------------------------------------|----------------------|--------------|--------------|
| `gpu`          | Native GPU compute via wgpu                    | *(always available)* | Fast         | N/A (GPU)    |
| `gpu-cranelift`| WGSL compiled to CPU via naga + Cranelift      | `cranelift-backend`  | ~1ms         | Good         |
| `gpu-llvm`     | WGSL compiled to CPU via naga + LLVM           | `llvm-backend`       | ~8ms         | Very good    |

`gpu-cranelift` is the default CPU backend. It compiles fast and produces good code. `gpu-llvm` compiles slower but can apply more optimizations (particularly beneficial for complex kernels). `gpu` runs natively on the GPU via wgpu compute shaders.

## Selecting a Backend

### In the .pdp file

```
settings {
  backend = "gpu-cranelift"
}
```

### Via CLI override

```bash
cargo run --release -- example.pdp --set backend=gpu
```

### Via .pds settings file

```
# my_machine.pds
backend = "gpu-llvm"
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

Without a JIT backend, `.pdp` files that use CPU backends (`gpu-cranelift`, `gpu-llvm`) will fail at compile time. The `gpu` backend (native GPU) remains available.

## Compilation Pipeline

```
.wgsl file  →  naga frontend  →  naga IR  →  backend  →  execution
                                     ├──→  gpu-cranelift  →  CPU machine code (Cranelift)
                                     ├──→  gpu-llvm       →  CPU machine code (LLVM)
                                     └──→  gpu            →  wgpu compute dispatch
```

All backends consume WGSL source via the naga shader compiler. The CPU backends lower naga IR to native machine code. The GPU backend dispatches compute shaders via wgpu.

### Two Kernel ABIs (CPU backends)

| ABI | Kernel Type | Used For |
|-----|-------------|----------|
| `TileKernelFn` | Pixel kernels | Per-pixel shaders (gradient, mandelbrot, SDF) |
| `SimTileKernelFn` | Simulation kernels | Buffer-based simulations (gray-scott, smoke) |

Both are `unsafe extern "C"` function pointers called from Rust via rayon parallel tile dispatch.

Both ABIs include a `user_args: *const u8` parameter -- a pointer to a packed byte buffer of user-defined argument values. Built-in parameters (`x`, `y`, `px`, `py`, `sample_index`, `time` for pixel; `px`, `py`, `width`, `height` for sim) are provided by the tile loop. Non-built-in kernel parameters are loaded from the `user_args` buffer at compile-time-determined byte offsets with typed loads (`f64` = 8 bytes, `u32` = 4 bytes, naturally aligned). When a kernel has no user arguments, `null` is passed.
