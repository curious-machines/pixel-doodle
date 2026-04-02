# Backends

pixel-doodle has two orthogonal settings that control execution:

- **render**: `"gpu"` (default) — native GPU compute via wgpu; `"cpu"` — JIT-compiled CPU fallback
- **codegen**: `"cranelift"` (default) or `"llvm"` — which JIT backend for CPU render

## Available Configurations

| render | codegen    | Description                               | Cargo Feature        | Compile Time | Code Quality |
|--------|------------|-------------------------------------------|----------------------|--------------|--------------|
| `gpu`  | —          | Native GPU compute via wgpu               | *(always available)* | Fast         | N/A (GPU)    |
| `cpu`  | `cranelift`| WGSL compiled to CPU via naga + Cranelift | `cranelift-backend`  | ~1ms         | Good         |
| `cpu`  | `llvm`     | WGSL compiled to CPU via naga + LLVM      | `llvm-backend`       | ~8ms         | Very good    |

GPU is the default render mode for both PDP and PDC files. Cranelift is the default codegen backend when CPU render is selected. LLVM compiles slower but can apply more optimizations (particularly beneficial for complex kernels).

## Selecting a Backend

### In the .pdp file

```
settings {
  render = "cpu"
  codegen = "cranelift"
}
```

### Via CLI override

```bash
cargo run --release -- example.pdp --set render=cpu,codegen=llvm
```

### Via .pds settings file

```
# my_machine.pds
render = "cpu"
codegen = "llvm"
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

Without a JIT backend, files that use `render=cpu` will fail at compile time. The `render=gpu` path (native GPU) remains available.

## Compilation Pipeline

```
.wgsl file  →  naga frontend  →  naga IR  →  backend  →  execution
                                     ├──→  cranelift  →  CPU machine code (render=cpu)
                                     ├──→  llvm       →  CPU machine code (render=cpu)
                                     └──→  wgpu       →  GPU compute dispatch (render=gpu)
```

All backends consume WGSL source via the naga shader compiler. The CPU backends lower naga IR to native machine code. The GPU backend dispatches compute shaders via wgpu.

### Two Kernel ABIs (CPU backends)

| ABI | Kernel Type | Used For |
|-----|-------------|----------|
| `TileKernelFn` | Pixel kernels | Per-pixel shaders (gradient, mandelbrot, SDF) |
| `SimTileKernelFn` | Simulation kernels | Buffer-based simulations (gray-scott, smoke) |

Both are `unsafe extern "C"` function pointers called from Rust via rayon parallel tile dispatch.

Both ABIs include a `user_args: *const u8` parameter -- a pointer to a packed byte buffer of user-defined argument values. Built-in parameters (`x`, `y`, `px`, `py`, `sample_index`, `time` for pixel; `px`, `py`, `width`, `height` for sim) are provided by the tile loop. Non-built-in kernel parameters are loaded from the `user_args` buffer at compile-time-determined byte offsets with typed loads (`f64` = 8 bytes, `u32` = 4 bytes, naturally aligned). When a kernel has no user arguments, `null` is passed.
