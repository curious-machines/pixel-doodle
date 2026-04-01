# pixel-doodle

Exploratory project for executing WGSL compute shaders that generate pixel data, displayed in a window. Supports native GPU execution (via wgpu) and CPU fallback (via Cranelift or LLVM JIT compilation of WGSL through naga).

## Constraints

- Must run on Linux, macOS, and Windows
- Must target both x86_64 and AArch64
- Minimize external dependencies (preference, not mandate — JIT libraries are acceptable)
- This is an exploration with no fixed long-term direction; expect experimentation

## Stack

| Layer         | Technology                           |
|---------------|--------------------------------------|
| Host language | Rust                                 |
| Shaders       | WGSL (parsed by naga)                |
| GPU           | wgpu                                 |
| CPU fallback  | Cranelift or LLVM (JIT from naga IR) |
| Display       | winit + softbuffer                   |
| Threading     | rayon                                |
| Build system  | Cargo                                |

## Architecture

### Kernel Model

Kernels are WGSL compute shaders. The runtime dispatches them either on the GPU (native wgpu) or on the CPU (compiled via naga → Cranelift/LLVM JIT).

### Pipeline Config (.pdp)

`.pdp` files describe how to orchestrate kernels:
- Declare buffers, textures, variables
- Reference `.wgsl` kernel files
- Define execution order, loops, swaps, mouse/keyboard handlers

### Backends

- **gpu** (default): Native GPU execution via wgpu compute shaders
- **gpu-cranelift**: CPU fallback — compiles WGSL to native code via naga + Cranelift
- **gpu-llvm**: CPU fallback — compiles WGSL to native code via naga + LLVM (optional feature)

## Implementation Strategy

- Parallel execution model: GPU dispatches workgroups; CPU fallback splits rows across threads
- Source image data is shared read-only across threads; output is partitioned so each thread owns its region exclusively
- Align work division to cache line boundaries (64 bytes) to avoid false sharing

## Testing

After making code changes that could affect rendering (backends, codegen, runtime), run the regression test suite:

```bash
./test_regression              # Test all sample×backend combos against golden references
./test_regression --no-build   # Skip rebuild if already built
./test_regression basic/gradient  # Test a single example
```

If the test fails because of an intentional change, update the golden images:
```bash
./test_regression --update     # Regenerate all golden images
```

Golden images are stored locally in `tests/golden/` (not tracked in git). Generate them on first clone with `./test_regression --update`.

### Testing Principles

- All code must have unit tests and integration tests
- Test known good cases, extreme/boundary cases, and unexpected/invalid input cases
- Testing must be thorough and exhaustive — all new code must include tests of this calibre
- Bug fixes: write a failing test first that reproduces the bug, then fix the code so it passes
- PDC functions must be testable from Rust via `compile_only` + `call_fn`
