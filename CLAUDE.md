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

### Pipeline Scripts (.pdc)

`.pdc` scripts orchestrate kernels via a JIT-compiled scripting language:
- Declare buffers, textures, variables
- Reference `.wgsl` kernel files
- Define execution order, loops, swaps, mouse/keyboard handlers

**PDC syntax notes:**
- No semicolons — statements are newline-separated
- Variables use `var` (mutable) or `const` (immutable), never `let`
- Number literals support type suffixes: `255u8`, `42i32`, `3.14f32`, `0xFFi32`

### Settings

Two orthogonal settings control execution:

- **render**: `"gpu"` (default) — native GPU via wgpu; `"cpu"` — JIT-compiled CPU fallback
- **codegen**: `"cranelift"` (default) or `"llvm"` — which JIT backend for CPU render and PDC

Set via `--set render=cpu,codegen=llvm` or `.pds` files.

When running with the LLVM backend, add `--features llvm-backend` to the cargo command:
```bash
cargo run --release --features llvm-backend -- example.pdc --set codegen=llvm
```

**Important:** When running `cargo check`, `cargo test`, or `cargo build` manually, always include `--features llvm-backend` to ensure the LLVM codegen path compiles. Without it, LLVM-specific code changes won't be checked and LLVM regression tests will time out or fail. The `test_regression` script already includes this flag automatically.

## Implementation Strategy

- Parallel execution model: GPU dispatches workgroups; CPU fallback splits rows across threads
- Source image data is shared read-only across threads; output is partitioned so each thread owns its region exclusively
- Align work division to cache line boundaries (64 bytes) to avoid false sharing

## Testing

After making code changes that could affect rendering (backends, codegen, runtime), run the regression test suite:

```bash
./test_regression              # Test all sample×config combos against golden references
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
- An implementation is not complete until both `cargo test` and `./test_regression` pass with zero failures
- Always run `./test_regression` after changes that could affect rendering (backends, codegen, runtime, shaders, PDC)

## VS Code Extension

The PDC syntax highlighting extension is at `editors/vscode-pdc/`. Whenever PDC syntax changes (new keywords, operators, built-in functions), update `syntaxes/pdc.tmLanguage.json` to match.
