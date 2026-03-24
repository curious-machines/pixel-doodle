# pixel-doodle

Exploratory project for executing code — possibly from a custom language — that leverages various approaches to parallelism (threads, SIMD) to generate pixel data displayed in a window.

## Constraints

- Must run on Linux, macOS, and Windows
- Must target both x86_64 and AArch64
- Minimize external dependencies (preference, not mandate — JIT libraries are acceptable)
- This is an exploration with no fixed long-term direction; expect experimentation

## Stack (under consideration, not finalized)

Two viable stacks were evaluated:

| Layer         | C++ option              | Rust option              |
|---------------|-------------------------|--------------------------|
| Host language | C++17                   | Rust                     |
| JIT           | LLVM (OrcJIT) or AsmJIT | Cranelift                |
| Display       | sokol_app + sokol_gfx   | egui or sokol bindings   |
| Threading     | std::thread / oneTBB    | rayon                    |
| Host SIMD     | Google Highway          | wide crate / std::simd   |
| Build system  | CMake + FetchContent    | Cargo                    |

Key JIT tradeoff: LLVM provides auto-vectorization; Cranelift does not but compiles faster. AsmJIT supports both x86_64 and AArch64.

## Custom Language (PDIR — Pixel Doodle Intermediate Representation)

The project includes a small custom language called **PDIR** (Pixel Doodle Intermediate Representation), designed for AI-assisted generation. Design principles:
- SSA form (every value assigned once, named)
- Explicit types everywhere — no inference, no implicit conversions
- Flat over nested — sequence of named assignments over deep expression trees
- Regular, non-contextual syntax
- Explicit SIMD width and parallel annotations (not inferred)

### IR-First Architecture

The SSA IR (`Kernel` type in `kernel_ir.rs`) is the central artifact. The text format (`.pdir` files) is a readable, writable serialization — not ugly machine format. A higher-level language may be added later as a second frontend targeting the same IR, so keep the IR clean and syntax-independent.

### Kernel Model

A kernel body describes **per-pixel** computation. Backends generate the tile loop wrapper (row/col iteration, coordinate math, pixel store). The kernel declares explicit parameters and a return type, and produces a value via `emit`.

### Text Format (.pdir)

```
kernel gradient(x: f64, y: f64) -> u32 {
    r: f64 = mul x 255.0
    r_u: u32 = f64_to_u32 r
    g: f64 = mul y 255.0
    g_u: u32 = f64_to_u32 g
    b: u32 = const 128
    pixel: u32 = pack_argb r_u g_u b
    emit pixel
}
```

### Control Flow (V2)

Structured `while` with explicit loop-carried values:
```
while carry(zx: f64 = z0, zy: f64 = z0, iter: u32 = i0) {
    # condition check
    cond cont
    # body
    yield new_zx new_zy new_iter
}
# carry vars (zx, zy, iter) are live here with final values
```

### Parser

Hand-written recursive descent — no external parser dependencies. Resolves names to `Var` indices, type-checks operands, reports errors with line:col positions.

## Implementation Strategy

- Start with the display loop before writing any JIT code
- Isolate windowing from JIT so they can be debugged independently
- Parallel execution model: split pixel buffer into tiles, distribute to worker threads, each thread calls JIT'd function on its tile
- Source image data is shared read-only across threads; output is partitioned so each thread owns its region exclusively
- Align work division to cache line boundaries (64 bytes) to avoid false sharing
