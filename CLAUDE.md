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

## Custom Language

The project will include a small custom language designed for AI-assisted generation. Design principles:
- SSA form (every value assigned once, named)
- Explicit types everywhere — no inference, no implicit conversions
- Flat over nested — sequence of named assignments over deep expression trees
- Regular, non-contextual syntax
- Explicit SIMD width and parallel annotations (not inferred)

## Implementation Strategy

- Start with the display loop before writing any JIT code
- Isolate windowing from JIT so they can be debugged independently
- Parallel execution model: split pixel buffer into tiles, distribute to worker threads, each thread calls JIT'd function on its tile
- Source image data is shared read-only across threads; output is partitioned so each thread owns its region exclusively
- Align work division to cache line boundaries (64 bytes) to avoid false sharing
