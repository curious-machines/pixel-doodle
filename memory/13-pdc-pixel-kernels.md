# PDC-Only Pixel Kernels

**Date:** 2026-04-04
**Status:** Design discussion

## Background

The project originally had a "pixel kernel" concept: a DSL where you defined a per-pixel kernel function, Rust used rayon to parallelize computation across cores, and the kernel emitted a pixel color. This was clean and fast.

Currently, CPU-mode pixel rendering uses WGSL compute shaders compiled to native code via Cranelift/LLVM. PDC orchestrates the pipeline (buffer management, kernel dispatch, multi-pass sequencing) but the actual per-pixel work is written in WGSL.

## Proposal

Allow PDC functions to serve as pixel kernels, dispatched in parallel by rayon. This makes PDC a complete software rendering system — pipeline orchestration AND per-pixel computation in one language, no WGSL required for CPU mode.

### How It Works

1. A PDC function with the right signature acts as a pixel kernel
2. `render(kernel)` auto-allocates an output buffer and returns its handle
3. `render(kernel, buffer)` writes to a caller-provided buffer (returns the same handle)
4. Rust receives the function pointer (and optional buffer handle), tiles rows across cores via rayon
5. Calls the kernel per pixel; return value (u32 color) written to output buffer at `y * w + x`

### The `render()` Function

Two forms:

```pdc
// Auto-allocate: creates/reuses an internal buffer, returns its handle
let result = render(my_kernel)
display_buffer(result)

// Explicit buffer: writes to a caller-provided buffer, returns the same handle
render(my_kernel, buf_b)
display_buffer(buf_b)
```

Both return the buffer handle, so the return value is always usable. The auto-allocate form lazy-creates a buffer on first call and reuses it (resizing if dimensions change). The explicit form gives control for ping-pong patterns without per-frame allocation.

### Fixed Kernel Contract

Every pixel kernel has the same signature — no binding, no introspection, no name matching:

```pdc
fn my_kernel(x: i32, y: i32, w: i32, h: i32) -> i32
```

Rayon knows exactly what to pass: coordinates and dimensions. The kernel returns a u32 color. This mirrors the simplicity of the original pixel kernel DSL.

For additional inputs (buffers from previous passes, configuration, textures), the kernel accesses them through **closure capture** or **module-level scope** — not through parameter binding:

```pdc
let buf_a = create_buffer(...)

fn blur_h(x: i32, y: i32, w: i32, h: i32) -> i32 {
    // buf_a is accessible here — module-level scope
    let src_pixel = buf_a[y * w + x]
    // ...
    return result
}

let result = render(blur_h)       // auto-allocate output
display_buffer(result)
```

This avoids reinventing the WGSL binding/params workflow. The kernel gets its own data from scope.

### Multi-Pass Pipelines

```pdc
let buf_a = create_buffer(...)
let buf_b = create_buffer(...)

fn blur_h(x: i32, y: i32, w: i32, h: i32) -> i32 {
    let src = buf_a[y * w + x]
    // horizontal blur...
    return result
}

fn blur_v(x: i32, y: i32, w: i32, h: i32) -> i32 {
    let src = buf_b[y * w + x]
    // vertical blur...
    return result
}

render(blur_h, buf_b)   // buf_a (scoped) -> buf_b (explicit output)
render(blur_v, buf_a)   // buf_b (scoped) -> buf_a (explicit output, ping-pong)
display_buffer(buf_a)
```

### Modules as Kernel Units

A module naturally scopes a kernel together with its configuration and helper functions:

```pdc
mod blur {
    var radius = 3
    var src: buffer

    fn sample(buf: buffer, x: int, y: int, w: int) -> int {
        return buf[y * w + x]
    }

    fn kernel(x: int, y: int, w: int, h: int) -> int {
        // radius, src, and sample() are all in module scope
        // ...
        return result
    }
}

mod sharpen {
    var amount = 1.5
    var src: buffer

    fn kernel(x: int, y: int, w: int, h: int) -> int {
        // different state, same fixed contract
        return result
    }
}

// Pipeline orchestration — configure then dispatch
blur.src = buf_a
blur.radius = 5
dispatch(blur.kernel, buf_b)

sharpen.src = buf_b
sharpen.amount = 2.0
dispatch(sharpen.kernel, buf_a)

present(buf_a)
```

With PDC imports, kernel modules can live in separate files — creating a reusable shader library:

```pdc
import blur from "effects/blur.pdc"
import tonemap from "effects/tonemap.pdc"

blur.src = buf_a
blur.radius = 3
dispatch(blur.kernel, buf_b)

tonemap.src = buf_b
tonemap.exposure = 1.2
dispatch(tonemap.kernel, buf_a)

present(buf_a)
```

Each module is a self-contained kernel with typed configuration (`var`s), private helpers (`fn`s), and a `kernel` function that follows the fixed dispatch contract. No binding boilerplate — just set vars and dispatch. This works entirely with module-scope variables, no closures required.

### Closures as Configurable Kernels (Future)

Closures enable factory functions that produce parameterized kernels while maintaining the fixed dispatch contract:

```pdc
fn make_blur(radius: int, src: buffer) -> fn(int, int, int, int) -> int {
    return fn(x: int, y: int, w: int, h: int) -> int {
        // radius and src captured from enclosing scope
        // sample neighbors within radius from src
        return result
    }
}

let h_blur = make_blur(3, buf_a)
let v_blur = make_blur(5, buf_b)
dispatch(h_blur, buf_b)
dispatch(v_blur, buf_a)
```

This is powerful — the dispatch contract stays fixed `(x, y, w, h) -> int`, but kernels can carry arbitrary configuration through captured state. No binding system needed.

**Note:** Closures that return closures are a significant language feature. PDC would need heap-allocated captured environments passed alongside the function pointer. This is valuable beyond pixel kernels — it's a general-purpose language capability.

### Thread Safety of Captured State

- **Reading** captured variables from kernel threads is safe (shared read across rayon threads).
- **Mutating** captured variables is a data race. Resolution: captured variables in dispatched kernels are **implicitly read-only**. If you need to accumulate across pixels, write to the output buffer — that's what it's for (partitioned by row, no thread contention).

## Why This Is Feasible

PDC functions already compile to C-ABI native function pointers via Cranelift/LLVM. They are `Send + Sync`. The existing WGSL CPU kernel dispatch (`pipeline_runtime.rs:560-601`) already calls raw function pointers from rayon `par_iter` — the same pattern applies directly to PDC function pointers.

Key infrastructure already in place:
- JIT compilation to native function pointers
- `compile_only` + `call_fn` for extracting user function pointers
- `PdcContext` passed explicitly (not global) — thread-safe by design
- Rayon thread pool integration
- Buffer data pointer exposure (Phase 2C) for inline element access

## Advantages Over WGSL-on-CPU

1. **Full software renderer in one language.** Pipeline logic and per-pixel computation unified. No context-switching between PDC and WGSL.

2. **Richer control flow.** WGSL has restrictions (no recursion, limited control flow). PDC compiles to unrestricted native code — recursion, arbitrary branching, dynamic buffer access all work. Important for ray marching, tree traversal, adaptive sampling.

3. **Shared functions between pipeline and kernel.** Math utilities, color conversion, noise functions — define once, call from both orchestration and per-pixel code. No duplication across languages.

4. **Simpler data plumbing.** No 256-byte params block with named member matching. PDC kernels are just typed function calls — the compiler handles the ABI.

## Display and Render Mode Interaction

### The Problem

PDC pixel kernels always run on CPU (rayon). But the display path depends on `render` mode:
- `render=cpu`: buffers are in CPU memory, softbuffer displays directly — no issue
- `render=gpu`: buffers live on the GPU device. A PDC kernel writes to CPU memory, so displaying its output requires uploading to the GPU.

### Resolution: Separate Computation from Display

**Where computation happens** is a per-kernel decision:
- WGSL kernels run on GPU or CPU depending on `render` mode
- PDC pixel kernels always run on CPU via rayon

**Where display happens** is the global `render` setting. The PDC script just calls `present(buf)` and the runtime handles the plumbing:
- **CPU mode:** no-op — buffer is already in CPU memory, softbuffer displays it directly
- **GPU mode:** runtime uploads the CPU buffer to a GPU texture/buffer, then presents via wgpu

From the PDC author's perspective, it's just "show this buffer." The upload cost is one memcpy per frame, which is acceptable — it happens once after dispatch completes, not per-pixel.

### Mixed Pipelines

This enables hybrid GPU+PDC pipelines:
1. WGSL kernel does a heavy compute pass on GPU
2. Result read back into a CPU buffer
3. PDC pixel kernel does post-processing that's hard in WGSL (recursion, complex branching)
4. `present()` the final buffer — runtime uploads to GPU if needed

The key insight: each kernel runs where it naturally runs. The runtime bridges the gap at display time.

### No New Render Mode Needed

PDC pixel kernels don't need a `render=pdc` mode. The existing `render` setting controls WGSL kernel execution and the display path. PDC kernels are always CPU. The `present()` function abstracts away the CPU↔GPU transfer when needed.

## What It Doesn't Change

- **Performance is equivalent** — both paths (WGSL-on-CPU and PDC) end up as native code via the same JIT backends.
- **GPU execution still needs WGSL** — PDC pixel kernels are CPU-only. GPU rendering continues to use WGSL compute shaders via wgpu.

## Design Decisions

### Resolved

1. **Kernel signature** — fixed contract: `fn(x: i32, y: i32, w: i32, h: i32) -> i32`. No parameter introspection or name-based binding.
2. **How kernels access data** — closure capture / module scope. Kernels read buffers and configuration from their enclosing scope, not from dispatch arguments.
3. **Render API** — two forms: `render(kernel)` auto-allocates and returns a buffer handle; `render(kernel, buffer)` writes to a provided buffer and returns the same handle. Both return the buffer handle so the result is always usable.
4. **Mutation in kernels** — captured state is implicitly read-only during dispatch. Output goes to the designated buffer.

### Still Open

1. **Texture access** — currently textures are opaque handles with no pixel read/write from PDC. PDC pixel kernels would benefit from texture sampling capability.
2. **`present()` vs `display_buffer()`** — whether to introduce a new `present()` function or extend the existing `display_buffer()` to handle CPU→GPU upload transparently.
3. **Closure implementation** — heap-allocated captured environments, lifetime management, and how they compose with the JIT function pointer model. This is a general PDC language feature, not pixel-kernel-specific.
4. **`render()` as language construct vs. host function** �� `render()` could be a compiler-recognized built-in (enabling static validation of the kernel signature) or a regular host function (simpler to implement). The plan leans toward a host function with type-checker special-casing for signature validation.
5. **Inline `mod` declarations** — currently PDC modules are 1:1 with files. Inline `mod` would allow defining multiple kernel modules in a single file, useful when kernels are small and tightly related (e.g., a multi-pass pipeline). Would need the compiler to handle both file-modules and inline-modules with the same scoping/resolution machinery.

### OOP Observation

Modules with `var`s and `fn`s that operate on them are structurally similar to classes. The key differences: no inheritance, no polymorphism, no `self`/`this`, no instantiation (one instance per declaration). It's closer to a singleton or stateful namespace.

The limitation shows when you want two instances of the same kernel with different configs — that's where closures fill the gap. `make_blur(3, buf_a)` gives you multiple "instances" without class machinery. Natural boundary: **modules for one-off kernel definitions, closures for multiple parameterized instances.** Neither requires OOP.

## Relationship to PDC Independence

This is a step toward PDC as a self-sufficient creative coding platform. The progression:
- Phase 2A-C (done): Reduced Rust boundary crossings for buffer/array access
- **PDC pixel kernels (this proposal):** Eliminate WGSL dependency for CPU rendering. Fixed-contract dispatch with module-scope data access.
- **Closures (future refinement):** Heap-allocated captured environments, functions returning functions. Enables configurable kernel factories (`make_blur(radius, src)`). Valuable as a general PDC language feature but NOT a prerequisite — module-level state is sufficient for v1.
- Future: PDC becomes the primary authoring environment, with WGSL generated from PDC for GPU acceleration

### Implementation Order

1. **Dispatch + module-scope access** — kernels read module-level variables. Change config by mutating module state between dispatch calls. Fully functional for multi-pass pipelines.
2. **Closures** — adds composability. Kernel factories, parameterized kernels, reusable pipeline building blocks. A language feature with benefits beyond pixel kernels.
