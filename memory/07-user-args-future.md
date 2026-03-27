# User-Defined Kernel Arguments — Future Discussion Items

Captured during the implementation of user-defined kernel arguments (2026-03-26).

## 1. Implicit built-in vs user parameter distinction

Kernels declare all parameters uniformly (`kernel foo(x: f64, y: f64, max_iter: u32)`), but the caller (`.pdp` `run` statement) only passes user-defined args — built-in values are injected automatically by the tile loop. The distinction is determined by a hardcoded set of built-in names.

This is slightly inconsistent: the kernel author declares params the same way, but the pdp author only passes some of them. The user noted this bothers them.

Possible directions:
- Annotate params in the kernel (e.g., `builtin x: f64` vs bare `max_iter: u32`)
- Require all params to be passed from pdp (including built-ins via special keywords)
- Keep implicit but improve documentation/error messages

## 2. ~~Leaking abstraction in JitBackend trait~~ (Resolved)

~~The runtime computes user arg layout (filtering out built-in names) and passes `&[UserArgSlot]` to the JIT backend. This means the runtime must know the built-in name set, which is conceptually a backend concern. Currently centralized as `PIXEL_BUILTINS` and `SIM_BUILTINS` constants in `jit/mod.rs`.~~

**Resolved:** Tile-loop builtin names are now defined by `KernelKind::tile_loop_params()` in `src/pdp/ast.rs`. The kernel kind (Pixel vs Standard) determines which parameters the tile loop injects — this is an execution model concern, not a backend concern. The runtime calls `decl.kind.tile_loop_params()` to compute the user arg layout, and the `JitBackend` trait has no knowledge of builtin names. The old `PIXEL_BUILTINS`/`SIM_BUILTINS` constants in `jit/mod.rs` have been removed.

## 3. GPU (WGSL) user-defined parameters

CPU backends (Cranelift + LLVM) support user args via a `*const u8` packed buffer. The GPU backend does not yet support user-defined parameters.

Agreed approach: add a **second uniform struct** for user args, separate from the existing `Params` struct:

```wgsl
// Existing, unchanged
@group(0) @binding(0) var<uniform> params: Params;

// New, generated per-kernel
struct UserArgs {
    max_iter: u32,
    threshold: f32,
}
@group(0) @binding(2) var<uniform> user_args: UserArgs;
```

This avoids modifying the fixed `Params` struct. The `UserArgs` struct would be generated to match each kernel's declared non-built-in parameters. On the Rust side, a second uniform buffer is created, packed with user arg values, and bound at the new binding slot.

Should be implemented soon after CPU to keep API parity between CPU and GPU pipelines.

## 4. Complex types (vec2, vec3)

The user wants to be able to pass complex types like `vec2<u32>` as kernel arguments (e.g., for Gray-Scott init kernels). The `*const u8` byte buffer approach was specifically chosen to support this in the future without ABI changes — just add the packing/loading logic for new types.
