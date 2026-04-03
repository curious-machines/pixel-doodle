# Design: Opaque Handle Types for Buffer and Kernel

Status: Future work
Date: 2026-04-02

## Problem

`Buffer()` and `Kernel()` return bare `i32` handles. A buffer handle and a kernel handle are interchangeable by accident — nothing prevents passing a kernel handle to `bind_buffer` or vice versa.

## Proposed Design

Introduce opaque builtin types `Buffer` and `Kernel` that wrap an `i32` handle but are distinct types at the PDC level.

```
var field  = Buffer(.Vec4F32, 0.0)       // returns Buffer, not i32
var advect = Kernel("advect", "smoke_advect.wgsl", .Sim)  // returns Kernel, not i32

bind_buffer("field_in", field, 0)   // requires Buffer
run_kernel(advect)                   // requires Kernel
```

### Properties

- **Opaque**: users cannot access the inner i32 or construct these types manually
- **Builtin constructors**: `Buffer(...)` and `Kernel(...)` call through to the host via the bridge
- **Type-safe APIs**: `bind_buffer`, `run_kernel`, `swap_buffers`, `display_buffer` etc. require the correct handle type
- **Zero overhead**: codegen treats them as plain i32 — no wrapping struct at the IR level

### Implementation Sketch

1. Add `PdcType::OpaqueHandle(String)` variant (or reuse `PdcType::Struct` with a flag)
2. Register `Buffer` and `Kernel` as builtin opaque types in the type checker
3. Register builtin constructors that emit host calls instead of struct packing
4. Update all host function signatures to use the opaque types instead of `i32`
5. Codegen: treat opaque handles identically to i32

### APIs Affected

**Buffer handle:**
- `Buffer(type, init_value) -> Buffer` (constructor, currently `create_buffer`)
- `bind_buffer(name, Buffer, is_output)`
- `swap_buffers(Buffer, Buffer)`
- `display_buffer(Buffer)`
- `scene_buffer(scene_handle, name) -> Buffer`

**Kernel handle:**
- `Kernel(name, path, kind) -> Kernel` (constructor, currently `load_kernel`)
- `run_kernel(Kernel)`

### Builtin Enum Discoverability

Builtin enums (`BufferType`, `KernelType`, `FillRule`, etc.) are currently registered only in Rust (`type_check.rs:register_builtins`). There's no way to discover valid variants from PDC source. Options to address:

1. **Better error messages** — when a variant doesn't match, list all valid variants in the error
2. **VS Code autocomplete** — suggest variants when typing `.` inside a typed call
3. **`builtin enum` declarations** — require `builtin enum BufferType` in `.pdc` files, making the type visible in source while the compiler verifies it matches the Rust-side definition

### Migration

The current `Buffer()` and `Kernel()` functions already return i32 and use the constructor naming convention. When opaque types are implemented, the function signatures change but calling code stays the same — only code that stores handles in `i32` variables would need updating (change `var x: i32 = Buffer(...)` to `var x = Buffer(...)`).
