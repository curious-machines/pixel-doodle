# Enum Variant Constructors

## Overview

PDC enums currently support two operations:
- **Plain variants** as values: `BufferType.F32`, `KernelType.Sim`
- **Destructuring** in match arms: `Option.Some(x) => ...`

Missing: constructing a variant with payload as an expression. You can pattern-match `Option.Some(x)` but you can't write `var x = Option.Some(42)`.

## Motivation

Enum variants with payload are a natural expression form. Without them, you need workarounds (wrapper functions, separate types) for values that are "one of several alternatives carrying data."

The immediate use case is `Bind.In(buffer)` / `Bind.Out(buffer)` for kernel virtual property assignment (see `design-opaque-handle-types.md`). But the concept applies broadly to any tagged union: `Option.Some(value)`, `Result.Ok(value)`, `Result.Err(msg)`, etc.

## Two-Phase Plan

### Phase 1: Bind-specific (compile-time only) — IMPLEMENTED

`Bind.In(buffer)` and `Bind.Out(buffer)` are recognized by the type checker and destructured at codegen time into a buffer handle + direction constant. No `Bind` value is ever materialized at runtime.

This works because `Bind` variants only appear in kernel virtual property assignments, where the compiler immediately lowers them to runtime calls. The enum is a compile-time abstraction that guides codegen.

**Scope:** Only `Bind` variants, only in kernel property assignment context.

**Implementation:**
- Type checker recognizes `Bind.In(expr)` and `Bind.Out(expr)` as valid expressions of type `Bind`
- Validates the payload is a `BufferHandle`
- Codegen pattern-matches on the variant to emit the correct `pdc_bind_buffer` call

### Phase 2: General enum variant constructors (runtime representation)

For enums like `Option` and `Result` that need to exist as runtime values, the compiler needs a concrete representation. This requires:

**Representation:** A tagged union. At the JIT level (i32/f64 values), this likely means:
- An i32 discriminant tag
- A payload value (type depends on the variant)
- Possibly a two-word struct, or a pointer to heap-allocated tagged data

**Open questions:**
- **Fixed-size vs variable-size:** If all variants have the same payload size, the representation is simple (tag + value). If sizes differ, you need either the max size or indirection.
- **Stack vs heap:** Small enums (tag + one scalar) can live on the stack. Enums with large or variable payloads may need heap allocation.
- **Generics:** `Option<f64>` vs `Option<BufferHandle>` — does the enum need to be generic, or do we define concrete types (`OptionF64`, `OptionHandle`)?

**Not needed until** there's a concrete use case for runtime enum values beyond Bind. The Bind-specific Phase 1 handles the immediate need with zero runtime cost.

## Syntax

Both phases use the same surface syntax:

```
// Construction
var x = Option.Some(42.0)
var b = Bind.In(grid)

// Pattern matching (already supported)
match x {
    Option.Some(val) => print(val),
    Option.None => print("nothing"),
}
```

The syntax `EnumName.Variant(payload)` is already parseable as a method call on an enum module. The type checker needs to distinguish "variant constructor" from "method call" based on whether `Variant` is a known variant of the enum.
