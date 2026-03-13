---
name: Custom language design rationale
description: Design decisions for the AI-optimized custom language — SSA, explicit SIMD, tile-based execution model
type: project
---

The custom language is designed to be easy for an AI to generate and verify, not for humans to write directly.

Key design choices:
- SSA form eliminates mutation tracking — every value has one definition, dataflow is trivially traceable
- Explicit types prevent ambiguity about valid operations on values
- SIMD width is declared in the type system (e.g. `f32x8`), not inferred by an optimizer
- Parallelism is declared (e.g. `parallel tiles(64, 64)`), not auto-detected

A tile-based execution model was sketched where a `TileContext` owns iteration and position. The stream/iterator pattern (`foreach pixel in tile`) was preferred over implicit cursors because it keeps write targets unambiguous and handles edge cases like non-power-of-two tile sizes cleanly.

**Why:** The user won't write in the language directly — the AI generates it. Optimizing for AI reasoning correctness (explicit, flat, regular) over human ergonomics is the right tradeoff. Auto-vectorization was deliberately excluded from the language's requirements because SIMD intent should be expressed explicitly.

**How to apply:** When designing language features, prefer explicitness and regularity over convenience. Every piece of information the codegen needs should be visible in the source — no inference, no implicit context.
