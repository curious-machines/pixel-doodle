# PDC (Pixel Doodle Code) Language Specification

**Date:** 2026-03-31 (design), 2026-04-01 (Phase 1-7+ implemented)
**Status:** Core language implemented. See `docs/pdc.md` for the complete user-facing reference.

**Implemented:** All numeric types (i8-i64, u8-u64, f32, f64), bool, Path, strings (handle-based, UTF-8, concat/slice/len/char_at), slices (slice\<T\> views into arrays), structs, enums (simple + data variants), match (with destructuring, dot-shorthand, exhaustiveness), tuples (construction, element access, destructuring, len), arrays (Array\<T\> with proper element-sized storage, bracket notation, for-each, map, broadcasting), imports (direct, namespaced, and file-based with circular detection), `pub` visibility with per-module isolation, stdlib (geometry, math), compound assignment (all operators), named function arguments, const enforcement, UFCS, alpha blending, `**` exponentiation, bitwise operators (`& | ^ ~ << >>`), ternary `?:`, function references/map, function overloading, for-each destructuring, richer fill/stroke styling (FillRule, LineCap, LineJoin), block comments (`/* */` with nesting), type aliases (`type Name = Type`), inclusive range loops (`..=`), exp2/inversesqrt math builtins.

**Not yet implemented:** SOA layout annotation, LLVM backend, PDP integration.

PDC is a JIT-compiled language for describing vector scenes and defining processing kernels. It combines VFS's expression language with PDP's resource model. Compiled via Cranelift (portable) and optionally LLVM (performance).

## Types

### Core types

| Category | Types |
|---|---|
| Integers | i8, i16, i32, i64, u8, u16, u32, u64 |
| Floats | f32, f64 |
| Bool | bool |
| Arrays | `array<T>` — dynamic-length, T is any type |
| Strings | `string` — UTF-8, copy-on-write |
| Structs | User-defined, with optional layout annotation |
| Enums | Rust-style with data variants |
| Tuples | `(T, U, ...)` |
| Slices | `slice<T>` — view into array/buffer without copying |

### Notes

- f16 deferred until Rust support improves.
- Generics only for container types (array, slice) — no user-defined generics or constraints.
- No built-in vec2/vec3/vec4 or matrix types — defined in standard library as structs with f32 fields (e.g., `struct Vec2 { x: f32, y: f32 }`). Default to f32 since that's the GPU type. If f64 variants are needed, define them with explicit names (e.g., `Vec2d`).
- Color, Path, Mat4, etc. defined in standard library.
- Structs support layout annotation for GPU interop:
  ```
  #[layout(soa)]
  struct Curve { p0x: f32, p0y: f32, p1x: f32, p1y: f32 }

  #[layout(aos)]  // default
  struct Style { width: f32, cap: LineCap }
  ```
  SOA layout stores `array<Curve>` as separate arrays per field — maps directly to GPU buffer format.

### Unified Constructor Syntax

`TypeName(args)` is a single syntax for type casts, struct construction, and type constructors. The compiler resolves based on what the name refers to:

```
// Type cast (built-in numeric type)
const x = f32(some_int)

// Struct construction (user-defined type, supports named args)
const s = Style(width: 2.0, color: 0xFF000000)

// Type constructor (built-in like Path)
var p = Path()
```

## Variables

| Feature | Syntax | Notes |
|---|---|---|
| Immutable | `const x: f32 = 5.0` | JIT folds when value known at compile time |
| Mutable | `var count: i32 = 0` | |
| Type annotation | Optional (inferred when possible) | Explicit for documentation or disambiguation |
| Builtins | `builtin const time: f64` | Declared explicitly, provided by runtime |
| Scope | Global, function, block | |

### Examples

```
const tolerance = 0.5           // inferred f64, JIT can fold
var frame_count: u64 = 0        // explicit type, mutable
const color = rgb(1.0, 0.5, 0)  // runtime immutable

builtin const time: f64
builtin const mouse_x: f64
builtin var paused: bool
```

### Notes

- Range constraints (`range<f64>(0.1..10.0)`) are PDP-only for UI slider generation.

## Flow Control

| Feature | Syntax |
|---|---|
| Conditional | `if cond { } elsif cond { } else { }` |
| Ternary | `cond ? expr_a : expr_b` |
| Counted loop | `for i in start..end { }` (exclusive) |
| Inclusive loop | `for i in start..=end { }` |
| Collection loop | `for item in array { }` |
| While loop | `while cond { }` |
| Infinite loop | `loop { }` |
| Break | `break` |
| Continue | `continue` |
| Return | `return value` |
| Pattern match | `match value { Variant => ..., }` |

### Notes

- `if` is a statement, not an expression. Ternary covers expression-level conditionals.
- Destructuring in for loops and match arms deferred.
- Pipeline constructs (run, display, swap, loop iterations, accumulate, event handlers, init blocks) remain in PDP.

### Example

```
const n = segments.length
var i = 0
while i < n {
    match curves[i].kind {
        CurveKind::Line => flatten_line(curves[i]),
        CurveKind::Cubic(c) => {
            if deviation(c) < tolerance {
                emit_segment(c.p0, c.p3)
            } else {
                subdivide(c)
            }
        },
    }
    i += 1
}
```

## Functions

| Feature | Decision |
|---|---|
| Definition | `fn name(params) -> Type { }` |
| Recursion | Supported. LLVM can optimize tail calls. |
| Function references | Supported — pass functions to map, etc. |
| Multiple return | Via tuples |
| Overloading | By parameter types, not return type |
| Method syntax | UFCS: `a.foo(b)` is sugar for `foo(a, b)` |
| Named arguments | Optional, order-independent when named |
| Kernels | Just functions, referenced by name from orchestration |
| Path builder | Rust-provided primitives, shapes built in PDC stdlib |
| Draw model | Geometry (paths) separate from styling (fill/stroke) |

### Named Arguments

Function arguments can be positional, named, or mixed. Named arguments can appear in any order. Positional arguments must come first and match declaration order.

```
fn Circle(cx: f32, cy: f32, radius: f32, color: u32) { ... }

// All equivalent:
circle(100.0, 200.0, 50.0, 0xFFFF8800)
circle(cx: 100.0, cy: 200.0, radius: 50.0, color: 0xFFFF8800)
circle(radius: 50.0, cx: 100.0, color: 0xFFFF8800, cy: 200.0)
circle(100.0, 200.0, color: 0xFFFF8800, radius: 50.0)
```

The type checker matches named arguments to parameters by name and reorders them. Positional arguments fill parameters left-to-right. Mixing positional after named is an error.

### Dot Operator Resolution

The `.` operator resolves in order:
1. **Struct field access** — if the left side is a struct with a matching field, access it directly
2. **UFCS function call** — otherwise, rewrite `a.foo(b)` to `foo(a, b)`

This allows natural field access on structs and method-style chaining on any type:

```
struct Circle { cx: f32, cy: f32, radius: f32 }

const c = Circle(cx: 100.0, cy: 200.0, radius: 50.0)
const r = c.radius                    // field access (rule 1)
const moved = c.translate(10.0, 20.0) // UFCS: translate(c, 10.0, 20.0) (rule 2)
```

### Default Parameter Values

Parameters can have default values. Omitted arguments use the default. Defaults must be trailing — required parameters cannot follow defaulted ones. Default expressions are evaluated fresh at each call site.

```
fn RoundedRect(x: f64, y: f64, w: f64, h: f64, r: f64 = 0.0) -> Path {
    if r == 0.0 { return Rect(x, y, w, h) }
    // ... rounded rect logic
}

RoundedRect(x: 10.0, y: 10.0, w: 100.0, h: 50.0)           // r defaults to 0.0
RoundedRect(x: 10.0, y: 10.0, w: 100.0, h: 50.0, r: 8.0)   // r = 8.0
```

### Deferred

- Closures/lambdas
- Tail call annotation

## Operators

### Arithmetic
`+ - * / % **` and unary `-`

### Bitwise
`& | ^ ~ << >>`

### Comparison
`== != < <= > >=`

### Logical
`&& || !`

### Assignment
`= += -= *= /= %= **= &= |= ^= <<= >>=`

### Other

| Operator | Syntax |
|---|---|
| Construction/Cast | `TypeName(args)` — unified syntax for casts, struct construction, and type constructors |
| Indexing | `[]` for arrays/slices |
| Field access | `.` for struct fields and UFCS |
| Range | `..` exclusive, `..=` inclusive |
| Ternary | `cond ? a : b` |

### Array broadcasting

| Operation | Behavior |
|---|---|
| `array op array` | Element-wise, same length required |
| `scalar op array` | Broadcast scalar to all elements |
| `array op scalar` | Broadcast scalar to all elements |
| `array.map(fn_ref)` | Apply any function element-wise: `a.map(sin)`, `a.map(my_func)` |

Math functions do not auto-broadcast on arrays — use `array.map(sin)` instead. This keeps the type checker simpler (every math function has a single signature) while still supporting element-wise application via function references.

## Built-in Functions

### Rust-provided (require platform intrinsics)

| Category | Functions |
|---|---|
| Trig | sin, cos, tan, asin, acos, atan, atan2 |
| Exponential | exp, ln, log2, log10, exp2 |
| Root | sqrt, inversesqrt (fast-math) |
| Rounding | floor, ceil, round |
| Comparison | min, max, abs |
| Misc | fract, fmod |

All support f32 and f64 overloads. Auto-broadcast on arrays.

### Standard library (written in PDC)

lerp, clamp, smoothstep, step, sign, saturate, degrees, radians, distance, dot, cross, length, normalize, reflect, Vec2, Vec3, Vec4, Mat3, Mat4, Color, geometry/shape construction functions.

### Deferred

- Noise functions (Perlin, simplex) — can be implemented in PDC

## Path Builder and Draw Model

Geometry (paths) is separate from styling (fill/stroke). Paths are pure geometry constructed via Rust-provided primitives. Drawing commands apply styles to paths and submit them to the scene.

### Path Primitives (Rust-provided built-ins)

| Function | Description |
|---|---|
| `Path() -> u32` | Create empty path, returns handle. Uses unified constructor syntax. |
| `move_to(path, x, y)` | Start new subpath at (x, y) |
| `line_to(path, x, y)` | Line from current point to (x, y) |
| `quad_to(path, cx, cy, x, y)` | Quadratic bezier (1 control point) |
| `cubic_to(path, c1x, c1y, c2x, c2y, x, y)` | Cubic bezier (2 control points) |
| `close(path)` | Close subpath back to last move_to |

Note: arc_to is not a primitive. Arcs are approximated with cubics in the PDC standard library.

With UFCS, these read naturally as method calls:
```
var p = Path()
p.move_to(100.0, 50.0)
p.cubic_to(150.0, 50.0, 150.0, 100.0, 100.0, 100.0)
p.close()
```

### Draw Commands (Rust-provided built-ins)

Drawing commands reference a path handle and submit styled geometry to the scene. The same path can be drawn multiple times with different styles.

| Function | Description |
|---|---|
| `fill(path, color)` | Draw path as filled shape (even-odd rule) |
| `stroke(path, width, color)` | Draw path as stroked outline (nonzero rule) |

```
var p = Path()
p.move_to(0.0, 0.0)
p.line_to(100.0, 0.0)
p.line_to(50.0, 86.0)
p.close()

fill(p, 0xFFFF8800)             // orange fill
stroke(p, 2.0, 0xFF000000)      // black outline
```

The path `p` is not consumed — both `fill` and `stroke` reference the same geometry. Each draw call creates a separate entry in the scene with its own path_id, color, and fill rule.

### Shapes (PDC standard library, written in PDC)

Higher-level shape constructors built on path primitives:

```
fn Circle(cx: f32, cy: f32, r: f32) -> u32 {
    const k = r * 0.5522847498
    var p = Path()
    p.move_to(cx + r, cy)
    p.cubic_to(cx + r, cy - k, cx + k, cy - r, cx, cy - r)
    p.cubic_to(cx - k, cy - r, cx - r, cy - k, cx - r, cy)
    p.cubic_to(cx - r, cy + k, cx - k, cy + r, cx, cy + r)
    p.cubic_to(cx + k, cy + r, cx + r, cy + k, cx + r, cy)
    return p
}

fn Rect(x: f32, y: f32, w: f32, h: f32) -> u32 {
    var p = Path()
    p.move_to(x, y)
    p.line_to(x + w, y)
    p.line_to(x + w, y + h)
    p.line_to(x, y + h)
    p.close()
    return p
}

fn Arc(cx: f32, cy: f32, r: f32, start_angle: f32, end_angle: f32) -> u32 {
    // Approximate with cubics...
}
```

### Future: Richer Styling

Fill and stroke are currently simple function calls. In future phases, they could accept struct-based style objects:

```
draw(p, Fill { color: 0xFFFF8800, rule: FillRule::EvenOdd })
draw(p, Stroke { width: 2.0, color: 0xFF000000, cap: LineCap::Round, join: LineJoin::Miter(4.0) })
```

This is deferred until structs and enums are implemented.

## Resources and Declarations

| Feature | Syntax | Notes |
|---|---|---|
| Language-defined buffer | `var data: array<Type>` | Language creates and manages |
| External buffer | `extern buffer name: array<Type>` | Provided by pipeline/runtime |
| GPU annotation | `#[gpu]` | Marks buffer for GPU upload |
| Textures | PDP concern | Possibly received as extern handle |
| Settings | PDP only | |

### Example

```
var segments: array<Segment>

extern buffer vertices: array<f32>

#[gpu]
var output: array<u32>
```

## Modules and Imports

| Feature | Decision |
|---|---|
| Module = file | Each `.pdc` file is a module |
| Namespaced import | `import math` → `math.sin(1.0)` |
| Direct import | `import { sin, cos } from math` |
| Visibility | Everything public for now |
| Standard library | Embedded in binary, overridable by local `stdlib/` directory |
| Circular imports | Error at parse time |
| File extension | `.pdc` |

### Example

```
import math
import { circle, rect } from geometry
import "./my_shapes"

const c = circle(100.0, 200.0, 50.0)
const angle = math.atan2(y, x)
const custom = my_shapes.star(5, 30.0)
```

### Deferred

- Re-exports

## Comments

| Feature | Syntax |
|---|---|
| Line comment | `// comment` |
| Block comment | `/* comment */` |
| Nested block comment | `/* outer /* inner */ still comment */` |

Doc comments deferred.

## Error Handling

| Situation | Behavior |
|---|---|
| Float div by zero | IEEE 754 (NaN/infinity) |
| Integer div by zero | Trap with error message |
| Integer overflow | Wrapping |
| Array out-of-bounds | Trap with error message |
| Type errors | Caught at compile time |

Result/try/catch deferred.

## Memory Model

| Feature | Decision |
|---|---|
| Model | Value semantics with copy-on-write |
| User mental model | Everything behaves like a copy |
| Implementation | Reference counting, COW on mutation |
| Garbage collection | None — deterministic cleanup via refcount |
| Ownership/borrowing | Not exposed to user |

## Interop (Language ↔ Rust)

| Feature | Decision |
|---|---|
| Language → Rust | Pre-registered built-in functions via C ABI |
| Rust → language | JIT'd functions as native function pointers |
| Data passing | Scalars in registers, structs/arrays/strings as pointer+length |
| Buffer sharing | Shared memory, no copying |
| Callbacks | Pre-registered built-ins only |
| Async | None — synchronous execution |

## Relationship to PDP

PDC handles computation, scene description, and processing kernels. PDP handles pipeline orchestration and UI concerns.

**Stays in PDP:**
- Pipeline constructs (run, display, swap, loop iterations, accumulate)
- Event handlers (on keypress, on mousedown)
- Init blocks
- Texture loading
- Settings and title
- Range-constrained variables (UI slider generation)
- WGSL kernel references

**In PDC:**
- All computation and logic
- Scene construction (paths, styles, transforms)
- Processing kernels (flatten, stroke — dispatched by runtime)
- Standard library (math, geometry, color)
- User-defined types and functions

## Functional Programming

PDC supports function references (named only, no closures) and `map`. The dividing line for what can be implemented in PDC vs what requires Rust is **closures** — any function that needs a predicate or callback with captured context requires closures, which PDC lacks.

### Status Table

| Function | Description | PDC or Rust? | Why |
|----------|-------------|:------------:|-----|
| **map** | Transform each element | PDC ✅ (exists) | Built-in; works with named fn refs |
| **forEach** | Side-effect per element | PDC ✅ | `for x in arr { ... }` — just a loop |
| **filter** | Keep elements matching predicate | **Rust** | Requires closure for predicate with context |
| **reduce / fold** | Accumulate into single value | **Rust** | Accumulator fn needs 2 params; map dispatch wired for 1-param fns; needs captured initial value |
| **find** | First element matching predicate | **Rust** | Closure problem (same as filter) |
| **any / some** | True if any element matches | **Rust** | Closure problem |
| **all / every** | True if all elements match | **Rust** | Closure problem |
| **flatMap** | Map then flatten | **Rust** | Needs map + flatten; also closure issue |
| **zip** | Pair elements from two arrays | PDC ✅ | `for i in 0..len(a)` building tuple array |
| **enumerate** | Pair each element with its index | PDC ✅ | `for i in 0..len(arr)`, build `(i, arr[i])` tuples |
| **reverse** | Reverse array order | PDC ✅ | Loop from `len-1` down to `0`, push to new array |
| **sort** | Sort elements | **Either** | PDC can do basic sort; performant sort better in Rust |
| **take / drop** | First/last N elements | PDC ✅ | `slice(arr, 0, n)` already exists |
| **contains** | Check if element exists | PDC ✅ | Loop + equality check |
| **sum / product** | Reduce with +/* | PDC ✅ | Simple accumulator loop — no closure needed |
| **min / max** | Find extreme value | PDC ✅ | Simple accumulator loop |
| **groupBy** | Group into map by key | **Rust** | PDC has no map/dictionary type |
| **partition** | Split into two arrays by predicate | **Rust** | Closure problem for predicate |
| **scan** | Like reduce but emit intermediates | **Rust** | 2-param accumulator + closure issue |
| **compose / pipe** | Chain functions | **Rust** | Can't return functions or build pipelines |
| **curry / partial** | Partially apply arguments | **Rust** | No closures = no captured partial args |

### Key Limitation

Adding closures to PDC would move almost everything to the PDC column except:
- **groupBy** — needs a dictionary type
- **compose / curry** — needs returning functions

### Current PDC FP Capabilities

- ✅ Named functions as values (function references)
- ✅ Higher-order functions via `map()` on arrays
- ✅ Nested data structures (arrays of arrays)
- ✅ Pattern matching (enums)
- ✅ Immutable variables (`const`)
- ❌ Closures / lambdas
- ❌ Function type annotations in parameter lists
- ❌ Returning functions
- ❌ Generics (uses overloading instead)
- ❌ Lazy evaluation / iterators
