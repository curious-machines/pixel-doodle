# PDC (Pixel Doodle Code) Language Reference

PDC is a JIT-compiled language for describing vector scenes. It compiles to native code via Cranelift and renders through a tile-based GPU rasterization pipeline.

## Table of Contents

- [Getting Started](#getting-started)
- [Comments](#comments)
- [Types](#types)
  - [Numeric Types](#numeric-types)
  - [Boolean](#boolean)
  - [Path](#path)
  - [Arrays](#arrays)
  - [Tuples](#tuples)
  - [Structs](#structs)
  - [Enums](#enums)
- [Variables](#variables)
  - [Constants](#constants)
  - [Mutable Variables](#mutable-variables)
  - [Builtin Variables](#builtin-variables)
  - [Const Enforcement](#const-enforcement)
- [Operators](#operators)
  - [Arithmetic](#arithmetic)
  - [Comparison](#comparison)
  - [Logical](#logical)
  - [Assignment](#assignment)
  - [Compound Assignment](#compound-assignment)
- [Type Casting](#type-casting)
- [Control Flow](#control-flow)
  - [If / Elsif / Else](#if--elsif--else)
  - [While Loop](#while-loop)
  - [For Range Loop](#for-range-loop)
  - [For-Each Loop](#for-each-loop)
  - [Loop (Infinite)](#loop-infinite)
  - [Break and Continue](#break-and-continue)
  - [Loop Variable Mutability](#loop-variable-mutability)
- [Functions](#functions)
  - [Definition](#definition)
  - [Void Functions](#void-functions)
  - [Recursion](#recursion)
  - [Named Arguments](#named-arguments)
- [Structs](#structs-1)
  - [Definition](#struct-definition)
  - [Construction](#struct-construction)
  - [Field Access](#field-access)
- [Enums](#enums-1)
  - [Simple Enums](#simple-enums)
  - [Data Variant Enums](#data-variant-enums)
  - [Enum Variant Access](#enum-variant-access)
- [Match Statements](#match-statements)
  - [Basic Match](#basic-match)
  - [Dot-Shorthand](#dot-shorthand)
  - [Destructuring](#destructuring)
  - [Wildcard Bindings](#wildcard-bindings)
  - [Catch-All Arm](#catch-all-arm)
  - [Exhaustiveness](#exhaustiveness)
- [Arrays](#arrays-1)
  - [Creating Arrays](#creating-arrays)
  - [Bracket Notation](#bracket-notation)
  - [Array Methods](#array-methods)
  - [For-Each Iteration](#for-each-iteration-1)
  - [Array Storage](#array-storage)
- [Tuples](#tuples-1)
  - [Construction](#tuple-construction)
  - [Element Access](#element-access)
  - [Tuple Length](#tuple-length)
  - [Destructuring](#tuple-destructuring)
  - [Empty and Single-Element Tuples](#empty-and-single-element-tuples)
- [Imports and Standard Library](#imports-and-standard-library)
  - [Import Syntax](#import-syntax)
  - [Geometry Module](#geometry-module)
  - [Math Module](#math-module)
  - [Built-in Math Functions](#built-in-math-functions)
- [Path Builder](#path-builder)
  - [Creating Paths](#creating-paths)
  - [Adding Segments](#adding-segments)
- [Drawing](#drawing)
  - [Fill](#fill)
  - [Stroke](#stroke)
  - [Color Format](#color-format)
  - [Alpha Blending](#alpha-blending)
  - [Painter's Algorithm](#painters-algorithm)
- [UFCS (Method Syntax)](#ufcs-method-syntax)

## Getting Started

Run a PDC file:

```bash
cargo run --example pdc_basic --release
```

The example loads `examples/pdc/basic.pdc` and renders the scene in a window.

A minimal PDC program:

```
import { Circle } from geometry

builtin const width: f32
builtin const height: f32

const c = Circle(f64(width) / 2.0, f64(height) / 2.0, 100.0)
fill(c, 0xFFFF8800)
```

## Comments

```
// Line comment
```

Block comments (`/* */`) are planned but not yet implemented.

## Types

### Numeric Types

| Type | Description | Size |
|------|-------------|------|
| `i8` | 8-bit signed integer | 1 byte |
| `i16` | 16-bit signed integer | 2 bytes |
| `i32` | 32-bit signed integer | 4 bytes |
| `i64` | 64-bit signed integer | 8 bytes |
| `u8` | 8-bit unsigned integer | 1 byte |
| `u16` | 16-bit unsigned integer | 2 bytes |
| `u32` | 32-bit unsigned integer | 4 bytes |
| `u64` | 64-bit unsigned integer | 8 bytes |
| `f32` | 32-bit float | 4 bytes |
| `f64` | 64-bit float (default for float literals) | 8 bytes |

Integer literals default to `i32`. Hex literals (e.g., `0xFFFF8800`) that exceed `i32` range default to `u32`. Values exceeding `u32` range default to `i64`.

Float literals default to `f64`.

### Boolean

```
const flag = true
const check = false
```

### Path

`Path` is an opaque handle to vector geometry managed by the runtime. See [Path Builder](#path-builder).

### Arrays

See [Arrays](#arrays-1) section for full details.

```
var arr = Array<f64>()
arr.push(1.0)
const val = arr[0]
```

### Tuples

See [Tuples](#tuples-1) section for full details.

```
const point = (100.0, 200.0)
const x = point.0
```

### Structs

See [Structs](#structs-1) section for full details.

### Enums

See [Enums](#enums-1) section for full details.

## Variables

### Constants

Constants are immutable. Type annotation is optional (inferred from the value).

```
const x = 42              // inferred as i32
const pi = 3.14159        // inferred as f64
const color = 0xFFFF8800  // inferred as u32 (hex literal)
const flag = true          // inferred as bool
```

With explicit type annotation:

```
const x: f32 = 42.0
```

### Mutable Variables

```
var count = 0
count = count + 1
count += 1              // compound assignment
```

### Builtin Variables

Builtin variables are provided by the runtime and are always const:

```
builtin const width: f32
builtin const height: f32
```

### Const Enforcement

Assigning to a `const` variable produces a compile error:

```
const x = 10
x = 20          // ERROR: cannot assign to const variable 'x'
```

This applies to `const` declarations, `builtin const`, const loop variables, and const tuple destructure bindings.

## Operators

### Arithmetic

| Operator | Description |
|----------|-------------|
| `+` | Addition |
| `-` | Subtraction |
| `*` | Multiplication |
| `/` | Division |
| `%` | Modulo (remainder) |
| `-x` | Unary negation |

Mixed numeric types are automatically widened (e.g., `i32 + f64` produces `f64`).

### Comparison

| Operator | Description |
|----------|-------------|
| `==` | Equal |
| `!=` | Not equal |
| `<` | Less than |
| `<=` | Less than or equal |
| `>` | Greater than |
| `>=` | Greater than or equal |

Comparison operators return `bool`. Enum values of the same type can be compared with `==` and `!=`.

### Logical

| Operator | Description |
|----------|-------------|
| `&&` | Logical AND (short-circuit) |
| `\|\|` | Logical OR (short-circuit) |
| `!` | Logical NOT |

### Assignment

```
var x = 0
x = 10
```

### Compound Assignment

| Operator | Equivalent |
|----------|------------|
| `x += 1` | `x = x + 1` |
| `x -= 1` | `x = x - 1` |
| `x *= 2` | `x = x * 2` |
| `x /= 2` | `x = x / 2` |
| `x %= 3` | `x = x % 3` |

## Type Casting

Cast between types using the type name as a constructor:

```
const i: i32 = 42
const f = f64(i)       // i32 to f64: 42.0
const s = f32(f)       // f64 to f32: 42.0
const n = i32(3.7)     // f64 to i32: 3 (truncates)
const b = u8(255)      // i32 to u8
const big = i64(i)     // i32 to i64
```

All numeric types can be cast to any other numeric type. Float-to-integer truncates. Integer narrowing wraps.

## Control Flow

### If / Elsif / Else

```
if temperature > 100 {
    // hot
} elsif temperature > 50 {
    // warm
} else {
    // cold
}
```

### While Loop

```
var i = 0
while i < 10 {
    // body
    i += 1
}
```

### For Range Loop

Iterates over an exclusive integer range:

```
for i in 0..10 {
    // i goes from 0 to 9
}
```

### For-Each Loop

Iterates over array elements:

```
var arr = Array<f64>()
arr.push(1.0)
arr.push(2.0)
arr.push(3.0)

for val in arr {
    // val is 1.0, then 2.0, then 3.0
}
```

### Loop (Infinite)

```
loop {
    // runs forever until break
    break
}
```

### Break and Continue

```
var i = 0
while i < 100 {
    if i == 50 {
        break       // exit the loop
    }
    i += 1
}
```

### Loop Variable Mutability

Loop variables are `const` by default. Use `var` to make them mutable:

```
// Default: const (cannot reassign i in the body)
for i in 0..10 {
    // i = 5  // ERROR: cannot assign to const variable 'i'
}

// Explicit const
for const i in 0..10 { }

// Mutable
for var i in 0..10 {
    i = i * 2  // OK
}

// Same for for-each
for const val in arr { }
for var val in arr { }
```

## Functions

### Definition

```
fn add(a: f64, b: f64) -> f64 {
    return a + b
}

const result = add(3.0, 4.0)  // 7.0
```

### Void Functions

Functions without a return type:

```
fn draw_at(x: f64, y: f64) {
    const c = Circle(x, y, 10.0)
    fill(c, 0xFFFF0000)
}
```

### Recursion

Functions can call themselves:

```
fn factorial(n: i32) -> i32 {
    if n <= 1 {
        return 1
    }
    return n * factorial(n - 1)
}
```

### Named Arguments

Function arguments can be named for clarity. Named arguments can appear in any order:

```
fn draw_circle(cx: f64, cy: f64, radius: f64) -> Path {
    return Circle(cx, cy, radius)
}

// Positional
draw_circle(100.0, 200.0, 50.0)

// Named (order doesn't matter)
draw_circle(radius: 50.0, cx: 100.0, cy: 200.0)
```

Named arguments are also used for struct construction (see [Struct Construction](#struct-construction)).

## Structs {#structs-1}

### Struct Definition

```
struct Vec2 {
    x: f64,
    y: f64,
}

struct Color {
    r: f64,
    g: f64,
    b: f64,
    a: f64,
}
```

### Struct Construction

Structs are constructed with named arguments using the type name as a constructor:

```
const pos = Vec2(x: 100.0, y: 200.0)
```

Field order doesn't matter:

```
const pos = Vec2(y: 200.0, x: 100.0)  // same result
```

### Field Access

```
const px = pos.x    // 100.0
const py = pos.y    // 200.0
```

## Enums {#enums-1}

### Simple Enums

C-style enums with no data:

```
enum Direction { North, South, East, West }

var dir = Direction.North

if dir == Direction.South {
    // heading south
}
```

### Data Variant Enums

Variants can carry named data:

```
enum Shape {
    Circle(cx: f64, cy: f64, r: f64),
    Rect(x: f64, y: f64, w: f64, h: f64),
    None,
}
```

Construction:

```
var s = Shape.Circle(100.0, 200.0, 50.0)
s = Shape.None
```

### Enum Variant Access

Access variants with dot notation on the enum name:

```
const dir = Direction.North
const shape = Shape.Circle(0.0, 0.0, 10.0)
```

## Match Statements

### Basic Match

```
enum Color { Red, Green, Blue }

var c = Color.Red

match c {
    Color.Red => {
        fill(circle, 0xFFFF0000)
    },
    Color.Green => {
        fill(circle, 0xFF00FF00)
    },
    Color.Blue => {
        fill(circle, 0xFF0000FF)
    },
}
```

### Dot-Shorthand

When the scrutinee type is known, omit the enum name with a leading dot:

```
match direction {
    .North => { /* ... */ },
    .South => { /* ... */ },
    .East => { /* ... */ },
    .West => { /* ... */ },
}
```

The full syntax `Direction.North` is also accepted.

### Destructuring

Extract data from variant payloads:

```
match s {
    .Circle(cx, cy, r) => {
        const path = Circle(cx, cy, r)
        fill(path, 0xFFFF8800)
    },
    .Rect(x, y, w, h) => {
        const path = Rect(x, y, w, h)
        fill(path, 0xFF4488FF)
    },
    .None => { },
}
```

### Wildcard Bindings

Use `_` to ignore specific fields:

```
match s {
    .Circle(_, _, r) => {
        // only use radius
    },
    .Rect(x, y, _, _) => {
        // only use position
    },
    .None => { },
}
```

### Catch-All Arm

Use `_` as a pattern to match any remaining variants:

```
match s {
    .Circle(cx, cy, r) => {
        // handle circles
    },
    _ => {
        // handle everything else
    },
}
```

### Exhaustiveness

Match statements must cover all variants. Missing variants cause a compile error:

```
// ERROR: non-exhaustive match: missing variant 'Shape.Rect'
match s {
    .Circle(cx, cy, r) => { /* ... */ },
}
```

Add a `_` catch-all arm or cover all variants to fix this.

### Match Arm Scope

Destructured bindings are only visible within their arm's body. Each arm has its own scope.

## Arrays {#arrays-1}

### Creating Arrays

Arrays are created with the `Array<T>()` constructor, specifying the element type:

```
var floats = Array<f64>()
var ints = Array<i32>()
var bytes = Array<u8>()
```

### Bracket Notation

Read and write elements using bracket notation:

```
var arr = Array<f64>()
arr.push(10.0)
arr.push(20.0)
arr.push(30.0)

const val = arr[0]     // 10.0
arr[2] = 99.0          // modify third element
```

### Array Methods

| Method | Description |
|--------|-------------|
| `arr.push(value)` | Append an element |
| `arr.len()` | Get the number of elements (returns `i32`) |
| `arr.get(index)` | Get element at index (same as `arr[index]`) |
| `arr.set(index, value)` | Set element at index (same as `arr[index] = value`) |

All methods also work as regular function calls via UFCS:

```
push(arr, 42.0)        // same as arr.push(42.0)
const n = len(arr)     // same as arr.len()
const v = get(arr, 0)  // same as arr.get(0) or arr[0]
set(arr, 0, 99.0)      // same as arr.set(0, 99.0) or arr[0] = 99.0
```

### For-Each Iteration {#for-each-iteration-1}

Iterate directly over array elements:

```
var colors = Array<u32>()
colors.push(0xFFFF0000)
colors.push(0xFF00FF00)
colors.push(0xFF0000FF)

for color in colors {
    // color is each element in turn
}
```

### Array Storage

Arrays store elements at their natural size in contiguous memory:
- `Array<u8>`: 1 byte per element
- `Array<i16>`: 2 bytes per element
- `Array<f32>` / `Array<i32>`: 4 bytes per element
- `Array<f64>` / `Array<i64>`: 8 bytes per element

The underlying byte buffer is contiguous and suitable for GPU buffer upload.

## Tuples {#tuples-1}

### Tuple Construction

Create tuples with parentheses and commas:

```
const pair = (100.0, 200.0)
const triple = (1, 2.0, true)
```

### Element Access

Access elements with numeric indices after a dot:

```
const point = (100.0, 200.0, 50.0)
const x = point.0     // 100.0
const y = point.1     // 200.0
const z = point.2     // 50.0
```

Tuple indices must be compile-time integer literals.

### Tuple Length

```
const t = (1, 2, 3)
const n = t.len()     // 3 (compile-time constant)
```

### Tuple Destructuring

Bind tuple elements to variables:

```
const point = (100.0, 200.0, 50.0)
const (x, y, z) = point

var (a, b) = (1.0, 2.0)

// Use _ to ignore elements
const (first, _, last) = (10, 20, 30)
```

`const` destructuring creates immutable bindings. `var` destructuring creates mutable bindings.

### Empty and Single-Element Tuples

```
const unit = ()        // empty tuple (unit type)
const single = (42,)   // single-element tuple (trailing comma required)
```

Without the trailing comma, `(42)` is a parenthesized expression, not a tuple.

## Imports and Standard Library

### Import Syntax

Import specific items from a module:

```
import { Circle, Rect, RoundedRect } from geometry
import { lerp, clamp, PI } from math
```

### Geometry Module

Shape constructors that return `Path` handles:

| Function | Description |
|----------|-------------|
| `Circle(cx, cy, r)` | Circle from 4 cubic beziers |
| `Rect(x, y, w, h)` | Rectangle from line segments |
| `RoundedRect(x, y, w, h, r)` | Rectangle with rounded corners |
| `Line(x0, y0, x1, y1)` | Single line segment |
| `Triangle(x0, y0, x1, y1, x2, y2)` | Triangle from 3 points |

All parameters are `f64`. All functions return `Path`.

```
import { Circle, Rect } from geometry

const c = Circle(100.0, 200.0, 50.0)
fill(c, 0xFFFF8800)

const r = Rect(10.0, 10.0, 200.0, 100.0)
fill(r, 0xFF4488FF)
```

### Math Module

Constants and math utilities:

| Item | Description |
|------|-------------|
| `PI` | 3.14159265358979... |
| `TAU` | 6.28318530717958... (2 * PI) |
| `E` | 2.71828182845904... |
| `lerp(a, b, t)` | Linear interpolation |
| `clamp(x, lo, hi)` | Clamp to range |
| `saturate(x)` | Clamp to 0..1 |
| `sign(x)` | -1, 0, or 1 |
| `step(edge, x)` | 0 if x < edge, else 1 |
| `smoothstep(e0, e1, x)` | Smooth Hermite interpolation |
| `degrees(rad)` | Radians to degrees |
| `radians(deg)` | Degrees to radians |

All math module functions operate on `f64`.

### Built-in Math Functions

Always available (no import needed):

| Function | Description |
|----------|-------------|
| `sin(x)`, `cos(x)`, `tan(x)` | Trigonometric |
| `asin(x)`, `acos(x)`, `atan(x)` | Inverse trig |
| `atan2(y, x)` | Two-argument arctangent |
| `sqrt(x)` | Square root |
| `abs(x)` | Absolute value |
| `floor(x)`, `ceil(x)`, `round(x)` | Rounding |
| `exp(x)`, `ln(x)` | Exponential / natural log |
| `log2(x)`, `log10(x)` | Logarithms |
| `fract(x)` | Fractional part |
| `min(a, b)`, `max(a, b)` | Minimum / maximum |
| `fmod(a, b)` | Floating-point modulo |

All built-in math functions operate on `f64`.

## Path Builder

Paths are the core geometry primitive. Create a path, add segments, then draw it.

### Creating Paths

```
var p = Path()
```

### Adding Segments

| Method | Description |
|--------|-------------|
| `p.move_to(x, y)` | Start a new subpath |
| `p.line_to(x, y)` | Straight line to point |
| `p.quad_to(cx, cy, x, y)` | Quadratic bezier (1 control point) |
| `p.cubic_to(c1x, c1y, c2x, c2y, x, y)` | Cubic bezier (2 control points) |
| `p.close()` | Close subpath back to last `move_to` |

All coordinates are `f64`. Path math uses f64 precision internally; conversion to f32 happens at the GPU boundary.

Example — manual triangle:

```
var p = Path()
p.move_to(100.0, 50.0)
p.line_to(150.0, 150.0)
p.line_to(50.0, 150.0)
p.close()
fill(p, 0xFFFF0000)
```

Example — cubic bezier curve:

```
var p = Path()
p.move_to(50.0, 200.0)
p.cubic_to(100.0, 50.0, 200.0, 50.0, 250.0, 200.0)
stroke(p, 3.0, 0xFF000000)
```

## Drawing

Drawing commands submit paths to the scene for rendering. The same path can be drawn multiple times with different styles.

### Fill

```
fill(path, color)
// or via UFCS:
path.fill(color)
```

Fills the path interior using the even-odd fill rule.

- `path` — a `Path` handle
- `color` — packed ARGB `u32`

### Stroke

```
stroke(path, width, color)
// or via UFCS:
path.stroke(width, color)
```

Draws the path outline with a given width.

- `path` — a `Path` handle
- `width` — stroke width in pixels (`f32`)
- `color` — packed ARGB `u32`

Strokes use miter joins with bevel fallback and butt end caps. Strokes are rendered with nonzero winding fill rule to handle self-intersecting outlines correctly.

### Color Format

Colors are packed as `0xAARRGGBB`:

| Component | Bits | Example |
|-----------|------|---------|
| Alpha | 24-31 | `0xFF` = opaque, `0x88` = ~53% |
| Red | 16-23 | `0xFF` = full red |
| Green | 8-15 | `0xFF` = full green |
| Blue | 0-7 | `0xFF` = full blue |

Common colors:

```
0xFFFF0000  // opaque red
0xFF00FF00  // opaque green
0xFF0000FF  // opaque blue
0xFFFFFFFF  // opaque white
0xFF000000  // opaque black
0x80FF8800  // 50% transparent orange
```

### Alpha Blending

Semi-transparent fills composite over previously drawn shapes using source-over alpha blending:

```
fill(background, 0xFF000000)   // opaque black
fill(overlay, 0x8800FF00)      // 50% green overlay — background shows through
```

### Painter's Algorithm

Shapes are composited in draw order — later `fill`/`stroke` calls render on top of earlier ones.

## UFCS (Method Syntax)

Any function can be called with dot syntax. `a.foo(b)` is syntactic sugar for `foo(a, b)`:

```
// These are equivalent:
move_to(p, 100.0, 200.0)
p.move_to(100.0, 200.0)

// These are equivalent:
fill(c, 0xFFFF8800)
c.fill(0xFFFF8800)

// These are equivalent:
push(arr, 42.0)
arr.push(42.0)
```

The dot operator resolves in order:
1. **Struct field access** — if the left side is a struct with a matching field
2. **Enum variant** — if the left side is an enum type
3. **Tuple index** — if followed by a numeric literal (e.g., `.0`, `.1`)
4. **UFCS function call** — otherwise, rewrites `a.method(b)` to `method(a, b)`

This enables method-style chaining:

```
var p = Path()
p.move_to(0.0, 0.0)
p.line_to(100.0, 0.0)
p.line_to(100.0, 100.0)
p.close()
p.fill(0xFFFF0000)
p.stroke(2.0, 0xFF000000)
```
