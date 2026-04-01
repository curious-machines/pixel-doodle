# PDC (Pixel Doodle Code) Language Reference

PDC is a JIT-compiled language for describing vector scenes. It compiles to native code via Cranelift and renders through a tile-based GPU rasterization pipeline.

## Table of Contents

- [Getting Started](#getting-started)
- [Comments](#comments)
- [Types](#types)
- [Variables](#variables)
- [Operators](#operators)
- [Control Flow](#control-flow)
- [Functions](#functions)
- [Structs](#structs)
- [Enums](#enums)
- [Match Statements](#match-statements)
- [Imports and Standard Library](#imports-and-standard-library)
- [Path Builder](#path-builder)
- [Drawing](#drawing)
- [Builtin Variables](#builtin-variables)
- [Type Casting](#type-casting)
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

// Block comments are planned but not yet implemented
```

## Types

### Numeric Types

| Type | Description |
|------|-------------|
| `i32` | 32-bit signed integer |
| `u32` | 32-bit unsigned integer |
| `f32` | 32-bit float |
| `f64` | 64-bit float (default for float literals) |
| `bool` | Boolean (`true` or `false`) |

### Special Types

| Type | Description |
|------|-------------|
| `Path` | Opaque path handle for vector geometry |

### User-Defined Types

Structs and enums can define custom types. See [Structs](#structs) and [Enums](#enums).

## Variables

### Constants

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
```

### Builtin Variables

Builtin variables are provided by the runtime:

```
builtin const width: f32
builtin const height: f32
```

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

```
const area = width * height
const half = total / 2.0
const remainder = 17 % 5    // 2
```

### Comparison

| Operator | Description |
|----------|-------------|
| `==` | Equal |
| `!=` | Not equal |
| `<` | Less than |
| `<=` | Less than or equal |
| `>` | Greater than |
| `>=` | Greater than or equal |

```
if x > 0 {
    // positive
}
```

### Logical

| Operator | Description |
|----------|-------------|
| `&&` | Logical AND (short-circuit) |
| `\|\|` | Logical OR (short-circuit) |
| `!` | Logical NOT |

```
if x > 0 && x < 100 {
    // in range
}
```

### Assignment

| Operator | Description |
|----------|-------------|
| `=` | Assign |

```
var x = 0
x = x + 1
```

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
    i = i + 1
}
```

### For Loop

Iterates over an exclusive integer range:

```
for i in 0..10 {
    // i goes from 0 to 9
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
    i = i + 1
    continue        // skip to next iteration
}
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

## Structs

### Definition

```
struct Vec2 {
    x: f64,
    y: f64,
}
```

### Construction (Named Arguments)

```
const pos = Vec2(x: 100.0, y: 200.0)
```

### Field Access

```
const px = pos.x    // 100.0
const py = pos.y    // 200.0
```

### Using Structs in Functions

```
struct Color {
    r: f64,
    g: f64,
    b: f64,
    a: f64,
}

fn pack_color(c: Color) -> u32 {
    // ... convert to packed ARGB
}
```

## Enums

### Simple Enums (C-style)

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

var s = Shape.Circle(cx: 100.0, cy: 200.0, r: 50.0)
```

Construction can also be positional:

```
var s = Shape.Circle(100.0, 200.0, 50.0)
```

## Match Statements

### Basic Match

```
enum Color { Red, Green, Blue }

var c = Color.Red

match c {
    .Red => {
        fill(circle, 0xFFFF0000)
    },
    .Green => {
        fill(circle, 0xFF00FF00)
    },
    .Blue => {
        fill(circle, 0xFF0000FF)
    },
}
```

### Dot-Shorthand

When the scrutinee type is known, you can omit the enum name with a leading dot:

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
enum Shape {
    Circle(cx: f64, cy: f64, r: f64),
    Rect(x: f64, y: f64, w: f64, h: f64),
}

var s = Shape.Circle(100.0, 200.0, 50.0)

match s {
    .Circle(cx, cy, r) => {
        const path = Circle(cx, cy, r)
        fill(path, 0xFFFF8800)
    },
    .Rect(x, y, w, h) => {
        const path = Rect(x, y, w, h)
        fill(path, 0xFF4488FF)
    },
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

## Imports and Standard Library

### Import Syntax

Import specific items:

```
import { Circle, Rect, RoundedRect } from geometry
import { lerp, clamp, PI } from math
```

### Standard Library Modules

#### `geometry`

Shape constructors that return `Path` handles:

| Function | Description |
|----------|-------------|
| `Circle(cx, cy, r)` | Circle from 4 cubic beziers |
| `Rect(x, y, w, h)` | Rectangle from line segments |
| `RoundedRect(x, y, w, h, r)` | Rectangle with rounded corners |
| `Line(x0, y0, x1, y1)` | Single line segment |
| `Triangle(x0, y0, x1, y1, x2, y2)` | Triangle from 3 points |

```
import { Circle, Rect } from geometry

const c = Circle(100.0, 200.0, 50.0)
fill(c, 0xFFFF8800)

const r = Rect(10.0, 10.0, 200.0, 100.0)
fill(r, 0xFF4488FF)
```

#### `math`

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

### Built-in Math Functions

These are provided by the runtime (not importable, always available):

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

All math functions operate on `f64`.

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

All coordinates are `f64`.

### Example: Manual Path

```
var p = Path()
p.move_to(100.0, 50.0)
p.line_to(150.0, 150.0)
p.line_to(50.0, 150.0)
p.close()
fill(p, 0xFFFF0000)
```

### Example: Cubic Bezier Curve

```
var p = Path()
p.move_to(50.0, 200.0)
p.cubic_to(100.0, 50.0, 200.0, 50.0, 250.0, 200.0)
stroke(p, 3.0, 0xFF000000)
```

## Drawing

Drawing commands submit paths to the scene for rendering. The same path can be drawn multiple times.

### Fill

```
fill(path, color)
```

Fills the path interior using the even-odd fill rule.

- `path` — a `Path` handle
- `color` — packed ARGB `u32` (e.g., `0xFFFF8800` = opaque orange)

### Stroke

```
stroke(path, width, color)
```

Draws the path outline with a given width.

- `path` — a `Path` handle
- `width` — stroke width in pixels (`f32`)
- `color` — packed ARGB `u32`

Strokes use miter joins with bevel fallback and butt end caps.

### Color Format

Colors are packed as `0xAARRGGBB`:

| Component | Bits | Example |
|-----------|------|---------|
| Alpha | 24-31 | `0xFF` = opaque, `0x88` = ~53% |
| Red | 16-23 | `0xFF` = full red |
| Green | 8-15 | `0xFF` = full green |
| Blue | 0-7 | `0xFF` = full blue |

Alpha blending is supported. Semi-transparent fills composite over previously drawn shapes.

```
fill(background, 0xFF000000)   // opaque black
fill(overlay, 0x8800FF00)      // 53% green overlay
```

### Painter's Algorithm

Shapes are composited in draw order — later `fill`/`stroke` calls render on top of earlier ones.

## Builtin Variables

These are declared with `builtin const` and provided by the runtime:

```
builtin const width: f32    // window width in pixels
builtin const height: f32   // window height in pixels
```

## Type Casting

Cast between types using the type name as a constructor:

```
const i: i32 = 42
const f = f64(i)       // i32 to f64: 42.0
const s = f32(f)       // f64 to f32: 42.0
const n = i32(3.7)     // f64 to i32: 3 (truncates)
```

## UFCS (Method Syntax)

Any function can be called with dot syntax. `a.foo(b)` is sugar for `foo(a, b)`:

```
// These are equivalent:
move_to(p, 100.0, 200.0)
p.move_to(100.0, 200.0)
```

The dot operator resolves in order:
1. **Struct field access** — if the left side is a struct with a matching field
2. **UFCS function call** — otherwise, rewrites to a regular function call

This enables method-style chaining:

```
var p = Path()
p.move_to(0.0, 0.0)
p.line_to(100.0, 0.0)
p.line_to(100.0, 100.0)
p.close()
```
