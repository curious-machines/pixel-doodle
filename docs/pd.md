# PD — Pixel Doodle (Higher-Level Language)

PD is a higher-level language for writing per-pixel compute kernels. It compiles to the same Kernel IR as PDIR but uses conventional infix syntax, expression nesting, and type inference for local variables. Kernels are JIT-compiled and run in parallel across tiles of the output image.

## Includes

Use `use` to import function definitions from other PD files:

```
use "helpers.pd";
use "shapes.pd";

kernel my_kernel(x: f64, y: f64) -> u32 {
    let d = sd_circle(vec2(x, y), 0.5);
    ...
}
```

- Paths are relative to the current file's directory.
- Included files can contain `use` and `fn` definitions, but not a `kernel`.
- Nested includes are supported (A uses B, B uses C).
- Each file is only processed once — duplicate `use` of the same file is silently skipped.
- Unused functions from included files have no runtime cost (they are never inlined into the IR).

## Kernel structure

A kernel declares its name, parameters, return type, a body of statements, and an `emit` that produces the final value.

```
kernel my_kernel(x: f64, y: f64) -> u32 {
    // body: sequence of statements
    ...
    emit result_expr;
}
```

- **Parameters** `x` and `y` receive normalized pixel coordinates (typically mapped to a viewport range like -2.0 to 2.0).
- Additional parameters `px: u32, py: u32, sample_index: u32` are available for pixel coordinates and progressive sampling.
- `time: f64` provides elapsed time in seconds for animation. Kernels that declare this parameter automatically render continuously.
- **Return type** is always `u32` (an ARGB pixel value).
- **`emit`** evaluates an expression and uses the result as the pixel color.
- Comments start with `//` and run to end of line.
- Statements end with `;`.

## Types

| Type   | Description                    |
|--------|--------------------------------|
| `f64`  | 64-bit floating point          |
| `u32`  | 32-bit unsigned integer        |
| `bool` | Boolean (true/false)           |
| `vec2` | 2-component vector (2× f64)   |
| `vec3` | 3-component vector (3× f64)   |

There are no implicit conversions between scalar types. Use `as` casts or conversion functions to move between types. Vec types support arithmetic operators with automatic dispatch (see Vector types below).

## Variables

Variables are declared with `let`. Type annotations are optional when the type can be inferred:

```
let r = x * 255.0;          // inferred as f64
let count: u32 = 0;         // explicit type
let flag = iter >= 256u32;   // inferred as bool
```

## Literals

```
3.14        // f64 (any number with a decimal point)
42          // bare integer (context-dependent: f64 or u32)
256u32      // explicitly u32
true        // bool
false       // bool
```

Bare integers (without suffix or decimal point) are resolved by context: they become `u32` in u32 expressions and `f64` in f64 expressions. Use the `u32` suffix when the context is ambiguous.

## Operators

### Arithmetic (f64, u32, vec2, vec3)

| Operator | Description     |
|----------|-----------------|
| `+`      | Addition        |
| `-`      | Subtraction     |
| `*`      | Multiplication  |
| `/`      | Division        |
| `%`      | Remainder (f64, u32 only) |

When both operands are the same vec type, `+`, `-`, `*`, `/` apply component-wise. Mixed scalar-vector multiply is also supported: `f64 * vec` or `vec * f64` scales each component. `vec / f64` divides each component by the scalar.

```
let sum = pos + offset;       // vec2 + vec2 -> vec2
let scaled = 2.0 * direction; // f64 * vec3 -> vec3
let half = pos / 2.0;         // vec2 / f64 -> vec2
```

### Unary

| Operator | Description   | Types            |
|----------|---------------|------------------|
| `-`      | Negate        | f64, u32, vec2, vec3 |
| `!`      | Logical NOT   | bool             |

### Comparison (produces bool)

| Operator | Description           |
|----------|-----------------------|
| `==`     | Equal                 |
| `!=`     | Not equal             |
| `<`      | Less than             |
| `<=`     | Less than or equal    |
| `>`      | Greater than          |
| `>=`     | Greater than or equal |

Operands must be the same type (f64 or u32). The result is always `bool`.

### Logical (bool)

| Operator | Description |
|----------|-------------|
| `&&`     | Logical AND |
| `\|\|`   | Logical OR  |

### Bitwise (u32)

| Operator | Description         |
|----------|---------------------|
| `&`      | Bitwise AND         |
| `\|`     | Bitwise OR          |
| `^`      | Bitwise XOR         |
| `~`      | Bitwise NOT         |
| `<<`     | Shift left          |
| `>>`     | Shift right (logical) |

### Operator precedence (highest to lowest)

| Precedence | Operators                  |
|------------|----------------------------|
| 1 (highest)| `as` (type cast)           |
| 2          | `*` `/` `%`                |
| 3          | `+` `-`                    |
| 4          | `<<` `>>`                  |
| 5          | `&`                        |
| 6          | `^`                        |
| 7          | `\|`                       |
| 8          | `<` `<=` `>` `>=`          |
| 9          | `==` `!=`                  |
| 10         | `&&`                       |
| 11 (lowest)| `\|\|`                     |

Parentheses `()` can override precedence.

## Type casts

Use `as` to convert between numeric types:

```
let channel = intensity as u32;
let coord = pixel_count as f64;
```

Only `f64 <-> u32` casts are supported.

## If-else expressions

`if`/`else` is an expression that returns a value:

```
let color = if is_inside { 0u32 } else { bright_color };
emit if d < 0.0 { surface_color } else { background };
```

Chained conditions use `else if`:

```
let v = if x > 0.5 { 1.0 }
        else if x > 0.0 { 0.5 }
        else { 0.0 };
```

Both/all branches must produce the same type. The braces `{}` around each branch are required (except after `else if`, which starts a new if-expression).

## Vector types

### Construction

```
let pos = vec2(x, y);
let color = vec3(1.0, 0.5, 0.0);
```

`vec2()` takes two f64 arguments, `vec3()` takes three.

### Component access

Use `.x`, `.y`, `.z` to extract components as f64:

```
let px = pos.x;
let py = pos.y;
let blue = color.z;   // vec3 only
```

`vec2` has `.x` and `.y`. `vec3` has `.x`, `.y`, and `.z`. Accessing `.z` on a `vec2` is an error.

### If-else with vectors

Both branches must be the same vec type:

```
let chosen = if d < 0.0 { inside_color } else { outside_color };
```

### While loops with vectors

Vec types work as loop-carried variables:

```
while pos: vec2 = vec2(0.0, 0.0), iter: u32 = 0 {
    break_if iter >= 100u32;
    yield pos + delta, iter + 1u32;
}
```

## Control flow

### While loops

Structured loops with explicit loop-carried variables:

```
while zx = 0.0, zy = 0.0, iter: u32 = 0 {
    break_if zx * zx + zy * zy > 4.0 || iter >= 256u32;
    yield zx * zx - zy * zy + x,
          2.0 * zx * zy + y,
          iter + 1u32;
}
// zx, zy, iter are live here with their final values
```

- **Carry variables** are declared inline after `while` with optional type annotations and initial values, separated by commas.
- **`break_if`** specifies the exit condition — the loop exits when the expression is true.
- **`yield`** provides the updated values for carry variables (must match count and types), separated by commas.
- After the loop, carry variables remain in scope with their final values.
- Variables defined inside the loop body (via `let`) are **not** visible after the loop.

## Functions

Functions define reusable blocks of computation. They are inlined at each call site during lowering — the IR remains completely flat.

```
fn circle_sdf(px: f64, py: f64, cx: f64, cy: f64, r: f64) -> f64 {
    return length(px - cx, py - cy) - r;
}
```

- **`fn`** keyword introduces a function definition. Functions must appear before the kernel.
- **`return`** provides the result value (distinct from `emit` in kernels).
- Functions are called with conventional syntax: `circle_sdf(x, y, 0.0, 0.0, 0.5)`.
- Function bodies can contain `let` bindings, `while` loops, and `return`.
- Functions have **isolated scope** — they can only reference their own parameters and locally defined variables, not outer kernel variables.
- Trailing commas in argument lists are allowed.

## Builtin functions

### Unary math (f64 -> f64)

| Function | Description              |
|----------|--------------------------|
| `sqrt(x)` | Square root             |
| `floor(x)` | Floor                  |
| `ceil(x)` | Ceiling                 |
| `round(x)` | Round to nearest integer |
| `trunc(x)` | Truncate toward zero    |
| `fract(x)` | Fractional part (x - floor(x)) |
| `sin(x)` | Sine                     |
| `cos(x)` | Cosine                   |
| `tan(x)` | Tangent                  |
| `asin(x)` | Arcsine                 |
| `acos(x)` | Arccosine               |
| `atan(x)` | Arctangent (unary)      |
| `exp(x)` | e^x                      |
| `exp2(x)` | 2^x                     |
| `log(x)` | Natural logarithm (ln)   |
| `log2(x)` | Base-2 logarithm        |
| `log10(x)` | Base-10 logarithm      |

### Binary math (f64 x f64 -> f64, unless noted)

| Function       | Description                |
|----------------|----------------------------|
| `atan2(y, x)`  | Arctangent of y/x          |
| `pow(base, exp)` | Power                    |
| `hash(a, b)`   | Pseudo-random (u32 x u32 -> u32) |

### Overloaded functions (scalar and vector)

These functions work on both scalar and vec types:

| Function                    | Scalar form                            | Vector form                             |
|-----------------------------|----------------------------------------|-----------------------------------------|
| `abs(x)`                    | f64 -> f64                             | vec -> vec (component-wise)             |
| `min(a, b)`                 | f64 x f64 -> f64, u32 x u32 -> u32    | vec x vec -> vec (component-wise)       |
| `max(a, b)`                 | f64 x f64 -> f64, u32 x u32 -> u32    | vec x vec -> vec (component-wise)       |
| `length(...)`               | f64 x f64 -> f64 (`sqrt(x*x+y*y)`)    | vec -> f64 (magnitude)                  |
| `distance(...)`             | f64 x4 -> f64                          | vec x vec -> f64 (`length(a - b)`)      |
| `mix(a, b, t)`              | f64 x f64 x f64 -> f64                | vec x vec x f64 -> vec                  |

```
let d = length(pos);                    // vec2 -> f64
let n = normalize(direction);           // vec3 -> vec3
let closest = min(pos_a, pos_b);        // vec2, vec2 -> vec2
let blended = mix(color_a, color_b, t); // vec3, vec3, f64 -> vec3
```

### Vector-only functions

| Function              | Description                          |
|-----------------------|--------------------------------------|
| `vec2(x, y)`          | Construct vec2 from two f64          |
| `vec3(x, y, z)`       | Construct vec3 from three f64        |
| `dot(a, b)`           | Dot product (vec x vec -> f64)       |
| `normalize(v)`        | Normalize to unit length (vec -> vec)|
| `cross(a, b)`         | Cross product (vec3 x vec3 -> vec3)  |

```
let pos = vec2(x, y);
let d = dot(normal, light_dir);
let n = cross(edge1, edge2);
```

### Convenience math (f64)

| Function                  | Description                                |
|---------------------------|--------------------------------------------|
| `clamp(x, lo, hi)`       | `min(max(x, lo), hi)`                      |
| `saturate(x)`             | `clamp(x, 0, 1)`                           |
| `smoothstep(e0, e1, x)`   | Hermite interpolation with clamped t       |
| `step(edge, x)`           | `0.0` if `x < edge`, `1.0` otherwise       |

### Color (f64 channels -> u32 pixel)

| Function           | Description                                  |
|--------------------|----------------------------------------------|
| `rgb(r, g, b)`     | Scales [0,1] to [0,255], packs ARGB          |
| `rgb255(r, g, b)`  | Packs f64 values (already 0-255) as ARGB     |
| `gray(v)`          | `rgb(v, v, v)`                               |
| `gray255(v)`       | `rgb255(v, v, v)`                             |
| `pack_argb(r, g, b)` | Packs three u32 channels (0-255) as ARGB   |

### Conversions

| Function        | Description                     |
|-----------------|---------------------------------|
| `f64_to_u32(x)` | Convert f64 to u32             |
| `u32_to_f64(x)` | Convert u32 to f64             |
| `norm(x)`       | Normalize u32 to f64 in [0, 1] |

### Selection

| Function                    | Description                                    |
|-----------------------------|------------------------------------------------|
| `select(cond, then, else)`  | Conditional value selection (both branches same type) |

Note: `if`/`else` expressions are generally preferred over `select`.

## Complete example

```
// SDF scene: circle + box with distance-based glow
kernel sdf(x: f64, y: f64) -> u32 {
    let d_circle = length(x + 0.5, y) - 0.4;
    let d_box = max(abs(x - 0.5) - 0.3, abs(y) - 0.25);
    let d = min(d_circle, d_box);
    let intensity = saturate(1.0 - d * 4.0);
    emit rgb255(intensity * 100.0, intensity * 200.0, intensity * 255.0);
}
```

## Example with functions, loops, and vec types

```
fn circle_sdf(p: vec2, center: vec2, r: f64) -> f64 {
    return length(p - center) - r;
}

fn scene_sdf(p: vec2) -> f64 {
    let d1 = circle_sdf(p, vec2(-0.3, 0.0), 0.4);
    let d2 = circle_sdf(p, vec2(0.3, 0.0), 0.3);
    return min(d1, d2);
}

kernel ray_march_ao(x: f64, y: f64, px: u32, py: u32, sample_index: u32) -> u32 {
    let seed = hash(px, py);
    let seed2 = hash(seed, sample_index);
    let theta = norm(seed2) * 6.283185307179586;
    let dir = vec2(cos(theta), sin(theta));
    let pos = vec2(x, y);

    while ray_pos: vec2 = pos, total_d = 0.0, steps: u32 = 0 {
        let d = abs(scene_sdf(ray_pos));
        let new_total = total_d + d;
        break_if d < 0.001 || new_total > 2.0;
        yield ray_pos + dir * d, new_total, steps + 1u32;
    }

    let ao = saturate(total_d / 2.0);
    let in_shape = scene_sdf(pos) <= 0.0;
    let brightness = if in_shape { min(ao * 200.0, 255.0) } else { 20.0 };
    emit rgb255(brightness, brightness, brightness);
}
```

## PD vs PDIR

PD and PDIR both compile to the same Kernel IR. PD is designed for readability and ease of writing; PDIR is the lower-level SSA text format.

| Feature              | PD                         | PDIR                         |
|----------------------|----------------------------|------------------------------|
| Syntax style         | Infix expressions          | `name: type = op args`       |
| Type annotations     | Optional on `let`          | Required on every statement  |
| Nesting              | Expressions nest freely    | Flat — one op per statement  |
| Functions            | `fn` with conventional calls | `inline` with op-style calls |
| Comments             | `//`                       | `#`                          |
| Emit                 | `emit expr;`               | `emit var`                   |
| Statement terminator | `;`                        | None (newline-delimited)     |
| While syntax         | `while x = init, ... {}`   | `while carry(x: t = init) {}` |
| Break condition      | `break_if expr;`           | `cond bool_var`              |
