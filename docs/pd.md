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
- Additional built-in parameters `px: u32, py: u32, sample_index: u32` are available for pixel coordinates and progressive sampling.
- `time: f64` provides elapsed time in seconds for animation. Kernels that declare this parameter automatically render continuously.
- **User-defined parameters** — any parameter whose name is not a built-in (e.g., `max_iter: u32`, `threshold: f64`) is a user-defined argument that must be supplied via the `run` statement in the `.pdp` file. See the [PDP reference](pdp.md#user-defined-kernel-arguments) for details.
- Simulation kernels have built-in names `px`, `py`, `width`, `height`; all other params are user-defined.
- **Return type** is always `u32` (an ARGB pixel value).
- **`emit`** evaluates an expression and uses the result as the pixel color.
- Comments start with `//` and run to end of line.
- Statements end with `;`.

## Types

### Scalars

| Type   | Description                    |
|--------|--------------------------------|
| `f32`  | 32-bit floating point          |
| `f64`  | 64-bit floating point          |
| `i8`   | 8-bit signed integer           |
| `u8`   | 8-bit unsigned integer         |
| `i16`  | 16-bit signed integer          |
| `u16`  | 16-bit unsigned integer        |
| `i32`  | 32-bit signed integer          |
| `u32`  | 32-bit unsigned integer        |
| `i64`  | 64-bit signed integer          |
| `u64`  | 64-bit unsigned integer        |
| `bool` | Boolean (true/false)           |

### Vectors

| Type       | Description                              |
|------------|------------------------------------------|
| `vec2<T>`  | 2-component vector parameterized by `T`  |
| `vec3<T>`  | 3-component vector parameterized by `T`  |
| `vec4<T>`  | 4-component vector parameterized by `T`  |

For example, `vec2<f64>` is a 2-component vector of f64, `vec3<f32>` is a 3-component vector of f32.

### Matrices

| Type       | Description                              |
|------------|------------------------------------------|
| `mat2<T>`  | 2x2 matrix parameterized by `T`         |
| `mat3<T>`  | 3x3 matrix parameterized by `T`         |
| `mat4<T>`  | 4x4 matrix parameterized by `T`         |

### Arrays

| Type          | Description                           |
|---------------|---------------------------------------|
| `array<T; N>` | Fixed-size array of `N` elements of type `T` |

### Structs

User-defined value types (see the Structs section below).

There are no implicit conversions between scalar types. Use `as` casts or conversion functions to move between types. Vec and matrix types support arithmetic operators with automatic dispatch (see Vector types below).

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
1.0f32      // explicitly f32
1.0f64      // explicitly f64
42          // bare integer (context-dependent: matches the surrounding integer type)
256u32      // explicitly u32
42i32       // explicitly i32
42u8        // explicitly u8
42i16       // explicitly i16
42u64       // explicitly u64
true        // bool
false       // bool
```

Bare integers (without suffix or decimal point) are resolved by context: they adopt the integer type of the surrounding expression (e.g., `u32` in a u32 expression, `i32` in an i32 expression, `f64` in an f64 expression). Use a type suffix when the context is ambiguous.

## Operators

### Arithmetic (all numeric scalars, vec2, vec3, vec4)

| Operator | Description     |
|----------|-----------------|
| `+`      | Addition        |
| `-`      | Subtraction     |
| `*`      | Multiplication  |
| `/`      | Division        |
| `%`      | Remainder (scalar only) |

Arithmetic works on all numeric scalar types (f32, f64, i8, u8, i16, u16, i32, u32, i64, u64) and on vector types (vec2, vec3, vec4). When both operands are the same vec type, `+`, `-`, `*`, `/` apply component-wise. Mixed scalar-vector multiply is also supported: `scalar * vec` or `vec * scalar` scales each component. `vec / scalar` divides each component by the scalar.

Matrix multiply is supported: `mat * vec -> vec` transforms a vector, `mat * mat -> mat` composes matrices.

```
let sum = pos + offset;       // vec2 + vec2 -> vec2
let scaled = 2.0 * direction; // f64 * vec3 -> vec3
let half = pos / 2.0;         // vec2 / f64 -> vec2
let transformed = m * pos;    // mat2 * vec2 -> vec2
```

### Unary

| Operator | Description   | Types            |
|----------|---------------|------------------|
| `-`      | Negate        | all numeric scalars, vec2, vec3, vec4 |
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

Operands must be the same type (any numeric scalar). The result is always `bool`.

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
let small = big_value as u8;
let wide = narrow as f64;
let signed = x as i32;
```

Any scalar-to-scalar cast is valid, except casts involving `bool`. For example: `f64 as i32`, `u8 as f64`, `i16 as u32`, `f32 as f64` are all permitted.

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
let pos = vec2(x, y);                     // infers vec2<f64> from f64 args
let color = vec3(1.0, 0.5, 0.0);          // vec3<f64>
let rgba = vec4(1.0, 0.5, 0.0, 1.0);      // vec4<f64>
let pos_f32 = vec2(1.0f32, 2.0f32);       // vec2<f32>
```

`vec2()` takes two arguments, `vec3()` takes three, `vec4()` takes four. The element type is inferred from the arguments. Type annotations use the parameterized form: `vec2<f64>`, `vec3<f32>`, etc.

### Component access

Use `.x`, `.y`, `.z`, `.w` to extract components (the component type matches the vector's element type):

```
let px = pos.x;
let py = pos.y;
let blue = color.z;   // vec3 and vec4 only
let alpha = rgba.w;   // vec4 only
```

`vec2` has `.x` and `.y`. `vec3` has `.x`, `.y`, and `.z`. `vec4` has `.x`, `.y`, `.z`, and `.w`. Accessing a component beyond the vector's size is an error.

### If-else with vectors

Both branches must be the same vec type:

```
let chosen = if d < 0.0 { inside_color } else { outside_color };
```

### While loops with vectors

Vec types work as loop-carried variables:

```
while pos: vec2<f64> = vec2(0.0, 0.0), iter: u32 = 0 {
    break_if iter >= 100u32;
    yield pos + delta, iter + 1u32;
}
```

## Structs

User-defined value types group related data together:

```
struct Point {
    x: f64,
    y: f64,
}

struct Color {
    r: f32,
    g: f32,
    b: f32,
    a: f32,
}
```

### Construction

Structs are constructed by name with all fields specified:

```
let origin = Point { x: 0.0, y: 0.0 };
let red = Color { r: 1.0f32, g: 0.0f32, b: 0.0f32, a: 1.0f32 };
```

### Field access

Use dot notation to access fields:

```
let px = origin.x;
let py = origin.y;
```

Structs are value types — there are no references or pointers. Assigning a struct copies it.

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
| `length(...)`               | f64 x f64 -> f64 (`sqrt(x*x+y*y)`)    | vec -> scalar (magnitude; vec2, vec3, vec4) |
| `distance(...)`             | f64 x4 -> f64                          | vec x vec -> scalar (`length(a - b)`; vec2, vec3, vec4) |
| `mix(a, b, t)`              | f64 x f64 x f64 -> f64                | vec x vec x f64 -> vec                  |

```
let d = length(pos);                    // vec2 -> f64
let n = normalize(direction);           // vec3 -> vec3
let closest = min(pos_a, pos_b);        // vec2, vec2 -> vec2
let blended = mix(color_a, color_b, t); // vec3, vec3, f64 -> vec3
```

### Vector-only functions

| Function              | Description                                      |
|-----------------------|--------------------------------------------------|
| `vec2(x, y)`          | Construct vec2 (element type inferred from args) |
| `vec3(x, y, z)`       | Construct vec3 (element type inferred from args) |
| `vec4(x, y, z, w)`    | Construct vec4 (element type inferred from args) |
| `dot(a, b)`           | Dot product (vec x vec -> scalar)                |
| `normalize(v)`        | Normalize to unit length (vec -> vec)            |
| `cross(a, b)`         | Cross product (vec3 x vec3 -> vec3)              |

`dot`, `normalize`, and `cross` work with any numeric element type.

```
let pos = vec2(x, y);
let d = dot(normal, light_dir);
let n = cross(edge1, edge2);
```

### Matrix functions

| Function                     | Description                              |
|------------------------------|------------------------------------------|
| `mat2(col0, col1)`           | Construct mat2 from two column vectors   |
| `mat3(col0, col1, col2)`     | Construct mat3 from three column vectors |
| `mat4(col0, col1, col2, col3)` | Construct mat4 from four column vectors |
| `transpose(m)`               | Matrix transpose                         |
| `col(m, index)`              | Extract column as vector                 |

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
fn circle_sdf(p: vec2<f64>, center: vec2<f64>, r: f64) -> f64 {
    return length(p - center) - r;
}

fn scene_sdf(p: vec2<f64>) -> f64 {
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

    while ray_pos: vec2<f64> = pos, total_d = 0.0, steps: u32 = 0 {
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
