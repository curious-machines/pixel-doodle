# PDL — Pixel Doodle Language

PDL is a small language for writing per-pixel compute kernels. It uses SSA form (every value assigned once), explicit types, and a flat sequence of named assignments. Kernels are JIT-compiled and run in parallel across tiles of the output image.

## Kernel structure

A kernel declares its name, parameters, return type, a body of statements, and an `emit` that produces the final value.

```
kernel my_kernel(x: f64, y: f64) -> u32 {
    # body: sequence of statements
    ...
    emit result_var
}
```

- **Parameters** `x` and `y` receive normalized pixel coordinates (typically mapped to a viewport range like -2.0 to 2.0).
- **Return type** is always `u32` (an ARGB pixel value).
- **`emit`** names the variable whose value becomes the pixel color.
- Comments start with `#` and run to end of line.

## Types

| Type   | Description                    |
|--------|--------------------------------|
| `f64`  | 64-bit floating point          |
| `u32`  | 32-bit unsigned integer        |
| `bool` | Boolean (true/false)           |

There are no implicit conversions. Use explicit conversion ops to move between types.

## Statements

Every statement has the form:

```
name: type = op operands...
```

This binds a new variable `name` of the given `type` to the result of `op`. Each variable name can only be assigned once (SSA).

Operands can be variable names or inline numeric/boolean literals:

```
scaled: f64 = mul x 255.0
clamped: u32 = add count 1
```

## Operations

### Constants

```
value: f64 = const 3.14
count: u32 = const 42
flag: bool = const true
```

### Arithmetic (f64, u32)

| Op    | Description     | Types      |
|-------|-----------------|------------|
| `add` | Addition        | f64, u32   |
| `sub` | Subtraction     | f64, u32   |
| `mul` | Multiplication  | f64, u32   |
| `div` | Division        | f64, u32   |
| `rem` | Remainder       | f64, u32   |
| `min` | Minimum         | f64, u32   |
| `max` | Maximum         | f64, u32   |

```
sum: f64 = add a b
d: f64 = min d_circle d_box
```

### Unary (f64 unless noted)

| Op      | Description              | Types      |
|---------|--------------------------|------------|
| `neg`   | Negate                   | f64, u32   |
| `abs`   | Absolute value           | f64, u32   |
| `sqrt`  | Square root              | f64        |
| `floor` | Floor                    | f64        |
| `ceil`  | Ceiling                  | f64        |
| `round` | Round to nearest integer | f64        |
| `trunc` | Truncate toward zero     | f64        |
| `fract` | Fractional part (x - floor(x)) | f64  |
| `not`   | Logical NOT              | bool       |

```
dist: f64 = sqrt sum_sq
positive: f64 = abs val
```

### Trigonometry (f64)

| Op      | Description              |
|---------|--------------------------|
| `sin`   | Sine                     |
| `cos`   | Cosine                   |
| `tan`   | Tangent                  |
| `asin`  | Arcsine                  |
| `acos`  | Arccosine                |
| `atan`  | Arctangent (unary)       |
| `atan2` | Arctangent of y/x (binary) |

```
angle: f64 = atan2 dy dx
sx: f64 = sin angle
cx: f64 = cos angle
```

### Exponential & Logarithmic (f64)

| Op      | Description              |
|---------|--------------------------|
| `exp`   | e^x                      |
| `exp2`  | 2^x                      |
| `log`   | Natural logarithm (ln)   |
| `log2`  | Base-2 logarithm         |
| `log10` | Base-10 logarithm        |
| `pow`   | Power (binary: x^y)      |

```
falloff: f64 = exp neg_dist_sq
brightness: f64 = pow base exponent
```

### Bitwise (u32)

| Op        | Description         |
|-----------|---------------------|
| `bit_and` | Bitwise AND         |
| `bit_or`  | Bitwise OR          |
| `bit_xor` | Bitwise XOR         |
| `shl`     | Shift left          |
| `shr`     | Shift right (logical) |

```
masked: u32 = bit_and color 0xFF
shifted: u32 = shl r 16
```

Note: inline literals in bitwise ops must be integers (e.g. `16`, not `16.0`).

### Logical (bool)

| Op    | Description |
|-------|-------------|
| `and` | Logical AND |
| `or`  | Logical OR  |

```
both: bool = and cond_a cond_b
```

### Comparison (produces bool)

| Op   | Description          |
|------|----------------------|
| `eq` | Equal                |
| `ne` | Not equal            |
| `lt` | Less than            |
| `le` | Less than or equal   |
| `gt` | Greater than         |
| `ge` | Greater than or equal |

Operands must be the same type (f64 or u32). The result is always `bool`.

```
inside: bool = lt dist radius
```

### Conversions

| Op          | From | To  |
|-------------|------|-----|
| `f64_to_u32`| f64  | u32 |
| `u32_to_f64`| u32  | f64 |

```
channel: u32 = f64_to_u32 intensity
coord: f64 = u32_to_f64 pixel_count
```

### Select

Conditional value selection (ternary):

```
result: f64 = select condition then_var else_var
```

`condition` must be `bool`. Both branches must have the same type, which becomes the result type.

### Pack ARGB

Packs three u32 channel values (0–255) into a single ARGB pixel with full alpha:

```
pixel: u32 = pack_argb r g b
```

Equivalent to `0xFF000000 | (r << 16) | (g << 8) | b`.

## Control flow

### While loops

Structured loops with explicit loop-carried variables:

```
while carry(var1: type = init1, var2: type = init2) {
    # condition body — compute whether to continue
    cont: bool = lt var1 limit
    cond cont

    # loop body — compute next values
    next1: type = add var1 step
    next2: type = mul var2 factor
    yield next1 next2
}
# var1 and var2 are live here with their final values
```

- **`carry`** declares loop-carried variables with their initial values.
- **`cond`** names a bool variable; the loop continues while it is true.
- **`yield`** provides the updated values for carry variables (must match count and types).
- After the loop, carry variables remain in scope with their final values.
- Variables defined inside the loop body are **not** visible after the loop.

## Complete example

```
# SDF scene: circle + box with distance-based glow
kernel sdf(x: f64, y: f64) -> u32 {
    cx: f64 = add x 0.5
    cx2: f64 = mul cx cx
    cy2: f64 = mul y y
    cd: f64 = sqrt (add cx2 cy2)  # note: no nesting — use a temp
    d_circle: f64 = sub cd 0.4

    bx: f64 = sub x 0.5
    dx: f64 = sub (abs bx) 0.3
    dy: f64 = sub (abs y) 0.25
    d_box: f64 = max dx dy

    d: f64 = min d_circle d_box

    scaled: f64 = mul d 4.0
    raw: f64 = sub 1.0 scaled
    intensity: f64 = max raw 0.0
    intensity: f64 = min intensity 1.0  # clamp to [0, 1]

    r_f: f64 = mul intensity 100.0
    g_f: f64 = mul intensity 200.0
    b_f: f64 = mul intensity 255.0
    r: u32 = f64_to_u32 r_f
    g: u32 = f64_to_u32 g_f
    b: u32 = f64_to_u32 b_f
    pixel: u32 = pack_argb r g b
    emit pixel
}
```

Note: the example above shows the intent. PDL does not support nested expressions like `sqrt(add cx2 cy2)` or reassignment of `intensity` — every value needs its own unique name and each operation is a separate statement. See `examples/sdf.pdl` for the working version.
