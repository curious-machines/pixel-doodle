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
| `vec2` | 2-component vector (2× f64)   |
| `vec3` | 3-component vector (3× f64)   |

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

## Vector operations

### Construction

```
pos: vec2 = make_vec2 x y
color: vec3 = make_vec3 r g b
```

Both `make_vec2` and `make_vec3` take f64 components and produce a vec2 or vec3.

### Component extraction

```
px: f64 = extract_x pos
py: f64 = extract_y pos
pz: f64 = extract_z color    # vec3 only
```

Extracts a single f64 component from a vector. `extract_z` is only valid for vec3.

### Component-wise arithmetic (vec op vec -> vec)

| Op         | Description              |
|------------|--------------------------|
| `vec_add`  | Component-wise addition  |
| `vec_sub`  | Component-wise subtraction |
| `vec_mul`  | Component-wise multiplication |
| `vec_div`  | Component-wise division  |
| `vec_min`  | Component-wise minimum   |
| `vec_max`  | Component-wise maximum   |

Both operands must be the same vec type. The result is the same vec type.

```
sum: vec2 = vec_add a b
diff: vec3 = vec_sub position offset
```

### Scalar-vector multiply (f64 × vec -> vec)

```
scaled: vec2 = vec_scale factor pos
```

The first operand must be f64, the second must be vec2 or vec3. The result matches the vec type.

### Unary vector operations (vec -> vec)

| Op              | Description                        |
|-----------------|------------------------------------|
| `vec_neg`       | Negate each component              |
| `vec_abs`       | Absolute value of each component   |
| `vec_normalize` | Normalize to unit length           |

```
flipped: vec2 = vec_neg pos
unit: vec3 = vec_normalize direction
```

### Reduction operations (vec -> f64)

| Op           | Description                          |
|--------------|--------------------------------------|
| `vec_dot`    | Dot product of two vectors           |
| `vec_length` | Length (magnitude) of a vector       |

```
d: f64 = vec_dot a b
len: f64 = vec_length pos
```

Both operands of `vec_dot` must be the same vec type.

### Cross product (vec3 × vec3 -> vec3)

```
normal: vec3 = vec_cross a b
```

Both operands must be vec3.

### Select with vectors

`select` works with vec types — both branches must be the same vec type:

```
chosen: vec2 = select cond pos_a pos_b
```

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

## Inline functions

Inline functions define reusable blocks of computation that are expanded at each call site during parsing. The IR remains completely flat — backends never see function calls.

### Syntax

```
inline smin(a: f64, b: f64, k: f64) -> f64 {
    ka: f64 = mul k a
    neg_ka: f64 = neg ka
    kb: f64 = mul k b
    neg_kb: f64 = neg kb
    ea: f64 = exp neg_ka
    eb: f64 = exp neg_kb
    esum: f64 = add ea eb
    log_sum: f64 = log esum
    neg_log: f64 = neg log_sum
    result: f64 = div neg_log k
    return result
}

kernel my_kernel(x: f64, y: f64) -> u32 {
    # ... compute d_a and d_b ...
    d: f64 = smin d_a d_b 12.0
    # ... rest ...
    emit pixel
}
```

### Rules

- **`inline`** keyword introduces a definition. Inline functions must appear before the kernel.
- **`return`** names the variable whose value becomes the result (distinct from `emit` in kernels).
- Call syntax uses the function name as the instruction: `d: f64 = smin a b k`.
- Arguments can be variable names or inline literals, matching the declared parameter types.
- Inline functions have **isolated scope** — they can only reference their own parameters and locally defined variables, not outer kernel variables.
- `while` loops are allowed inside inline function bodies.
- Each call site generates uniquely prefixed variables (e.g., `__smin_0_ka`, `__smin_1_ka`) to avoid collisions.

### Nesting

- Inline functions **can call other inline functions**, as long as the callee is defined before the caller.
- Inline functions containing `while` loops cannot be called from within another `while` loop body (the IR's while body is flat statements only).

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

Note: the example above shows the intent. PDL does not support nested expressions like `sqrt(add cx2 cy2)` or reassignment of `intensity` — every value needs its own unique name and each operation is a separate statement. See `examples/sdf/sdf/sdf.pdl` for the working version.
