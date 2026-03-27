# PDIR — Pixel Doodle Intermediate Representation

PDIR is the textual representation of the pixel-doodle kernel IR. It uses SSA form (every value assigned once), explicit types, and a flat sequence of named assignments. Kernels are JIT-compiled and run in parallel across tiles of the output image.

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

Parameterized by element type. The element type must always be specified — bare `vec2` is not valid.

| Type       | Description                                |
|------------|--------------------------------------------|
| `vec2<T>`  | 2-component vector (e.g., `vec2<f64>`)     |
| `vec3<T>`  | 3-component vector (e.g., `vec3<f32>`)     |
| `vec4<T>`  | 4-component vector (e.g., `vec4<u8>`)      |

`T` can be any numeric scalar type (not `bool`).

### Matrices

Column-major storage, parameterized by element type.

| Type       | Description                                |
|------------|--------------------------------------------|
| `mat2<T>`  | 2x2 matrix (e.g., `mat2<f64>`)            |
| `mat3<T>`  | 3x3 matrix (e.g., `mat3<f32>`)            |
| `mat4<T>`  | 4x4 matrix (e.g., `mat4<f64>`)            |

### Arrays

Fixed-size arrays with explicit element type and length.

| Type          | Description                              |
|---------------|------------------------------------------|
| `array<T; N>` | Fixed-size array (e.g., `array<f64; 8>`) |

### Structs

User-defined named field types, declared before use. (Syntax TBD — structs are defined in the kernel preamble and referenced by name.)

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

### Arithmetic (all numeric types)

| Op    | Description     |
|-------|-----------------|
| `add` | Addition        |
| `sub` | Subtraction     |
| `mul` | Multiplication  |
| `div` | Division        |
| `rem` | Remainder       |
| `min` | Minimum         |
| `max` | Maximum         |

All numeric scalar types are supported: f32, f64, i8, u8, i16, u16, i32, u32, i64, u64. Both operands must be the same type. Signed types (i8, i16, i32, i64) use signed division and signed remainder.

```
sum: f64 = add a b
d: f64 = min d_circle d_box
offset: i32 = add base 1i32
```

### Unary

| Op      | Description              | Types              |
|---------|--------------------------|---------------------|
| `neg`   | Negate                   | all numeric         |
| `abs`   | Absolute value           | all numeric         |
| `sqrt`  | Square root              | f32, f64            |
| `floor` | Floor                    | f32, f64            |
| `ceil`  | Ceiling                  | f32, f64            |
| `round` | Round to nearest integer | f32, f64            |
| `trunc` | Truncate toward zero     | f32, f64            |
| `fract` | Fractional part (x - floor(x)) | f32, f64     |
| `not`   | Logical NOT              | bool                |

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

### Bitwise (all integer types)

| Op        | Description                                      |
|-----------|--------------------------------------------------|
| `bit_and` | Bitwise AND                                      |
| `bit_or`  | Bitwise OR                                       |
| `bit_xor` | Bitwise XOR                                      |
| `shl`     | Shift left                                       |
| `shr`     | Shift right (logical for unsigned, arithmetic for signed) |

All integer types are supported: i8, u8, i16, u16, i32, u32, i64, u64. Both operands must be the same type. Signed types (i8, i16, i32, i64) use arithmetic shift right (sign-extending).

```
masked: u32 = bit_and color 0xFF
shifted: u32 = shl r 16
flags: i32 = bit_or a b
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

Operands must be the same numeric type. All numeric scalar types are supported. Signed types use signed comparisons. The result is always `bool`.

```
inside: bool = lt dist radius
positive: bool = gt count 0i32
```

### Conversions

Any scalar-to-scalar cast is valid (except conversions to or from `bool`). The instruction name follows the pattern `{from}_to_{to}`:

```
channel: u32 = f64_to_u32 intensity
coord: f64 = u32_to_f64 pixel_count
half: f32 = f64_to_f32 precise_val
wide: i64 = i32_to_i64 narrow_val
byte: u8 = u32_to_u8 color_channel
signed: i32 = u32_to_i32 raw_bits
```

### Literal suffixes

Numeric literals can carry a type suffix to disambiguate. Unsuffixed literals default to `f64` (floating point) or `u32` (integer) for backward compatibility.

```
a: f32 = const 1.0f32
b: i32 = const 42i32
c: u8 = const 255u8
d: i64 = const -1i64
e: f64 = const 3.14        # defaults to f64
f: u32 = const 42           # defaults to u32
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

Vectors can have any numeric element type (not just f64). All vector operations require operands to have the same vector type (same dimension and element type).

### Construction

```
pos: vec2<f64> = make_vec2 x y
color: vec3<f32> = make_vec3 r g b
rgba: vec4<u8> = make_vec4 r g b a
```

`make_vec2`, `make_vec3`, and `make_vec4` take scalar components and produce the corresponding vector type. The component types must match the vector's element type.

### Component extraction

```
px: f64 = extract_x pos
py: f64 = extract_y pos
pz: f64 = extract_z color    # vec3, vec4 only
pw: f64 = extract_w quat     # vec4 only
```

Extracts a single scalar component from a vector. The result type matches the vector's element type.

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
sum: vec2<f64> = vec_add a b
diff: vec3<f32> = vec_sub position offset
```

### Scalar-vector multiply (scalar x vec -> vec)

```
scaled: vec2<f64> = vec_scale factor pos
```

The first operand must be a scalar matching the vector's element type. The second must be a vec. The result matches the vec type.

### Unary vector operations (vec -> vec)

| Op              | Description                        |
|-----------------|------------------------------------|
| `vec_neg`       | Negate each component              |
| `vec_abs`       | Absolute value of each component   |
| `vec_normalize` | Normalize to unit length           |

```
flipped: vec2<f64> = vec_neg pos
unit: vec3<f32> = vec_normalize direction
```

### Reduction operations (vec -> scalar)

| Op           | Description                          |
|--------------|--------------------------------------|
| `vec_dot`    | Dot product of two vectors           |
| `vec_length` | Length (magnitude) of a vector       |

```
d: f64 = vec_dot a b
len: f32 = vec_length pos
```

Both operands of `vec_dot` must be the same vec type. The result type matches the vector's element type.

### Cross product (vec3 x vec3 -> vec3)

```
normal: vec3<f64> = vec_cross a b
```

Both operands must be the same vec3 type.

### Select with vectors

`select` works with vec types — both branches must be the same vec type:

```
chosen: vec2<f64> = select cond pos_a pos_b
```

## Matrix operations

Matrices use column-major storage and are parameterized by element type. All matrix operations require consistent element types across operands.

### Construction

Construct a matrix from column vectors:

```
col0: vec2<f64> = make_vec2 a b
col1: vec2<f64> = make_vec2 c d
m: mat2<f64> = make_mat2 col0 col1

c0: vec3<f32> = make_vec3 a b c
c1: vec3<f32> = make_vec3 d e f
c2: vec3<f32> = make_vec3 g h i
m3: mat3<f32> = make_mat3 c0 c1 c2

# make_mat4 takes 4 vec4 columns
m4: mat4<f64> = make_mat4 col0 col1 col2 col3
```

### Matrix-vector multiply

```
transformed: vec3<f64> = mat_mul_vec m v
```

The matrix column count must match the vector dimension. The result is a vector with dimension equal to the matrix row count.

### Matrix-matrix multiply

```
combined: mat3<f64> = mat_mul a b
```

Both operands must be the same matrix type. The result is the same matrix type.

### Transpose

```
mt: mat3<f64> = mat_transpose m
```

### Column extraction

```
c0: vec3<f64> = mat_col0 m
c1: vec3<f64> = mat_col1 m
c2: vec3<f64> = mat_col2 m    # mat3, mat4 only
c3: vec4<f64> = mat_col3 m4   # mat4 only
```

Extracts a column vector from a matrix. The result type matches the matrix's column vector type.

## Array operations

Fixed-size arrays with explicit element type and length.

### Construction

```
arr: array<f64; 4> = array_new a b c d
```

The number of arguments must match the array length. All arguments must match the element type.

### Element access

```
val: f64 = array_get arr 2
```

The index operand is a constant integer. Zero-based indexing.

### Element update

```
arr2: array<f64; 4> = array_set arr 2 new_val
```

Produces a new array with the element at the given index replaced. The original array is unchanged (SSA).

## Struct operations

User-defined structs with named fields. Fields are accessed by index (matching declaration order).

### Construction

```
pt: MyStruct = struct_new x_val y_val z_val
```

Arguments must match the struct's field types in declaration order.

### Field access

```
x: f64 = struct_get pt 0
```

The index operand is a constant integer identifying the field (zero-based, matching declaration order).

### Field update

```
pt2: MyStruct = struct_set pt 0 new_x
```

Produces a new struct with the field at the given index replaced. The original struct is unchanged (SSA).

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

Note: the example above shows the intent. PDIR does not support nested expressions like `sqrt(add cx2 cy2)` or reassignment of `intensity` — every value needs its own unique name and each operation is a separate statement. See `examples/sdf/sdf/sdf.pdir` for the working version.
