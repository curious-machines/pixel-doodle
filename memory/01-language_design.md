---
name: Language & IR design document
description: Comprehensive design for the custom language — IR-first architecture, IR types, text format grammar, control flow, backend lowering, kernel model
type: project
---

# Pixel Doodle Language Design

## Overview

IR-first architecture: the SSA IR (`Kernel` in `kernel_ir.rs`) is the central artifact. The text format (`.pdir` files) is a readable, writable serialization — designed to look like a language you'd want to write, not ugly machine format. A higher-level language may be added later as a second frontend targeting the same IR without changing backends.

The custom language is designed to be easy for an AI to generate and verify, not for humans to write directly. Optimizing for AI reasoning correctness (explicit, flat, regular) over human ergonomics is the right tradeoff.

```
[ .pdir source ]  ──parse──>  [ Kernel IR ]  ──lower──>  [ Cranelift / LLVM ]
                                   ^
                  [ built-in kernels ]  (programmatic construction)
                                   ^
                  [ future high-level lang ]  (compiles to same IR)
```

**Why IR-first:** The SSA text format may prove too verbose in practice. By making the IR the stable interface between frontends and backends, we can add a higher-level language without touching backend code.

## Kernel Model

A kernel body describes **per-pixel** computation. Backends generate the tile loop wrapper (row/col iteration, coordinate math `cx = x_min + col * x_step`, pixel store). The kernel declares its inputs explicitly as parameters:

```
kernel gradient(x: f64, y: f64) -> u32 { ... }
kernel mandelbrot(x: f64, y: f64, max_iter: u32) -> u32 { ... }
```

Parameters are assigned `Var` indices in declaration order (`x` → `Var(0)`, `y` → `Var(1)`, etc.). User-defined variables start after the last parameter. The return type (declared with `->`) specifies the type of the `emit` value — currently always `u32` (ARGB pixel color).

Coordinate parameters (`x`, `y`) are view-space coordinates computed by the tile loop wrapper. Additional parameters (like `max_iter`) are user-defined values passed in from the host.

> **Future alternative:** If parameter lists grow long, consider a `params { ... }` block:
> ```
> kernel mandelbrot {
>     params { x: f64, y: f64, max_iter: u32 }
>     ...
> }
> ```
> This keeps the kernel header clean and allows grouping/commenting params. Decision deferred until we see real usage patterns.

## IR Data Structures

Defined in `src/kernel_ir.rs`.

```rust
pub enum ScalarType {
    F32, F64, I8, U8, I16, U16, I32, U32, I64, U64, Bool,
}

pub enum ValType {
    Scalar(ScalarType),
    Vec { len: u8, elem: ScalarType },    // vec2<T>, vec3<T>, vec4<T>; len in {2, 3, 4}
    Mat { size: u8, elem: ScalarType },   // mat2<T>, mat3<T>, mat4<T>; column-major
    Array { elem: Box<ValType>, size: u32 }, // fixed-size array
    Struct(String),                       // references StructDef by name
}

pub struct StructDef { name: String, fields: Vec<(String, ValType)> }

pub struct Var(pub u32);
pub struct Binding { var: Var, name: String, ty: ValType }

pub enum Const {
    F32(f32), F64(f64), I8(i8), U8(u8), I16(i16), U16(u16),
    I32(i32), U32(u32), I64(i64), U64(u64), Bool(bool),
}

pub enum BinOp {
    Add, Sub, Mul, Div, Rem,           // arithmetic (any matching numeric type)
    BitAnd, BitOr, BitXor, Shl, Shr,  // bitwise (integer types only)
    And, Or,                           // logical (bool only)
    Min, Max,                          // numeric min/max
    Atan2, Pow, Hash,                  // math builtins
}
pub enum CmpOp { Eq, Ne, Lt, Le, Gt, Ge }
pub enum UnaryOp {
    Neg, Not, Abs, Sqrt, Floor, Ceil,
    Sin, Cos, Tan, Asin, Acos, Atan,
    Exp, Exp2, Log, Log2, Log10,
    Round, Trunc, Fract,
}

/// Type conversion. `norm` = normalizing conversion (e.g. u32 -> f64 / 2^32).
pub struct ConvOp { from: ScalarType, to: ScalarType, norm: bool }

// Vector-specific operation enums
pub enum VecBinOp { Add, Sub, Mul, Div, Min, Max }
pub enum VecUnaryOp { Neg, Abs, Normalize }

pub enum Inst {
    Const(Const),
    Binary { op: BinOp, lhs: Var, rhs: Var },
    Unary { op: UnaryOp, arg: Var },
    Cmp { op: CmpOp, lhs: Var, rhs: Var },
    Conv { op: ConvOp, arg: Var },
    Select { cond: Var, then_val: Var, else_val: Var },
    PackArgb { r: Var, g: Var, b: Var },

    // Vector instructions (unified: length of Vec<Var> determines vec2/3/4)
    MakeVec(Vec<Var>),
    VecExtract { vec: Var, index: u8 },              // vec -> scalar (0=x, 1=y, 2=z, 3=w)
    VecBinary { op: VecBinOp, lhs: Var, rhs: Var },
    VecScale { scalar: Var, vec: Var },
    VecUnary { op: VecUnaryOp, arg: Var },
    VecDot { lhs: Var, rhs: Var },
    VecLength { arg: Var },
    VecCross { lhs: Var, rhs: Var },                 // vec3 x vec3 -> vec3

    // Matrix instructions
    MakeMat(Vec<Var>),                               // from column vectors
    MatMulVec { mat: Var, vec: Var },                 // matN * vecN -> vecN
    MatMul { lhs: Var, rhs: Var },                    // matN * matN -> matN
    MatTranspose { arg: Var },                        // matN -> matN
    MatCol { mat: Var, index: u8 },                   // matN -> vecN

    // Struct operations
    StructNew(Vec<Var>),                              // field values in definition order
    StructGet { val: Var, field: u32 },               // get field by index
    StructSet { val: Var, field: u32, new_val: Var }, // set field, produce new struct

    // Array operations
    ArrayNew(Vec<Var>),                               // create fixed-size array
    ArrayGet { array: Var, index: Var },              // get element at index
    ArraySet { array: Var, index: Var, val: Var },    // set element, produce new array

    // Buffer operations (simulation kernels)
    BufLoad { buf: u32, x: Var, y: Var },
    BufStore { buf: u32, x: Var, y: Var, val: Var },
}

pub struct Statement { binding: Binding, inst: Inst }

pub struct CarryVar {
    binding: Binding,  // name and type for use inside loop
    init: Var,         // initial value (defined before the while)
}

pub struct While {
    carry: Vec<CarryVar>,
    cond_body: Vec<BodyItem>,   // computes condition (can contain nested while)
    cond: Var,                  // Bool — loop continues while true
    body: Vec<BodyItem>,        // computes next values (can contain nested while)
    yields: Vec<Var>,           // new values for carry vars
}

pub enum BodyItem {
    Stmt(Statement),
    While(While),
}

pub struct BufDecl { name: String, is_output: bool }

pub struct Kernel {
    name: String,
    params: Vec<Binding>,
    return_ty: ValType,
    body: Vec<BodyItem>,
    emit: Var,
    buffers: Vec<BufDecl>,         // buffer declarations (simulation kernels)
    struct_defs: Vec<StructDef>,   // struct type definitions used by this kernel
}
```

Carry variable names are live after the loop with their final values.

## Text Format Syntax

### Examples

**Gradient (per-pixel, no loops):**
```
kernel gradient(x: f64, y: f64) -> u32 {
    r: f64 = mul x 255.0
    r_u: u32 = f64_to_u32 r
    g: f64 = mul y 255.0
    g_u: u32 = f64_to_u32 g
    b: u32 = const 128
    pixel: u32 = pack_argb r_u g_u b
    emit pixel
}
```

**Mandelbrot (with while loop):**
```
kernel mandelbrot(x: f64, y: f64, max_iter: u32) -> u32 {
    zero: f64 = const 0.0
    four: f64 = const 4.0
    i_zero: u32 = const 0

    while carry(zx: f64 = zero, zy: f64 = zero, iter: u32 = i_zero) {
        zx2: f64 = mul zx zx
        zy2: f64 = mul zy zy
        mag: f64 = add zx2 zy2
        escaped: bool = gt mag four
        too_many: bool = ge iter max_iter
        done: bool = or escaped too_many
        cont: bool = not done
        cond cont

        new_zx: f64 = sub zx2 zy2
        new_zx2: f64 = add new_zx x
        zxzy: f64 = mul zx zy
        new_zy: f64 = mul zxzy 2.0
        new_zy2: f64 = add new_zy y
        new_iter: u32 = add iter 1
        yield new_zx2 new_zy2 new_iter
    }

    is_inside: bool = eq iter max_iter
    # ... color computation using iter ...
    emit pixel
}
```

### Grammar

```
program     = kernel
kernel      = "kernel" IDENT "(" param_list? ")" "->" type "{" body_item* "emit" IDENT "}"
param_list  = param ("," param)*
param       = IDENT ":" type
body_item   = statement | while_loop
statement   = IDENT ":" type "=" instruction
while_loop  = "while" "carry" "(" carry_list ")" "{"
                  statement* "cond" IDENT
                  statement* "yield" ident_list
              "}"
carry_list  = carry_var ("," carry_var)*
carry_var   = IDENT ":" type "=" IDENT
ident_list  = IDENT (IDENT)*

instruction = const_inst | binary_inst | unary_inst | cmp_inst
            | conv_inst | select_inst | pack_inst | vec_inst
            | mat_inst | struct_inst | array_inst | buf_inst
const_inst  = "const" literal
binary_inst = binop operand operand
unary_inst  = unaryop operand
cmp_inst    = cmpop operand operand
conv_inst   = convop operand
select_inst = "select" operand operand operand
pack_inst   = "pack_argb" operand operand operand
vec_inst    = vec_make | vec_extract | vec_binop | vec_unaryop
            | vec_scale | vec_dot | vec_length | vec_cross
mat_inst    = "make_mat" operand+ | "mat_mul_vec" operand operand
            | "mat_mul" operand operand | "mat_transpose" operand
            | "mat_col" operand INT
struct_inst = "struct_new" operand* | "struct_get" operand INT
            | "struct_set" operand INT operand
array_inst  = "array_new" operand* | "array_get" operand operand
            | "array_set" operand operand operand
buf_inst    = "buf_load" INT operand operand | "buf_store" INT operand operand operand
vec_make    = "make_vec" operand+       (2-4 operands determine vec size)
vec_extract = ("extract_x" | "extract_y" | "extract_z" | "extract_w") operand
vec_binop   = ("vec_add"|"vec_sub"|"vec_mul"|"vec_div"|"vec_min"|"vec_max") operand operand
vec_unaryop = ("vec_neg" | "vec_abs" | "vec_normalize") operand
vec_scale   = "vec_scale" operand operand     (f64, vec -> vec)
vec_dot     = "vec_dot" operand operand       (vec, vec -> f64)
vec_length  = "vec_length" operand            (vec -> f64)
vec_cross   = "vec_cross" operand operand     (vec3, vec3 -> vec3)
operand     = IDENT | literal   (inline literals create implicit consts)

type        = scalar_type | vec_type | mat_type | array_type | IDENT
scalar_type = "f32" | "f64" | "i8" | "u8" | "i16" | "u16" | "i32" | "u32" | "i64" | "u64" | "bool"
vec_type    = ("vec2" | "vec3" | "vec4") "<" scalar_type ">"
mat_type    = ("mat2" | "mat3" | "mat4") "<" scalar_type ">"
array_type  = "array" "<" type ";" INT ">"
binop       = "add" | "sub" | "mul" | "div" | "rem"
            | "bit_and" | "bit_or" | "bit_xor" | "shl" | "shr"
            | "and" | "or" | "min" | "max" | "atan2" | "pow" | "hash"
cmpop       = "eq" | "ne" | "lt" | "le" | "gt" | "ge"
unaryop     = "neg" | "not" | "abs" | "sqrt" | "floor" | "ceil"
            | "sin" | "cos" | "tan" | "asin" | "acos" | "atan"
            | "exp" | "exp2" | "log" | "log2" | "log10"
            | "round" | "trunc" | "fract"
convop      = SCALAR_TYPE "_to_" SCALAR_TYPE  (e.g. "f64_to_u32", "i32_to_f64", "f32_to_u32")
literal     = FLOAT | INT | "true" | "false"

IDENT       = [a-zA-Z_][a-zA-Z0-9_]*
FLOAT       = digit+ "." digit+
INT         = digit+
comment     = "#" ... newline
```

### Syntax Design Principles

- **One statement per line**: `name: type = op args...`
- **All op names are keywords**: non-contextual, no ambiguity
- **Operands are variable names** (resolved by parser) or inline literals (for `const`)
- **`emit varname`** as the final line produces the return value (must match declared return type)
- **`#` line comments**
- **All inputs are explicit parameters** — no magic implicit variables
- Every piece of information the codegen needs is visible in the source — no inference, no implicit context

## Type Checking Rules

### Scalar operations
- `add/sub/mul/div/rem`: both operands same numeric type (any of f32, f64, i8..u64), result same type
- `min/max`: both operands same numeric type, result same type
- `bit_and/bit_or/bit_xor/shl/shr`: integer types only (i8..u64)
- `and/or`: bool only
- `eq/ne/lt/le/gt/ge`: both operands same type, result is bool
- `neg/abs`: any numeric type, result same type
- `not`: bool only
- `sqrt/floor/ceil/sin/cos/tan/asin/acos/atan/exp/exp2/log/log2/log10/round/trunc/fract`: float types only (f32, f64)
- `atan2/pow`: both operands same float type, result same type
- `hash`: integer operands, result same type
- Conversion (`ConvOp`): any scalar-to-scalar conversion via `from_to_to` syntax (e.g. `f64_to_u32`, `i32_to_f64`, `f32_to_u32`). `norm` flag for normalizing conversions (e.g. u32 -> f64 divides by 2^32)
- `select`: cond must be bool, then/else must be same type (including compound types), result is that type
- `pack_argb`: three u32 args (r, g, b in [0,255]), result is u32
- `emit`: must reference a variable matching the declared return type
- SSA: each name defined exactly once (carry vars scope to their while block and after)

### Vector operations
- `make_vec`: 2-4 scalar args of same type → vec{2,3,4}<T> (length determined by arg count)
- `extract_x/extract_y`: vec2+ → scalar element type
- `extract_z`: vec3+ → scalar element type
- `extract_w`: vec4 → scalar element type
- `vec_add/vec_sub/vec_mul/vec_div/vec_min/vec_max`: both operands same vec type, result same vec type
- `vec_scale`: scalar × vec → vec (scalar must match vec element type)
- `vec_neg/vec_abs`: vec → vec (same type)
- `vec_normalize`: vec → vec (same type)
- `vec_dot`: both operands same vec type → scalar (element type)
- `vec_length`: vec → scalar (element type)
- `vec_cross`: vec3 × vec3 → vec3

### Matrix operations
- `make_mat`: N column vectors of vecN → matN (N in {2, 3, 4})
- `mat_mul_vec`: matN × vecN → vecN
- `mat_mul`: matN × matN → matN
- `mat_transpose`: matN → matN
- `mat_col`: matN, index → vecN (extract column by index)

### Struct operations
- `struct_new`: field values in definition order → Struct
- `struct_get`: struct value, field index → field type
- `struct_set`: struct value, field index, new field value → Struct (produces new value)

### Array operations
- `array_new`: element values → Array (all elements must match declared element type)
- `array_get`: array, integer index → element type
- `array_set`: array, integer index, value → Array (produces new value)

### PD operator overloading for compound types
In the PD higher-level language, standard operators work on vec and mat types:
- `vec + vec`, `vec - vec`, `vec * vec`, `vec / vec` → component-wise, same vec type
- `scalar * vec`, `vec * scalar` → scalar-vector multiply (lowers to `VecScale`)
- `mat * vec` → matrix-vector multiply (lowers to `MatMulVec`)
- `mat * mat` → matrix-matrix multiply (lowers to `MatMul`)
- `-vec` → component-wise negation
- `vec.x`, `vec.y`, `vec.z`, `vec.w` → field access, extracts scalar component
- `struct.field_name` → field access (lowers to `StructGet`)
- `array[index]` → element access (lowers to `ArrayGet`)
- Built-in functions `dot()`, `length()`, `normalize()`, `cross()`, `distance()`, `mix()`, `min()`, `max()`, `abs()` are overloaded for vec types

## Backend Lowering Strategy

Both backends split into:

1. **Tile loop scaffolding** (backend-specific):
   - Function signature matching `TileKernelFn` (9 params)
   - Outer row loop: `row_start..row_end`
   - Inner col loop: `0..width`
   - Coordinate computation: `cx = x_min + col * x_step`, `cy = y_min + row * y_step`
   - Pixel store: `output[(row - row_start) * width + col] = color`

2. **Body lowering** (walks `Kernel` IR):
   - Maps parameter `Var`s to their backend values (e.g., coordinates from tile loop, uniforms from host)
   - Walks `BodyItem`s, emitting backend instructions for each `Inst`
   - Handles `While` by creating loop blocks with carry variable management
   - Returns the backend value for the `emit` var

### Instruction Mapping

| IR Inst | Cranelift | LLVM (inkwell) |
|---------|-----------|----------------|
| `Const(F64(v))` | `f64const(v)` | `f64_type().const_float(v)` |
| `Const(U32(v))` | `iconst(I32, v)` | `i32_type().const_int(v)` |
| `Binary(Add) [f64]` | `fadd` | `build_float_add` |
| `Binary(Add) [u32]` | `iadd` | `build_int_add` |
| `Cmp(Lt) [f64]` | `fcmp(LessThan)` | `build_float_compare(OLT)` |
| `Conv(F64ToU32)` | `fcvt_to_sint(I32)` | `build_float_to_signed_int` |
| `PackArgb(r,g,b)` | shift+or sequence | shift+or sequence |
| `Select(c,t,f)` | `select` | `build_select` |

### Vector Decomposition in Backends

Vec, mat, struct, and array types are first-class in the IR but decomposed to scalars by the backends. Neither Cranelift nor LLVM has a native geometric vec3 type, so backends store each compound var as N scalar values using a `VarValues` enum:

```rust
enum VarValues {
    Scalar(Value),
    Multi(Vec<Value>),  // vec2..4, mat2..4, struct fields, array elements
}
```

- `MakeVec` → store component values as `VarValues::Multi`
- `VecExtract` → read indexed component from VarValues
- `VecBinary` → apply scalar op to each component pair
- `VecDot` → multiply pairwise, sum
- `VecLength` → dot(v,v), sqrt
- `VecCross` → standard cross product formula on components
- `VecNormalize` → divide each component by length
- `MakeMat` → flatten column vectors into scalar components
- `MatMulVec` / `MatMul` → expanded to scalar multiply-accumulate
- Struct/array ops → index into the flattened scalar components
- While loop carry vars with compound types expand to N individual `Variable`s (Cranelift) or phi nodes (LLVM)

### While Loop Lowering

- **Cranelift**: Carry vars become `Variable`s (Cranelift's SSA construction handles phi insertion). Loop header block reads vars, runs cond_body, branches to body or exit. Body runs, updates vars with yield values, jumps back to header.
- **LLVM**: Carry vars become explicit phi nodes in loop header block. Pre-block provides initial values, loop body provides yield values. Standard loop structure with conditional branch.

## Parser Implementation

Hand-written recursive descent in `src/lang/parser.rs`. No external dependencies.

- Lexer tokenizes into keywords, idents, literals, punctuation
- All op names are keywords (lookup table, fallback to Ident)
- Parser resolves variable names to `Var` indices during parsing
- Tracks `HashMap<String, Var>` for name resolution
- Type-checks operands at parse time
- Scoping: while body introduces carry vars + local vars; after while, only carry vars survive
- Returns `Result<Kernel, ParseError>` with line:col positions

## Printer

`src/lang/printer.rs` — converts `Kernel` back to text format.
Roundtrip property: `parse(print(k))` produces an equivalent kernel.
Handles while loops with proper indentation and carry var formatting.

## Use Directive

`use "path.pd";` imports function definitions from external PD files. Paths are relative to the current file's directory. Files are only processed once (dedup by canonical path). Nested includes are supported. Included files can contain `use` and `fn` but not `kernel`. Unused functions have no runtime cost — the lowerer only inlines functions that are actually called.

## Future: Annotated IR Dump (not yet implemented)

**Status:** Discussed, deferred for future implementation.

A `--dump-ir --annotate` flag (or similar) would include original PD source lines as comments in the generated PDIR output, making it easier for humans to understand the relationship between PD source and lowered IR.

**Design considerations:**
- Preferred approach: side-channel — pass original PD source text to the printer rather than adding annotations to the IR (keeps the IR syntax-independent).
- The lowerer could tag each IR statement with a source line number during lowering. The printer would look up the line in the original source and emit it as a `# comment`.
- A single PD `let` statement can produce multiple IR statements (e.g., `let d = length(p) - r;` → `vec_length` + `sub`). The comment should appear before the first statement in the group.
- Also discussed: using PD variable names as a base for generated temp names (instead of `_pd_tmp_4`, something more descriptive). The lowerer already preserves user-given names from `let` bindings; only intermediates use generated names.
- This is a human readability aid, not needed by the LLM.
