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
pub enum ValType { F64, U32, Bool, Vec2, Vec3 }  // Vec2/Vec3 are 2x/3x f64
pub struct Var(pub u32);
pub struct Binding { var: Var, name: String, ty: ValType }
pub enum Const { F64(f64), U32(u32), Bool(bool) }

pub enum BinOp {
    Add, Sub, Mul, Div, Rem,           // arithmetic (f64 or u32)
    BitAnd, BitOr, BitXor, Shl, Shr,  // bitwise (u32 only)
    And, Or,                           // logical (bool only)
}
pub enum CmpOp { Eq, Ne, Lt, Le, Gt, Ge }
pub enum UnaryOp { Neg, Not, Abs, Sqrt, Floor, Ceil }
pub enum ConvOp { F64ToU32, U32ToF64 }

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

    // Vector instructions
    MakeVec2 { x: Var, y: Var },                    // f64, f64 -> vec2
    MakeVec3 { x: Var, y: Var, z: Var },            // f64, f64, f64 -> vec3
    VecExtract { vec: Var, index: u8 },              // vec -> f64 (0=x, 1=y, 2=z)
    VecBinary { op: VecBinOp, lhs: Var, rhs: Var },  // vec op vec -> vec
    VecScale { scalar: Var, vec: Var },              // f64 * vec -> vec
    VecUnary { op: VecUnaryOp, arg: Var },           // vec -> vec
    VecDot { lhs: Var, rhs: Var },                   // vec, vec -> f64
    VecLength { arg: Var },                          // vec -> f64
    VecCross { lhs: Var, rhs: Var },                 // vec3, vec3 -> vec3
}

pub struct Statement { binding: Binding, inst: Inst }

pub struct CarryVar {
    binding: Binding,  // name and type for use inside loop
    init: Var,         // initial value (defined before the while)
}

pub struct While {
    carry: Vec<CarryVar>,
    cond_body: Vec<Statement>,  // computes condition from carry vars
    cond: Var,                  // Bool — loop continues while true
    body: Vec<Statement>,       // computes next values
    yields: Vec<Var>,           // new values for carry vars
}

pub enum BodyItem {
    Stmt(Statement),
    While(While),
}

pub struct Kernel {
    name: String,
    params: Vec<Binding>,
    return_ty: ValType,
    body: Vec<BodyItem>,
    emit: Var,
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
const_inst  = "const" literal
binary_inst = binop operand operand
unary_inst  = unaryop operand
cmp_inst    = cmpop operand operand
conv_inst   = convop operand
select_inst = "select" operand operand operand
pack_inst   = "pack_argb" operand operand operand
vec_inst    = vec_make | vec_extract | vec_binop | vec_unaryop
            | vec_scale | vec_dot | vec_length | vec_cross
vec_make    = "make_vec2" operand operand | "make_vec3" operand operand operand
vec_extract = ("extract_x" | "extract_y" | "extract_z") operand
vec_binop   = ("vec_add"|"vec_sub"|"vec_mul"|"vec_div"|"vec_min"|"vec_max") operand operand
vec_unaryop = ("vec_neg" | "vec_abs" | "vec_normalize") operand
vec_scale   = "vec_scale" operand operand     (f64, vec -> vec)
vec_dot     = "vec_dot" operand operand       (vec, vec -> f64)
vec_length  = "vec_length" operand            (vec -> f64)
vec_cross   = "vec_cross" operand operand     (vec3, vec3 -> vec3)
operand     = IDENT | literal   (inline literals create implicit consts)

type        = "f64" | "u32" | "bool" | "vec2" | "vec3"
binop       = "add" | "sub" | "mul" | "div" | "rem"
            | "bit_and" | "bit_or" | "bit_xor" | "shl" | "shr"
            | "and" | "or"
cmpop       = "eq" | "ne" | "lt" | "le" | "gt" | "ge"
unaryop     = "neg" | "not" | "abs" | "sqrt" | "floor" | "ceil"
convop      = "f64_to_u32" | "u32_to_f64"
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
- `add/sub/mul/div/rem`: both operands same type (f64 or u32), result same type
- `bit_and/bit_or/bit_xor/shl/shr`: u32 only
- `and/or`: bool only
- `eq/ne/lt/le/gt/ge`: both operands same type, result is bool
- `neg/abs`: f64 or u32, result same type
- `not`: bool only
- `sqrt/floor/ceil`: f64 only
- `f64_to_u32`: f64 → u32 (truncation toward zero)
- `u32_to_f64`: u32 → f64
- `select`: cond must be bool, then/else must be same type (including vec types), result is that type
- `pack_argb`: three u32 args (r, g, b in [0,255]), result is u32
- `emit`: must reference a variable matching the declared return type
- SSA: each name defined exactly once (carry vars scope to their while block and after)

### Vector operations
- `make_vec2`: two f64 args → vec2
- `make_vec3`: three f64 args → vec3
- `extract_x/extract_y`: vec2 or vec3 → f64
- `extract_z`: vec3 only → f64
- `vec_add/vec_sub/vec_mul/vec_div/vec_min/vec_max`: both operands same vec type, result same vec type
- `vec_scale`: f64 × vec → vec (same vec type)
- `vec_neg/vec_abs`: vec → vec (same type)
- `vec_normalize`: vec → vec (same type)
- `vec_dot`: both operands same vec type → f64
- `vec_length`: vec → f64
- `vec_cross`: vec3 × vec3 → vec3

### PD operator overloading for vec types
In the PD higher-level language, standard operators work on vec types:
- `vec + vec`, `vec - vec`, `vec * vec`, `vec / vec` → component-wise, same vec type
- `f64 * vec`, `vec * f64` → scalar-vector multiply (lowers to `VecScale`)
- `-vec` → component-wise negation
- `vec.x`, `vec.y`, `vec.z` → field access, extracts f64 component
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

Vec types are first-class in the IR but decomposed to scalars by the backends. Neither Cranelift nor LLVM has a native geometric vec3 type, so backends store each vec var as 2–3 scalar f64 values using a `VarValues` enum:

```rust
enum VarValues {
    Scalar(Value),
    Vec2(Value, Value),
    Vec3(Value, Value, Value),
}
```

- `MakeVec2/3` → store component values as `VarValues::Vec2/3`
- `VecExtract` → read indexed component from VarValues
- `VecBinary` → apply scalar op to each component pair
- `VecDot` → multiply pairwise, sum
- `VecLength` → dot(v,v), sqrt
- `VecCross` → standard cross product formula on components
- `VecNormalize` → divide each component by length
- While loop carry vars with vec type expand to 2–3 individual `Variable`s (Cranelift) or phi nodes (LLVM)

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
