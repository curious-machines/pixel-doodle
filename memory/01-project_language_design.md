---
name: Language & IR design document
description: Comprehensive design for the custom language — IR-first architecture, IR types, text format grammar, control flow, backend lowering, kernel model
type: project
---

# Pixel Doodle Language Design

## Overview

IR-first architecture: the SSA IR (`Kernel` in `kernel_ir.rs`) is the central artifact. The text format (`.pdl` files) is a readable, writable serialization — designed to look like a language you'd want to write, not ugly machine format. A higher-level language may be added later as a second frontend targeting the same IR without changing backends.

The custom language is designed to be easy for an AI to generate and verify, not for humans to write directly. Optimizing for AI reasoning correctness (explicit, flat, regular) over human ergonomics is the right tradeoff.

```
[ .pdl source ]  ──parse──>  [ Kernel IR ]  ──lower──>  [ Cranelift / LLVM ]
                                   ^
                  [ built-in kernels ]  (programmatic construction)
                                   ^
                  [ future high-level lang ]  (compiles to same IR)
```

**Why IR-first:** The SSA text format may prove too verbose in practice. By making the IR the stable interface between frontends and backends, we can add a higher-level language without touching backend code.

## Kernel Model

A kernel body describes **per-pixel** computation. Backends generate the tile loop wrapper (row/col iteration, coordinate math `cx = x_min + col * x_step`, pixel store). The kernel receives:

- `x: f64` — view-space x coordinate (implicit, `Var(0)`)
- `y: f64` — view-space y coordinate (implicit, `Var(1)`)

And produces a `u32` ARGB color via `emit`. User-defined variables start at `Var(2)`.

## IR Data Structures

Replace `src/kernel_ir.rs` entirely.

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarType { F64, U32, Bool }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Var(pub u32);

#[derive(Debug, Clone)]
pub struct Binding {
    pub var: Var,
    pub name: String,
    pub ty: ScalarType,
}

#[derive(Debug, Clone, Copy)]
pub enum Const { F64(f64), U32(u32), Bool(bool) }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add, Sub, Mul, Div, Rem,           // arithmetic (f64 or u32)
    BitAnd, BitOr, BitXor, Shl, Shr,  // bitwise (u32 only)
    And, Or,                           // logical (bool only)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp { Eq, Ne, Lt, Le, Gt, Ge }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp { Neg, Not, Abs, Sqrt, Floor, Ceil }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvOp { F64ToU32, U32ToF64 }

#[derive(Debug, Clone)]
pub enum Inst {
    Const(Const),
    Binary { op: BinOp, lhs: Var, rhs: Var },
    Unary { op: UnaryOp, arg: Var },
    Cmp { op: CmpOp, lhs: Var, rhs: Var },
    Conv { op: ConvOp, arg: Var },
    Select { cond: Var, then_val: Var, else_val: Var },
    PackArgb { r: Var, g: Var, b: Var },
}

#[derive(Debug, Clone)]
pub struct Statement {
    pub binding: Binding,
    pub inst: Inst,
}

// V1: body is Vec<Statement>
// V2: body becomes Vec<BodyItem> to include While
#[derive(Debug, Clone)]
pub struct Kernel {
    pub name: String,
    pub body: Vec<Statement>,
    pub emit: Var,
}
```

### V2 Additions for Control Flow

```rust
#[derive(Debug, Clone)]
pub enum BodyItem {
    Stmt(Statement),
    While {
        carry: Vec<CarryVar>,
        cond_body: Vec<Statement>,  // computes condition from carry vars
        cond: Var,                  // Bool — loop continues while true
        body: Vec<Statement>,       // main body
        yields: Vec<Var>,           // new values for carry vars
    },
}

#[derive(Debug, Clone)]
pub struct CarryVar {
    pub binding: Binding,  // name and type for use inside loop
    pub init: Var,         // initial value (defined before the while)
}
```

Carry variable names are live after the loop with their final values.

## Text Format Syntax

### Examples

**V1 — gradient (per-pixel, no loops):**
```
kernel gradient {
    r: f64 = mul x 255.0
    r_u: u32 = f64_to_u32 r
    g: f64 = mul y 255.0
    g_u: u32 = f64_to_u32 g
    b: u32 = const 128
    pixel: u32 = pack_argb r_u g_u b
    emit pixel
}
```

**V2 — Mandelbrot (with while loop):**
```
kernel mandelbrot {
    four: f64 = const 4.0
    max: u32 = const 256
    z0: f64 = const 0.0
    i0: u32 = const 0

    while carry(zx: f64 = z0, zy: f64 = z0, iter: u32 = i0) {
        zx2: f64 = mul zx zx
        zy2: f64 = mul zy zy
        mag2: f64 = add zx2 zy2
        not_esc: bool = le mag2 four
        not_max: bool = lt iter max
        cont: bool = and not_esc not_max
        cond cont

        diff: f64 = sub zx2 zy2
        new_zx: f64 = add diff x
        two: f64 = const 2.0
        two_zx: f64 = mul two zx
        prod: f64 = mul two_zx zy
        new_zy: f64 = add prod y
        one: u32 = const 1
        new_iter: u32 = add iter one
        yield new_zx new_zy new_iter
    }

    is_max: bool = eq iter max
    # ... color computation ...
    emit pixel
}
```

### Grammar

```
program     = kernel
kernel      = "kernel" IDENT "{" body_item* "emit" IDENT "}"
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
            | conv_inst | select_inst | pack_inst
const_inst  = "const" literal
binary_inst = binop IDENT IDENT
unary_inst  = unaryop IDENT
cmp_inst    = cmpop IDENT IDENT
conv_inst   = convop IDENT
select_inst = "select" IDENT IDENT IDENT
pack_inst   = "pack_argb" IDENT IDENT IDENT

type        = "f64" | "u32" | "bool"
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
- **`emit varname`** as the final line produces the pixel color
- **`#` line comments**
- **`x` and `y`** are predefined implicit inputs
- Every piece of information the codegen needs is visible in the source — no inference, no implicit context

## Type Checking Rules

- `add/sub/mul/div/rem`: both operands same type (f64 or u32), result same type
- `bit_and/bit_or/bit_xor/shl/shr`: u32 only
- `and/or`: bool only
- `eq/ne/lt/le/gt/ge`: both operands same type, result is bool
- `neg/abs`: f64 or u32, result same type
- `not`: bool only
- `sqrt/floor/ceil`: f64 only
- `f64_to_u32`: f64 → u32 (truncation)
- `u32_to_f64`: u32 → f64
- `select`: cond must be bool, then/else must be same type, result is that type
- `pack_argb`: three u32 args (r, g, b in [0,255]), result is u32
- `emit`: must reference a u32 variable
- SSA: each name defined exactly once (carry vars scope to their while block and after)

## Backend Lowering Strategy

Both backends split into:

1. **Tile loop scaffolding** (backend-specific, stays mostly unchanged):
   - Function signature matching `TileKernelFn` (9 params)
   - Outer row loop: `row_start..row_end`
   - Inner col loop: `0..width`
   - Coordinate computation: `cx = x_min + col * x_step`, `cy = y_min + row * y_step`
   - Pixel store: `output[(row - row_start) * width + col] = color`

2. **Body lowering** (new, walks `Kernel` IR):
   - Maps `Var(0)` → cx, `Var(1)` → cy
   - Walks statements, emits backend instructions for each `Inst`
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

### While Loop Lowering

- **Cranelift**: Carry vars become `Variable`s. Loop header block with defs from init/yield. Cond check branch to body or exit. Uses Cranelift's implicit phi via Variable.
- **LLVM**: Carry vars become allocas (promoted to SSA by mem2reg in O3). Store init before loop, load in cond/body, store yields. Standard loop structure with br/cond_br.

## Parser Implementation

Hand-written recursive descent in `src/lang/parser.rs`. No external dependencies.

- Lexer tokenizes into keywords, idents, literals, punctuation
- All op names are keywords (lookup table, fallback to Ident)
- Parser resolves variable names to `Var` indices during parsing
- Tracks `HashMap<String, Var>` for name resolution
- Type-checks operands at parse time
- Returns `Result<Kernel, ParseError>` with line:col positions

## Printer

`src/lang/printer.rs` — converts `Kernel` back to text format.
Roundtrip property: `parse(print(k))` produces an equivalent kernel.

## V1 vs V2 Scope

- **V1:** Per-pixel only (no loops). `Kernel.body` is `Vec<Statement>`. Enough for solid colors, gradients, checkerboards, distance fields.
- **V2:** Add `while` with carry/cond/yield. `Kernel.body` becomes `Vec<BodyItem>`. Enough for Mandelbrot and iterative algorithms.
