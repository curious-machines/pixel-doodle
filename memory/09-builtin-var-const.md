# Builtin, Var, and Const Keywords for PDP

## Problem

The PDP language has 10 implicit "intrinsic" variables (`width`, `height`, `time`, `zoom`, etc.) that are magically available without declaration. User variables use bare assignment syntax (`iterations: range(1..10) = 1`) with no mutability distinction. A reader must know these names exist from documentation alone — violating the project's "explicit over magic" principle.

## Design Decisions

- **`builtin` keyword** declares usage of a runtime-provided value
- **`var` keyword** marks a user-defined mutable variable
- **`const` keyword** marks a user-defined or builtin immutable value
- Using an undeclared intrinsic is a **hard error**
- Declarations include **explicit types** (`builtin const time: f64`)
- **Pipeline-only scope** — kernel built-ins (`x`, `y`, etc.) are unchanged
- `const` means "not assignable by the pipeline" — the runtime can still update `time` each frame

## Syntax

```
# Runtime-provided, read-only
builtin const width: u32
builtin const time: f64

# Runtime-provided, assignable by key handlers
builtin var center_x: f64
builtin var paused: bool

# User-defined mutable (with optional range)
var iterations: range(1..10) = 1

# User-defined with type annotation
var speed: f64 = 1.0

# User-defined immutable
const max_detail: u32 = 256
```

### Placement

Builtin and variable declarations can appear:
- **Top level** — shared across all pipelines in the file
- **Inside a pipeline block** — scoped to that pipeline
- **In included files** — merged into the including file

### Duplicate Declarations

Duplicate builtin declarations are allowed across files. Rules:
- **Types must match** — `builtin var zoom: f64` and `builtin var zoom: u32` is a hard error
- **var/const can differ** — most permissive wins (if any says `var`, effective is `var`)
- Duplicate user `var`/`const` declarations are still an error

### Recommended Pattern

A shared `builtins.pdp` declares all builtins in one place:

```
# examples/shared/builtins.pdp
builtin const width: u32
builtin const height: u32
builtin const time: f64
builtin const mouse_x: f64
builtin const mouse_y: f64
builtin var center_x: f64
builtin var center_y: f64
builtin var zoom: f64
builtin var paused: bool
builtin var frame: u64
```

Most `.pdp` files include this single file. Specialized includes like `pan_zoom.pdp` can also declare the builtins they use directly — duplicates merge harmlessly.

## Intrinsic Registry

| Name | Type | Mutable | Description |
|------|------|---------|-------------|
| `width` | `u32` | no | Canvas width in pixels |
| `height` | `u32` | no | Canvas height in pixels |
| `time` | `f64` | no | Elapsed seconds since window opened |
| `mouse_x` | `f64` | no | Mouse X position |
| `mouse_y` | `f64` | no | Mouse Y position |
| `center_x` | `f64` | yes | Viewport center X |
| `center_y` | `f64` | yes | Viewport center Y |
| `zoom` | `f64` | yes | Viewport zoom level |
| `paused` | `bool` | yes | Whether frame auto-increments |
| `frame` | `u64` | yes | Current frame number |

A user can declare a mutable intrinsic as `builtin const` (opting into less power). Declaring a const-only intrinsic as `builtin var` is an error.

## Four-Way Declaration Distinction

| Declaration | Meaning |
|---|---|
| `builtin const` | Runtime-provided, read-only |
| `builtin var` | Runtime-provided, assignable (key handlers can modify) |
| `const` | User-defined, immutable |
| `var` | User-defined, mutable |

## Error Messages

| Mistake | Error |
|---|---|
| Undeclared intrinsic | `use of undeclared builtin 'time'. Add 'builtin const time: f64' to your file` |
| Wrong type | `builtin 'width' has type u32, but was declared as f64` |
| Unknown builtin name | `unknown builtin 'foo'. Known builtins: width (u32), height (u32), ...` |
| `builtin var` on const intrinsic | `builtin 'time' is read-only and cannot be declared as 'var'` |
| Assign to const | `cannot assign to 'max_detail': it is declared as 'const'` |
| Assign to builtin const | `cannot assign to 'time': it is declared as 'builtin const'` |
| Bare variable (no keyword) | `unexpected identifier 'iterations'. Use 'var iterations: ...' or 'const iterations: ...'` |
| Conflicting duplicate type | `conflicting builtin declarations for 'zoom': declared as f64 and u32` |
