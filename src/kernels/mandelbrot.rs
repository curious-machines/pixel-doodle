use crate::kernel_ir::*;

/// Build a gradient kernel programmatically.
/// V1 simplified version — no loops, so we can't do Mandelbrot yet.
/// Maps x to red, y to green, constant blue.
pub fn gradient_kernel() -> Kernel {
    // Var(0) = x, Var(1) = y (declared as params)
    let mut next = 2u32;
    let mut alloc = || { let v = Var(next); next += 1; v };

    let c255 = alloc(); // Var(2)
    let r_f = alloc();  // Var(3)
    let r_u = alloc();  // Var(4)
    let g_f = alloc();  // Var(5)
    let g_u = alloc();  // Var(6)
    let b = alloc();    // Var(7)
    let pixel = alloc(); // Var(8)

    let s = |stmt: Statement| BodyItem::Stmt(stmt);

    Kernel {
        name: "gradient".to_string(),
        params: vec![
            Binding { var: Var(0), name: "x".into(), ty: ScalarType::F64 },
            Binding { var: Var(1), name: "y".into(), ty: ScalarType::F64 },
        ],
        body: vec![
            s(Statement {
                binding: Binding { var: c255, name: "c255".into(), ty: ScalarType::F64 },
                inst: Inst::Const(Const::F64(255.0)),
            }),
            s(Statement {
                binding: Binding { var: r_f, name: "r_f".into(), ty: ScalarType::F64 },
                inst: Inst::Binary { op: BinOp::Mul, lhs: Var(0), rhs: c255 },
            }),
            s(Statement {
                binding: Binding { var: r_u, name: "r_u".into(), ty: ScalarType::U32 },
                inst: Inst::Conv { op: ConvOp::F64ToU32, arg: r_f },
            }),
            s(Statement {
                binding: Binding { var: g_f, name: "g_f".into(), ty: ScalarType::F64 },
                inst: Inst::Binary { op: BinOp::Mul, lhs: Var(1), rhs: c255 },
            }),
            s(Statement {
                binding: Binding { var: g_u, name: "g_u".into(), ty: ScalarType::U32 },
                inst: Inst::Conv { op: ConvOp::F64ToU32, arg: g_f },
            }),
            s(Statement {
                binding: Binding { var: b, name: "b".into(), ty: ScalarType::U32 },
                inst: Inst::Const(Const::U32(128)),
            }),
            s(Statement {
                binding: Binding { var: pixel, name: "pixel".into(), ty: ScalarType::U32 },
                inst: Inst::PackArgb { r: r_u, g: g_u, b },
            }),
        ],
        emit: pixel,
    }
}

/// Build a solid color kernel programmatically.
#[allow(dead_code)]
pub fn solid_color_kernel(r: u32, g: u32, b: u32) -> Kernel {
    // solid_color doesn't use x/y but still declares them for ABI compatibility
    let rv = Var(2);
    let gv = Var(3);
    let bv = Var(4);
    let pixel = Var(5);

    let s = |stmt: Statement| BodyItem::Stmt(stmt);

    Kernel {
        name: "solid".to_string(),
        params: vec![
            Binding { var: Var(0), name: "x".into(), ty: ScalarType::F64 },
            Binding { var: Var(1), name: "y".into(), ty: ScalarType::F64 },
        ],
        body: vec![
            s(Statement {
                binding: Binding { var: rv, name: "r".into(), ty: ScalarType::U32 },
                inst: Inst::Const(Const::U32(r)),
            }),
            s(Statement {
                binding: Binding { var: gv, name: "g".into(), ty: ScalarType::U32 },
                inst: Inst::Const(Const::U32(g)),
            }),
            s(Statement {
                binding: Binding { var: bv, name: "b".into(), ty: ScalarType::U32 },
                inst: Inst::Const(Const::U32(b)),
            }),
            s(Statement {
                binding: Binding { var: pixel, name: "pixel".into(), ty: ScalarType::U32 },
                inst: Inst::PackArgb { r: rv, g: gv, b: bv },
            }),
        ],
        emit: pixel,
    }
}
