use crate::kernel_ir::*;

/// Print a Kernel IR back to .pdir text format.
#[cfg_attr(not(test), allow(dead_code))]
pub fn print(kernel: &Kernel) -> String {
    let mut out = String::new();
    out.push_str(&format!("kernel {}(", kernel.name));
    for (i, p) in kernel.params.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(&p.name);
        out.push_str(": ");
        out.push_str(&format!("{}", p.ty));
    }
    out.push_str(&format!(") -> {} {{\n", kernel.return_ty));

    print_body(&mut out, &kernel.body, kernel, 1);

    out.push_str("    emit ");
    out.push_str(kernel.var_name(kernel.emit).unwrap_or("?"));
    out.push_str("\n}\n");
    out
}

fn print_body(out: &mut String, body: &[BodyItem], kernel: &Kernel, indent: usize) {
    let prefix = "    ".repeat(indent);
    for item in body {
        match item {
            BodyItem::Stmt(stmt) => {
                // Skip implicit literal bindings
                if stmt.binding.name.starts_with("__lit_") {
                    continue;
                }
                out.push_str(&prefix);
                out.push_str(&stmt.binding.name);
                out.push_str(": ");
                out.push_str(&format!("{}", stmt.binding.ty));
                out.push_str(" = ");
                print_inst(out, &stmt.inst, kernel);
                out.push('\n');
            }
            BodyItem::While(w) => {
                print_while(out, w, kernel, indent);
            }
        }
    }
}

fn print_while(out: &mut String, w: &While, kernel: &Kernel, indent: usize) {
    let prefix = "    ".repeat(indent);
    let inner_prefix = "    ".repeat(indent + 1);

    out.push_str(&prefix);
    out.push_str("while carry(");
    for (i, cv) in w.carry.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(&cv.binding.name);
        out.push_str(": ");
        out.push_str(&format!("{}", cv.binding.ty));
        out.push_str(" = ");
        print_operand(out, cv.init, kernel);
    }
    out.push_str(") {\n");

    // Print cond_body
    print_body(out, &w.cond_body, kernel, indent + 1);

    // Print cond
    out.push_str(&inner_prefix);
    out.push_str("cond ");
    out.push_str(kernel.var_name(w.cond).unwrap_or("?"));
    out.push('\n');

    // Print body
    print_body(out, &w.body, kernel, indent + 1);

    // Print yield
    out.push_str(&inner_prefix);
    out.push_str("yield");
    for yv in &w.yields {
        out.push(' ');
        print_operand(out, *yv, kernel);
    }
    out.push('\n');

    out.push_str(&prefix);
    out.push_str("}\n");
}

fn print_inst(out: &mut String, inst: &Inst, kernel: &Kernel) {
    match inst {
        Inst::Const(c) => {
            out.push_str("const ");
            match c {
                Const::F32(v) => {
                    let s = format!("{v}");
                    out.push_str(&s);
                    if !s.contains('.') {
                        out.push_str(".0");
                    }
                    out.push_str("f32");
                }
                Const::F64(v) => {
                    let s = format!("{v}");
                    out.push_str(&s);
                    // Ensure it has a decimal point
                    if !s.contains('.') {
                        out.push_str(".0");
                    }
                }
                Const::I8(v) => out.push_str(&format!("{v}i8")),
                Const::U8(v) => out.push_str(&format!("{v}u8")),
                Const::I16(v) => out.push_str(&format!("{v}i16")),
                Const::U16(v) => out.push_str(&format!("{v}u16")),
                Const::I32(v) => out.push_str(&format!("{v}i32")),
                Const::U32(v) => out.push_str(&format!("{v}")),
                Const::I64(v) => out.push_str(&format!("{v}i64")),
                Const::U64(v) => out.push_str(&format!("{v}u64")),
                Const::Bool(v) => out.push_str(if *v { "true" } else { "false" }),
            }
        }
        Inst::Binary { op, lhs, rhs } => {
            out.push_str(binop_name(*op));
            out.push(' ');
            print_operand(out, *lhs, kernel);
            out.push(' ');
            print_operand(out, *rhs, kernel);
        }
        Inst::Unary { op, arg } => {
            out.push_str(unaryop_name(*op));
            out.push(' ');
            print_operand(out, *arg, kernel);
        }
        Inst::Cmp { op, lhs, rhs } => {
            out.push_str(cmpop_name(*op));
            out.push(' ');
            print_operand(out, *lhs, kernel);
            out.push(' ');
            print_operand(out, *rhs, kernel);
        }
        Inst::Conv { op, arg } => {
            out.push_str(convop_name(*op));
            out.push(' ');
            print_operand(out, *arg, kernel);
        }
        Inst::Select { cond, then_val, else_val } => {
            out.push_str("select ");
            print_operand(out, *cond, kernel);
            out.push(' ');
            print_operand(out, *then_val, kernel);
            out.push(' ');
            print_operand(out, *else_val, kernel);
        }
        Inst::PackArgb { r, g, b } => {
            out.push_str("pack_argb ");
            print_operand(out, *r, kernel);
            out.push(' ');
            print_operand(out, *g, kernel);
            out.push(' ');
            print_operand(out, *b, kernel);
        }
        Inst::MakeVec(components) => {
            out.push_str(&format!("make_vec{} ", components.len()));
            for (i, c) in components.iter().enumerate() {
                if i > 0 {
                    out.push(' ');
                }
                print_operand(out, *c, kernel);
            }
        }
        Inst::VecExtract { vec, index } => {
            let name = match index {
                0 => "extract_x",
                1 => "extract_y",
                2 => "extract_z",
                3 => "extract_w",
                _ => "extract_?",
            };
            out.push_str(name);
            out.push(' ');
            print_operand(out, *vec, kernel);
        }
        Inst::VecBinary { op, lhs, rhs } => {
            out.push_str(vecbinop_name(*op));
            out.push(' ');
            print_operand(out, *lhs, kernel);
            out.push(' ');
            print_operand(out, *rhs, kernel);
        }
        Inst::VecScale { scalar, vec } => {
            out.push_str("vec_scale ");
            print_operand(out, *scalar, kernel);
            out.push(' ');
            print_operand(out, *vec, kernel);
        }
        Inst::VecUnary { op, arg } => {
            out.push_str(vecunaryop_name(*op));
            out.push(' ');
            print_operand(out, *arg, kernel);
        }
        Inst::VecDot { lhs, rhs } => {
            out.push_str("vec_dot ");
            print_operand(out, *lhs, kernel);
            out.push(' ');
            print_operand(out, *rhs, kernel);
        }
        Inst::VecLength { arg } => {
            out.push_str("vec_length ");
            print_operand(out, *arg, kernel);
        }
        Inst::VecCross { lhs, rhs } => {
            out.push_str("vec_cross ");
            print_operand(out, *lhs, kernel);
            out.push(' ');
            print_operand(out, *rhs, kernel);
        }
        Inst::BufLoad { buf, x, y } => {
            let buf_name = &kernel.buffers[*buf as usize].name;
            out.push_str(&format!("buf_load {} ", buf_name));
            print_operand(out, *x, kernel);
            out.push(' ');
            print_operand(out, *y, kernel);
        }
        Inst::BufStore { buf, x, y, val } => {
            let buf_name = &kernel.buffers[*buf as usize].name;
            out.push_str(&format!("buf_store {} ", buf_name));
            print_operand(out, *x, kernel);
            out.push(' ');
            print_operand(out, *y, kernel);
            out.push(' ');
            print_operand(out, *val, kernel);
        }
    }
}

fn print_operand(out: &mut String, var: Var, kernel: &Kernel) {
    // Check if this is an implicit literal binding — if so, inline the value
    if let Some(b) = kernel.binding(var) {
        if b.name.starts_with("__lit_") {
            // Find the statement to get the const value
            if let Some(c) = find_const_in_body(&kernel.body, var) {
                match c {
                    Const::F32(v) => {
                        let s = format!("{v}");
                        out.push_str(&s);
                        if !s.contains('.') {
                            out.push_str(".0");
                        }
                        out.push_str("f32");
                        return;
                    }
                    Const::F64(v) => {
                        let s = format!("{v}");
                        out.push_str(&s);
                        if !s.contains('.') {
                            out.push_str(".0");
                        }
                        return;
                    }
                    Const::I8(v) => {
                        out.push_str(&format!("{v}i8"));
                        return;
                    }
                    Const::U8(v) => {
                        out.push_str(&format!("{v}u8"));
                        return;
                    }
                    Const::I16(v) => {
                        out.push_str(&format!("{v}i16"));
                        return;
                    }
                    Const::U16(v) => {
                        out.push_str(&format!("{v}u16"));
                        return;
                    }
                    Const::I32(v) => {
                        out.push_str(&format!("{v}i32"));
                        return;
                    }
                    Const::U32(v) => {
                        out.push_str(&format!("{v}"));
                        return;
                    }
                    Const::I64(v) => {
                        out.push_str(&format!("{v}i64"));
                        return;
                    }
                    Const::U64(v) => {
                        out.push_str(&format!("{v}u64"));
                        return;
                    }
                    Const::Bool(v) => {
                        out.push_str(if v { "true" } else { "false" });
                        return;
                    }
                }
            }
        }
    }
    out.push_str(kernel.var_name(var).unwrap_or("?"));
}

/// Search body items recursively for a Const instruction bound to `var`.
fn find_const_in_body(body: &[BodyItem], var: Var) -> Option<Const> {
    for item in body {
        match item {
            BodyItem::Stmt(stmt) => {
                if stmt.binding.var == var {
                    if let Inst::Const(c) = &stmt.inst {
                        return Some(*c);
                    }
                }
            }
            BodyItem::While(w) => {
                if let Some(c) = find_const_in_body(&w.cond_body, var) {
                    return Some(c);
                }
                if let Some(c) = find_const_in_body(&w.body, var) {
                    return Some(c);
                }
            }
        }
    }
    None
}

fn binop_name(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "add",
        BinOp::Sub => "sub",
        BinOp::Mul => "mul",
        BinOp::Div => "div",
        BinOp::Rem => "rem",
        BinOp::BitAnd => "bit_and",
        BinOp::BitOr => "bit_or",
        BinOp::BitXor => "bit_xor",
        BinOp::Shl => "shl",
        BinOp::Shr => "shr",
        BinOp::And => "and",
        BinOp::Or => "or",
        BinOp::Min => "min",
        BinOp::Max => "max",
        BinOp::Atan2 => "atan2",
        BinOp::Pow => "pow",
        BinOp::Hash => "hash",
    }
}

fn cmpop_name(op: CmpOp) -> &'static str {
    match op {
        CmpOp::Eq => "eq",
        CmpOp::Ne => "ne",
        CmpOp::Lt => "lt",
        CmpOp::Le => "le",
        CmpOp::Gt => "gt",
        CmpOp::Ge => "ge",
    }
}

fn unaryop_name(op: UnaryOp) -> &'static str {
    match op {
        UnaryOp::Neg => "neg",
        UnaryOp::Not => "not",
        UnaryOp::Abs => "abs",
        UnaryOp::Sqrt => "sqrt",
        UnaryOp::Floor => "floor",
        UnaryOp::Ceil => "ceil",
        UnaryOp::Sin => "sin",
        UnaryOp::Cos => "cos",
        UnaryOp::Tan => "tan",
        UnaryOp::Asin => "asin",
        UnaryOp::Acos => "acos",
        UnaryOp::Atan => "atan",
        UnaryOp::Exp => "exp",
        UnaryOp::Exp2 => "exp2",
        UnaryOp::Log => "log",
        UnaryOp::Log2 => "log2",
        UnaryOp::Log10 => "log10",
        UnaryOp::Round => "round",
        UnaryOp::Trunc => "trunc",
        UnaryOp::Fract => "fract",
    }
}

fn convop_name(op: ConvOp) -> &'static str {
    use ScalarType::*;
    match (op.from, op.to, op.norm) {
        (F64, U32, false) => "f64_to_u32",
        (U32, F64, false) => "u32_to_f64",
        (U32, F64, true) => "u32_to_f64_norm",
        (F32, F64, false) => "f32_to_f64",
        (F64, F32, false) => "f64_to_f32",
        (I32, U32, false) => "i32_to_u32",
        (U32, I32, false) => "u32_to_i32",
        (I32, F64, false) => "i32_to_f64",
        (F64, I32, false) => "f64_to_i32",
        (I32, F32, false) => "i32_to_f32",
        (F32, I32, false) => "f32_to_i32",
        (F32, U32, false) => "f32_to_u32",
        (U32, F32, false) => "u32_to_f32",
        _ => "conv_unknown",
    }
}

fn vecbinop_name(op: VecBinOp) -> &'static str {
    match op {
        VecBinOp::Add => "vec_add",
        VecBinOp::Sub => "vec_sub",
        VecBinOp::Mul => "vec_mul",
        VecBinOp::Div => "vec_div",
        VecBinOp::Min => "vec_min",
        VecBinOp::Max => "vec_max",
    }
}

fn vecunaryop_name(op: VecUnaryOp) -> &'static str {
    match op {
        VecUnaryOp::Neg => "vec_neg",
        VecUnaryOp::Abs => "vec_abs",
        VecUnaryOp::Normalize => "vec_normalize",
    }
}
