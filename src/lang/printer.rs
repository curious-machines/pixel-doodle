use crate::kernel_ir::*;

/// Print a Kernel IR back to .pdl text format.
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

    // Print cond_body statements
    for stmt in &w.cond_body {
        if stmt.binding.name.starts_with("__lit_") {
            continue;
        }
        out.push_str(&inner_prefix);
        out.push_str(&stmt.binding.name);
        out.push_str(": ");
        out.push_str(&format!("{}", stmt.binding.ty));
        out.push_str(" = ");
        print_inst(out, &stmt.inst, kernel);
        out.push('\n');
    }

    // Print cond
    out.push_str(&inner_prefix);
    out.push_str("cond ");
    out.push_str(kernel.var_name(w.cond).unwrap_or("?"));
    out.push('\n');

    // Print body statements
    for stmt in &w.body {
        if stmt.binding.name.starts_with("__lit_") {
            continue;
        }
        out.push_str(&inner_prefix);
        out.push_str(&stmt.binding.name);
        out.push_str(": ");
        out.push_str(&format!("{}", stmt.binding.ty));
        out.push_str(" = ");
        print_inst(out, &stmt.inst, kernel);
        out.push('\n');
    }

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
                Const::F64(v) => {
                    let s = format!("{v}");
                    out.push_str(&s);
                    // Ensure it has a decimal point
                    if !s.contains('.') {
                        out.push_str(".0");
                    }
                }
                Const::U32(v) => out.push_str(&format!("{v}")),
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
    }
}

fn print_operand(out: &mut String, var: Var, kernel: &Kernel) {
    // Check if this is an implicit literal binding — if so, inline the value
    if let Some(b) = kernel.binding(var) {
        if b.name.starts_with("__lit_") {
            // Find the statement to get the const value
            if let Some(c) = find_const_in_body(&kernel.body, var) {
                match c {
                    Const::F64(v) => {
                        let s = format!("{v}");
                        out.push_str(&s);
                        if !s.contains('.') {
                            out.push_str(".0");
                        }
                        return;
                    }
                    Const::U32(v) => {
                        out.push_str(&format!("{v}"));
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
                for s in &w.cond_body {
                    if s.binding.var == var {
                        if let Inst::Const(c) = &s.inst {
                            return Some(*c);
                        }
                    }
                }
                for s in &w.body {
                    if s.binding.var == var {
                        if let Inst::Const(c) = &s.inst {
                            return Some(*c);
                        }
                    }
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
    match op {
        ConvOp::F64ToU32 => "f64_to_u32",
        ConvOp::U32ToF64 => "u32_to_f64",
    }
}
