use crate::kernel_ir::*;
use super::ast::{BinOpKind, UnaryOpKind};
use super::typecheck::*;
use std::collections::HashMap;

struct Lowerer {
    next_var: u32,
    /// Maps names to Var indices.
    names: HashMap<String, Var>,
    /// Type cache.
    var_types: HashMap<Var, ValType>,
    /// Inline function definitions.
    fn_defs: HashMap<String, TFnDef>,
    /// Call counter for unique SSA names in inline expansion.
    call_count: u32,
    /// Maps buffer names to indices.
    buf_names: HashMap<String, u32>,
}

impl Lowerer {
    fn new() -> Self {
        Self {
            next_var: 0,
            names: HashMap::new(),
            var_types: HashMap::new(),
            fn_defs: HashMap::new(),
            call_count: 0,
            buf_names: HashMap::new(),
        }
    }

    fn fresh_var(&mut self, name: &str, ty: ValType) -> Var {
        let var = Var(self.next_var);
        self.next_var += 1;
        self.names.insert(name.to_string(), var);
        self.var_types.insert(var, ty);
        var
    }

    fn auto_var(&mut self, prefix: &str, ty: ValType) -> Var {
        let name = format!("_pd_{}_{}", prefix, self.next_var);
        self.fresh_var(&name, ty)
    }

    fn lookup(&self, name: &str) -> Var {
        self.names[name]
    }

    fn emit_stmt(&self, var: Var, name: &str, ty: ValType, inst: Inst) -> Statement {
        Statement {
            binding: Binding {
                var,
                name: name.to_string(),
                ty,
            },
            inst,
        }
    }

    /// Lower an expression, emitting SSA statements into `out`.
    /// Returns the Var holding the result.
    fn lower_expr(&mut self, expr: &TExpr, out: &mut Vec<BodyItem>) -> Var {
        match expr {
            TExpr::FloatLit(v) => {
                let var = self.auto_var("lit", ValType::F64);
                let name = self.binding_name(var);
                out.push(BodyItem::Stmt(self.emit_stmt(var, &name, ValType::F64, Inst::Const(Const::F64(*v)))));
                var
            }
            TExpr::F32Lit(v) => {
                let var = self.auto_var("lit", ValType::F32);
                let name = self.binding_name(var);
                out.push(BodyItem::Stmt(self.emit_stmt(var, &name, ValType::F32, Inst::Const(Const::F32(*v)))));
                var
            }
            TExpr::IntLit(v, ty) => {
                let var = self.auto_var("lit", ty.clone());
                let name = self.binding_name(var);
                let c = match ty {
                    ValType::Scalar(ScalarType::F64) => Const::F64(*v as f64),
                    ValType::Scalar(ScalarType::F32) => Const::F32(*v as f32),
                    ValType::Scalar(ScalarType::I8) => Const::I8(*v as i8),
                    ValType::Scalar(ScalarType::U8) => Const::U8(*v as u8),
                    ValType::Scalar(ScalarType::I16) => Const::I16(*v as i16),
                    ValType::Scalar(ScalarType::U16) => Const::U16(*v as u16),
                    ValType::Scalar(ScalarType::U32) => Const::U32(*v as u32),
                    ValType::Scalar(ScalarType::I32) => Const::I32(*v as i32),
                    ValType::Scalar(ScalarType::I64) => Const::I64(*v as i64),
                    ValType::Scalar(ScalarType::U64) => Const::U64(*v),
                    ValType::Scalar(ScalarType::Bool) | ValType::Vec { .. } | ValType::Mat { .. } | ValType::Array { .. } => unreachable!(),
                };
                out.push(BodyItem::Stmt(self.emit_stmt(var, &name, ty.clone(), Inst::Const(c))));
                var
            }
            TExpr::I8Lit(v) => {
                let var = self.auto_var("lit", ValType::I8);
                let name = self.binding_name(var);
                out.push(BodyItem::Stmt(self.emit_stmt(var, &name, ValType::I8, Inst::Const(Const::I8(*v)))));
                var
            }
            TExpr::U8Lit(v) => {
                let var = self.auto_var("lit", ValType::U8);
                let name = self.binding_name(var);
                out.push(BodyItem::Stmt(self.emit_stmt(var, &name, ValType::U8, Inst::Const(Const::U8(*v)))));
                var
            }
            TExpr::I16Lit(v) => {
                let var = self.auto_var("lit", ValType::I16);
                let name = self.binding_name(var);
                out.push(BodyItem::Stmt(self.emit_stmt(var, &name, ValType::I16, Inst::Const(Const::I16(*v)))));
                var
            }
            TExpr::U16Lit(v) => {
                let var = self.auto_var("lit", ValType::U16);
                let name = self.binding_name(var);
                out.push(BodyItem::Stmt(self.emit_stmt(var, &name, ValType::U16, Inst::Const(Const::U16(*v)))));
                var
            }
            TExpr::U32Lit(v) => {
                let var = self.auto_var("lit", ValType::U32);
                let name = self.binding_name(var);
                out.push(BodyItem::Stmt(self.emit_stmt(var, &name, ValType::U32, Inst::Const(Const::U32(*v)))));
                var
            }
            TExpr::I32Lit(v) => {
                let var = self.auto_var("lit", ValType::I32);
                let name = self.binding_name(var);
                out.push(BodyItem::Stmt(self.emit_stmt(var, &name, ValType::I32, Inst::Const(Const::I32(*v)))));
                var
            }
            TExpr::I64Lit(v) => {
                let var = self.auto_var("lit", ValType::I64);
                let name = self.binding_name(var);
                out.push(BodyItem::Stmt(self.emit_stmt(var, &name, ValType::I64, Inst::Const(Const::I64(*v)))));
                var
            }
            TExpr::U64Lit(v) => {
                let var = self.auto_var("lit", ValType::U64);
                let name = self.binding_name(var);
                out.push(BodyItem::Stmt(self.emit_stmt(var, &name, ValType::U64, Inst::Const(Const::U64(*v)))));
                var
            }
            TExpr::BoolLit(v) => {
                let var = self.auto_var("lit", ValType::BOOL);
                let name = self.binding_name(var);
                out.push(BodyItem::Stmt(self.emit_stmt(var, &name, ValType::BOOL, Inst::Const(Const::Bool(*v)))));
                var
            }
            TExpr::Ident(name, _) => {
                self.lookup(name)
            }
            TExpr::FieldAccess { expr, field, .. } => {
                let vec_var = self.lower_expr(expr, out);
                let index = match field.as_str() {
                    "x" => 0u8,
                    "y" => 1,
                    "z" => 2,
                    "w" => 3,
                    _ => unreachable!(),
                };
                let elem_ty = expr.ty().element_scalar();
                let var = self.auto_var("tmp", ValType::Scalar(elem_ty));
                let vname = self.binding_name(var);
                out.push(BodyItem::Stmt(self.emit_stmt(var, &vname, ValType::Scalar(elem_ty),
                    Inst::VecExtract { vec: vec_var, index })));
                var
            }
            TExpr::BinOp { op, lhs, rhs, ty } => {
                let l = self.lower_expr(lhs, out);
                let r = self.lower_expr(rhs, out);
                let var = self.auto_var("tmp", ty.clone());
                let name = self.binding_name(var);

                // Handle matrix operations
                if *op == BinOpKind::Mul && lhs.ty().is_mat() && rhs.ty().is_vec() {
                    // mat * vec -> vec (MatMulVec)
                    out.push(BodyItem::Stmt(self.emit_stmt(var, &name, ty.clone(),
                        Inst::MatMulVec { mat: l, vec: r })));
                    return var;
                }
                if *op == BinOpKind::Mul && lhs.ty().is_mat() && rhs.ty().is_mat() {
                    // mat * mat -> mat (MatMul)
                    out.push(BodyItem::Stmt(self.emit_stmt(var, &name, ty.clone(),
                        Inst::MatMul { lhs: l, rhs: r })));
                    return var;
                }

                // Handle vec operations
                if ty.is_vec() {
                    let inst = if *op == BinOpKind::Mul && lhs.ty() == ValType::F64 {
                        // f64 * vec
                        Inst::VecScale { scalar: l, vec: r }
                    } else if *op == BinOpKind::Mul && rhs.ty() == ValType::F64 {
                        // vec * f64
                        Inst::VecScale { scalar: r, vec: l }
                    } else if *op == BinOpKind::Div && rhs.ty() == ValType::F64 {
                        // vec / f64 → vec * (1/f64)
                        let one = self.auto_var("lit", ValType::F64);
                        let one_name = self.binding_name(one);
                        out.push(BodyItem::Stmt(self.emit_stmt(one, &one_name, ValType::F64, Inst::Const(Const::F64(1.0)))));
                        let recip = self.auto_var("tmp", ValType::F64);
                        let recip_name = self.binding_name(recip);
                        out.push(BodyItem::Stmt(self.emit_stmt(recip, &recip_name, ValType::F64,
                            Inst::Binary { op: BinOp::Div, lhs: one, rhs: r })));
                        Inst::VecScale { scalar: recip, vec: l }
                    } else {
                        let vec_op = match op {
                            BinOpKind::Add => VecBinOp::Add,
                            BinOpKind::Sub => VecBinOp::Sub,
                            BinOpKind::Mul => VecBinOp::Mul,
                            BinOpKind::Div => VecBinOp::Div,
                            _ => unreachable!("unsupported vec binop"),
                        };
                        Inst::VecBinary { op: vec_op, lhs: l, rhs: r }
                    };
                    out.push(BodyItem::Stmt(self.emit_stmt(var, &name, ty.clone(), inst)));
                    return var;
                }

                let inst = match op {
                    // Comparisons → Cmp
                    BinOpKind::Eq => Inst::Cmp { op: CmpOp::Eq, lhs: l, rhs: r },
                    BinOpKind::Ne => Inst::Cmp { op: CmpOp::Ne, lhs: l, rhs: r },
                    BinOpKind::Lt => Inst::Cmp { op: CmpOp::Lt, lhs: l, rhs: r },
                    BinOpKind::Le => Inst::Cmp { op: CmpOp::Le, lhs: l, rhs: r },
                    BinOpKind::Gt => Inst::Cmp { op: CmpOp::Gt, lhs: l, rhs: r },
                    BinOpKind::Ge => Inst::Cmp { op: CmpOp::Ge, lhs: l, rhs: r },
                    // Logical → Binary
                    BinOpKind::And => Inst::Binary { op: BinOp::And, lhs: l, rhs: r },
                    BinOpKind::Or => Inst::Binary { op: BinOp::Or, lhs: l, rhs: r },
                    // Arithmetic/bitwise → Binary
                    BinOpKind::Add => Inst::Binary { op: BinOp::Add, lhs: l, rhs: r },
                    BinOpKind::Sub => Inst::Binary { op: BinOp::Sub, lhs: l, rhs: r },
                    BinOpKind::Mul => Inst::Binary { op: BinOp::Mul, lhs: l, rhs: r },
                    BinOpKind::Div => Inst::Binary { op: BinOp::Div, lhs: l, rhs: r },
                    BinOpKind::Rem => Inst::Binary { op: BinOp::Rem, lhs: l, rhs: r },
                    BinOpKind::BitAnd => Inst::Binary { op: BinOp::BitAnd, lhs: l, rhs: r },
                    BinOpKind::BitOr => Inst::Binary { op: BinOp::BitOr, lhs: l, rhs: r },
                    BinOpKind::BitXor => Inst::Binary { op: BinOp::BitXor, lhs: l, rhs: r },
                    BinOpKind::Shl => Inst::Binary { op: BinOp::Shl, lhs: l, rhs: r },
                    BinOpKind::Shr => Inst::Binary { op: BinOp::Shr, lhs: l, rhs: r },
                };
                out.push(BodyItem::Stmt(self.emit_stmt(var, &name, ty.clone(), inst)));
                var
            }
            TExpr::UnaryOp { op, expr, ty } => {
                let arg = self.lower_expr(expr, out);
                let var = self.auto_var("tmp", ty.clone());
                let name = self.binding_name(var);
                if ty.is_vec() && *op == UnaryOpKind::Neg {
                    out.push(BodyItem::Stmt(self.emit_stmt(var, &name, ty.clone(),
                        Inst::VecUnary { op: VecUnaryOp::Neg, arg })));
                    return var;
                }
                let inst = match op {
                    UnaryOpKind::Neg => Inst::Unary { op: UnaryOp::Neg, arg },
                    UnaryOpKind::Not => Inst::Unary { op: UnaryOp::Not, arg },
                };
                out.push(BodyItem::Stmt(self.emit_stmt(var, &name, ty.clone(), inst)));
                var
            }
            TExpr::Call { name, args, ty } => {
                self.lower_call(name, args, ty.clone(), out)
            }
            TExpr::Cast { expr, from: _, to } => {
                let arg = self.lower_expr(expr, out);
                let var = self.auto_var("conv", to.clone());
                let vname = self.binding_name(var);
                let conv_op = match (expr.ty(), to) {
                    (ValType::Scalar(from_s), ValType::Scalar(to_s)) => {
                        ConvOp { from: from_s, to: *to_s, norm: false }
                    }
                    _ => unreachable!(),
                };
                out.push(BodyItem::Stmt(self.emit_stmt(var, &vname, to.clone(), Inst::Conv { op: conv_op, arg })));
                var
            }
            TExpr::IfElse { cond, then_expr, else_expr, ty } => {
                let c = self.lower_expr(cond, out);
                let t = self.lower_expr(then_expr, out);
                let e = self.lower_expr(else_expr, out);
                // For vec types, decompose into per-component selects
                if ty.is_vec() {
                    let n = ty.component_count();
                    let mut components = Vec::new();
                    for i in 0..n {
                        let t_comp = self.auto_var("tmp", ValType::F64);
                        let t_comp_name = self.binding_name(t_comp);
                        out.push(BodyItem::Stmt(self.emit_stmt(t_comp, &t_comp_name, ValType::F64,
                            Inst::VecExtract { vec: t, index: i as u8 })));
                        let e_comp = self.auto_var("tmp", ValType::F64);
                        let e_comp_name = self.binding_name(e_comp);
                        out.push(BodyItem::Stmt(self.emit_stmt(e_comp, &e_comp_name, ValType::F64,
                            Inst::VecExtract { vec: e, index: i as u8 })));
                        let selected = self.auto_var("sel", ValType::F64);
                        let sel_name = self.binding_name(selected);
                        out.push(BodyItem::Stmt(self.emit_stmt(selected, &sel_name, ValType::F64,
                            Inst::Select { cond: c, then_val: t_comp, else_val: e_comp })));
                        components.push(selected);
                    }
                    let var = self.auto_var("sel", ty.clone());
                    let name = self.binding_name(var);
                    out.push(BodyItem::Stmt(self.emit_stmt(var, &name, ty.clone(),
                        Inst::MakeVec(components))));
                    return var;
                }
                let var = self.auto_var("sel", ty.clone());
                let name = self.binding_name(var);
                out.push(BodyItem::Stmt(self.emit_stmt(var, &name, ty.clone(),
                    Inst::Select { cond: c, then_val: t, else_val: e })));
                var
            }
        }
    }

    fn lower_call(&mut self, name: &str, args: &[TExpr], ret_ty: ValType, out: &mut Vec<BodyItem>) -> Var {
        // Check builtins first
        if let Some(var) = self.lower_builtin_call(name, args, ret_ty, out) {
            return var;
        }

        // User-defined function: inline expansion
        let fdef = self.fn_defs.get(name).cloned()
            .expect("function not found (should have been caught by typechecker)");

        let call_id = self.call_count;
        self.call_count += 1;
        let prefix = format!("__{name}_{call_id}");

        // Save outer scope for params
        let saved_names: HashMap<String, Var> = self.names.clone();

        // Lower args and bind to param names
        let mut arg_vars = Vec::new();
        for arg in args {
            arg_vars.push(self.lower_expr(arg, out));
        }

        // Map param names to arg vars
        for (param, avar) in fdef.params.iter().zip(arg_vars.iter()) {
            self.names.insert(param.name.clone(), *avar);
        }

        // Lower function body statements
        let mut result_var = None;
        for stmt in &fdef.body {
            match stmt {
                TStmt::Emit { expr } => {
                    // `return expr` in fn = compute and use as result
                    result_var = Some(self.lower_expr_prefixed(expr, out, &prefix));
                }
                TStmt::Let { name: vname, ty, expr } => {
                    let v = self.lower_expr_prefixed(expr, out, &prefix);
                    let aliased = format!("{prefix}_{vname}");
                    let new_var = self.fresh_var(&aliased, ty.clone());
                    // Create a copy (identity) by const or just alias
                    // Actually we can just map the name
                    self.names.insert(vname.clone(), v);
                    // But we also need the prefixed name for nested calls
                    self.names.insert(aliased, v);
                    // The var_types entry for new_var is set but unused, that's fine
                    let _ = new_var; // suppress warning; we use name mapping instead
                }
                TStmt::While { carry, body } => {
                    self.lower_while_prefixed(carry, body, out, &prefix);
                }
                _ => {
                    // BreakIf, Yield shouldn't appear at fn top level
                }
            }
        }

        // Restore outer scope
        self.names = saved_names;

        // Re-insert any carry vars from while loops that were defined in the fn
        // This is handled by the while lowering defining vars in self.names

        result_var.expect("function must have a return statement")
    }

    fn lower_expr_prefixed(&mut self, expr: &TExpr, out: &mut Vec<BodyItem>, _prefix: &str) -> Var {
        // The prefix is applied through auto_var naming; the key thing is that
        // name lookups go through self.names which has been set up correctly
        self.lower_expr(expr, out)
    }

    fn lower_builtin_call(&mut self, name: &str, args: &[TExpr], ret_ty: ValType, out: &mut Vec<BodyItem>) -> Option<Var> {
        // Vec-specific builtins (must come before scalar fallbacks)
        match name {
            "mat2" | "mat3" | "mat4" => {
                let columns: Vec<Var> = args.iter().map(|a| self.lower_expr(a, out)).collect();
                let var = self.auto_var("tmp", ret_ty.clone());
                let vname = self.binding_name(var);
                out.push(BodyItem::Stmt(self.emit_stmt(var, &vname, ret_ty,
                    Inst::MakeMat(columns))));
                return Some(var);
            }
            "transpose" => {
                let a = self.lower_expr(&args[0], out);
                let var = self.auto_var("tmp", ret_ty.clone());
                let vname = self.binding_name(var);
                out.push(BodyItem::Stmt(self.emit_stmt(var, &vname, ret_ty,
                    Inst::MatTranspose { arg: a })));
                return Some(var);
            }
            "col" => {
                let mat = self.lower_expr(&args[0], out);
                let index = match &args[1] {
                    TExpr::U32Lit(i) => *i as u8,
                    TExpr::IntLit(i, _) => *i as u8,
                    _ => unreachable!("col index must be integer literal"),
                };
                let var = self.auto_var("tmp", ret_ty.clone());
                let vname = self.binding_name(var);
                out.push(BodyItem::Stmt(self.emit_stmt(var, &vname, ret_ty,
                    Inst::MatCol { mat, index })));
                return Some(var);
            }
            "vec2" | "vec3" | "vec4" => {
                let components: Vec<Var> = args.iter().map(|a| self.lower_expr(a, out)).collect();
                let var = self.auto_var("tmp", ret_ty.clone());
                let vname = self.binding_name(var);
                out.push(BodyItem::Stmt(self.emit_stmt(var, &vname, ret_ty,
                    Inst::MakeVec(components))));
                return Some(var);
            }
            "dot" => {
                let a = self.lower_expr(&args[0], out);
                let b = self.lower_expr(&args[1], out);
                let var = self.auto_var("tmp", ret_ty.clone());
                let vname = self.binding_name(var);
                out.push(BodyItem::Stmt(self.emit_stmt(var, &vname, ret_ty,
                    Inst::VecDot { lhs: a, rhs: b })));
                return Some(var);
            }
            "normalize" => {
                let a = self.lower_expr(&args[0], out);
                let ty = args[0].ty();
                let var = self.auto_var("tmp", ty.clone());
                let vname = self.binding_name(var);
                out.push(BodyItem::Stmt(self.emit_stmt(var, &vname, ty,
                    Inst::VecUnary { op: VecUnaryOp::Normalize, arg: a })));
                return Some(var);
            }
            "cross" => {
                let a = self.lower_expr(&args[0], out);
                let b = self.lower_expr(&args[1], out);
                let var = self.auto_var("tmp", ret_ty.clone());
                let vname = self.binding_name(var);
                out.push(BodyItem::Stmt(self.emit_stmt(var, &vname, ret_ty,
                    Inst::VecCross { lhs: a, rhs: b })));
                return Some(var);
            }
            "length" if args.len() == 1 && args[0].ty().is_vec() => {
                let a = self.lower_expr(&args[0], out);
                let var = self.auto_var("tmp", ret_ty.clone());
                let vname = self.binding_name(var);
                out.push(BodyItem::Stmt(self.emit_stmt(var, &vname, ret_ty,
                    Inst::VecLength { arg: a })));
                return Some(var);
            }
            "distance" if args.len() == 2 && args[0].ty().is_vec() => {
                let a = self.lower_expr(&args[0], out);
                let b = self.lower_expr(&args[1], out);
                let vec_ty = args[0].ty();
                let diff = self.auto_var("tmp", vec_ty.clone());
                let diff_name = self.binding_name(diff);
                out.push(BodyItem::Stmt(self.emit_stmt(diff, &diff_name, vec_ty,
                    Inst::VecBinary { op: VecBinOp::Sub, lhs: a, rhs: b })));
                let var = self.auto_var("tmp", ret_ty.clone());
                let vname = self.binding_name(var);
                out.push(BodyItem::Stmt(self.emit_stmt(var, &vname, ret_ty,
                    Inst::VecLength { arg: diff })));
                return Some(var);
            }
            "mix" if args.len() >= 1 && args[0].ty().is_vec() => {
                let a = self.lower_expr(&args[0], out);
                let b = self.lower_expr(&args[1], out);
                let t = self.lower_expr(&args[2], out);
                let vec_ty = args[0].ty();
                let elem_ty = ValType::Scalar(vec_ty.element_scalar());
                let one_const = match vec_ty.element_scalar() {
                    ScalarType::F32 => Const::F32(1.0),
                    ScalarType::F64 => Const::F64(1.0),
                    _ => Const::F64(1.0), // fallback
                };
                // mix(a, b, t) = a*(1-t) + b*t
                let one = self.auto_var("lit", elem_ty.clone());
                let one_name = self.binding_name(one);
                out.push(BodyItem::Stmt(self.emit_stmt(one, &one_name, elem_ty.clone(), Inst::Const(one_const))));
                let one_minus_t = self.auto_var("tmp", elem_ty.clone());
                let omt_name = self.binding_name(one_minus_t);
                out.push(BodyItem::Stmt(self.emit_stmt(one_minus_t, &omt_name, elem_ty,
                    Inst::Binary { op: BinOp::Sub, lhs: one, rhs: t })));
                let a_scaled = self.auto_var("tmp", vec_ty.clone());
                let as_name = self.binding_name(a_scaled);
                out.push(BodyItem::Stmt(self.emit_stmt(a_scaled, &as_name, vec_ty.clone(),
                    Inst::VecScale { scalar: one_minus_t, vec: a })));
                let b_scaled = self.auto_var("tmp", vec_ty.clone());
                let bs_name = self.binding_name(b_scaled);
                out.push(BodyItem::Stmt(self.emit_stmt(b_scaled, &bs_name, vec_ty.clone(),
                    Inst::VecScale { scalar: t, vec: b })));
                let var = self.auto_var("tmp", vec_ty.clone());
                let vname = self.binding_name(var);
                out.push(BodyItem::Stmt(self.emit_stmt(var, &vname, vec_ty,
                    Inst::VecBinary { op: VecBinOp::Add, lhs: a_scaled, rhs: b_scaled })));
                return Some(var);
            }
            "abs" if args.len() == 1 && args[0].ty().is_vec() => {
                let a = self.lower_expr(&args[0], out);
                let ty = args[0].ty();
                let var = self.auto_var("tmp", ty.clone());
                let vname = self.binding_name(var);
                out.push(BodyItem::Stmt(self.emit_stmt(var, &vname, ty,
                    Inst::VecUnary { op: VecUnaryOp::Abs, arg: a })));
                return Some(var);
            }
            "min" if args.len() >= 1 && args[0].ty().is_vec() => {
                let a = self.lower_expr(&args[0], out);
                let b = self.lower_expr(&args[1], out);
                let ty = args[0].ty();
                let var = self.auto_var("tmp", ty.clone());
                let vname = self.binding_name(var);
                out.push(BodyItem::Stmt(self.emit_stmt(var, &vname, ty,
                    Inst::VecBinary { op: VecBinOp::Min, lhs: a, rhs: b })));
                return Some(var);
            }
            "max" if args.len() >= 1 && args[0].ty().is_vec() => {
                let a = self.lower_expr(&args[0], out);
                let b = self.lower_expr(&args[1], out);
                let ty = args[0].ty();
                let var = self.auto_var("tmp", ty.clone());
                let vname = self.binding_name(var);
                out.push(BodyItem::Stmt(self.emit_stmt(var, &vname, ty,
                    Inst::VecBinary { op: VecBinOp::Max, lhs: a, rhs: b })));
                return Some(var);
            }
            _ => {}
        }

        // Map unary f64 builtins to UnaryOp
        let unary_op = match name {
            "abs" => Some(UnaryOp::Abs),
            "sqrt" => Some(UnaryOp::Sqrt),
            "floor" => Some(UnaryOp::Floor),
            "ceil" => Some(UnaryOp::Ceil),
            "sin" => Some(UnaryOp::Sin),
            "cos" => Some(UnaryOp::Cos),
            "tan" => Some(UnaryOp::Tan),
            "asin" => Some(UnaryOp::Asin),
            "acos" => Some(UnaryOp::Acos),
            "atan" => Some(UnaryOp::Atan),
            "exp" => Some(UnaryOp::Exp),
            "exp2" => Some(UnaryOp::Exp2),
            "log" => Some(UnaryOp::Log),
            "log2" => Some(UnaryOp::Log2),
            "log10" => Some(UnaryOp::Log10),
            "round" => Some(UnaryOp::Round),
            "trunc" => Some(UnaryOp::Trunc),
            "fract" => Some(UnaryOp::Fract),
            _ => None,
        };
        if let Some(uop) = unary_op {
            let arg = self.lower_expr(&args[0], out);
            let var = self.auto_var("tmp", ret_ty.clone());
            let vname = self.binding_name(var);
            out.push(BodyItem::Stmt(self.emit_stmt(var, &vname, ret_ty,
                Inst::Unary { op: uop, arg })));
            return Some(var);
        }

        // Binary f64 builtins
        let bin_op = match name {
            "min" => Some(BinOp::Min),
            "max" => Some(BinOp::Max),
            "atan2" => Some(BinOp::Atan2),
            "pow" => Some(BinOp::Pow),
            "hash" => Some(BinOp::Hash),
            _ => None,
        };
        if let Some(bop) = bin_op {
            let l = self.lower_expr(&args[0], out);
            let r = self.lower_expr(&args[1], out);
            let var = self.auto_var("tmp", ret_ty.clone());
            let vname = self.binding_name(var);
            out.push(BodyItem::Stmt(self.emit_stmt(var, &vname, ret_ty,
                Inst::Binary { op: bop, lhs: l, rhs: r })));
            return Some(var);
        }

        // Conversion builtins
        match name {
            "f64_to_u32" => {
                let arg = self.lower_expr(&args[0], out);
                let var = self.auto_var("conv", ValType::U32);
                let vname = self.binding_name(var);
                out.push(BodyItem::Stmt(self.emit_stmt(var, &vname, ValType::U32,
                    Inst::Conv { op: ConvOp::F64_TO_U32, arg })));
                return Some(var);
            }
            "u32_to_f64" => {
                let arg = self.lower_expr(&args[0], out);
                let var = self.auto_var("conv", ValType::F64);
                let vname = self.binding_name(var);
                out.push(BodyItem::Stmt(self.emit_stmt(var, &vname, ValType::F64,
                    Inst::Conv { op: ConvOp::U32_TO_F64, arg })));
                return Some(var);
            }
            "norm" => {
                let arg = self.lower_expr(&args[0], out);
                let var = self.auto_var("conv", ValType::F64);
                let vname = self.binding_name(var);
                out.push(BodyItem::Stmt(self.emit_stmt(var, &vname, ValType::F64,
                    Inst::Conv { op: ConvOp::U32_TO_F64_NORM, arg })));
                return Some(var);
            }
            _ => {}
        }

        // select(cond, then, else)
        if name == "select" {
            let c = self.lower_expr(&args[0], out);
            let t = self.lower_expr(&args[1], out);
            let e = self.lower_expr(&args[2], out);
            let var = self.auto_var("sel", ret_ty.clone());
            let vname = self.binding_name(var);
            out.push(BodyItem::Stmt(self.emit_stmt(var, &vname, ret_ty,
                Inst::Select { cond: c, then_val: t, else_val: e })));
            return Some(var);
        }

        // buf_load:buf_name(x, y) — encoded as "buf_load:name" by typechecker
        if let Some(buf_name) = name.strip_prefix("buf_load:") {
            let buf = self.buf_names[buf_name];
            let x = self.lower_expr(&args[0], out);
            let y = self.lower_expr(&args[1], out);
            let var = self.auto_var("bld", ValType::F64);
            let vname = self.binding_name(var);
            out.push(BodyItem::Stmt(self.emit_stmt(var, &vname, ValType::F64,
                Inst::BufLoad { buf, x, y })));
            return Some(var);
        }

        // pack_argb(r, g, b)
        if name == "pack_argb" {
            let r = self.lower_expr(&args[0], out);
            let g = self.lower_expr(&args[1], out);
            let b = self.lower_expr(&args[2], out);
            let var = self.auto_var("tmp", ValType::U32);
            let vname = self.binding_name(var);
            out.push(BodyItem::Stmt(self.emit_stmt(var, &vname, ValType::U32,
                Inst::PackArgb { r, g, b })));
            return Some(var);
        }

        // Convenience builtins (expand to multiple IR ops)
        match name {
            "clamp" => {
                // clamp(x, lo, hi) = min(max(x, lo), hi)
                let x = self.lower_expr(&args[0], out);
                let lo = self.lower_expr(&args[1], out);
                let hi = self.lower_expr(&args[2], out);
                let v1 = self.auto_var("tmp", ValType::F64);
                let n1 = self.binding_name(v1);
                out.push(BodyItem::Stmt(self.emit_stmt(v1, &n1, ValType::F64,
                    Inst::Binary { op: BinOp::Max, lhs: x, rhs: lo })));
                let v2 = self.auto_var("tmp", ValType::F64);
                let n2 = self.binding_name(v2);
                out.push(BodyItem::Stmt(self.emit_stmt(v2, &n2, ValType::F64,
                    Inst::Binary { op: BinOp::Min, lhs: v1, rhs: hi })));
                return Some(v2);
            }
            "saturate" => {
                // saturate(x) = clamp(x, 0, 1)
                let x = self.lower_expr(&args[0], out);
                let z = self.auto_var("lit", ValType::F64);
                let zn = self.binding_name(z);
                out.push(BodyItem::Stmt(self.emit_stmt(z, &zn, ValType::F64, Inst::Const(Const::F64(0.0)))));
                let o = self.auto_var("lit", ValType::F64);
                let on = self.binding_name(o);
                out.push(BodyItem::Stmt(self.emit_stmt(o, &on, ValType::F64, Inst::Const(Const::F64(1.0)))));
                let v1 = self.auto_var("tmp", ValType::F64);
                let n1 = self.binding_name(v1);
                out.push(BodyItem::Stmt(self.emit_stmt(v1, &n1, ValType::F64,
                    Inst::Binary { op: BinOp::Max, lhs: x, rhs: z })));
                let v2 = self.auto_var("tmp", ValType::F64);
                let n2 = self.binding_name(v2);
                out.push(BodyItem::Stmt(self.emit_stmt(v2, &n2, ValType::F64,
                    Inst::Binary { op: BinOp::Min, lhs: v1, rhs: o })));
                return Some(v2);
            }
            "length" => {
                // length(x, y) = sqrt(x*x + y*y)
                let x = self.lower_expr(&args[0], out);
                let y = self.lower_expr(&args[1], out);
                let xx = self.auto_var("tmp", ValType::F64);
                let xxn = self.binding_name(xx);
                out.push(BodyItem::Stmt(self.emit_stmt(xx, &xxn, ValType::F64,
                    Inst::Binary { op: BinOp::Mul, lhs: x, rhs: x })));
                let yy = self.auto_var("tmp", ValType::F64);
                let yyn = self.binding_name(yy);
                out.push(BodyItem::Stmt(self.emit_stmt(yy, &yyn, ValType::F64,
                    Inst::Binary { op: BinOp::Mul, lhs: y, rhs: y })));
                let sum = self.auto_var("tmp", ValType::F64);
                let sn = self.binding_name(sum);
                out.push(BodyItem::Stmt(self.emit_stmt(sum, &sn, ValType::F64,
                    Inst::Binary { op: BinOp::Add, lhs: xx, rhs: yy })));
                let r = self.auto_var("tmp", ValType::F64);
                let rn = self.binding_name(r);
                out.push(BodyItem::Stmt(self.emit_stmt(r, &rn, ValType::F64,
                    Inst::Unary { op: UnaryOp::Sqrt, arg: sum })));
                return Some(r);
            }
            "distance" => {
                // distance(x1,y1,x2,y2) = length(x2-x1, y2-y1)
                let x1 = self.lower_expr(&args[0], out);
                let y1 = self.lower_expr(&args[1], out);
                let x2 = self.lower_expr(&args[2], out);
                let y2 = self.lower_expr(&args[3], out);
                let dx = self.auto_var("tmp", ValType::F64);
                let dxn = self.binding_name(dx);
                out.push(BodyItem::Stmt(self.emit_stmt(dx, &dxn, ValType::F64,
                    Inst::Binary { op: BinOp::Sub, lhs: x2, rhs: x1 })));
                let dy = self.auto_var("tmp", ValType::F64);
                let dyn_ = self.binding_name(dy);
                out.push(BodyItem::Stmt(self.emit_stmt(dy, &dyn_, ValType::F64,
                    Inst::Binary { op: BinOp::Sub, lhs: y2, rhs: y1 })));
                let dxdx = self.auto_var("tmp", ValType::F64);
                let dxdxn = self.binding_name(dxdx);
                out.push(BodyItem::Stmt(self.emit_stmt(dxdx, &dxdxn, ValType::F64,
                    Inst::Binary { op: BinOp::Mul, lhs: dx, rhs: dx })));
                let dydy = self.auto_var("tmp", ValType::F64);
                let dydyn = self.binding_name(dydy);
                out.push(BodyItem::Stmt(self.emit_stmt(dydy, &dydyn, ValType::F64,
                    Inst::Binary { op: BinOp::Mul, lhs: dy, rhs: dy })));
                let s = self.auto_var("tmp", ValType::F64);
                let sn = self.binding_name(s);
                out.push(BodyItem::Stmt(self.emit_stmt(s, &sn, ValType::F64,
                    Inst::Binary { op: BinOp::Add, lhs: dxdx, rhs: dydy })));
                let r = self.auto_var("tmp", ValType::F64);
                let rn = self.binding_name(r);
                out.push(BodyItem::Stmt(self.emit_stmt(r, &rn, ValType::F64,
                    Inst::Unary { op: UnaryOp::Sqrt, arg: s })));
                return Some(r);
            }
            "mix" => {
                // mix(a, b, t) = a + t * (b - a)
                let a = self.lower_expr(&args[0], out);
                let b = self.lower_expr(&args[1], out);
                let t = self.lower_expr(&args[2], out);
                let diff = self.auto_var("tmp", ValType::F64);
                let dn = self.binding_name(diff);
                out.push(BodyItem::Stmt(self.emit_stmt(diff, &dn, ValType::F64,
                    Inst::Binary { op: BinOp::Sub, lhs: b, rhs: a })));
                let td = self.auto_var("tmp", ValType::F64);
                let tdn = self.binding_name(td);
                out.push(BodyItem::Stmt(self.emit_stmt(td, &tdn, ValType::F64,
                    Inst::Binary { op: BinOp::Mul, lhs: t, rhs: diff })));
                let r = self.auto_var("tmp", ValType::F64);
                let rn = self.binding_name(r);
                out.push(BodyItem::Stmt(self.emit_stmt(r, &rn, ValType::F64,
                    Inst::Binary { op: BinOp::Add, lhs: a, rhs: td })));
                return Some(r);
            }
            "smoothstep" => {
                // smoothstep(e0, e1, x): t = clamp((x-e0)/(e1-e0), 0, 1); return t*t*(3-2*t)
                let e0 = self.lower_expr(&args[0], out);
                let e1 = self.lower_expr(&args[1], out);
                let x = self.lower_expr(&args[2], out);

                let diff = self.auto_var("tmp", ValType::F64);
                let dn = self.binding_name(diff);
                out.push(BodyItem::Stmt(self.emit_stmt(diff, &dn, ValType::F64,
                    Inst::Binary { op: BinOp::Sub, lhs: x, rhs: e0 })));
                let range = self.auto_var("tmp", ValType::F64);
                let rn = self.binding_name(range);
                out.push(BodyItem::Stmt(self.emit_stmt(range, &rn, ValType::F64,
                    Inst::Binary { op: BinOp::Sub, lhs: e1, rhs: e0 })));
                let raw = self.auto_var("tmp", ValType::F64);
                let rwn = self.binding_name(raw);
                out.push(BodyItem::Stmt(self.emit_stmt(raw, &rwn, ValType::F64,
                    Inst::Binary { op: BinOp::Div, lhs: diff, rhs: range })));

                // clamp to [0, 1]
                let z = self.auto_var("lit", ValType::F64);
                let zn = self.binding_name(z);
                out.push(BodyItem::Stmt(self.emit_stmt(z, &zn, ValType::F64, Inst::Const(Const::F64(0.0)))));
                let o = self.auto_var("lit", ValType::F64);
                let on = self.binding_name(o);
                out.push(BodyItem::Stmt(self.emit_stmt(o, &on, ValType::F64, Inst::Const(Const::F64(1.0)))));
                let clamped1 = self.auto_var("tmp", ValType::F64);
                let c1n = self.binding_name(clamped1);
                out.push(BodyItem::Stmt(self.emit_stmt(clamped1, &c1n, ValType::F64,
                    Inst::Binary { op: BinOp::Max, lhs: raw, rhs: z })));
                let t = self.auto_var("tmp", ValType::F64);
                let tn = self.binding_name(t);
                out.push(BodyItem::Stmt(self.emit_stmt(t, &tn, ValType::F64,
                    Inst::Binary { op: BinOp::Min, lhs: clamped1, rhs: o })));

                // t * t * (3 - 2*t)
                let tt = self.auto_var("tmp", ValType::F64);
                let ttn = self.binding_name(tt);
                out.push(BodyItem::Stmt(self.emit_stmt(tt, &ttn, ValType::F64,
                    Inst::Binary { op: BinOp::Mul, lhs: t, rhs: t })));
                let two = self.auto_var("lit", ValType::F64);
                let twon = self.binding_name(two);
                out.push(BodyItem::Stmt(self.emit_stmt(two, &twon, ValType::F64, Inst::Const(Const::F64(2.0)))));
                let three = self.auto_var("lit", ValType::F64);
                let threen = self.binding_name(three);
                out.push(BodyItem::Stmt(self.emit_stmt(three, &threen, ValType::F64, Inst::Const(Const::F64(3.0)))));
                let twot = self.auto_var("tmp", ValType::F64);
                let twotn = self.binding_name(twot);
                out.push(BodyItem::Stmt(self.emit_stmt(twot, &twotn, ValType::F64,
                    Inst::Binary { op: BinOp::Mul, lhs: two, rhs: t })));
                let three_minus = self.auto_var("tmp", ValType::F64);
                let tmn = self.binding_name(three_minus);
                out.push(BodyItem::Stmt(self.emit_stmt(three_minus, &tmn, ValType::F64,
                    Inst::Binary { op: BinOp::Sub, lhs: three, rhs: twot })));
                let result = self.auto_var("tmp", ValType::F64);
                let resn = self.binding_name(result);
                out.push(BodyItem::Stmt(self.emit_stmt(result, &resn, ValType::F64,
                    Inst::Binary { op: BinOp::Mul, lhs: tt, rhs: three_minus })));
                return Some(result);
            }
            "step" => {
                // step(edge, x) = if x < edge { 0.0 } else { 1.0 }
                let edge = self.lower_expr(&args[0], out);
                let x = self.lower_expr(&args[1], out);
                let cond = self.auto_var("tmp", ValType::BOOL);
                let cn = self.binding_name(cond);
                out.push(BodyItem::Stmt(self.emit_stmt(cond, &cn, ValType::BOOL,
                    Inst::Cmp { op: CmpOp::Lt, lhs: x, rhs: edge })));
                let z = self.auto_var("lit", ValType::F64);
                let zn = self.binding_name(z);
                out.push(BodyItem::Stmt(self.emit_stmt(z, &zn, ValType::F64, Inst::Const(Const::F64(0.0)))));
                let o = self.auto_var("lit", ValType::F64);
                let on = self.binding_name(o);
                out.push(BodyItem::Stmt(self.emit_stmt(o, &on, ValType::F64, Inst::Const(Const::F64(1.0)))));
                let r = self.auto_var("sel", ValType::F64);
                let rn = self.binding_name(r);
                out.push(BodyItem::Stmt(self.emit_stmt(r, &rn, ValType::F64,
                    Inst::Select { cond, then_val: z, else_val: o })));
                return Some(r);
            }
            "rgb" => {
                // rgb(r, g, b) = scale [0,1] by 255, convert, pack
                return Some(self.lower_rgb_scale(&args[0], &args[1], &args[2], 255.0, out));
            }
            "rgb255" => {
                // rgb255(r, g, b) = f64_to_u32 + pack_argb (values already 0-255)
                return Some(self.lower_rgb_direct(&args[0], &args[1], &args[2], out));
            }
            "gray" => {
                // gray(v) = rgb(v, v, v)
                return Some(self.lower_rgb_scale(&args[0], &args[0], &args[0], 255.0, out));
            }
            "gray255" => {
                return Some(self.lower_rgb_direct(&args[0], &args[0], &args[0], out));
            }
            _ => {}
        }

        None
    }

    fn lower_rgb_scale(&mut self, r: &TExpr, g: &TExpr, b: &TExpr, scale: f64, out: &mut Vec<BodyItem>) -> Var {
        let rv = self.lower_expr(r, out);
        let gv = self.lower_expr(g, out);
        let bv = self.lower_expr(b, out);

        let s = self.auto_var("lit", ValType::F64);
        let sn = self.binding_name(s);
        out.push(BodyItem::Stmt(self.emit_stmt(s, &sn, ValType::F64, Inst::Const(Const::F64(scale)))));

        let rs = self.auto_var("tmp", ValType::F64);
        let rsn = self.binding_name(rs);
        out.push(BodyItem::Stmt(self.emit_stmt(rs, &rsn, ValType::F64,
            Inst::Binary { op: BinOp::Mul, lhs: rv, rhs: s })));
        let gs = self.auto_var("tmp", ValType::F64);
        let gsn = self.binding_name(gs);
        out.push(BodyItem::Stmt(self.emit_stmt(gs, &gsn, ValType::F64,
            Inst::Binary { op: BinOp::Mul, lhs: gv, rhs: s })));
        let bs = self.auto_var("tmp", ValType::F64);
        let bsn = self.binding_name(bs);
        out.push(BodyItem::Stmt(self.emit_stmt(bs, &bsn, ValType::F64,
            Inst::Binary { op: BinOp::Mul, lhs: bv, rhs: s })));

        let ru = self.auto_var("conv", ValType::U32);
        let run = self.binding_name(ru);
        out.push(BodyItem::Stmt(self.emit_stmt(ru, &run, ValType::U32,
            Inst::Conv { op: ConvOp::F64_TO_U32, arg: rs })));
        let gu = self.auto_var("conv", ValType::U32);
        let gun = self.binding_name(gu);
        out.push(BodyItem::Stmt(self.emit_stmt(gu, &gun, ValType::U32,
            Inst::Conv { op: ConvOp::F64_TO_U32, arg: gs })));
        let bu = self.auto_var("conv", ValType::U32);
        let bun = self.binding_name(bu);
        out.push(BodyItem::Stmt(self.emit_stmt(bu, &bun, ValType::U32,
            Inst::Conv { op: ConvOp::F64_TO_U32, arg: bs })));

        let packed = self.auto_var("tmp", ValType::U32);
        let pn = self.binding_name(packed);
        out.push(BodyItem::Stmt(self.emit_stmt(packed, &pn, ValType::U32,
            Inst::PackArgb { r: ru, g: gu, b: bu })));
        packed
    }

    fn lower_rgb_direct(&mut self, r: &TExpr, g: &TExpr, b: &TExpr, out: &mut Vec<BodyItem>) -> Var {
        let rv = self.lower_expr(r, out);
        let gv = self.lower_expr(g, out);
        let bv = self.lower_expr(b, out);

        let ru = self.auto_var("conv", ValType::U32);
        let run = self.binding_name(ru);
        out.push(BodyItem::Stmt(self.emit_stmt(ru, &run, ValType::U32,
            Inst::Conv { op: ConvOp::F64_TO_U32, arg: rv })));
        let gu = self.auto_var("conv", ValType::U32);
        let gun = self.binding_name(gu);
        out.push(BodyItem::Stmt(self.emit_stmt(gu, &gun, ValType::U32,
            Inst::Conv { op: ConvOp::F64_TO_U32, arg: gv })));
        let bu = self.auto_var("conv", ValType::U32);
        let bun = self.binding_name(bu);
        out.push(BodyItem::Stmt(self.emit_stmt(bu, &bun, ValType::U32,
            Inst::Conv { op: ConvOp::F64_TO_U32, arg: bv })));

        let packed = self.auto_var("tmp", ValType::U32);
        let pn = self.binding_name(packed);
        out.push(BodyItem::Stmt(self.emit_stmt(packed, &pn, ValType::U32,
            Inst::PackArgb { r: ru, g: gu, b: bu })));
        packed
    }

    fn binding_name(&self, var: Var) -> String {
        // Reverse lookup — find the name we gave this var
        for (name, &v) in &self.names {
            if v == var {
                return name.clone();
            }
        }
        format!("__v{}", var.0)
    }

    fn lower_stmts(&mut self, stmts: &[TStmt], out: &mut Vec<BodyItem>) -> Option<Var> {
        let mut emit_var = None;
        for stmt in stmts {
            match stmt {
                TStmt::Let { name, ty, expr } => {
                    let v = self.lower_expr(expr, out);
                    // Register the name to point to the computed var
                    self.names.insert(name.clone(), v);
                    // Also update var_types
                    self.var_types.insert(v, ty.clone());
                    // Rename the last emitted statement to use this name
                    if let Some(BodyItem::Stmt(last)) = out.last_mut() {
                        if last.binding.var == v {
                            last.binding.name = name.clone();
                        }
                    }
                }
                TStmt::While { carry, body } => {
                    self.lower_while(carry, body, out);
                }
                TStmt::Emit { expr } => {
                    emit_var = Some(self.lower_expr(expr, out));
                }
                TStmt::BufStore { buf_name, x, y, val } => {
                    let buf = self.buf_names[buf_name.as_str()];
                    let xv = self.lower_expr(x, out);
                    let yv = self.lower_expr(y, out);
                    let vv = self.lower_expr(val, out);
                    let var = self.auto_var("bst", ValType::U32);
                    let vname = self.binding_name(var);
                    out.push(BodyItem::Stmt(self.emit_stmt(var, &vname, ValType::U32,
                        Inst::BufStore { buf, x: xv, y: yv, val: vv })));
                }
                TStmt::BreakIf { .. } | TStmt::Yield { .. } => {
                    // These are handled inside lower_while
                    unreachable!("break_if/yield outside while");
                }
            }
        }
        emit_var
    }

    fn lower_while(&mut self, carry: &[TCarryDef], body: &[TStmt], out: &mut Vec<BodyItem>) {
        self.lower_while_prefixed(carry, body, out, "");
    }

    fn lower_while_prefixed(&mut self, carry: &[TCarryDef], body: &[TStmt], out: &mut Vec<BodyItem>, _prefix: &str) {
        // Lower carry init expressions in outer scope
        let mut carry_vars = Vec::new();
        for c in carry {
            let init = self.lower_expr(&c.init, out);
            let var = self.fresh_var(&c.name, c.ty.clone());
            carry_vars.push(CarryVar {
                binding: Binding {
                    var,
                    name: c.name.clone(),
                    ty: c.ty.clone(),
                },
                init,
            });
        }

        // Split body into: pre-break_if (cond_body), break_if (cond), post-break_if (loop body)
        let mut break_idx = None;
        for (i, stmt) in body.iter().enumerate() {
            if matches!(stmt, TStmt::BreakIf { .. }) {
                break_idx = Some(i);
                break;
            }
        }

        let break_idx = break_idx.expect("while loop must have break_if");

        // Cond body: statements before break_if
        let mut cond_body = Vec::new();
        for stmt in &body[..break_idx] {
            self.lower_stmt_in_loop(stmt, &mut cond_body);
        }

        // Break condition: break_if expr → cond = not(expr)
        let break_cond = match &body[break_idx] {
            TStmt::BreakIf { cond } => cond,
            _ => unreachable!(),
        };
        let break_var = self.lower_expr(break_cond, &mut cond_body);
        let cont_var = self.auto_var("cont", ValType::BOOL);
        let cont_name = self.binding_name(cont_var);
        cond_body.push(BodyItem::Stmt(self.emit_stmt(cont_var, &cont_name, ValType::BOOL,
            Inst::Unary { op: UnaryOp::Not, arg: break_var })));

        // Loop body: statements after break_if (excluding yield)
        let mut loop_body = Vec::new();
        let mut yield_vars = Vec::new();

        for stmt in &body[break_idx + 1..] {
            match stmt {
                TStmt::Yield { values } => {
                    for val in values {
                        yield_vars.push(self.lower_expr(val, &mut loop_body));
                    }
                }
                other => {
                    self.lower_stmt_in_loop(other, &mut loop_body);
                }
            }
        }

        out.push(BodyItem::While(While {
            carry: carry_vars,
            cond_body,
            cond: cont_var,
            body: loop_body,
            yields: yield_vars,
        }));
    }

    fn lower_stmt_in_loop(&mut self, stmt: &TStmt, out: &mut Vec<BodyItem>) {
        match stmt {
            TStmt::Let { name, ty, expr } => {
                let v = self.lower_expr(expr, out);
                self.names.insert(name.clone(), v);
                self.var_types.insert(v, ty.clone());
                if let Some(BodyItem::Stmt(last)) = out.last_mut() {
                    if last.binding.var == v {
                        last.binding.name = name.clone();
                    }
                }
            }
            TStmt::While { carry, body } => {
                self.lower_while(carry, body, out);
            }
            _ => {}
        }
    }
}

pub fn lower(program: &TProgram) -> Kernel {
    let mut lowerer = Lowerer::new();

    // Register inline functions
    for f in &program.fns {
        lowerer.fn_defs.insert(f.name.clone(), f.clone());
    }

    // Create kernel params
    let mut params = Vec::new();
    for p in &program.kernel.params {
        let var = lowerer.fresh_var(&p.name, p.ty.clone());
        params.push(Binding {
            var,
            name: p.name.clone(),
            ty: p.ty.clone(),
        });
    }

    // Register buffer names → indices
    let mut buf_decls = Vec::new();
    for b in &program.kernel.buffers {
        let idx = buf_decls.len() as u32;
        lowerer.buf_names.insert(b.name.clone(), idx);
        buf_decls.push(BufDecl { name: b.name.clone(), is_output: b.is_output });
    }

    // Lower kernel body
    let mut body = Vec::new();
    let emit_var = lowerer.lower_stmts(&program.kernel.body, &mut body);

    let emit = emit_var.expect("kernel must have emit");

    Kernel {
        name: program.kernel.name.clone(),
        params,
        return_ty: program.kernel.return_ty.clone(),
        body,
        emit,
        buffers: buf_decls,
    }
}
