//! Mechanical translator from PDP (declarative pipeline config) to PDC
//! (imperative pipeline script).
//!
//! Takes a parsed `pdp::ast::Config` and generates equivalent PDC source text
//! that can be executed by `PdcRuntime`.

use super::ast::*;

/// Convert a PDP config to equivalent PDC source text.
///
/// Uses the first pipeline in the config. Generates:
/// - Builtin declarations (const/var)
/// - State variables for buffer and kernel handles
/// - `fn init()` — creates buffers, loads kernels, runs init steps
/// - `fn frame() -> bool` — executes pipeline steps, returns false (static)
/// - Event handler functions (keyboard + mouse)
pub fn config_to_pdc(config: &Config) -> String {
    let mut out = String::new();

    // Title as comment
    if let Some(ref title) = config.title {
        out.push_str(&format!("// {title}\n\n"));
    }

    // Builtin declarations
    emit_builtins(&mut out, &config.builtins);
    if let Some(pipeline) = config.pipelines.first() {
        emit_builtins(&mut out, &pipeline.builtins);
    }
    out.push('\n');

    // User variables as module-level state
    for var in &config.variables {
        let ty = var_type_name(var);
        let val = literal_to_pdc(&var.default);
        match var.mutability {
            Mutability::Var => out.push_str(&format!("var {}: {} = {}\n", var.name, ty, val)),
            Mutability::Const => out.push_str(&format!("const {}: {} = {}\n", var.name, ty, val)),
        }
    }

    let pipeline = match config.pipelines.first() {
        Some(p) => p,
        None => return out,
    };

    // Handle variables for buffers, kernels, textures
    for buf in &pipeline.buffers {
        out.push_str(&format!("var _buf_{}: i32 = 0\n", buf.name));
    }
    for kernel in &pipeline.kernels {
        out.push_str(&format!("var _kern_{}: i32 = 0\n", kernel.name));
    }
    for tex in &pipeline.textures {
        out.push_str(&format!("var _tex_{}: i32 = 0\n", tex.name));
    }
    out.push('\n');

    // Build name→handle mappings for steps
    let buf_names: Vec<&str> = pipeline.buffers.iter().map(|b| b.name.as_str()).collect();
    let kern_names: Vec<&str> = pipeline.kernels.iter().map(|k| k.name.as_str()).collect();

    // init() function
    out.push_str("fn init() {\n");

    // Create buffers
    for buf in &pipeline.buffers {
        let type_name = match buf.gpu_type {
            Some(GpuElementType::F32) => "gpu_f32",
            Some(GpuElementType::I32) => "gpu_i32",
            Some(GpuElementType::U32) => "gpu_u32",
            Some(GpuElementType::Vec2F32) => "gpu_vec2_f32",
            Some(GpuElementType::Vec3F32) => "gpu_vec3_f32",
            Some(GpuElementType::Vec4F32) => "gpu_vec4_f32",
            None => "cpu_f64",
        };
        let init_val = match &buf.init {
            BufferInit::Constant(v) => format!("{v}"),
        };
        out.push_str(&format!(
            "    _buf_{} = create_buffer(\"{type_name}\", {init_val})\n",
            buf.name
        ));
    }

    // Load kernels
    for kernel in &pipeline.kernels {
        let kind = match kernel.kind {
            KernelKind::Pixel => 0,
            KernelKind::Standard => 1,
            KernelKind::Scene => 2,
        };
        out.push_str(&format!(
            "    _kern_{} = load_kernel(\"{}\", \"{}\", {})\n",
            kernel.name, kernel.name, kernel.path, kind
        ));
    }

    // Load textures
    for tex in &pipeline.textures {
        let path = match &tex.init {
            TextureInit::File(p) => p.as_str(),
        };
        out.push_str(&format!(
            "    _tex_{} = load_texture(\"{}\", \"{path}\")\n",
            tex.name, tex.name
        ));
    }

    // Init steps (from Init blocks in the pipeline)
    for step in &pipeline.steps {
        if let PipelineStep::Init { body, .. } = step {
            emit_steps(&mut out, body, &buf_names, &kern_names, 1);
        }
    }

    out.push_str("}\n\n");

    // frame() function — non-init steps
    out.push_str("fn frame() -> bool {\n");
    for step in &pipeline.steps {
        if matches!(step, PipelineStep::Init { .. }) {
            continue;
        }
        emit_steps(&mut out, std::slice::from_ref(step), &buf_names, &kern_names, 1);
    }
    out.push_str("    return false\n}\n");

    // Event bindings → handler functions
    for binding in &config.event_bindings {
        let kind = match binding.kind {
            EventKind::Keypress => "keypress",
            EventKind::Keydown => "keydown",
            EventKind::Keyup => "keyup",
        };
        let fn_name = format!("on_{kind}_{}", binding.key_name);
        out.push_str(&format!("\nfn {fn_name}() {{\n"));
        for action in &binding.actions {
            emit_action(&mut out, action, 1);
        }
        out.push_str("}\n");
    }

    // Mouse event handlers from pipeline steps
    for step in &pipeline.steps {
        if let PipelineStep::OnMouse { kind, body, .. } = step {
            let fn_name = match kind {
                MouseEventKind::Mousedown => "on_mousedown",
                MouseEventKind::Click => "on_click",
                MouseEventKind::Mouseup => "on_mouseup",
            };
            out.push_str(&format!("\nfn {fn_name}() {{\n"));
            emit_steps(&mut out, body, &buf_names, &kern_names, 1);
            out.push_str("}\n");
        }
    }

    out
}

fn emit_builtins(out: &mut String, builtins: &[BuiltinDecl]) {
    for b in builtins {
        let kw = if b.mutable { "var" } else { "const" };
        let ty = match b.ty {
            BuiltinType::U32 => "u32",
            BuiltinType::U64 => "u64",
            BuiltinType::F64 => "f64",
            BuiltinType::Bool => "bool",
        };
        out.push_str(&format!("builtin {kw} {}: {ty}\n", b.name));
    }
}

fn emit_steps(
    out: &mut String,
    steps: &[PipelineStep],
    buf_names: &[&str],
    kern_names: &[&str],
    indent: usize,
) {
    let pad = "    ".repeat(indent);
    for step in steps {
        match step {
            PipelineStep::Run { kernel_name, args, input_bindings, .. } => {
                // Bind buffers
                for binding in input_bindings {
                    let handle = buf_handle(&binding.buffer_name, buf_names);
                    let is_out = if binding.is_output { "1" } else { "0" };
                    out.push_str(&format!(
                        "{pad}bind_buffer(\"{}\", {handle}, {is_out})\n",
                        binding.param_name
                    ));
                }
                // Set kernel args
                for arg in args {
                    let val = literal_to_pdc(&arg.value);
                    out.push_str(&format!(
                        "{pad}set_kernel_arg_f64(\"{}\", {val})\n",
                        arg.name
                    ));
                }
                // Dispatch
                let handle = kern_handle(kernel_name, kern_names);
                out.push_str(&format!("{pad}run_kernel({handle})\n"));
            }
            PipelineStep::Display { buffer_name, .. } => {
                if let Some(name) = buffer_name {
                    let handle = buf_handle(name, buf_names);
                    out.push_str(&format!("{pad}display_buffer({handle})\n"));
                } else {
                    out.push_str(&format!("{pad}display()\n"));
                }
            }
            PipelineStep::Swap { a, b, .. } => {
                let ha = buf_handle(a, buf_names);
                let hb = buf_handle(b, buf_names);
                out.push_str(&format!("{pad}swap_buffers({ha}, {hb})\n"));
            }
            PipelineStep::Loop { iterations, body, .. } => {
                let count = match iterations {
                    IterCount::Fixed(n) => format!("{n}"),
                    IterCount::Variable(name) => format!("i32({name})"),
                };
                out.push_str(&format!("{pad}for _i in 0..{count} {{\n"));
                emit_steps(out, body, buf_names, kern_names, indent + 1);
                out.push_str(&format!("{pad}}}\n"));
            }
            PipelineStep::Accumulate { samples, body, .. } => {
                out.push_str(&format!("{pad}set_max_samples({samples})\n"));
                out.push_str(&format!("{pad}if !is_converged() {{\n"));
                emit_steps(out, body, buf_names, kern_names, indent + 1);
                let inner_pad = "    ".repeat(indent + 1);
                out.push_str(&format!("{inner_pad}accumulate_sample()\n"));
                out.push_str(&format!("{inner_pad}display_accumulated()\n"));
                out.push_str(&format!("{pad}}}\n"));
            }
            PipelineStep::Init { .. } => {
                // Init blocks handled separately in init()
            }
            PipelineStep::OnMouse { .. } => {
                // Mouse handlers emitted as separate functions
            }
        }
    }
}

fn emit_action(out: &mut String, action: &Action, indent: usize) {
    let pad = "    ".repeat(indent);
    match action {
        Action::Toggle(var) => {
            out.push_str(&format!("{pad}{var} = !{var}\n"));
        }
        Action::CompoundAssign { target, op, value } => {
            let op_str = compound_op_str(*op);
            let val = value_expr_to_pdc(value);
            out.push_str(&format!("{pad}{target} = {target} {op_str} {val}\n"));
        }
        Action::BinAssign { target, op, value } => {
            let op_str = compound_op_str(*op);
            let val = value_expr_to_pdc(value);
            out.push_str(&format!("{pad}{target} = {target} {op_str} {val}\n"));
        }
        Action::Assign { target, value } => {
            out.push_str(&format!("{pad}{target} = {}\n", format_f64(*value)));
        }
        Action::Quit => {
            out.push_str(&format!("{pad}// quit — not yet supported in PDC\n"));
        }
    }
}

fn compound_op_str(op: CompoundOp) -> &'static str {
    match op {
        CompoundOp::Add => "+",
        CompoundOp::Sub => "-",
        CompoundOp::Mul => "*",
        CompoundOp::Div => "/",
    }
}

fn value_expr_to_pdc(expr: &ValueExpr) -> String {
    match expr {
        ValueExpr::Literal(v) => format_f64(*v),
        ValueExpr::BinOp { left, op, right } => {
            format!("{} {} {}", format_f64(*left), compound_op_str(*op), right)
        }
    }
}

fn literal_to_pdc(lit: &Literal) -> String {
    match lit {
        Literal::Float(v) => format_f64(*v),
        Literal::Int(v) => format!("{v}"),
        Literal::Bool(v) => format!("{v}"),
        Literal::Str(s) => format!("\"{s}\""),
        Literal::VarRef(name) => name.clone(),
    }
}

fn format_f64(v: f64) -> String {
    let s = format!("{v}");
    if !s.contains('.') && !s.contains('e') && !s.contains('E') {
        format!("{s}.0")
    } else {
        s
    }
}

fn var_type_name(var: &VarDecl) -> &'static str {
    match var.ty {
        Some(BuiltinType::U32) => "u32",
        Some(BuiltinType::U64) => "u64",
        Some(BuiltinType::F64) => "f64",
        Some(BuiltinType::Bool) => "bool",
        None => "f64",
    }
}

fn buf_handle(name: &str, buf_names: &[&str]) -> String {
    if buf_names.contains(&name) {
        format!("_buf_{name}")
    } else {
        format!("0 /* unknown buffer '{name}' */")
    }
}

fn kern_handle(name: &str, kern_names: &[&str]) -> String {
    if kern_names.contains(&name) {
        format!("_kern_{name}")
    } else {
        format!("0 /* unknown kernel '{name}' */")
    }
}
