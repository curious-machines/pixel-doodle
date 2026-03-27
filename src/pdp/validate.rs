use std::collections::{HashMap, HashSet};

use super::ast::*;

#[derive(Debug)]
pub struct ValidationError {
    pub line: usize,
    pub col: usize,
    pub message: String,
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}: {}", self.line, self.col, self.message)
    }
}

// ── Intrinsic registry ──

struct IntrinsicDef {
    name: &'static str,
    ty: BuiltinType,
    mutable: bool,
}

const INTRINSIC_DEFS: &[IntrinsicDef] = &[
    IntrinsicDef { name: "width",    ty: BuiltinType::U32,  mutable: false },
    IntrinsicDef { name: "height",   ty: BuiltinType::U32,  mutable: false },
    IntrinsicDef { name: "time",     ty: BuiltinType::F64,  mutable: false },
    IntrinsicDef { name: "mouse_x",  ty: BuiltinType::F64,  mutable: false },
    IntrinsicDef { name: "mouse_y",  ty: BuiltinType::F64,  mutable: false },
    IntrinsicDef { name: "center_x", ty: BuiltinType::F64,  mutable: true },
    IntrinsicDef { name: "center_y", ty: BuiltinType::F64,  mutable: true },
    IntrinsicDef { name: "zoom",     ty: BuiltinType::F64,  mutable: true },
    IntrinsicDef { name: "paused",   ty: BuiltinType::Bool, mutable: true },
    IntrinsicDef { name: "frame",    ty: BuiltinType::U64,  mutable: true },
];

fn find_intrinsic(name: &str) -> Option<&'static IntrinsicDef> {
    INTRINSIC_DEFS.iter().find(|d| d.name == name)
}

fn intrinsic_names_list() -> String {
    INTRINSIC_DEFS
        .iter()
        .map(|d| format!("{} ({})", d.name, d.ty))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Merged builtin info: type and effective mutability (most permissive wins).
#[derive(Clone)]
struct ResolvedBuiltin {
    ty: BuiltinType,
    mutable: bool,
}

pub fn validate(config: &Config) -> Result<(), Vec<ValidationError>> {
    let mut errors = Vec::new();

    // ── Validate and merge builtin declarations ──
    let mut resolved_builtins: HashMap<String, ResolvedBuiltin> = HashMap::new();
    validate_builtin_decls(&config.builtins, &mut resolved_builtins, &mut errors);

    // ── Build effective variable namespace ──
    let mut var_names: HashSet<String> = HashSet::new();
    let mut const_names: HashSet<String> = HashSet::new();

    for (name, rb) in &resolved_builtins {
        var_names.insert(name.clone());
        if !rb.mutable {
            const_names.insert(name.clone());
        }
    }

    // Check user variable declarations for duplicates
    for v in &config.variables {
        if var_names.contains(&v.name) {
            errors.push(ValidationError {
                line: v.span.line,
                col: v.span.col,
                message: format!("duplicate variable name '{}'", v.name),
            });
        }
        var_names.insert(v.name.clone());
        if v.mutability == Mutability::Const {
            const_names.insert(v.name.clone());
        }
    }

    // ── Check key bindings reference valid variables and respect const-ness ──
    for kb in &config.key_bindings {
        for action in &kb.actions {
            let target = match action {
                Action::Toggle(var) => var,
                Action::CompoundAssign { target, .. } => target,
                Action::BinAssign { target, .. } => target,
                Action::Assign { target, .. } => target,
                Action::Quit => continue,
            };
            if !var_names.contains(target) {
                // Check if it's a known intrinsic that wasn't declared
                if let Some(def) = find_intrinsic(target) {
                    let constness = if def.mutable { "var" } else { "const" };
                    errors.push(ValidationError {
                        line: kb.span.line,
                        col: kb.span.col,
                        message: format!(
                            "use of undeclared builtin '{target}'. \
                             Add 'builtin {constness} {target}: {}' to your file",
                            def.ty
                        ),
                    });
                } else {
                    errors.push(ValidationError {
                        line: kb.span.line,
                        col: kb.span.col,
                        message: format!(
                            "key binding references undeclared variable '{target}'"
                        ),
                    });
                }
            } else if const_names.contains(target) {
                let kind = if resolved_builtins.contains_key(target) {
                    "builtin const"
                } else {
                    "const"
                };
                errors.push(ValidationError {
                    line: kb.span.line,
                    col: kb.span.col,
                    message: format!(
                        "cannot assign to '{target}': it is declared as '{kind}'"
                    ),
                });
            }
        }
    }

    // ── Validate each pipeline ──
    for pipeline in &config.pipelines {
        // Merge pipeline-scoped builtins with top-level
        let mut pipe_var_names = var_names.clone();
        let mut pipe_const_names = const_names.clone();
        let mut pipe_resolved = resolved_builtins.clone();

        validate_builtin_decls(&pipeline.builtins, &mut pipe_resolved, &mut errors);
        for (name, rb) in &pipe_resolved {
            pipe_var_names.insert(name.clone());
            if !rb.mutable {
                pipe_const_names.insert(name.clone());
            } else {
                // If a pipeline-scoped builtin is var, remove from const set
                // (most permissive wins)
                pipe_const_names.remove(name);
            }
        }

        let mut pipe_kernels = HashSet::new();
        let mut pipe_buffers = HashSet::new();

        for k in &pipeline.kernels {
            if !pipe_kernels.insert(k.name.clone()) {
                errors.push(ValidationError {
                    line: k.span.line,
                    col: k.span.col,
                    message: format!("duplicate kernel name '{}'", k.name),
                });
            }
        }
        for b in &pipeline.buffers {
            if !pipe_buffers.insert(b.name.clone()) {
                errors.push(ValidationError {
                    line: b.span.line,
                    col: b.span.col,
                    message: format!("duplicate buffer name '{}'", b.name),
                });
            }
        }

        let mut has_display = false;
        validate_steps(
            &pipeline.steps,
            &pipe_kernels,
            &pipe_buffers,
            &pipe_var_names,
            &pipe_resolved,
            &mut has_display,
            &mut errors,
        );
        if !has_display {
            errors.push(ValidationError {
                line: pipeline.span.line,
                col: pipeline.span.col,
                message: format!(
                    "pipeline{} has no 'display' step",
                    pipeline.name.as_ref().map(|n| format!(" '{n}'")).unwrap_or_default()
                ),
            });
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Validate a list of builtin declarations and merge into the resolved map.
/// Duplicates with matching type are allowed (most permissive mutability wins).
/// Type mismatches are errors.
fn validate_builtin_decls(
    decls: &[BuiltinDecl],
    resolved: &mut HashMap<String, ResolvedBuiltin>,
    errors: &mut Vec<ValidationError>,
) {
    for decl in decls {
        // Check name is a known intrinsic
        let Some(def) = find_intrinsic(&decl.name) else {
            errors.push(ValidationError {
                line: decl.span.line,
                col: decl.span.col,
                message: format!(
                    "unknown builtin '{}'. Known builtins: {}",
                    decl.name,
                    intrinsic_names_list()
                ),
            });
            continue;
        };

        // Check declared type matches the intrinsic's actual type
        if decl.ty != def.ty {
            errors.push(ValidationError {
                line: decl.span.line,
                col: decl.span.col,
                message: format!(
                    "builtin '{}' has type {}, but was declared as {}",
                    decl.name, def.ty, decl.ty
                ),
            });
            continue;
        }

        // Check mutability: can't declare builtin var on a const-only intrinsic
        if decl.mutable && !def.mutable {
            errors.push(ValidationError {
                line: decl.span.line,
                col: decl.span.col,
                message: format!(
                    "builtin '{}' is read-only and cannot be declared as 'var'",
                    decl.name
                ),
            });
            continue;
        }

        // Merge: if already declared, check for type conflict (most permissive mutability wins)
        if let Some(existing) = resolved.get_mut(&decl.name) {
            if existing.ty != decl.ty {
                errors.push(ValidationError {
                    line: decl.span.line,
                    col: decl.span.col,
                    message: format!(
                        "conflicting builtin declarations for '{}': declared as {} and {}",
                        decl.name, existing.ty, decl.ty
                    ),
                });
            } else {
                // Most permissive wins
                existing.mutable = existing.mutable || decl.mutable;
            }
        } else {
            resolved.insert(
                decl.name.clone(),
                ResolvedBuiltin {
                    ty: decl.ty,
                    mutable: decl.mutable,
                },
            );
        }
    }
}

fn validate_steps(
    steps: &[PipelineStep],
    kernels: &HashSet<String>,
    buffers: &HashSet<String>,
    vars: &HashSet<String>,
    resolved_builtins: &HashMap<String, ResolvedBuiltin>,
    has_display: &mut bool,
    errors: &mut Vec<ValidationError>,
) {
    for step in steps {
        match step {
            PipelineStep::Run {
                kernel_name,
                args,
                input_bindings,
                span,
            } => {
                if !kernels.contains(kernel_name) {
                    errors.push(ValidationError {
                        line: span.line,
                        col: span.col,
                        message: format!("undeclared kernel '{kernel_name}'"),
                    });
                }
                // Validate variable references in run args
                for arg in args {
                    if let Literal::VarRef(ref name) = arg.value {
                        if !vars.contains(name) {
                            check_undeclared_var(name, arg.span, errors);
                        }
                    }
                }
                validate_buffer_refs(input_bindings, buffers, *span, errors);
            }
            PipelineStep::Display { buffer_name, span } => {
                *has_display = true;
                if let Some(name) = buffer_name {
                    if !buffers.contains(name) {
                        errors.push(ValidationError {
                            line: span.line,
                            col: span.col,
                            message: format!("display references undeclared buffer '{name}'"),
                        });
                    }
                }
            }
            PipelineStep::Init { body, .. } => {
                validate_steps(body, kernels, buffers, vars, resolved_builtins, has_display, errors);
            }
            PipelineStep::Swap { a, b, span } => {
                if !buffers.contains(a.as_str()) {
                    errors.push(ValidationError {
                        line: span.line,
                        col: span.col,
                        message: format!("swap references undeclared buffer '{a}'"),
                    });
                }
                if !buffers.contains(b.as_str()) {
                    errors.push(ValidationError {
                        line: span.line,
                        col: span.col,
                        message: format!("swap references undeclared buffer '{b}'"),
                    });
                }
            }
            PipelineStep::Loop {
                iterations,
                body,
                span,
            } => {
                if let IterCount::Variable(name) = iterations {
                    if !vars.contains(name) {
                        check_undeclared_var(name, *span, errors);
                    }
                }
                validate_steps(body, kernels, buffers, vars, resolved_builtins, has_display, errors);
            }
            PipelineStep::Accumulate { body, .. } => {
                validate_steps(body, kernels, buffers, vars, resolved_builtins, has_display, errors);
            }
            PipelineStep::OnClick { body, .. } => {
                validate_steps(body, kernels, buffers, vars, resolved_builtins, has_display, errors);
            }
        }
    }
}

/// When a variable reference is undeclared, check if it's a known intrinsic
/// and produce a targeted error message.
fn check_undeclared_var(name: &str, span: Span, errors: &mut Vec<ValidationError>) {
    if let Some(def) = find_intrinsic(name) {
        let constness = if def.mutable { "var" } else { "const" };
        errors.push(ValidationError {
            line: span.line,
            col: span.col,
            message: format!(
                "use of undeclared builtin '{name}'. \
                 Add 'builtin {constness} {name}: {}' to your file",
                def.ty
            ),
        });
    } else {
        errors.push(ValidationError {
            line: span.line,
            col: span.col,
            message: format!("undeclared variable '{name}'"),
        });
    }
}

fn validate_buffer_refs(
    bindings: &[BufferBinding],
    buffers: &HashSet<String>,
    span: Span,
    errors: &mut Vec<ValidationError>,
) {
    let _ = span; // kept for future use
    for binding in bindings {
        if !buffers.contains(&binding.buffer_name) {
            errors.push(ValidationError {
                line: binding.span.line,
                col: binding.span.col,
                message: format!(
                    "binding references undeclared buffer '{}'",
                    binding.buffer_name
                ),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pdp::lexer::lex;
    use crate::pdp::parser::Parser;

    fn parse_and_validate(input: &str) -> Result<(), Vec<ValidationError>> {
        let tokens = lex(input).unwrap();
        let mut parser = Parser::new(tokens);
        let config = parser.parse_config().unwrap();
        validate(&config)
    }

    #[test]
    fn valid_gradient() {
        parse_and_validate(
            r#"
            pipeline {
              pixel kernel "gradient.pd"
              run gradient
              display
            }
            "#,
        )
        .unwrap();
    }

    #[test]
    fn undeclared_kernel_in_pipeline() {
        let result = parse_and_validate(
            r#"
            pipeline {
              pixel kernel "gradient.pd"
              run nonexistent
              display
            }
            "#,
        );
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors[0].message.contains("undeclared kernel"));
    }

    #[test]
    fn undeclared_buffer_in_swap() {
        let result = parse_and_validate(
            r#"
            pipeline {
              pixel kernel "test.pd"
              buffer a = constant(0.0)
              run test
              display
              swap a, b
            }
            "#,
        );
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| e.message.contains("undeclared buffer 'b'")));
    }

    #[test]
    fn no_display_step() {
        let result = parse_and_validate(
            r#"
            pipeline {
              kernel "test.pd"
              buffer a = constant(0.0)
              run test
            }
            "#,
        );
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| e.message.contains("no 'display' step")));
    }

    #[test]
    fn undeclared_var_in_key_binding() {
        let result = parse_and_validate(
            r#"
            on key(space) nonexistent = !nonexistent
            pipeline {
              pixel kernel "test.pd"
              display test
            }
            "#,
        );
        assert!(result.is_err());
    }

    #[test]
    fn intrinsic_var_in_key_binding() {
        parse_and_validate(
            r#"
            builtin var paused: bool
            on key(space) paused = !paused
            pipeline {
              pixel kernel "test.pd"
              run test
              display
            }
            "#,
        )
        .unwrap();
    }

    #[test]
    fn undeclared_intrinsic_in_key_binding() {
        let result = parse_and_validate(
            r#"
            on key(space) paused = !paused
            pipeline {
              pixel kernel "test.pd"
              run test
              display
            }
            "#,
        );
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors[0].message.contains("undeclared builtin 'paused'"));
        assert!(errors[0].message.contains("builtin var paused: bool"));
    }

    #[test]
    fn builtin_wrong_type() {
        let result = parse_and_validate(
            r#"
            builtin const width: f64
            pipeline {
              pixel kernel "test.pd"
              run test
              display
            }
            "#,
        );
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors[0].message.contains("has type u32, but was declared as f64"));
    }

    #[test]
    fn builtin_unknown_name() {
        let result = parse_and_validate(
            r#"
            builtin const foo: f64
            pipeline {
              pixel kernel "test.pd"
              run test
              display
            }
            "#,
        );
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors[0].message.contains("unknown builtin 'foo'"));
    }

    #[test]
    fn builtin_var_on_const_intrinsic() {
        let result = parse_and_validate(
            r#"
            builtin var time: f64
            pipeline {
              pixel kernel "test.pd"
              run test
              display
            }
            "#,
        );
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors[0].message.contains("read-only and cannot be declared as 'var'"));
    }

    #[test]
    fn assign_to_const_in_key_binding() {
        let result = parse_and_validate(
            r#"
            const max_iter = 256
            on key(space) max_iter += 1
            pipeline {
              pixel kernel "test.pd"
              run test
              display
            }
            "#,
        );
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors[0].message.contains("cannot assign to 'max_iter'"));
        assert!(errors[0].message.contains("'const'"));
    }

    #[test]
    fn assign_to_builtin_const_in_key_binding() {
        let result = parse_and_validate(
            r#"
            builtin const time: f64
            on key(space) time += 1.0
            pipeline {
              pixel kernel "test.pd"
              run test
              display
            }
            "#,
        );
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors[0].message.contains("cannot assign to 'time'"));
        assert!(errors[0].message.contains("'builtin const'"));
    }

    #[test]
    fn duplicate_builtin_same_type_ok() {
        parse_and_validate(
            r#"
            builtin var zoom: f64
            builtin var zoom: f64
            on key(plus) zoom *= 1.1
            pipeline {
              pixel kernel "test.pd"
              run test
              display
            }
            "#,
        )
        .unwrap();
    }

    #[test]
    fn duplicate_builtin_var_const_mismatch_ok() {
        // Most permissive wins: var + const = var
        parse_and_validate(
            r#"
            builtin const zoom: f64
            builtin var zoom: f64
            on key(plus) zoom *= 1.1
            pipeline {
              pixel kernel "test.pd"
              run test
              display
            }
            "#,
        )
        .unwrap();
    }

    #[test]
    fn duplicate_builtin_type_mismatch_error() {
        // zoom is f64, so declaring it as u32 hits "wrong type" before duplicate check.
        // Use two intrinsics that could conflict: declare width as u32 then u64.
        // But width's actual type is u32, so u64 hits "wrong type" too.
        // The "conflicting" error only fires when both declarations pass individual
        // validation but disagree. Since all intrinsics have fixed types, a true
        // type conflict between two declarations of the same intrinsic can't happen
        // (the second would fail type check against the registry first).
        // So we test that the wrong-type error fires on the second declaration.
        let result = parse_and_validate(
            r#"
            builtin var zoom: f64
            builtin var zoom: u32
            pipeline {
              pixel kernel "test.pd"
              run test
              display
            }
            "#,
        );
        assert!(result.is_err());
        let errors = result.unwrap_err();
        // The second declaration fails because zoom's intrinsic type is f64, not u32
        assert!(errors[0].message.contains("has type f64, but was declared as u32"));
    }

    #[test]
    fn pipeline_scoped_builtin() {
        parse_and_validate(
            r#"
            pipeline {
              builtin const time: f64
              pixel kernel "test.pd"
              run test(t: time)
              display
            }
            "#,
        )
        .unwrap();
    }

    #[test]
    fn undeclared_intrinsic_in_run_arg() {
        let result = parse_and_validate(
            r#"
            pipeline {
              pixel kernel "test.pd"
              run test(t: time)
              display
            }
            "#,
        );
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors[0].message.contains("undeclared builtin 'time'"));
    }

    #[test]
    fn inject_kernel_must_be_declared() {
        // inject is no longer a built-in — it must be declared like any other kernel
        let result = parse_and_validate(
            r#"
            pipeline {
              kernel "test.pd"
              buffer state = constant(0.0)
              on click(continuous: true) {
                run inject(value: 1.0, radius: 3) with(target: out state)
              }
              run test
              display
            }
            "#,
        );
        assert!(result.is_err(), "inject should require a kernel declaration");
    }

    #[test]
    fn duplicate_kernel_name() {
        let result = parse_and_validate(
            r#"
            pipeline {
              pixel kernel "a.pd"
              pixel kernel a = "b.pd"
              run a
              display
            }
            "#,
        );
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| e.message.contains("duplicate kernel name")));
    }
}
