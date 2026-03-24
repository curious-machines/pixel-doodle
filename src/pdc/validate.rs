use std::collections::HashSet;

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

/// Known intrinsic global names that don't need to be declared.
const INTRINSICS: &[&str] = &[
    "width",
    "height",
    "center_x",
    "center_y",
    "zoom",
    "mouse_x",
    "mouse_y",
    "time",
    "paused",
    "frame",
];

pub fn validate(config: &Config) -> Result<(), Vec<ValidationError>> {
    let mut errors = Vec::new();

    let mut var_names: HashSet<String> = INTRINSICS.iter().map(|s| (*s).to_string()).collect();

    // Check variable declarations for duplicates
    for v in &config.variables {
        if var_names.contains(&v.name) {
            errors.push(ValidationError {
                line: v.span.line,
                col: v.span.col,
                message: format!("duplicate variable name '{}'", v.name),
            });
        }
        var_names.insert(v.name.clone());
    }

    // Check key bindings reference valid variables
    for kb in &config.key_bindings {
        let target = match &kb.action {
            Action::Toggle(var) => var,
            Action::CompoundAssign { target, .. } => target,
            Action::BinAssign { target, .. } => target,
        };
        if !var_names.contains(target) {
            errors.push(ValidationError {
                line: kb.span.line,
                col: kb.span.col,
                message: format!(
                    "key binding references undeclared variable '{target}'"
                ),
            });
        }
    }

    // Validate each pipeline
    for pipeline in &config.pipelines {
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
            if let BufferInit::InitKernel { kernel_name, .. } = &b.init {
                if !pipe_kernels.contains(kernel_name) {
                    errors.push(ValidationError {
                        line: b.span.line,
                        col: b.span.col,
                        message: format!(
                            "buffer '{}' references undeclared init kernel '{kernel_name}'",
                            b.name
                        ),
                    });
                }
            }
        }

        let mut has_display = false;
        validate_steps(
            &pipeline.steps,
            &pipe_kernels,
            &pipe_buffers,
            &var_names,
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

fn validate_steps(
    steps: &[PipelineStep],
    kernels: &HashSet<String>,
    buffers: &HashSet<String>,
    vars: &HashSet<String>,
    has_display: &mut bool,
    errors: &mut Vec<ValidationError>,
) {
    for step in steps {
        match step {
            PipelineStep::Run {
                outputs,
                kernel_name,
                input_bindings,
                span,
                ..
            } => {
                // 'inject' is a built-in, skip kernel check
                if kernel_name != "inject" && !kernels.contains(kernel_name) {
                    errors.push(ValidationError {
                        line: span.line,
                        col: span.col,
                        message: format!("undeclared kernel '{kernel_name}'"),
                    });
                }
                validate_buffer_refs(outputs, input_bindings, buffers, *span, errors);
            }
            PipelineStep::Display {
                outputs,
                kernel_name,
                input_bindings,
                span,
                ..
            } => {
                *has_display = true;
                if !kernels.contains(kernel_name) {
                    errors.push(ValidationError {
                        line: span.line,
                        col: span.col,
                        message: format!("undeclared kernel '{kernel_name}'"),
                    });
                }
                validate_buffer_refs(outputs, input_bindings, buffers, *span, errors);
            }
            PipelineStep::Swap { pairs, span } => {
                for (a, b) in pairs {
                    if !buffers.contains(a) {
                        errors.push(ValidationError {
                            line: span.line,
                            col: span.col,
                            message: format!("swap references undeclared buffer '{a}'"),
                        });
                    }
                    if !buffers.contains(b) {
                        errors.push(ValidationError {
                            line: span.line,
                            col: span.col,
                            message: format!("swap references undeclared buffer '{b}'"),
                        });
                    }
                }
            }
            PipelineStep::Loop {
                iterations,
                body,
                span,
            } => {
                if let IterCount::Variable(name) = iterations {
                    if !vars.contains(name) {
                        errors.push(ValidationError {
                            line: span.line,
                            col: span.col,
                            message: format!(
                                "loop iterations references undeclared variable '{name}'"
                            ),
                        });
                    }
                }
                validate_steps(body, kernels, buffers, vars, has_display, errors);
            }
            PipelineStep::Accumulate { body, .. } => {
                validate_steps(body, kernels, buffers, vars, has_display, errors);
            }
            PipelineStep::OnClick { body, .. } => {
                validate_steps(body, kernels, buffers, vars, has_display, errors);
            }
        }
    }
}

fn validate_buffer_refs(
    outputs: &[String],
    bindings: &[BufferBinding],
    buffers: &HashSet<String>,
    span: Span,
    errors: &mut Vec<ValidationError>,
) {
    for out in outputs {
        if !buffers.contains(out) {
            errors.push(ValidationError {
                line: span.line,
                col: span.col,
                message: format!("output references undeclared buffer '{out}'"),
            });
        }
    }
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
    use crate::pdc::lexer::lex;
    use crate::pdc::parser::Parser;

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
              display gradient
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
              display nonexistent
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
              display test
              swap a <-> b
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
              sim kernel "test.pd"
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
            on key(space) paused = !paused
            pipeline {
              pixel kernel "test.pd"
              display test
            }
            "#,
        )
        .unwrap();
    }

    #[test]
    fn inject_is_builtin() {
        parse_and_validate(
            r#"
            pipeline {
              sim kernel "test.pd"
              buffer state = constant(0.0)
              on click(continuous: true) {
                state = run inject(value: 1.0, radius: 3)
              }
              display test
            }
            "#,
        )
        .unwrap();
    }

    #[test]
    fn duplicate_kernel_name() {
        let result = parse_and_validate(
            r#"
            pipeline {
              pixel kernel "a.pd"
              pixel kernel a = "b.pd"
              display a
            }
            "#,
        );
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| e.message.contains("duplicate kernel name")));
    }
}
