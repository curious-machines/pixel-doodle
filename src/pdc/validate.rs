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

    let mut kernel_names = HashSet::new();
    let mut buffer_names = HashSet::new();
    let mut var_names: HashSet<String> = INTRINSICS.iter().map(|s| (*s).to_string()).collect();

    // Check kernel declarations for duplicates
    for k in &config.kernels {
        if !kernel_names.insert(k.name.clone()) {
            errors.push(ValidationError {
                line: k.span.line,
                col: k.span.col,
                message: format!("duplicate kernel name '{}'", k.name),
            });
        }
    }

    // Check buffer declarations for duplicates
    for b in &config.buffers {
        if !buffer_names.insert(b.name.clone()) {
            errors.push(ValidationError {
                line: b.span.line,
                col: b.span.col,
                message: format!("duplicate buffer name '{}'", b.name),
            });
        }
        // Check that init kernel references exist
        if let BufferInit::InitKernel { kernel_name, .. } = &b.init {
            if !kernel_names.contains(kernel_name) {
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

    // Check variable declarations for duplicates and conflicts
    for v in &config.variables {
        if kernel_names.contains(&v.name)
            || buffer_names.contains(&v.name)
        {
            errors.push(ValidationError {
                line: v.span.line,
                col: v.span.col,
                message: format!("variable '{}' conflicts with a kernel or buffer name", v.name),
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

    // Validate pipeline
    if let Some(pipeline) = &config.pipeline {
        let mut has_display = false;
        validate_steps(
            &pipeline.steps,
            &kernel_names,
            &buffer_names,
            &var_names,
            &mut has_display,
            &mut errors,
        );
        if !has_display {
            errors.push(ValidationError {
                line: pipeline.span.line,
                col: pipeline.span.col,
                message: "pipeline has no 'display' step".into(),
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
            pixel kernel "gradient.pd"
            pipeline { display gradient }
            "#,
        )
        .unwrap();
    }

    #[test]
    fn undeclared_kernel_in_pipeline() {
        let result = parse_and_validate(
            r#"
            pixel kernel "gradient.pd"
            pipeline { display nonexistent }
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
            pixel kernel "test.pd"
            buffer a = constant(0.0)
            pipeline {
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
            sim kernel "test.pd"
            buffer a = constant(0.0)
            pipeline {
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
            pixel kernel "test.pd"
            on key(space) nonexistent = !nonexistent
            pipeline { display test }
            "#,
        );
        assert!(result.is_err());
    }

    #[test]
    fn intrinsic_var_in_key_binding() {
        // paused is an intrinsic, should be valid
        parse_and_validate(
            r#"
            pixel kernel "test.pd"
            on key(space) paused = !paused
            pipeline { display test }
            "#,
        )
        .unwrap();
    }

    #[test]
    fn inject_is_builtin() {
        // 'inject' should not require a kernel declaration
        parse_and_validate(
            r#"
            sim kernel "test.pd"
            buffer state = constant(0.0)
            pipeline {
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
            pixel kernel "a.pd"
            pixel kernel a = "b.pd"
            pipeline { display a }
            "#,
        );
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| e.message.contains("duplicate kernel name")));
    }
}
