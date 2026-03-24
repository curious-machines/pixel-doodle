#[allow(dead_code)]
pub mod ast;
pub mod lexer;
pub mod parser;
pub mod runtime;
pub mod token;
pub mod validate;

use std::path::Path;

use ast::Config;
use parser::ParseError;

/// Parse a `.pdc` configuration file.
///
/// `source` is the file content. `base_dir` is the directory of the config file,
/// used for resolving relative kernel paths.
pub fn parse(source: &str, _base_dir: &Path) -> Result<Config, String> {
    let tokens = lexer::lex(source)?;
    let mut p = parser::Parser::new(tokens);
    let config = p.parse_config().map_err(|e: ParseError| e.to_string())?;
    validate::validate(&config).map_err(|errors| {
        errors
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join("\n")
    })?;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn parse_pdc(input: &str) -> Result<Config, String> {
        parse(input, &PathBuf::from("."))
    }

    #[test]
    fn end_to_end_gradient() {
        let config = parse_pdc(
            r#"
            pipeline {
              pixel kernel "gradient.pd"
              display gradient
            }
            "#,
        )
        .unwrap();
        assert_eq!(config.pipelines.len(), 1);
        assert_eq!(config.pipelines[0].kernels.len(), 1);
    }

    #[test]
    fn end_to_end_full_example() {
        let config = parse_pdc(
            r#"
            title = "Game of Life"

            iterations: range(1..10) = 1

            on key(space) paused = !paused
            on key(period) frame += 1
            on key(bracket_right) iterations += 1
            on key(bracket_left) iterations -= 1

            pipeline {
              sim kernel "game_of_life.pd"
              init kernel init_state = "init/random_binary.pd"

              buffer state = init_state(density: 0.3, seed: 42)
              buffer age = constant(0.0)
              buffer state_next = constant(0.0)
              buffer age_next = constant(0.0)

              on click(continuous: true) {
                state = run inject(value: 1.0, radius: 3)
                age = run inject(value: 0.0, radius: 3)
              }
              loop(iterations: iterations) {
                state_next, age_next = display game_of_life { state_in: state, age_in: age }
                swap state <-> state_next
                swap age <-> age_next
              }
            }
            "#,
        )
        .unwrap();

        assert_eq!(config.title.as_deref(), Some("Game of Life"));
        assert_eq!(config.pipelines[0].kernels.len(), 2);
        assert_eq!(config.pipelines[0].buffers.len(), 4);
        assert_eq!(config.variables.len(), 1);
        assert_eq!(config.key_bindings.len(), 4);
    }

    #[test]
    fn end_to_end_smoke() {
        let config = parse_pdc(
            r#"
            title = "Smoke Simulation"

            on key(space) paused = !paused
            on key(period) frame += 1

            pipeline {
              sim kernel advect = "smoke/advect.pd"
              sim kernel divergence = "smoke/divergence.pd"
              sim kernel jacobi = "smoke/jacobi.pd"
              sim kernel project = "smoke/project.pd"

              buffer vx = constant(0.0)
              buffer vy = constant(0.0)
              buffer density = constant(0.0)
              buffer vx0 = constant(0.0)
              buffer vy0 = constant(0.0)
              buffer density0 = constant(0.0)
              buffer pressure = constant(0.0)
              buffer pressure_tmp = constant(0.0)
              buffer divergence = constant(0.0)

              on click(continuous: true) {
                vy = run inject(value: -3.0, radius: 15)
                density = run inject(value: 0.5, radius: 15)
              }
              swap vx <-> vx0, vy <-> vy0, density <-> density0
              vx, vy, density = run advect { vx_in: vx0, vy_in: vy0, den_in: density0 }
              divergence = run divergence { vx_in: vx, vy_in: vy }
              loop(iterations: 40) {
                pressure_tmp = run jacobi { div_in: divergence, p_in: pressure }
                swap pressure <-> pressure_tmp
              }
              vx0, vy0 = display project { p_in: pressure, vx_in: vx, vy_in: vy, den_in: density }
              swap vx <-> vx0, vy <-> vy0
            }
            "#,
        )
        .unwrap();

        assert_eq!(config.pipelines[0].kernels.len(), 4);
        assert_eq!(config.pipelines[0].buffers.len(), 9);
    }

    #[test]
    fn validation_error_reported() {
        let result = parse_pdc(
            r#"
            pipeline {
              pixel kernel "test.pd"
              display nonexistent
            }
            "#,
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("undeclared kernel"));
    }

    #[test]
    fn toplevel_kernel_rejected() {
        let result = parse_pdc(
            r#"
            pixel kernel "test.pd"
            pipeline { display test }
            "#,
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("must be inside a pipeline"));
    }
}
