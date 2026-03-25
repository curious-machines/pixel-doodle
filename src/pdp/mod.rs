#[allow(dead_code)]
pub mod ast;
pub mod lexer;
pub mod parser;
pub mod runtime;
pub mod token;
pub mod validate;

use std::cell::RefCell;
use std::collections::HashSet;
use std::path::Path;
use std::rc::Rc;

use ast::Config;
use parser::ParseError;

/// Parse a `.pdp` configuration file.
///
/// `source` is the file content. `base_dir` is the directory of the config file,
/// used for resolving relative kernel and include paths.
pub fn parse(source: &str, base_dir: &Path) -> Result<Config, String> {
    let tokens = lexer::lex(source)?;
    let included = Rc::new(RefCell::new(HashSet::new()));
    let mut p = parser::Parser::new_with_context(tokens, base_dir.to_path_buf(), included);
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

    fn parse_pdp(input: &str) -> Result<Config, String> {
        parse(input, &PathBuf::from("."))
    }

    #[test]
    fn end_to_end_gradient() {
        let config = parse_pdp(
            r#"
            pipeline {
              pixel kernel "gradient.pd"
              run gradient
              display
            }
            "#,
        )
        .unwrap();
        assert_eq!(config.pipelines.len(), 1);
        assert_eq!(config.pipelines[0].kernels.len(), 1);
    }

    #[test]
    fn end_to_end_full_example() {
        let config = parse_pdp(
            r#"
            title = "Game of Life"

            iterations: range(1..10) = 1

            on key(space) paused = !paused
            on key(period) frame += 1
            on key(bracket_right) iterations += 1
            on key(bracket_left) iterations -= 1

            pipeline {
              kernel "game_of_life.pd"
              kernel init_state = "init/random_binary.pd"

              buffer state = constant(0.0)
              buffer age = constant(0.0)
              buffer state_next = constant(0.0)
              buffer age_next = constant(0.0)

              init {
                run init_state { out: out state }
              }
              on click(continuous: true) {
                run inject(value: 1.0, radius: 3) { target: out state }
                run inject(value: 0.0, radius: 3) { target: out age }
              }
              loop(iterations: iterations) {
                run game_of_life { state_in: state, age_in: age, state_out: out state_next, age_out: out age_next }
                display
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
        let config = parse_pdp(
            r#"
            title = "Smoke Simulation"

            on key(space) paused = !paused
            on key(period) frame += 1

            pipeline {
              kernel advect = "smoke/advect.pd"
              kernel divergence = "smoke/divergence.pd"
              kernel jacobi = "smoke/jacobi.pd"
              kernel project = "smoke/project.pd"

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
                run inject(value: -3.0, radius: 15) { target: out vy }
                run inject(value: 0.5, radius: 15) { target: out density }
              }
              swap vx <-> vx0, vy <-> vy0, density <-> density0
              run advect { vx_in: vx0, vy_in: vy0, den_in: density0, vx_out: out vx, vy_out: out vy, den_out: out density }
              run divergence { vx_in: vx, vy_in: vy, div_out: out divergence }
              loop(iterations: 40) {
                run jacobi { div_in: divergence, p_in: pressure, p_out: out pressure_tmp }
                swap pressure <-> pressure_tmp
              }
              run project { p_in: pressure, vx_in: vx, vy_in: vy, den_in: density, vx_out: out vx0, vy_out: out vy0 }
              display
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
        let result = parse_pdp(
            r#"
            pipeline {
              pixel kernel "test.pd"
              run nonexistent
              display
            }
            "#,
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("undeclared kernel"));
    }

    #[test]
    fn toplevel_kernel_rejected() {
        let result = parse_pdp(
            r#"
            pixel kernel "test.pd"
            pipeline { run test
            display }
            "#,
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("must be inside a pipeline"));
    }

    fn make_test_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("pdp_test_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn include_key_bindings() {
        let dir = make_test_dir("include_kb");
        std::fs::write(
            dir.join("pan_zoom.pdp"),
            "on key(left) center_x -= 0.1\non key(right) center_x += 0.1\n",
        )
        .unwrap();

        let source = concat!(
            "include \"pan_zoom.pdp\"\n",
            "\n",
            "pipeline {\n",
            "  pixel kernel \"gradient.pd\"\n",
            "  run gradient\n",
            "  display\n",
            "}\n",
        );
        let config = parse(source, &dir).unwrap();
        assert_eq!(config.key_bindings.len(), 2);
        assert_eq!(config.key_bindings[0].key_name, "left");
        assert_eq!(config.key_bindings[1].key_name, "right");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn include_rejects_pipeline() {
        let dir = make_test_dir("include_reject");
        std::fs::write(
            dir.join("bad.pdp"),
            "pipeline { pixel kernel \"test.pd\"\n run test\n display }\n",
        )
        .unwrap();

        let source = "include \"bad.pdp\"\npipeline { pixel kernel \"test.pd\"\n run test\n display }\n";
        let result = parse(source, &dir);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("must not contain pipeline"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn include_circular_dedup() {
        let dir = make_test_dir("include_circular");
        std::fs::write(
            dir.join("a.pdp"),
            "include \"b.pdp\"\non key(left) center_x -= 0.1\n",
        )
        .unwrap();
        std::fs::write(
            dir.join("b.pdp"),
            "include \"a.pdp\"\non key(right) center_x += 0.1\n",
        )
        .unwrap();

        let source = std::fs::read_to_string(dir.join("a.pdp")).unwrap();
        let config = parse(
            &format!("{source}\npipeline {{\n  pixel kernel \"test.pd\"\n  run test\n  display\n}}\n"),
            &dir,
        )
        .unwrap();
        // Should have key bindings from both files without infinite loop
        assert!(config.key_bindings.len() >= 2);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
