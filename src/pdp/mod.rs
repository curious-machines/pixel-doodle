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
              pixel kernel "gradient.wgsl"
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

            builtin var paused: bool
            builtin var frame: u64

            var iterations: range<u32>(1..10) = 1

            on keypress(space) paused = !paused
            on keypress(period) frame += 1
            on keypress(bracket_right) iterations += 1
            on keypress(bracket_left) iterations -= 1

            pipeline {
              kernel "game_of_life.wgsl"
              kernel init_state = "init/random_binary.wgsl"
              kernel inject = "shared/inject.wgsl"

              buffer state = constant(0.0)
              buffer age = constant(0.0)
              buffer state_next = constant(0.0)
              buffer age_next = constant(0.0)

              init {
                run init_state with(out: out state)
              }
              on mousedown {
                run inject(inject_x: 0.0, inject_y: 0.0, radius: 3.0, value: 1.0, falloff_quadratic: 0) with(target: state, target_out: out state)
                run inject(inject_x: 0.0, inject_y: 0.0, radius: 3.0, value: 0.0, falloff_quadratic: 0) with(target: age, target_out: out age)
              }
              loop(iterations: iterations) {
                run game_of_life with(state_in: state, age_in: age, state_out: out state_next, age_out: out age_next)
                display
                swap state, state_next
                swap age, age_next
              }
            }
            "#,
        )
        .unwrap();

        assert_eq!(config.title.as_deref(), Some("Game of Life"));
        assert_eq!(config.pipelines[0].kernels.len(), 3);
        assert_eq!(config.pipelines[0].buffers.len(), 4);
        assert_eq!(config.variables.len(), 1);
        assert_eq!(config.event_bindings.len(), 4);
    }

    #[test]
    fn end_to_end_smoke() {
        let config = parse_pdp(
            r#"
            title = "Smoke Simulation"

            builtin var paused: bool
            builtin var frame: u64

            on keypress(space) paused = !paused
            on keypress(period) frame += 1

            pipeline {
              kernel advect = "smoke/advect.wgsl"
              kernel divergence = "smoke/divergence.wgsl"
              kernel jacobi = "smoke/jacobi.wgsl"
              kernel project = "smoke/project.wgsl"
              kernel inject = "shared/inject.wgsl"

              buffer vx = constant(0.0)
              buffer vy = constant(0.0)
              buffer density = constant(0.0)
              buffer vx0 = constant(0.0)
              buffer vy0 = constant(0.0)
              buffer density0 = constant(0.0)
              buffer pressure = constant(0.0)
              buffer pressure_tmp = constant(0.0)
              buffer divergence = constant(0.0)

              on mousedown {
                run inject(inject_x: 0.0, inject_y: 0.0, radius: 15.0, value: -3.0, falloff_quadratic: 1) with(target: vy, target_out: out vy)
                run inject(inject_x: 0.0, inject_y: 0.0, radius: 15.0, value: 0.5, falloff_quadratic: 1) with(target: density, target_out: out density)
              }
              swap vx, vx0
              swap vy, vy0
              swap density, density0
              run advect with(vx_in: vx0, vy_in: vy0, den_in: density0, vx_out: out vx, vy_out: out vy, den_out: out density)
              run divergence with(vx_in: vx, vy_in: vy, div_out: out divergence)
              loop(iterations: 40) {
                run jacobi with(div_in: divergence, p_in: pressure, p_out: out pressure_tmp)
                swap pressure, pressure_tmp
              }
              run project with(p_in: pressure, vx_in: vx, vy_in: vy, den_in: density, vx_out: out vx0, vy_out: out vy0)
              display
              swap vx, vx0
              swap vy, vy0
            }
            "#,
        )
        .unwrap();

        assert_eq!(config.pipelines[0].kernels.len(), 5);
        assert_eq!(config.pipelines[0].buffers.len(), 9);
    }

    #[test]
    fn validation_error_reported() {
        let result = parse_pdp(
            r#"
            pipeline {
              pixel kernel "test.wgsl"
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
            pixel kernel "test.wgsl"
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
            "builtin var center_x: f64\non keydown(left) center_x -= 0.1\non keydown(right) center_x += 0.1\n",
        )
        .unwrap();

        let source = concat!(
            "include \"pan_zoom.pdp\"\n",
            "\n",
            "pipeline {\n",
            "  pixel kernel \"gradient.wgsl\"\n",
            "  run gradient\n",
            "  display\n",
            "}\n",
        );
        let config = parse(source, &dir).unwrap();
        assert_eq!(config.event_bindings.len(), 2);
        assert_eq!(config.event_bindings[0].key_name, "left");
        assert_eq!(config.event_bindings[1].key_name, "right");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn include_rejects_pipeline() {
        let dir = make_test_dir("include_reject");
        std::fs::write(
            dir.join("bad.pdp"),
            "pipeline { pixel kernel \"test.wgsl\"\n run test\n display }\n",
        )
        .unwrap();

        let source = "include \"bad.pdp\"\npipeline { pixel kernel \"test.wgsl\"\n run test\n display }\n";
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
            "builtin var center_x: f64\ninclude \"b.pdp\"\non keydown(left) center_x -= 0.1\n",
        )
        .unwrap();
        std::fs::write(
            dir.join("b.pdp"),
            "builtin var center_x: f64\ninclude \"a.pdp\"\non keydown(right) center_x += 0.1\n",
        )
        .unwrap();

        let source = std::fs::read_to_string(dir.join("a.pdp")).unwrap();
        let config = parse(
            &format!("{source}\npipeline {{\n  pixel kernel \"test.wgsl\"\n  run test\n  display\n}}\n"),
            &dir,
        )
        .unwrap();
        // Should have key bindings from both files without infinite loop
        assert!(config.event_bindings.len() >= 2);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
