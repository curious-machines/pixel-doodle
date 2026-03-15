pub mod ast;
pub mod lexer;
pub mod parser;
pub mod typecheck;
pub mod lower;

use crate::kernel_ir::Kernel;

pub fn parse(source: &str) -> Result<Kernel, String> {
    let tokens = lexer::lex(source)?;
    let mut p = parser::Parser::new(tokens);
    let program = p.parse_program().map_err(|e| e.to_string())?;
    let typed = typecheck::typecheck(&program).map_err(|e| e.to_string())?;
    let kernel = lower::lower(&typed);
    Ok(kernel)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_ir::ScalarType;

    #[test]
    fn parse_gradient_pd() {
        let src = std::fs::read_to_string("examples/gradient.pd").unwrap();
        let kernel = parse(&src).unwrap();
        assert_eq!(kernel.name, "gradient");
        assert_eq!(kernel.params.len(), 2);
        assert_eq!(kernel.return_ty, ScalarType::U32);
    }

    #[test]
    fn parse_mandelbrot_pd() {
        let src = std::fs::read_to_string("examples/mandelbrot.pd").unwrap();
        let kernel = parse(&src).unwrap();
        assert_eq!(kernel.name, "mandelbrot");
        assert_eq!(kernel.params.len(), 2);
        // Should have a while loop in body
        let has_while = kernel.body.iter().any(|b| matches!(b, crate::kernel_ir::BodyItem::While(_)));
        assert!(has_while, "mandelbrot should have a while loop");
    }

    #[test]
    fn parse_sdf_flower_pd() {
        let src = std::fs::read_to_string("examples/sdf_flower.pd").unwrap();
        let kernel = parse(&src).unwrap();
        assert_eq!(kernel.name, "sdf_flower");
    }

    #[test]
    fn parse_cornell_2d_pd() {
        let src = std::fs::read_to_string("examples/cornell_2d.pd").unwrap();
        let kernel = parse(&src).unwrap();
        assert_eq!(kernel.name, "cornell_2d");
        assert_eq!(kernel.params.len(), 5);
    }

    #[test]
    fn gradient_pd_produces_valid_ir() {
        let src = std::fs::read_to_string("examples/gradient.pd").unwrap();
        let kernel = parse(&src).unwrap();
        // The emit var should exist and be u32
        let emit_ty = kernel.var_type(kernel.emit);
        assert_eq!(emit_ty, Some(ScalarType::U32));
    }

    #[test]
    fn mandelbrot_ir_has_carry_vars() {
        let src = std::fs::read_to_string("examples/mandelbrot.pd").unwrap();
        let kernel = parse(&src).unwrap();
        // Find the while loop and check it has 3 carry vars
        for item in &kernel.body {
            if let crate::kernel_ir::BodyItem::While(w) = item {
                assert_eq!(w.carry.len(), 3);
                assert_eq!(w.yields.len(), 3);
                return;
            }
        }
        panic!("no while loop found");
    }

    #[test]
    fn pd_roundtrip_via_printer() {
        let src = std::fs::read_to_string("examples/gradient.pd").unwrap();
        let kernel = parse(&src).unwrap();
        // Print to PDL text — should produce valid PDL
        let pdl = crate::lang::printer::print(&kernel);
        // Re-parse the PDL output
        let kernel2 = crate::lang::parser::parse(&pdl).unwrap();
        assert_eq!(kernel2.name, kernel.name);
        assert_eq!(kernel2.params.len(), kernel.params.len());
    }
}
