pub mod ast;
pub mod lexer;
pub mod parser;
pub mod typecheck;
pub mod lower;

use crate::kernel_ir::Kernel;
use std::path::Path;

pub fn parse(source: &str, file_path: Option<&Path>) -> Result<Kernel, String> {
    let tokens = lexer::lex(source)?;
    let p = if let Some(path) = file_path {
        use std::cell::RefCell;
        use std::collections::HashSet;
        use std::rc::Rc;

        let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        let base_dir = canonical.parent().unwrap_or(Path::new(".")).to_path_buf();
        let included = Rc::new(RefCell::new(HashSet::new()));
        included.borrow_mut().insert(canonical);
        parser::Parser::new_with_context(tokens, base_dir, included)
    } else {
        parser::Parser::new(tokens)
    };
    let mut p = p;
    let program = p.parse_program().map_err(|e| e.to_string())?;
    let typed = typecheck::typecheck(&program).map_err(|e| e.to_string())?;
    let kernel = lower::lower(&typed);
    Ok(kernel)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_ir::ValType;

    #[test]
    fn parse_gradient_pd() {
        let path = "examples/basic/gradient/gradient.pd";
        let src = std::fs::read_to_string(path).unwrap();
        let kernel = parse(&src, Some(std::path::Path::new(path))).unwrap();
        assert_eq!(kernel.name, "gradient");
        assert_eq!(kernel.params.len(), 2);
        assert_eq!(kernel.return_ty, ValType::U32);
    }

    fn parse_file(path: &str) -> crate::kernel_ir::Kernel {
        let src = std::fs::read_to_string(path).unwrap();
        parse(&src, Some(std::path::Path::new(path))).unwrap()
    }

    #[test]
    fn parse_mandelbrot_pd() {
        let kernel = parse_file("examples/basic/mandelbrot/mandelbrot.pd");
        assert_eq!(kernel.name, "mandelbrot");
        assert_eq!(kernel.params.len(), 2);
        let has_while = kernel.body.iter().any(|b| matches!(b, crate::kernel_ir::BodyItem::While(_)));
        assert!(has_while, "mandelbrot should have a while loop");
    }

    #[test]
    fn parse_sdf_flower_pd() {
        let kernel = parse_file("examples/sdf/sdf_flower/sdf_flower.pd");
        assert_eq!(kernel.name, "sdf_flower");
    }

    #[test]
    fn parse_cornell_2d_pd() {
        let kernel = parse_file("examples/lighting/cornell_2d/cornell_2d.pd");
        assert_eq!(kernel.name, "cornell_2d");
        assert_eq!(kernel.params.len(), 5);
    }

    #[test]
    fn gradient_pd_produces_valid_ir() {
        let kernel = parse_file("examples/basic/gradient/gradient.pd");
        let emit_ty = kernel.var_type(kernel.emit);
        assert_eq!(emit_ty, Some(ValType::U32));
    }

    #[test]
    fn mandelbrot_ir_has_carry_vars() {
        let kernel = parse_file("examples/basic/mandelbrot/mandelbrot.pd");
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
        let kernel = parse_file("examples/basic/gradient/gradient.pd");
        // Print to PDL text — should produce valid PDL
        let pdl = crate::lang::printer::print(&kernel);
        // Re-parse the PDL output
        let kernel2 = crate::lang::parser::parse(&pdl).unwrap();
        assert_eq!(kernel2.name, kernel.name);
        assert_eq!(kernel2.params.len(), kernel.params.len());
    }
}
