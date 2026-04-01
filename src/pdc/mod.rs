pub mod ast;
pub mod codegen;
pub mod error;
pub mod lexer;
pub mod parser;
pub mod runtime;
pub mod span;
pub mod token;
pub mod type_check;

use std::collections::HashSet;

use crate::vector::flatten::{self, Path};
use crate::vector::stroke::{self, StrokeStyle};
use crate::vector::{self, VectorScene, FILL_EVEN_ODD, FILL_NONZERO};

use error::PdcError;
use runtime::{DrawCommand, PdcContext, SceneBuilder};
use span::Span;

/// Embedded standard library modules.
const STDLIB_GEOMETRY: &str = include_str!("stdlib/geometry.pdc");
const STDLIB_MATH: &str = include_str!("stdlib/math.pdc");

/// Resolve a module name to its source code.
fn resolve_module(name: &str) -> Result<&'static str, PdcError> {
    match name {
        "geometry" => Ok(STDLIB_GEOMETRY),
        "math" => Ok(STDLIB_MATH),
        _ => Err(PdcError::Parse {
            span: Span::new(0, 0),
            message: format!("unknown module '{name}'"),
        }),
    }
}

/// Pre-process source: scan for import statements and prepend the imported
/// module source code. This ensures all code is parsed with a single ID
/// allocator, so node IDs are consistent for type checking and codegen.
fn preprocess_imports(source: &str) -> Result<String, PdcError> {
    let mut imported_modules: HashSet<String> = HashSet::new();
    let mut prefix = String::new();

    // Quick scan for import lines (before full parsing)
    for line in source.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("import ") {
            // Extract module name from:
            //   import module_name
            //   import { ... } from module_name
            let module_name = if trimmed.contains(" from ") {
                trimmed.rsplit(" from ").next().unwrap().trim()
            } else {
                trimmed.strip_prefix("import ").unwrap().trim()
            };

            if !imported_modules.contains(module_name) {
                imported_modules.insert(module_name.to_string());
                let module_source = resolve_module(module_name)?;
                prefix.push_str(module_source);
                prefix.push('\n');
            }
        }
    }

    // Prepend module code, then the original source (with import lines kept —
    // the parser will parse and skip them)
    prefix.push_str(source);
    Ok(prefix)
}

/// Compile and execute a PDC source file, producing a VectorScene.
pub fn compile_and_run(
    source: &str,
    width: u32,
    height: u32,
    tolerance: f32,
    tile_size: u32,
) -> Result<VectorScene, PdcError> {
    // 1. Preprocess imports (prepend stdlib source)
    let full_source = preprocess_imports(source)?;

    // 2. Lex
    let tokens = lexer::lex(&full_source)?;

    // 3. Parse (single parse with consistent IDs)
    let program = parser::parse(tokens)?;

    // 4. Type check
    let mut checker = type_check::TypeChecker::new();
    checker.check_program(&program)?;

    // 5. Compile
    let builtins_layout: Vec<(&str, ast::PdcType)> = vec![
        ("width", ast::PdcType::F32),
        ("height", ast::PdcType::F32),
    ];
    let compiled = codegen::compile(&program, &checker.types, &builtins_layout, &checker.user_fns, &checker.structs)?;

    // 6. Execute
    let builtins = [width as f64, height as f64];
    let mut scene_builder = SceneBuilder::new();
    let mut ctx = PdcContext {
        builtins: builtins.as_ptr(),
        scene: &mut scene_builder as *mut _,
    };
    unsafe {
        (compiled.fn_ptr)(&mut ctx);
    }

    // 7. Convert draw commands to paths
    let mut paths: Vec<Path> = Vec::new();
    let mut path_colors: Vec<u32> = Vec::new();
    let mut path_fill_rules: Vec<u32> = Vec::new();

    for (draw_idx, draw) in scene_builder.draws.iter().enumerate() {
        let path_id = draw_idx as u32;
        match draw {
            DrawCommand::Fill {
                path_handle,
                color,
            } => {
                let curves = scene_builder.paths[*path_handle as usize].curves.clone();
                paths.push(Path {
                    curves,
                    path_id,
                });
                path_colors.push(*color);
                path_fill_rules.push(FILL_EVEN_ODD);
            }
            DrawCommand::Stroke {
                path_handle,
                width: stroke_width,
                color,
            } => {
                let source_curves = &scene_builder.paths[*path_handle as usize].curves;
                let temp_path = Path {
                    curves: source_curves.clone(),
                    path_id: 0,
                };
                let (segments, _) = flatten::flatten_paths(&[temp_path], tolerance);
                let style = StrokeStyle {
                    width: *stroke_width as f64,
                    miter_limit: 4.0,
                };
                let stroked = stroke::stroke_path(&segments, &style, path_id);
                paths.push(stroked);
                path_colors.push(*color);
                path_fill_rules.push(FILL_NONZERO);
            }
        }
    }

    // 8. Flatten all paths and bin tiles
    let (segments, seg_path_ids) = flatten::flatten_paths(&paths, tolerance);
    Ok(vector::bin_tiles_pub(
        &segments,
        &seg_path_ids,
        path_colors,
        path_fill_rules,
        tile_size,
        width,
        height,
    ))
}
