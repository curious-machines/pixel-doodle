pub mod ast;
pub mod codegen;
pub mod error;
pub mod lexer;
pub mod parser;
pub mod runtime;
pub mod span;
pub mod token;
pub mod type_check;

use crate::vector::flatten::{self, Path};
use crate::vector::stroke::{self, StrokeStyle};
use crate::vector::{self, VectorScene, FILL_EVEN_ODD, FILL_NONZERO};

use error::PdcError;
use runtime::{DrawCommand, PdcContext, SceneBuilder};

/// Compile and execute a PDC source file, producing a VectorScene.
pub fn compile_and_run(
    source: &str,
    width: u32,
    height: u32,
    tolerance: f32,
    tile_size: u32,
) -> Result<VectorScene, PdcError> {
    // 1. Lex
    let tokens = lexer::lex(source)?;

    // 2. Parse
    let program = parser::parse(tokens)?;

    // 3. Type check
    let mut checker = type_check::TypeChecker::new();
    checker.check_program(&program)?;

    // 4. Compile
    let builtins_layout: Vec<(&str, ast::PdcType)> = vec![
        ("width", ast::PdcType::F32),
        ("height", ast::PdcType::F32),
    ];
    let compiled = codegen::compile(&program, &checker.types, &builtins_layout)?;

    // 5. Execute
    let builtins = [width as f64, height as f64];
    let mut scene_builder = SceneBuilder::new();
    let mut ctx = PdcContext {
        builtins: builtins.as_ptr(),
        scene: &mut scene_builder as *mut _,
    };
    unsafe {
        (compiled.fn_ptr)(&mut ctx);
    }

    // 6. Convert draw commands to paths
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
                // Flatten the source path, then stroke expand
                let source_curves = &scene_builder.paths[*path_handle as usize].curves;
                let temp_path = Path {
                    curves: source_curves.clone(),
                    path_id: 0,
                };
                let (segments, _) = flatten::flatten_paths(&[temp_path], tolerance);
                let style = StrokeStyle {
                    width: *stroke_width,
                    miter_limit: 4.0,
                };
                let stroked = stroke::stroke_path(&segments, &style, path_id);
                paths.push(stroked);
                path_colors.push(*color);
                path_fill_rules.push(FILL_NONZERO);
            }
        }
    }

    // 7. Flatten all paths and bin tiles
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
