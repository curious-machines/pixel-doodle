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
use std::path::Path as FsPath;

use crate::vector::flatten::{self, Path};
use crate::vector::stroke::{self, StrokeStyle};
use crate::vector::{self, VectorScene, FILL_EVEN_ODD, FILL_NONZERO};

use error::PdcError;
use runtime::{DrawCommand, PdcContext, SceneBuilder};
use span::Span;

/// Embedded standard library modules.
const STDLIB_GEOMETRY: &str = include_str!("stdlib/geometry.pdc");
const STDLIB_MATH: &str = include_str!("stdlib/math.pdc");

/// Resolve a module name to source code. Checks stdlib first, then filesystem.
fn resolve_module(name: &str, base_dir: Option<&FsPath>) -> Result<String, PdcError> {
    match name {
        "geometry" => Ok(STDLIB_GEOMETRY.to_string()),
        "math" => Ok(STDLIB_MATH.to_string()),
        _ => {
            // File-based import: resolve relative to the importing file's directory
            let base = base_dir.unwrap_or_else(|| FsPath::new("."));
            let mut path = base.join(name);
            if path.extension().is_none() {
                path.set_extension("pdc");
            }
            std::fs::read_to_string(&path).map_err(|e| PdcError::Parse {
                span: Span::new(0, 0),
                message: format!("cannot import '{}': {e}", path.display()),
            })
        }
    }
}

/// Extract the module name from an import line (quick text scan before parsing).
fn extract_module_name(trimmed: &str) -> &str {
    if trimmed.contains(" from ") {
        let raw = trimmed.rsplit(" from ").next().unwrap().trim();
        // Strip surrounding quotes if present: `"./path"` → `./path`
        raw.trim_matches('"')
    } else {
        let raw = trimmed.strip_prefix("import ").unwrap().trim();
        raw.trim_matches('"')
    }
}

/// Pre-process source: scan for import statements and prepend the imported
/// module source code. This ensures all code is parsed with a single ID
/// allocator, so node IDs are consistent for type checking and codegen.
///
/// `base_dir` is the directory of the source file being processed (for
/// resolving relative imports). `import_stack` tracks the chain of files
/// currently being processed to detect circular imports.
fn preprocess_imports(
    source: &str,
    base_dir: Option<&FsPath>,
    imported_modules: &mut HashSet<String>,
    import_stack: &mut Vec<String>,
) -> Result<String, PdcError> {
    let mut prefix = String::new();

    // Quick scan for import lines (before full parsing)
    for line in source.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("import ") {
            let module_name = extract_module_name(trimmed);

            // Circular import detection
            if import_stack.contains(&module_name.to_string()) {
                return Err(PdcError::Parse {
                    span: Span::new(0, 0),
                    message: format!(
                        "circular import detected: {} → {}",
                        import_stack.join(" → "),
                        module_name,
                    ),
                });
            }

            if !imported_modules.contains(module_name) {
                imported_modules.insert(module_name.to_string());
                let module_source = resolve_module(module_name, base_dir)?;

                // Determine base dir for the imported module's own imports
                let child_base = if module_name.contains('/') || module_name.starts_with('.') {
                    let base = base_dir.unwrap_or_else(|| FsPath::new("."));
                    let mut child_path = base.join(module_name);
                    if child_path.extension().is_none() {
                        child_path.set_extension("pdc");
                    }
                    child_path.parent().map(|p| p.to_path_buf())
                } else {
                    None
                };

                // Recursively process imports in the imported module
                import_stack.push(module_name.to_string());
                let processed = preprocess_imports(
                    &module_source,
                    child_base.as_deref().or(base_dir),
                    imported_modules,
                    import_stack,
                )?;
                import_stack.pop();

                prefix.push_str(&processed);
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
///
/// `source_path` is the optional filesystem path of the `.pdc` file being
/// compiled, used to resolve relative `import` paths. Pass `None` for
/// in-memory sources that don't support file imports.
pub fn compile_and_run(
    source: &str,
    width: u32,
    height: u32,
    tolerance: f32,
    tile_size: u32,
    source_path: Option<&FsPath>,
) -> Result<VectorScene, PdcError> {
    // 1. Preprocess imports (prepend stdlib source)
    let base_dir = source_path.and_then(|p| p.parent());
    let mut imported_modules = HashSet::new();
    let mut import_stack = Vec::new();
    let full_source = preprocess_imports(source, base_dir, &mut imported_modules, &mut import_stack)?;

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
    let compiled = codegen::compile(&program, &checker.types, &builtins_layout, &checker.user_fns, &checker.structs, &checker.enums)?;

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
                rule,
            } => {
                let curves = scene_builder.paths[*path_handle as usize].curves.clone();
                paths.push(Path {
                    curves,
                    path_id,
                });
                path_colors.push(*color);
                path_fill_rules.push(match rule {
                    runtime::FillRule::EvenOdd => FILL_EVEN_ODD,
                    runtime::FillRule::NonZero => FILL_NONZERO,
                });
            }
            DrawCommand::Stroke {
                path_handle,
                width: stroke_width,
                color,
                ..
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
