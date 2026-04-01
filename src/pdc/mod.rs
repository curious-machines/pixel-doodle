pub mod ast;
pub mod codegen;
pub mod error;
pub mod lexer;
pub mod parser;
pub mod runtime;
pub mod span;
pub mod token;
pub mod type_check;

use std::collections::HashMap;
use std::path::Path as FsPath;

use crate::vector::flatten::{self, Path};
use crate::vector::stroke::{self, StrokeStyle};
use crate::vector::{self, VectorScene, FILL_EVEN_ODD, FILL_NONZERO};

use ast::{ModuleUnit, Program};
use error::PdcError;
use runtime::{DrawCommand, PdcContext, SceneBuilder};
use span::{IdAlloc, Span};

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
        raw.trim_matches('"')
    } else {
        let raw = trimmed.strip_prefix("import ").unwrap().trim();
        raw.trim_matches('"')
    }
}

/// Recursively collect module source texts from import statements.
/// Returns a map of module name → source text.
fn load_modules(
    source: &str,
    base_dir: Option<&FsPath>,
    modules: &mut HashMap<String, String>,
    import_stack: &mut Vec<String>,
) -> Result<(), PdcError> {
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

            if !modules.contains_key(module_name) {
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

                modules.insert(module_name.to_string(), module_source.clone());

                // Recursively load modules that this module imports
                import_stack.push(module_name.to_string());
                load_modules(
                    &module_source,
                    child_base.as_deref().or(base_dir),
                    modules,
                    import_stack,
                )?;
                import_stack.pop();
            }
        }
    }
    Ok(())
}

/// Load and parse all modules and the main source into a Program.
fn load_and_parse(
    source: &str,
    base_dir: Option<&FsPath>,
) -> Result<Program, PdcError> {
    // 1. Collect all module sources
    let mut module_sources: HashMap<String, String> = HashMap::new();
    let mut import_stack = Vec::new();
    load_modules(source, base_dir, &mut module_sources, &mut import_stack)?;

    // 2. Parse each module with a shared IdAlloc
    let mut ids = IdAlloc::new();
    let mut modules = Vec::new();

    for (name, mod_source) in &module_sources {
        let tokens = lexer::lex(mod_source)?;
        let stmts = parser::parse(tokens, &mut ids)?;
        modules.push(ModuleUnit {
            name: name.clone(),
            stmts,
        });
    }

    // 3. Parse main source
    let main_tokens = lexer::lex(source)?;
    let main_stmts = parser::parse(main_tokens, &mut ids)?;

    Ok(Program {
        modules,
        stmts: main_stmts,
    })
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
    // 1. Load and parse all modules + main
    let base_dir = source_path.and_then(|p| p.parent());
    let program = load_and_parse(source, base_dir)?;

    // 2. Type check
    let mut checker = type_check::TypeChecker::new();
    checker.check_program(&program)?;

    // 3. Compile
    let builtins_layout: Vec<(&str, ast::PdcType)> = vec![
        ("width", ast::PdcType::F32),
        ("height", ast::PdcType::F32),
    ];
    let compiled = codegen::compile(&program, &checker.types, &builtins_layout, &checker.user_fns, &checker.structs, &checker.enums, &checker.fn_aliases)?;

    // 4. Execute
    let builtins = [width as f64, height as f64];
    let mut scene_builder = SceneBuilder::new();
    let mut ctx = PdcContext {
        builtins: builtins.as_ptr(),
        scene: &mut scene_builder as *mut _,
    };
    unsafe {
        (compiled.fn_ptr)(&mut ctx);
    }

    // 5. Convert draw commands to paths
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

    // 6. Flatten all paths and bin tiles
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
