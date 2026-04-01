pub mod ast;
pub mod codegen;
pub mod error;
pub mod lexer;
pub mod parser;
pub mod runtime;
pub mod span;
pub mod token;
pub mod type_check;

use std::collections::{HashMap, HashSet};
use std::path::Path as FsPath;

use crate::vector::flatten::{self, Path};
use crate::vector::stroke::{self, StrokeStyle};
use crate::vector::{self, VectorScene, FILL_EVEN_ODD, FILL_NONZERO};

use ast::{Block, Expr, FnDef, ModuleUnit, Program, Stmt};
use error::PdcError;
use runtime::{DrawCommand, PdcContext, SceneBuilder};
use span::{IdAlloc, Span, Spanned};

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

// ── Dead code elimination ──
// Walks the raw AST to find which module functions are reachable,
// then strips unreferenced FnDef statements from module units.
// This runs before type-checking so unreachable code is never analyzed.

fn collect_expr_refs(expr: &Spanned<Expr>, refs: &mut HashSet<String>) {
    match &expr.node {
        Expr::Call { name, args, .. } => {
            refs.insert(name.clone());
            for arg in args { collect_expr_refs(arg, refs); }
        }
        Expr::MethodCall { object, method, args } => {
            collect_expr_refs(object, refs);
            refs.insert(method.clone());
            for arg in args { collect_expr_refs(arg, refs); }
        }
        Expr::FieldAccess { object, .. } => collect_expr_refs(object, refs),
        Expr::BinaryOp { left, right, .. } => {
            collect_expr_refs(left, refs);
            collect_expr_refs(right, refs);
        }
        Expr::UnaryOp { operand, .. } => collect_expr_refs(operand, refs),
        Expr::StructConstruct { name, fields } => {
            refs.insert(name.clone());
            for (_, e) in fields { collect_expr_refs(e, refs); }
        }
        Expr::TupleConstruct { elements } => {
            for e in elements { collect_expr_refs(e, refs); }
        }
        Expr::TupleIndex { object, .. } => collect_expr_refs(object, refs),
        Expr::Index { object, index } => {
            collect_expr_refs(object, refs);
            collect_expr_refs(index, refs);
        }
        Expr::Ternary { condition, then_expr, else_expr } => {
            collect_expr_refs(condition, refs);
            collect_expr_refs(then_expr, refs);
            collect_expr_refs(else_expr, refs);
        }
        Expr::Variable(name) => { refs.insert(name.clone()); }
        Expr::Literal(_) => {}
    }
}

fn collect_stmt_refs(stmt: &Spanned<Stmt>, refs: &mut HashSet<String>) {
    match &stmt.node {
        Stmt::ConstDecl { value, .. } | Stmt::VarDecl { value, .. } => collect_expr_refs(value, refs),
        Stmt::Assign { value, .. } => collect_expr_refs(value, refs),
        Stmt::IndexAssign { object, index, value } => {
            collect_expr_refs(object, refs);
            collect_expr_refs(index, refs);
            collect_expr_refs(value, refs);
        }
        Stmt::TupleDestructure { value, .. } => collect_expr_refs(value, refs),
        Stmt::ExprStmt(expr) => collect_expr_refs(expr, refs),
        Stmt::If { condition, then_body, elsif_clauses, else_body } => {
            collect_expr_refs(condition, refs);
            collect_block_refs(then_body, refs);
            for (c, b) in elsif_clauses { collect_expr_refs(c, refs); collect_block_refs(b, refs); }
            if let Some(b) = else_body { collect_block_refs(b, refs); }
        }
        Stmt::While { condition, body } => {
            collect_expr_refs(condition, refs);
            collect_block_refs(body, refs);
        }
        Stmt::For { start, end, body, .. } => {
            collect_expr_refs(start, refs);
            collect_expr_refs(end, refs);
            collect_block_refs(body, refs);
        }
        Stmt::ForEach { collection, body, .. } => {
            collect_expr_refs(collection, refs);
            collect_block_refs(body, refs);
        }
        Stmt::Loop { body } => collect_block_refs(body, refs),
        Stmt::Match { scrutinee, arms } => {
            collect_expr_refs(scrutinee, refs);
            for arm in arms { collect_block_refs(&arm.body, refs); }
        }
        Stmt::Return(Some(expr)) => collect_expr_refs(expr, refs),
        _ => {}
    }
}

fn collect_block_refs(block: &Block, refs: &mut HashSet<String>) {
    for stmt in &block.stmts { collect_stmt_refs(stmt, refs); }
}

/// Build a map of qualified function names from modules.
fn build_fn_map<'a>(program: &'a Program) -> HashMap<String, &'a FnDef> {
    let mut map = HashMap::new();
    for module in &program.modules {
        for stmt in &module.stmts {
            if let Stmt::FnDef(fndef) = &stmt.node {
                map.insert(format!("{}::{}", module.name, fndef.name), fndef);
            }
        }
    }
    for stmt in &program.stmts {
        if let Stmt::FnDef(fndef) = &stmt.node {
            map.insert(fndef.name.clone(), fndef);
        }
    }
    map
}

/// Build a simple alias map from import statements (no type checker needed).
fn build_import_aliases(program: &Program) -> HashMap<String, String> {
    let mut aliases = HashMap::new();
    for stmt in &program.stmts {
        if let Stmt::Import { module, names } = &stmt.node {
            for name in names {
                aliases.insert(name.clone(), format!("{module}::{name}"));
            }
        }
    }
    aliases
}

/// Compute reachable function qualified names via transitive AST walk.
fn reachable_functions(program: &Program) -> HashSet<String> {
    let fn_map = build_fn_map(program);
    let aliases = build_import_aliases(program);

    // Seed: references from main stmts + module init stmts
    let mut raw_refs = HashSet::new();
    for stmt in &program.stmts {
        collect_stmt_refs(stmt, &mut raw_refs);
    }
    for module in &program.modules {
        for stmt in &module.stmts {
            if !matches!(&stmt.node, Stmt::FnDef(_) | Stmt::StructDef(_) | Stmt::EnumDef(_) | Stmt::TypeAlias { .. } | Stmt::Import { .. }) {
                collect_stmt_refs(stmt, &mut raw_refs);
            }
        }
    }

    let resolve = |name: &str| -> Vec<String> {
        let mut out = Vec::new();
        if fn_map.contains_key(name) { out.push(name.to_string()); }
        if let Some(q) = aliases.get(name) {
            if fn_map.contains_key(q.as_str()) { out.push(q.clone()); }
        }
        for key in fn_map.keys() {
            if key.ends_with(&format!("::{name}")) && !out.contains(key) {
                out.push(key.clone());
            }
        }
        out
    };

    let mut reachable = HashSet::new();
    let mut worklist = Vec::new();

    for name in &raw_refs {
        for q in resolve(name) {
            if reachable.insert(q.clone()) { worklist.push(q); }
        }
    }

    while let Some(qualified) = worklist.pop() {
        if let Some(fndef) = fn_map.get(qualified.as_str()) {
            let mut body_refs = HashSet::new();
            collect_block_refs(&fndef.body, &mut body_refs);

            let module_prefix = qualified.rfind("::").map(|i| &qualified[..i]);

            for name in &body_refs {
                let mut candidates = resolve(name);
                if let Some(prefix) = module_prefix {
                    let same = format!("{prefix}::{name}");
                    if fn_map.contains_key(same.as_str()) && !candidates.contains(&same) {
                        candidates.push(same);
                    }
                }
                for c in candidates {
                    if reachable.insert(c.clone()) { worklist.push(c); }
                }
            }
        }
    }

    reachable
}

/// Strip unreferenced FnDef statements from module units.
fn eliminate_dead_code(program: &mut Program) {
    let reachable = reachable_functions(program);

    for module in &mut program.modules {
        let mod_name = &module.name;
        module.stmts.retain(|stmt| {
            if let Stmt::FnDef(fndef) = &stmt.node {
                let qualified = format!("{mod_name}::{}", fndef.name);
                reachable.contains(&qualified)
            } else {
                true // keep all non-FnDef statements
            }
        });
    }
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
    let mut program = load_and_parse(source, base_dir)?;

    // 2. Dead code elimination (before type checking)
    eliminate_dead_code(&mut program);

    // 3. Type check
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
