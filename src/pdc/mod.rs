pub mod ast;
pub mod codegen_common;
#[cfg(feature = "cranelift-backend")]
pub(crate) mod codegen_cranelift;
#[cfg(feature = "llvm-backend")]
pub(crate) mod codegen_llvm;
pub mod error;
pub mod lexer;
pub mod parser;
pub mod runtime;
pub mod span;
pub mod token;
pub mod type_check;

/// Re-export the active codegen backend as `codegen`.
/// When both features are enabled, Cranelift takes priority.
pub mod codegen {
    pub use super::codegen_common::*;

    #[cfg(feature = "cranelift-backend")]
    pub use super::codegen_cranelift::compile;

    #[cfg(all(feature = "llvm-backend", not(feature = "cranelift-backend")))]
    pub use super::codegen_llvm::compile;
}

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
const STDLIB_VECTOR2D: &str = include_str!("stdlib/vector2d.pdc");
const STDLIB_VECTOR3D: &str = include_str!("stdlib/vector3d.pdc");
const STDLIB_VECTOR4D: &str = include_str!("stdlib/vector4d.pdc");
const STDLIB_POINT2D: &str = include_str!("stdlib/point2d.pdc");
const STDLIB_POINT3D: &str = include_str!("stdlib/point3d.pdc");
const STDLIB_POINT4D: &str = include_str!("stdlib/point4d.pdc");
const STDLIB_AFFINE2D: &str = include_str!("stdlib/affine2d.pdc");
const STDLIB_AFFINE3D: &str = include_str!("stdlib/affine3d.pdc");

/// Resolve a module name to source code. Checks stdlib first, then filesystem.
fn resolve_module(name: &str, base_dir: Option<&FsPath>) -> Result<String, PdcError> {
    match name {
        "geometry" => Ok(STDLIB_GEOMETRY.to_string()),
        "math" => Ok(STDLIB_MATH.to_string()),
        "vector2d" => Ok(STDLIB_VECTOR2D.to_string()),
        "vector3d" => Ok(STDLIB_VECTOR3D.to_string()),
        "vector4d" => Ok(STDLIB_VECTOR4D.to_string()),
        "point2d" => Ok(STDLIB_POINT2D.to_string()),
        "point3d" => Ok(STDLIB_POINT3D.to_string()),
        "point4d" => Ok(STDLIB_POINT4D.to_string()),
        "affine2d" => Ok(STDLIB_AFFINE2D.to_string()),
        "affine3d" => Ok(STDLIB_AFFINE3D.to_string()),
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

/// Topologically sort modules so dependencies come before dependents.
fn toposort_modules(modules: &HashMap<String, String>) -> Vec<String> {
    // Build dependency graph: module → set of modules it imports
    let mut deps: HashMap<&str, Vec<&str>> = HashMap::new();
    for (name, source) in modules {
        let mut module_deps = Vec::new();
        for line in source.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("import ") {
                let dep_name = extract_module_name(trimmed);
                if modules.contains_key(dep_name) {
                    module_deps.push(dep_name);
                }
            }
        }
        deps.insert(name.as_str(), module_deps);
    }

    // Kahn's algorithm
    let mut in_degree: HashMap<&str, usize> = modules.keys().map(|k| (k.as_str(), 0)).collect();
    for dep_list in deps.values() {
        for dep in dep_list {
            *in_degree.entry(dep).or_insert(0) += 1;
        }
    }

    // Wait — in_degree should count how many modules depend on each module.
    // Actually, Kahn's: in_degree[X] = number of modules X depends on.
    // Then process modules with in_degree 0 first.
    let mut in_deg: HashMap<&str, usize> = HashMap::new();
    for (name, dep_list) in &deps {
        in_deg.insert(name, dep_list.len());
    }

    let mut queue: Vec<&str> = in_deg.iter()
        .filter(|(_, d)| **d == 0)
        .map(|(&n, _)| n)
        .collect();
    queue.sort(); // deterministic order for same-level modules

    let mut result = Vec::new();
    while let Some(name) = queue.pop() {
        result.push(name.to_string());
        // Find modules that depend on `name` and decrement their in-degree
        for (dependent, dep_list) in &deps {
            if dep_list.contains(&name) {
                if let Some(d) = in_deg.get_mut(dependent) {
                    *d -= 1;
                    if *d == 0 {
                        queue.push(dependent);
                        queue.sort();
                    }
                }
            }
        }
    }

    // Any remaining modules (cycles) — shouldn't happen due to circular import detection
    for name in modules.keys() {
        if !result.contains(name) {
            result.push(name.clone());
        }
    }

    result
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

    // 2. Topologically sort modules so dependencies come first
    let sorted_names = toposort_modules(&module_sources);

    // 3. Parse each module with a shared IdAlloc
    let mut ids = IdAlloc::new();
    let mut modules = Vec::new();

    for name in &sorted_names {
        let mod_source = &module_sources[name];
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
        Stmt::TestDef { body, .. } => collect_block_refs(body, refs),
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
                // Always keep operator overloads — they're referenced
                // implicitly by operator expressions, not by name.
                // TODO: a post-typecheck DCE pass could prune unused
                // operator overloads using operand type information.
                if fndef.name.starts_with("__op_") {
                    return true;
                }
                let qualified = format!("{mod_name}::{}", fndef.name);
                reachable.contains(&qualified)
            } else {
                true // keep all non-FnDef statements
            }
        });
    }
}

/// Strip test blocks from a program (for production builds).
fn strip_tests(program: &mut Program) {
    for module in &mut program.modules {
        module.stmts.retain(|stmt| !matches!(&stmt.node, Stmt::TestDef { .. }));
    }
    program.stmts.retain(|stmt| !matches!(&stmt.node, Stmt::TestDef { .. }));
}

/// Run all PDC test blocks in a source file, returning per-test results.
///
/// Each `test "name" { ... }` block is compiled and executed independently.
/// Assert failures are collected per test (all asserts run, not just the first).
pub fn run_pdc_tests(
    source: &str,
    source_path: Option<&FsPath>,
) -> Result<Vec<runtime::PdcTestResult>, PdcError> {
    let (compiled, state_layout) = compile_only(source, source_path)?;
    let mut builtins = [200.0f64, 200.0f64];

    let mut results = Vec::new();
    for (test_name, test_fn) in &compiled.test_fns {
        let mut scene = SceneBuilder::new();
        let mut state_block = state_layout.alloc();
        let mut ctx = PdcContext {
            builtins: builtins.as_mut_ptr(),
            scene: &mut scene as *mut _,
            state: state_block.as_mut_ptr(),
            host: std::ptr::null_mut(),
        };
        // Clear any previous failures
        runtime::take_assert_failures();
        unsafe { test_fn(&mut ctx); }
        let failures = runtime::take_assert_failures();
        let passed = failures.is_empty();
        let message = if passed {
            String::new()
        } else {
            failures.iter().map(|f| f.message.as_str()).collect::<Vec<_>>().join("; ")
        };
        results.push(runtime::PdcTestResult {
            name: test_name.clone(),
            passed,
            message,
        });
    }
    Ok(results)
}

/// Compile a PDC source without executing. Returns the compiled program
/// for inspection and individual function calls from test code.
///
/// Unlike `compile_and_run`, this skips dead code elimination so that all
/// user-defined functions remain available for testing via `call_fn`.
pub fn compile_only(
    source: &str,
    source_path: Option<&FsPath>,
) -> Result<(codegen::CompiledProgram, codegen::StateLayout), PdcError> {
    let builtins_layout: Vec<(&str, ast::PdcType)> = vec![
        ("width", ast::PdcType::F32),
        ("height", ast::PdcType::F32),
    ];
    compile_only_with_builtins(source, source_path, &builtins_layout)
}

/// Like `compile_only`, but with a custom builtins layout.
/// Use `codegen::PIPELINE_BUILTINS` for the full pipeline builtins set.
pub fn compile_only_with_builtins(
    source: &str,
    source_path: Option<&FsPath>,
    builtins_layout: &[(&str, ast::PdcType)],
) -> Result<(codegen::CompiledProgram, codegen::StateLayout), PdcError> {
    let base_dir = source_path.and_then(|p| p.parent());
    let program = load_and_parse(source, base_dir)?;
    // No DCE — we want all functions available for testing
    let mut checker = type_check::TypeChecker::new();
    checker.check_program(&program)?;

    codegen::compile(
        &program,
        &checker.types,
        builtins_layout,
        &checker.user_fns,
        &checker.structs,
        &checker.enums,
        &checker.fn_aliases,
        &checker.op_overloads,
    )
}

/// Compile and evaluate a PDC expression, returning its value.
///
/// Wraps the expression in a synthetic function, compiles, and calls it.
/// The `expected_type` must be a valid PDC type name (e.g., "f64", "i32", "bool").
pub fn eval_expr(expr: &str, expected_type: &str) -> Result<codegen::PdcValue, PdcError> {
    let source = format!(
        "builtin const width: f32\nbuiltin const height: f32\n\
         fn __eval__() -> {expected_type} {{ return {expr} }}\n"
    );
    let (compiled, _state_layout) = compile_only(&source, None)?;
    let mut builtins = [200.0f64, 200.0f64];
    let mut scene_builder = SceneBuilder::new();
    let mut ctx = PdcContext {
        builtins: builtins.as_mut_ptr(),
        scene: &mut scene_builder as *mut _,
        state: std::ptr::null_mut(),
            host: std::ptr::null_mut(),
    };
    unsafe { compiled.call_fn("__eval__", &mut ctx, &[]) }
}

/// Compile a PDC source for use in a PDP pipeline.
///
/// Performs dead code elimination and test stripping (unlike `compile_only`),
/// but does not execute the program. Returns a `CompiledProgram` ready for
/// execution via `fn_ptr` or `call_fn`.
pub fn compile_for_pipeline(
    source: &str,
    source_path: Option<&FsPath>,
) -> Result<(codegen::CompiledProgram, codegen::StateLayout), PdcError> {
    let base_dir = source_path.and_then(|p| p.parent());
    let mut program = load_and_parse(source, base_dir)?;

    eliminate_dead_code(&mut program);
    strip_tests(&mut program);

    let mut checker = type_check::TypeChecker::new();
    checker.check_program(&program)?;

    let builtins_layout: Vec<(&str, ast::PdcType)> = vec![
        ("width", ast::PdcType::F32),
        ("height", ast::PdcType::F32),
    ];
    codegen::compile(
        &program,
        &checker.types,
        &builtins_layout,
        &checker.user_fns,
        &checker.structs,
        &checker.enums,
        &checker.fn_aliases,
        &checker.op_overloads,
    )
}

/// Compile a PDC source for use in a PDP pipeline, selecting the codegen
/// backend at runtime by name ("cranelift" or "llvm").
pub fn compile_for_pipeline_with_codegen(
    source: &str,
    source_path: Option<&FsPath>,
    codegen_backend: &str,
) -> Result<(codegen::CompiledProgram, codegen::StateLayout), error::PdcError> {
    let base_dir = source_path.and_then(|p| p.parent());
    let mut program = load_and_parse(source, base_dir)?;

    eliminate_dead_code(&mut program);
    strip_tests(&mut program);

    let mut checker = type_check::TypeChecker::new();
    checker.check_program(&program)?;

    let builtins_layout: Vec<(&str, ast::PdcType)> = vec![
        ("width", ast::PdcType::F32),
        ("height", ast::PdcType::F32),
    ];

    match codegen_backend {
        #[cfg(feature = "cranelift-backend")]
        "cranelift" => codegen_cranelift::compile(
            &program, &checker.types, &builtins_layout, &checker.user_fns,
            &checker.structs, &checker.enums, &checker.fn_aliases, &checker.op_overloads,
        ),
        #[cfg(feature = "llvm-backend")]
        "llvm" => codegen_llvm::compile(
            &program, &checker.types, &builtins_layout, &checker.user_fns,
            &checker.structs, &checker.enums, &checker.fn_aliases, &checker.op_overloads,
        ),
        _ => Err(error::PdcError::Codegen {
            message: format!("unsupported codegen backend '{codegen_backend}'"),
        }),
    }
}

/// Extract a `VectorScene` from an executed `SceneBuilder`.
///
/// Converts draw commands to paths, flattens curves, expands strokes,
/// and bins segments into tiles for GPU rasterization.
pub fn extract_scene(
    scene_builder: &SceneBuilder,
    tolerance: f32,
    tile_size: u32,
    width: u32,
    height: u32,
) -> VectorScene {
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

    let (segments, seg_path_ids) = flatten::flatten_paths(&paths, tolerance);
    vector::bin_tiles_pub(
        &segments,
        &seg_path_ids,
        path_colors,
        path_fill_rules,
        tile_size,
        width,
        height,
    )
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

    // 2. Dead code elimination and strip test blocks (before type checking)
    eliminate_dead_code(&mut program);
    strip_tests(&mut program);

    // 3. Type check
    let mut checker = type_check::TypeChecker::new();
    checker.check_program(&program)?;

    // 3. Compile
    let builtins_layout: Vec<(&str, ast::PdcType)> = vec![
        ("width", ast::PdcType::F32),
        ("height", ast::PdcType::F32),
    ];
    let (compiled, state_layout) = codegen::compile(&program, &checker.types, &builtins_layout, &checker.user_fns, &checker.structs, &checker.enums, &checker.fn_aliases, &checker.op_overloads)?;

    // 4. Execute
    let mut builtins = [width as f64, height as f64];
    let mut scene_builder = SceneBuilder::new();
    let mut state_block = state_layout.alloc();
    let mut ctx = PdcContext {
        builtins: builtins.as_mut_ptr(),
        scene: &mut scene_builder as *mut _,
        state: state_block.as_mut_ptr(),
            host: std::ptr::null_mut(),
    };
    unsafe {
        (compiled.fn_ptr)(&mut ctx);
    }

    // 5. Extract scene
    Ok(extract_scene(&scene_builder, tolerance, tile_size, width, height))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn compile_ok(src: &str) {
        compile_and_run(src, 200, 200, 0.5, 16, None).expect("compile_and_run failed");
    }

    fn compile_err(src: &str) -> String {
        match compile_and_run(src, 200, 200, 0.5, 16, None) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected error"),
        }
    }

    #[test]
    fn break_exits_loop() {
        compile_ok(
            r#"
            import { Circle } from geometry
            builtin const width: f32
            builtin const height: f32
            var n: f64 = 0.0
            loop {
                n = n + 1.0
                if n >= 5.0 { break }
            }
            Circle(cx: n * 20.0, cy: 100.0, r: 10.0).fill(0xFF0000FF)
            "#,
        );
    }

    #[test]
    fn continue_in_for_loop() {
        compile_ok(
            r#"
            import { Circle } from geometry
            builtin const width: f32
            builtin const height: f32
            var count: f64 = 0.0
            for i in 0..10 {
                if i % 2 == 0 { continue }
                count = count + 1.0
            }
            Circle(cx: count * 20.0, cy: 100.0, r: 10.0).fill(0xFF0000FF)
            "#,
        );
    }

    #[test]
    fn break_in_for_loop() {
        compile_ok(
            r#"
            import { Circle } from geometry
            builtin const width: f32
            builtin const height: f32
            var n: f64 = 0.0
            for i in 0..100 {
                if i >= 3 { break }
                n = n + 1.0
            }
            Circle(cx: n * 20.0, cy: 100.0, r: 10.0).fill(0xFF0000FF)
            "#,
        );
    }

    #[test]
    fn continue_in_while_loop() {
        compile_ok(
            r#"
            import { Circle } from geometry
            builtin const width: f32
            builtin const height: f32
            var i: i32 = 0
            var count: f64 = 0.0
            while i < 10 {
                i = i + 1
                if i % 2 == 0 { continue }
                count = count + 1.0
            }
            Circle(cx: count * 20.0, cy: 100.0, r: 10.0).fill(0xFF0000FF)
            "#,
        );
    }

    #[test]
    fn break_outside_loop_errors() {
        let err = compile_err(
            r#"
            builtin const width: f32
            builtin const height: f32
            break
            "#,
        );
        assert!(err.contains("break outside of loop"), "got: {err}");
    }

    #[test]
    fn default_param_basic() {
        compile_ok(
            r#"
            import { Circle } from geometry
            builtin const width: f32
            builtin const height: f32
            fn draw_dot(x: f64, y: f64, r: f64 = 10.0) -> Path {
                return Circle(cx: x, cy: y, r: r)
            }
            draw_dot(100.0, 100.0).fill(0xFF0000FF)
            draw_dot(200.0, 100.0, 20.0).fill(0x00FF00FF)
            "#,
        );
    }

    #[test]
    fn default_param_multiple() {
        compile_ok(
            r#"
            import { Circle } from geometry
            builtin const width: f32
            builtin const height: f32
            fn make(x: f64, y: f64, r: f64 = 15.0, color: u32 = 0xFF0000FF) -> Path {
                return Circle(cx: x, cy: y, r: r)
            }
            make(50.0, 50.0).fill(0xFF0000FF)
            make(100.0, 50.0, 20.0).fill(0xFF0000FF)
            make(150.0, 50.0, 25.0, 0x00FF00FF).fill(0xFF0000FF)
            "#,
        );
    }

    #[test]
    fn rounded_rect_default_radius() {
        compile_ok(
            r#"
            import { RoundedRect } from geometry
            builtin const width: f32
            builtin const height: f32
            RoundedRect(x: 10.0, y: 10.0, w: 100.0, h: 50.0).fill(0xFF0000FF)
            RoundedRect(x: 10.0, y: 70.0, w: 100.0, h: 50.0, r: 8.0).fill(0x00FF00FF)
            "#,
        );
    }

    #[test]
    fn default_param_required_after_default_errors() {
        let err = compile_err(
            r#"
            builtin const width: f32
            builtin const height: f32
            fn bad(a: f64 = 1.0, b: f64) -> f64 {
                return a + b
            }
            "#,
        );
        assert!(err.contains("must have a default value"), "got: {err}");
    }

    // ---- PDC function call tests (compile_only + call_fn + eval_expr) ----

    use codegen::PdcValue;

    fn pdc_header() -> &'static str {
        "builtin const width: f32\nbuiltin const height: f32\n"
    }

    fn compile_and_call(source: &str, fn_name: &str, args: &[PdcValue]) -> PdcValue {
        let full = format!("{}{}", pdc_header(), source);
        let (compiled, state_layout) = compile_only(&full, None).expect("compile_only failed");
        let mut builtins = [200.0f64, 200.0f64];
        let mut scene = SceneBuilder::new();
        let mut state_block = state_layout.alloc();
        let mut ctx = PdcContext {
            builtins: builtins.as_mut_ptr(),
            scene: &mut scene as *mut _,
            state: state_block.as_mut_ptr(),
            host: std::ptr::null_mut(),
        };
        unsafe { compiled.call_fn(fn_name, &mut ctx, args) }.expect("call_fn failed")
    }

    // ---- eval_expr tests ----

    #[test]
    fn eval_expr_addition() {
        assert_eq!(eval_expr("2.0 + 3.0", "f64").unwrap(), PdcValue::F64(5.0));
    }

    #[test]
    fn eval_expr_subtraction() {
        assert_eq!(eval_expr("10.0 - 4.0", "f64").unwrap(), PdcValue::F64(6.0));
    }

    #[test]
    fn eval_expr_multiplication() {
        assert_eq!(eval_expr("3.0 * 7.0", "f64").unwrap(), PdcValue::F64(21.0));
    }

    #[test]
    fn eval_expr_division() {
        assert_eq!(eval_expr("15.0 / 3.0", "f64").unwrap(), PdcValue::F64(5.0));
    }

    #[test]
    fn eval_expr_integer_arithmetic() {
        assert_eq!(eval_expr("10 + 20", "i32").unwrap(), PdcValue::I32(30));
    }

    #[test]
    fn eval_expr_integer_division() {
        assert_eq!(eval_expr("10 / 3", "i32").unwrap(), PdcValue::I32(3));
    }

    #[test]
    fn eval_expr_integer_modulo() {
        assert_eq!(eval_expr("10 % 3", "i32").unwrap(), PdcValue::I32(1));
    }

    #[test]
    fn eval_expr_bool_true() {
        assert_eq!(eval_expr("true", "bool").unwrap(), PdcValue::Bool(true));
    }

    #[test]
    fn eval_expr_bool_false() {
        assert_eq!(eval_expr("false", "bool").unwrap(), PdcValue::Bool(false));
    }

    #[test]
    fn eval_expr_comparison() {
        assert_eq!(eval_expr("5 > 3", "bool").unwrap(), PdcValue::Bool(true));
        assert_eq!(eval_expr("2 > 3", "bool").unwrap(), PdcValue::Bool(false));
    }

    #[test]
    fn eval_expr_float_negation() {
        assert_eq!(eval_expr("-1.0", "f64").unwrap(), PdcValue::F64(-1.0));
    }

    #[test]
    fn eval_expr_division_by_zero_float() {
        let result = eval_expr("1.0 / 0.0", "f64").unwrap();
        assert_eq!(result, PdcValue::F64(f64::INFINITY));
    }

    // ---- call_fn tests ----

    #[test]
    fn call_fn_add_f64() {
        let result = compile_and_call(
            "fn add(a: f64, b: f64) -> f64 { return a + b }",
            "add", &[PdcValue::F64(2.0), PdcValue::F64(3.0)],
        );
        assert_eq!(result, PdcValue::F64(5.0));
    }

    #[test]
    fn call_fn_multiply_i32() {
        let result = compile_and_call(
            "fn mul(a: i32, b: i32) -> i32 { return a * b }",
            "mul", &[PdcValue::I32(6), PdcValue::I32(7)],
        );
        assert_eq!(result, PdcValue::I32(42));
    }

    #[test]
    fn call_fn_no_args() {
        let result = compile_and_call(
            "fn pi() -> f64 { return 3.14159 }",
            "pi", &[],
        );
        assert_eq!(result, PdcValue::F64(3.14159));
    }

    #[test]
    fn call_fn_bool_return() {
        let result = compile_and_call(
            "fn is_positive(x: f64) -> bool { return x > 0.0 }",
            "is_positive", &[PdcValue::F64(5.0)],
        );
        assert_eq!(result, PdcValue::Bool(true));

        let result2 = compile_and_call(
            "fn is_positive(x: f64) -> bool { return x > 0.0 }",
            "is_positive", &[PdcValue::F64(-1.0)],
        );
        assert_eq!(result2, PdcValue::Bool(false));
    }

    #[test]
    fn call_fn_conditional_logic() {
        let result = compile_and_call(
            r#"fn abs_val(x: f64) -> f64 {
                if x < 0.0 { return -x }
                return x
            }"#,
            "abs_val", &[PdcValue::F64(-42.5)],
        );
        assert_eq!(result, PdcValue::F64(42.5));
    }

    #[test]
    fn call_fn_recursive() {
        let result = compile_and_call(
            r#"fn factorial(n: i32) -> i32 {
                if n <= 1 { return 1 }
                return n * factorial(n - 1)
            }"#,
            "factorial", &[PdcValue::I32(5)],
        );
        assert_eq!(result, PdcValue::I32(120));
    }

    #[test]
    fn call_fn_three_args() {
        let result = compile_and_call(
            "fn clamp(x: f64, lo: f64, hi: f64) -> f64 { if x < lo { return lo } if x > hi { return hi } return x }",
            "clamp", &[PdcValue::F64(15.0), PdcValue::F64(0.0), PdcValue::F64(10.0)],
        );
        assert_eq!(result, PdcValue::F64(10.0));
    }

    #[test]
    fn call_fn_loop_with_accumulator() {
        let result = compile_and_call(
            r#"fn sum_to(n: i32) -> i32 {
                var total: i32 = 0
                for i in 1..=n {
                    total = total + i
                }
                return total
            }"#,
            "sum_to", &[PdcValue::I32(10)],
        );
        assert_eq!(result, PdcValue::I32(55));
    }

    // ---- Error cases ----

    #[test]
    fn call_fn_unknown_name() {
        let full = format!("{}fn dummy() -> i32 {{ return 0 }}", pdc_header());
        let (compiled, state_layout) = compile_only(&full, None).unwrap();
        let mut builtins = [200.0f64, 200.0f64];
        let mut scene = SceneBuilder::new();
        let mut state_block = state_layout.alloc();
        let mut ctx = PdcContext {
            builtins: builtins.as_mut_ptr(),
            scene: &mut scene as *mut _,
            state: state_block.as_mut_ptr(),
            host: std::ptr::null_mut(),
        };
        let err = unsafe { compiled.call_fn("nonexistent", &mut ctx, &[]) };
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("no compiled function"));
    }

    #[test]
    fn call_fn_wrong_arg_count() {
        let full = format!("{}fn add(a: f64, b: f64) -> f64 {{ return a + b }}", pdc_header());
        let (compiled, state_layout) = compile_only(&full, None).unwrap();
        let mut builtins = [200.0f64, 200.0f64];
        let mut scene = SceneBuilder::new();
        let mut state_block = state_layout.alloc();
        let mut ctx = PdcContext {
            builtins: builtins.as_mut_ptr(),
            scene: &mut scene as *mut _,
            state: state_block.as_mut_ptr(),
            host: std::ptr::null_mut(),
        };
        let err = unsafe { compiled.call_fn("add", &mut ctx, &[PdcValue::F64(1.0)]) };
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("expects 2 args"));
    }

    // ---- Edge cases ----

    #[test]
    fn eval_expr_large_integer() {
        assert_eq!(eval_expr("2147483647", "i32").unwrap(), PdcValue::I32(i32::MAX));
    }

    #[test]
    fn call_fn_zero_return() {
        let result = compile_and_call(
            "fn zero() -> f64 { return 0.0 }",
            "zero", &[],
        );
        assert_eq!(result, PdcValue::F64(0.0));
    }

    #[test]
    fn call_fn_negative_return() {
        let result = compile_and_call(
            "fn neg() -> i32 { return -1 }",
            "neg", &[],
        );
        assert_eq!(result, PdcValue::I32(-1));
    }

    // ── PDC test framework tests ──

    fn run_pdc_tests_and_assert(source: &str) {
        let source = format!("{}{}", pdc_header(), source);
        let results = run_pdc_tests(&source, None).expect("run_pdc_tests failed");
        let mut failures = Vec::new();
        for r in &results {
            let status = if r.passed { "PASS" } else { "FAIL" };
            println!("  {status}: {}", r.name);
            if !r.passed {
                println!("        {}", r.message);
                failures.push(format!("  '{}': {}", r.name, r.message));
            }
        }
        println!("  {}/{} passed", results.len() - failures.len(), results.len());
        assert!(
            failures.is_empty(),
            "PDC tests failed ({}/{} passed):\n{}",
            results.len() - failures.len(),
            results.len(),
            failures.join("\n"),
        );
        assert!(!results.is_empty(), "no PDC tests found in source");
    }

    #[test]
    fn pdc_test_framework_basic() {
        run_pdc_tests_and_assert(r#"
            test "one equals one" {
                assert_eq(1.0, 1.0)
            }
            test "true is true" {
                assert_true(true)
            }
            test "near works" {
                assert_near(1.0000001, 1.0, 0.001)
            }
        "#);
    }

    #[test]
    fn pdc_test_framework_assert_eq_integer() {
        run_pdc_tests_and_assert(r#"
            test "integer equality" {
                assert_eq(42, 42)
            }
        "#);
    }

    #[test]
    fn pdc_test_framework_assert_eq_bool() {
        run_pdc_tests_and_assert(r#"
            test "bool equality" {
                assert_eq(true, true)
                assert_eq(false, false)
            }
        "#);
    }

    #[test]
    fn pdc_test_framework_failure_reports() {
        let source = format!(
            "{}test \"should fail\" {{ assert_eq(1.0, 2.0) }}",
            pdc_header()
        );
        let results = run_pdc_tests(&source, None).unwrap();
        assert_eq!(results.len(), 1);
        assert!(!results[0].passed);
        assert!(results[0].message.contains("assert_eq failed"));
    }

    #[test]
    fn pdc_test_framework_multiple_failures_collected() {
        let source = format!(
            "{}test \"multi fail\" {{ assert_eq(1.0, 2.0)\nassert_eq(3.0, 4.0) }}",
            pdc_header()
        );
        let results = run_pdc_tests(&source, None).unwrap();
        assert_eq!(results.len(), 1);
        assert!(!results[0].passed);
        // Both failures should be collected
        assert!(results[0].message.contains("1") && results[0].message.contains("2"));
        assert!(results[0].message.contains("3") && results[0].message.contains("4"));
    }

    #[test]
    fn pdc_test_imported_module_tests_not_collected() {
        // math module has many tests, but importing it should not run them
        let source = format!(
            "{}\nimport math\ntest \"only this runs\" {{ assert_eq(1.0, 1.0) }}",
            pdc_header()
        );
        let results = run_pdc_tests(&source, None).unwrap();
        assert_eq!(results.len(), 1, "expected 1 test, got {}: {:?}",
            results.len(), results.iter().map(|r| &r.name).collect::<Vec<_>>());
        assert_eq!(results[0].name, "only this runs");
        assert!(results[0].passed);
    }

    #[test]
    fn pdc_test_strip_from_production() {
        let source = format!(
            "{}test \"should not run\" {{ assert_eq(1.0, 2.0) }}",
            pdc_header()
        );
        // compile_and_run should succeed (tests stripped, not executed)
        compile_and_run(&source, 200, 200, 0.5, 16, None).expect("should compile without error");
    }

    #[test]
    fn pdc_test_can_call_functions() {
        run_pdc_tests_and_assert(r#"
            fn add(a: f64, b: f64) -> f64 {
                return a + b
            }
            test "add works" {
                assert_eq(add(2.0, 3.0), 5.0)
            }
        "#);
    }

    // ── Stdlib PDC tests ──

    fn run_stdlib_tests(module_source: &str, module_name: &str) {
        let source = format!("{}\n{}", pdc_header(), module_source);
        let results = run_pdc_tests(&source, None)
            .unwrap_or_else(|e| panic!("{module_name} stdlib tests failed to compile: {e}"));
        let mut failures = Vec::new();
        for r in &results {
            let status = if r.passed { "PASS" } else { "FAIL" };
            println!("  {status}: {module_name}::{}", r.name);
            if !r.passed {
                println!("        {}", r.message);
                failures.push(format!("  '{}': {}", r.name, r.message));
            }
        }
        println!("  {}/{} passed", results.len() - failures.len(), results.len());
        assert!(
            failures.is_empty(),
            "{module_name} stdlib tests failed ({}/{} passed):\n{}",
            results.len() - failures.len(),
            results.len(),
            failures.join("\n"),
        );
        assert!(!results.is_empty(), "no tests found in {module_name} stdlib");
    }

    #[test]
    fn pdc_stdlib_math() { run_stdlib_tests(STDLIB_MATH, "math"); }

    #[test]
    fn pdc_stdlib_vector2d() { run_stdlib_tests(STDLIB_VECTOR2D, "vector2d"); }

    #[test]
    fn pdc_stdlib_vector3d() { run_stdlib_tests(STDLIB_VECTOR3D, "vector3d"); }

    #[test]
    fn pdc_stdlib_vector4d() { run_stdlib_tests(STDLIB_VECTOR4D, "vector4d"); }

    #[test]
    fn pdc_stdlib_point2d() { run_stdlib_tests(STDLIB_POINT2D, "point2d"); }

    #[test]
    fn pdc_stdlib_point3d() { run_stdlib_tests(STDLIB_POINT3D, "point3d"); }

    #[test]
    fn pdc_stdlib_point4d() { run_stdlib_tests(STDLIB_POINT4D, "point4d"); }

    #[test]
    fn pdc_stdlib_affine2d() { run_stdlib_tests(STDLIB_AFFINE2D, "affine2d"); }

    #[test]
    fn pdc_stdlib_affine3d() { run_stdlib_tests(STDLIB_AFFINE3D, "affine3d"); }

    #[test]
    fn compile_for_pipeline_basic() {
        let src = r#"
            builtin const width: f32
            builtin const height: f32
            var p = Path()
            move_to(p, 0.0, 0.0)
            line_to(p, 100.0, 0.0)
            line_to(p, 100.0, 100.0)
            close(p)
            fill(p, 0xFFFF0000)
        "#;
        // Should compile without error
        super::compile_for_pipeline(src, None).unwrap();
    }

    #[test]
    fn compile_for_pipeline_strips_tests() {
        let src = r#"
            builtin const width: f32
            builtin const height: f32
            var p = Path()
            move_to(p, 0.0, 0.0)
            line_to(p, 10.0, 10.0)
            close(p)
            fill(p, 0xFFFF0000)
            test "should not affect compilation" {
                assert_eq(1, 1)
            }
        "#;
        super::compile_for_pipeline(src, None).unwrap();
    }

    #[test]
    fn compile_for_pipeline_syntax_error() {
        let src = "??? invalid";
        let result = super::compile_for_pipeline(src, None);
        assert!(result.is_err());
    }

    fn run_scene_pipeline(src: &str, width: u32, height: u32) -> super::VectorScene {
        let (compiled, state_layout) = super::compile_for_pipeline(src, None).unwrap();
        let mut builtins = [width as f64, height as f64];
        let mut scene_builder = super::SceneBuilder::new();
        let mut state_block = state_layout.alloc();
        let mut ctx = super::PdcContext {
            builtins: builtins.as_mut_ptr(),
            scene: &mut scene_builder as *mut _,
            state: state_block.as_mut_ptr(),
            host: std::ptr::null_mut(),
        };
        unsafe { (compiled.fn_ptr)(&mut ctx); }
        super::extract_scene(&scene_builder, 0.5, 16, width, height)
    }

    #[test]
    fn compile_for_pipeline_matches_compile_and_run() {
        // Use raw Path API (no imports) to test pipeline equivalence.
        // PDC executes top-level statements, not a main() function.
        let src = r#"
            builtin const width: f32
            builtin const height: f32
            var p = Path()
            move_to(p, 10.0, 10.0)
            line_to(p, 50.0, 10.0)
            line_to(p, 50.0, 50.0)
            line_to(p, 10.0, 50.0)
            close(p)
            fill(p, 0xFFFF0000)
        "#;
        let width = 100u32;
        let height = 100u32;

        let scene_monolithic = super::compile_and_run(src, width, height, 0.5, 16, None).unwrap();
        let scene_pipeline = run_scene_pipeline(src, width, height);

        // Both should produce non-empty scenes
        assert!(!scene_monolithic.segments.is_empty(), "monolithic should have segments");
        assert_eq!(scene_monolithic.segments.len(), scene_pipeline.segments.len());
        assert_eq!(scene_monolithic.path_colors, scene_pipeline.path_colors);
        assert_eq!(scene_monolithic.tile_offsets, scene_pipeline.tile_offsets);
        assert_eq!(scene_monolithic.tile_counts, scene_pipeline.tile_counts);
        assert_eq!(scene_monolithic.tile_indices, scene_pipeline.tile_indices);
    }

    #[test]
    fn extract_scene_basic() {
        // PDC code runs at top level, not inside main()
        let src = r#"
            builtin const width: f32
            builtin const height: f32
            var p = Path()
            move_to(p, 10.0, 10.0)
            line_to(p, 50.0, 10.0)
            line_to(p, 50.0, 50.0)
            line_to(p, 10.0, 50.0)
            close(p)
            fill(p, 0xFFFF0000)
        "#;
        let scene = run_scene_pipeline(src, 100, 100);
        assert!(!scene.segments.is_empty(), "scene should have segments");
        assert_eq!(scene.path_colors.len(), 1);
        assert_eq!(scene.path_colors[0], 0xFFFF0000);
        let tiles_x = (100 + 15) / 16;
        let tiles_y = (100 + 15) / 16;
        assert_eq!(scene.tile_offsets.len(), (tiles_x * tiles_y) as usize);
        assert_eq!(scene.tile_counts.len(), (tiles_x * tiles_y) as usize);
        // Non-empty tiles should exist in the 10-50 pixel range
        let nonempty = scene.tile_counts.iter().filter(|&&c| c > 0).count();
        assert!(nonempty > 0, "should have non-empty tiles");
    }

    #[test]
    fn extract_scene_empty() {
        let src = r#"
            builtin const width: f32
            builtin const height: f32
        "#;
        let scene = run_scene_pipeline(src, 100, 100);
        assert!(scene.segments.is_empty());
        assert!(scene.path_colors.is_empty());
        let tiles_x = (100 + 15) / 16;
        let tiles_y = (100 + 15) / 16;
        assert_eq!(scene.tile_offsets.len(), (tiles_x * tiles_y) as usize);
    }

    #[test]
    fn extract_scene_multiple_paths() {
        let src = r#"
            builtin const width: f32
            builtin const height: f32
            var p1 = Path()
            move_to(p1, 0.0, 0.0)
            line_to(p1, 30.0, 0.0)
            line_to(p1, 30.0, 30.0)
            line_to(p1, 0.0, 30.0)
            close(p1)
            fill(p1, 0xFFFF0000)

            var p2 = Path()
            move_to(p2, 50.0, 50.0)
            line_to(p2, 80.0, 50.0)
            line_to(p2, 80.0, 80.0)
            line_to(p2, 50.0, 80.0)
            close(p2)
            fill(p2, 0xFF00FF00)
        "#;
        let scene = run_scene_pipeline(src, 100, 100);
        assert_eq!(scene.path_colors.len(), 2);
        assert_eq!(scene.path_colors[0], 0xFFFF0000);
        assert_eq!(scene.path_colors[1], 0xFF00FF00);
    }

    #[test]
    fn extract_scene_tiles_cover_dimensions() {
        // Odd dimensions that don't divide evenly by tile_size
        let src = r#"
            builtin const width: f32
            builtin const height: f32
        "#;
        let scene = run_scene_pipeline(src, 100, 70);
        let tiles_x = (100 + 15) / 16; // 7
        let tiles_y = (70 + 15) / 16;  // 5
        assert_eq!(scene.tile_offsets.len(), (tiles_x * tiles_y) as usize);
        assert_eq!(scene.tile_counts.len(), (tiles_x * tiles_y) as usize);
    }

    // ---- Module-level mutable state tests ----

    /// Helper: compile with state, run pdc_main to init state, return (compiled, ctx_pieces).
    fn compile_with_state(source: &str) -> (
        codegen::CompiledProgram,
        codegen::StateLayout,
        Box<[u8]>,
    ) {
        let full = format!("{}{}", pdc_header(), source);
        let (compiled, state_layout) = compile_only(&full, None).expect("compile_only failed");
        let state_block = state_layout.alloc();
        (compiled, state_layout, state_block)
    }

    #[test]
    fn state_var_persists_across_calls() {
        let (compiled, _layout, mut state_block) = compile_with_state(r#"
            var x: f64 = 0.0
            fn increment() { x = x + 1.0 }
            fn get_x() -> f64 { return x }
        "#);
        let mut builtins = [200.0f64, 200.0f64];
        let mut scene = SceneBuilder::new();
        let mut ctx = PdcContext {
            builtins: builtins.as_mut_ptr(),
            scene: &mut scene as *mut _,
            state: state_block.as_mut_ptr(),
            host: std::ptr::null_mut(),
        };

        // Run pdc_main to initialize state (var x = 0.0)
        unsafe { (compiled.fn_ptr)(&mut ctx); }

        // Call increment 3 times
        for _ in 0..3 {
            unsafe { compiled.call_fn("increment", &mut ctx, &[]).unwrap(); }
        }

        // Verify state persisted
        let result = unsafe { compiled.call_fn("get_x", &mut ctx, &[]).unwrap() };
        assert_eq!(result, PdcValue::F64(3.0));
    }

    #[test]
    fn state_var_initialized_by_main() {
        let (compiled, _layout, mut state_block) = compile_with_state(r#"
            var counter: f64 = 42.0
            fn get_counter() -> f64 { return counter }
        "#);
        let mut builtins = [200.0f64, 200.0f64];
        let mut scene = SceneBuilder::new();
        let mut ctx = PdcContext {
            builtins: builtins.as_mut_ptr(),
            scene: &mut scene as *mut _,
            state: state_block.as_mut_ptr(),
            host: std::ptr::null_mut(),
        };

        // Run pdc_main to initialize state
        unsafe { (compiled.fn_ptr)(&mut ctx); }

        let result = unsafe { compiled.call_fn("get_counter", &mut ctx, &[]).unwrap() };
        assert_eq!(result, PdcValue::F64(42.0));
    }

    #[test]
    fn state_var_multiple_types() {
        let (compiled, _layout, mut state_block) = compile_with_state(r#"
            var a: f64 = 1.0
            var b: i32 = 10
            var c: bool = false
            fn get_a() -> f64 { return a }
            fn get_b() -> i32 { return b }
            fn get_c() -> bool { return c }
            fn set_c(val: bool) { c = val }
        "#);
        let mut builtins = [200.0f64, 200.0f64];
        let mut scene = SceneBuilder::new();
        let mut ctx = PdcContext {
            builtins: builtins.as_mut_ptr(),
            scene: &mut scene as *mut _,
            state: state_block.as_mut_ptr(),
            host: std::ptr::null_mut(),
        };

        unsafe { (compiled.fn_ptr)(&mut ctx); }

        assert_eq!(unsafe { compiled.call_fn("get_a", &mut ctx, &[]).unwrap() }, PdcValue::F64(1.0));
        assert_eq!(unsafe { compiled.call_fn("get_b", &mut ctx, &[]).unwrap() }, PdcValue::I32(10));
        assert_eq!(unsafe { compiled.call_fn("get_c", &mut ctx, &[]).unwrap() }, PdcValue::Bool(false));

        unsafe { compiled.call_fn("set_c", &mut ctx, &[PdcValue::Bool(true)]).unwrap(); }
        assert_eq!(unsafe { compiled.call_fn("get_c", &mut ctx, &[]).unwrap() }, PdcValue::Bool(true));
    }

    #[test]
    fn state_var_init_frame_lifecycle() {
        let (compiled, _layout, mut state_block) = compile_with_state(r#"
            var counter: f64 = 0.0
            fn init() { counter = 10.0 }
            fn frame() { counter = counter + 1.0 }
            fn get_counter() -> f64 { return counter }
        "#);
        let mut builtins = [200.0f64, 200.0f64];
        let mut scene = SceneBuilder::new();
        let mut ctx = PdcContext {
            builtins: builtins.as_mut_ptr(),
            scene: &mut scene as *mut _,
            state: state_block.as_mut_ptr(),
            host: std::ptr::null_mut(),
        };

        // 1. pdc_main initializes state (counter = 0.0)
        unsafe { (compiled.fn_ptr)(&mut ctx); }

        // 2. init() sets counter = 10.0
        unsafe { compiled.call_fn("init", &mut ctx, &[]).unwrap(); }

        // 3. frame() x5 increments counter to 15.0
        for _ in 0..5 {
            unsafe { compiled.call_fn("frame", &mut ctx, &[]).unwrap(); }
        }

        let result = unsafe { compiled.call_fn("get_counter", &mut ctx, &[]).unwrap() };
        assert_eq!(result, PdcValue::F64(15.0));
    }

    #[test]
    fn state_layout_empty_for_no_vars() {
        let (_compiled, layout, _state_block) = compile_with_state(r#"
            fn add(a: f64, b: f64) -> f64 { return a + b }
        "#);
        assert_eq!(layout.total_size, 0);
        assert!(layout.vars.is_empty());
    }

    #[test]
    fn state_layout_alignment() {
        let (_compiled, layout, _state_block) = compile_with_state(r#"
            var a: bool = false
            var b: f64 = 0.0
            var c: i32 = 0
        "#);
        // bool is 1 byte, f64 needs 8-byte alignment, i32 needs 4-byte alignment
        let a = layout.vars.get("a").unwrap();
        let b = layout.vars.get("b").unwrap();
        let c = layout.vars.get("c").unwrap();
        assert_eq!(a.offset, 0);
        assert_eq!(b.offset % 8, 0); // f64 must be 8-byte aligned
        assert_eq!(c.offset % 4, 0); // i32 must be 4-byte aligned
    }

    #[test]
    fn state_var_coexists_with_local_vars() {
        let (compiled, _layout, mut state_block) = compile_with_state(r#"
            var global_x: f64 = 0.0
            fn accumulate(n: f64) -> f64 {
                var local_sum: f64 = 0.0
                var i: f64 = 0.0
                while i < n {
                    local_sum = local_sum + 1.0
                    i = i + 1.0
                }
                global_x = global_x + local_sum
                return global_x
            }
        "#);
        let mut builtins = [200.0f64, 200.0f64];
        let mut scene = SceneBuilder::new();
        let mut ctx = PdcContext {
            builtins: builtins.as_mut_ptr(),
            scene: &mut scene as *mut _,
            state: state_block.as_mut_ptr(),
            host: std::ptr::null_mut(),
        };

        unsafe { (compiled.fn_ptr)(&mut ctx); }

        let r1 = unsafe { compiled.call_fn("accumulate", &mut ctx, &[PdcValue::F64(3.0)]).unwrap() };
        assert_eq!(r1, PdcValue::F64(3.0));

        let r2 = unsafe { compiled.call_fn("accumulate", &mut ctx, &[PdcValue::F64(5.0)]).unwrap() };
        assert_eq!(r2, PdcValue::F64(8.0));
    }

    // ---- Lifecycle entry point tests (Phase 2) ----

    #[test]
    fn has_init_and_frame() {
        let (compiled, _, _) = compile_with_state(r#"
            fn init() {}
            fn frame() {}
        "#);
        assert!(compiled.has_init());
        assert!(compiled.has_frame());
    }

    #[test]
    fn has_init_and_frame_absent() {
        let (compiled, _, _) = compile_with_state(r#"
            fn foo() {}
        "#);
        assert!(!compiled.has_init());
        assert!(!compiled.has_frame());
    }

    #[test]
    fn call_init_and_frame_lifecycle() {
        let (compiled, _layout, mut state_block) = compile_with_state(r#"
            var counter: f64 = 0.0
            fn init() { counter = 10.0 }
            fn frame() { counter = counter + 1.0 }
            fn get_counter() -> f64 { return counter }
        "#);
        let mut builtins = [200.0f64, 200.0f64];
        let mut scene = SceneBuilder::new();
        let mut ctx = PdcContext {
            builtins: builtins.as_mut_ptr(),
            scene: &mut scene as *mut _,
            state: state_block.as_mut_ptr(),
            host: std::ptr::null_mut(),
        };

        // 1. pdc_main initializes state
        unsafe { (compiled.fn_ptr)(&mut ctx); }

        // 2. call_init
        unsafe { compiled.call_init(&mut ctx).unwrap(); }

        // 3. call_frame x3
        for _ in 0..3 {
            unsafe { compiled.call_frame(&mut ctx).unwrap(); }
        }

        let result = unsafe { compiled.call_fn("get_counter", &mut ctx, &[]).unwrap() };
        assert_eq!(result, PdcValue::F64(13.0));
    }

    #[test]
    fn call_init_noop_when_absent() {
        let (compiled, _layout, mut state_block) = compile_with_state(r#"
            var x: f64 = 5.0
            fn get_x() -> f64 { return x }
        "#);
        let mut builtins = [200.0f64, 200.0f64];
        let mut scene = SceneBuilder::new();
        let mut ctx = PdcContext {
            builtins: builtins.as_mut_ptr(),
            scene: &mut scene as *mut _,
            state: state_block.as_mut_ptr(),
            host: std::ptr::null_mut(),
        };

        unsafe { (compiled.fn_ptr)(&mut ctx); }
        // call_init is a no-op since init() doesn't exist
        unsafe { compiled.call_init(&mut ctx).unwrap(); }

        let result = unsafe { compiled.call_fn("get_x", &mut ctx, &[]).unwrap() };
        assert_eq!(result, PdcValue::F64(5.0));
    }

    // ---- Mutable builtins tests (Phase 3) ----

    #[test]
    fn mutable_builtin_read_and_write() {
        // Use the expanded pipeline builtins layout
        let source = r#"
            builtin const width: f32
            builtin const height: f32
            builtin const time: f64
            builtin const mouse_x: f64
            builtin const mouse_y: f64
            builtin var center_x: f64
            builtin var center_y: f64
            builtin var zoom: f64
            builtin var paused: bool

            fn get_time() -> f64 { return time }
            fn get_zoom() -> f64 { return zoom }
            fn set_zoom(z: f64) { zoom = z }
            fn toggle_pause() { paused = !paused }
            fn get_paused() -> bool { return paused }
            fn pan_left() { center_x = center_x - 0.02 / zoom }
            fn get_center_x() -> f64 { return center_x }
        "#;
        let (compiled, state_layout) = compile_only_with_builtins(source, None, codegen::PIPELINE_BUILTINS).expect("compile failed");

        // Pipeline builtins: width=200, height=200, time=1.5, mouse_x=100, mouse_y=50,
        //                     center_x=0.0, center_y=0.0, zoom=2.0, paused=0.0, frame=0, ...
        let mut builtins = [
            200.0f64, 200.0, 1.5, 100.0, 50.0,
            0.0, 0.0, 2.0, 0.0, 0.0,
            0.0, 0.0,
        ];
        let mut scene = SceneBuilder::new();
        let mut state_block = state_layout.alloc();
        let mut ctx = PdcContext {
            builtins: builtins.as_mut_ptr(),
            scene: &mut scene as *mut _,
            state: state_block.as_mut_ptr(),
            host: std::ptr::null_mut(),
        };

        // Run pdc_main (processes builtin declarations)
        unsafe { (compiled.fn_ptr)(&mut ctx); }

        // Read immutable builtin
        let time = unsafe { compiled.call_fn("get_time", &mut ctx, &[]).unwrap() };
        assert_eq!(time, PdcValue::F64(1.5));

        // Read mutable builtin
        let zoom = unsafe { compiled.call_fn("get_zoom", &mut ctx, &[]).unwrap() };
        assert_eq!(zoom, PdcValue::F64(2.0));

        // Write mutable builtin from PDC
        unsafe { compiled.call_fn("set_zoom", &mut ctx, &[PdcValue::F64(4.0)]).unwrap(); }

        // Read back from PDC — should see 4.0
        let zoom = unsafe { compiled.call_fn("get_zoom", &mut ctx, &[]).unwrap() };
        assert_eq!(zoom, PdcValue::F64(4.0));

        // Read back from Rust — builtins array should be updated
        assert_eq!(builtins[7], 4.0); // zoom is index 7

        // Toggle paused (false → true)
        unsafe { compiled.call_fn("toggle_pause", &mut ctx, &[]).unwrap(); }
        let paused = unsafe { compiled.call_fn("get_paused", &mut ctx, &[]).unwrap() };
        assert_eq!(paused, PdcValue::Bool(true));
        assert_eq!(builtins[8], 1.0); // paused is index 8, true = 1.0

        // Pan left: center_x = 0.0 - 0.02/4.0 = -0.005
        unsafe { compiled.call_fn("pan_left", &mut ctx, &[]).unwrap(); }
        let cx = unsafe { compiled.call_fn("get_center_x", &mut ctx, &[]).unwrap() };
        assert_eq!(cx, PdcValue::F64(-0.005));
        assert_eq!(builtins[5], -0.005); // center_x is index 5
    }

    #[test]
    fn host_can_update_mutable_builtins_between_frames() {
        let source = r#"
            builtin const width: f32
            builtin const height: f32
            builtin const time: f64
            builtin const mouse_x: f64
            builtin const mouse_y: f64
            builtin var center_x: f64
            builtin var center_y: f64
            builtin var zoom: f64
            builtin var paused: bool
            builtin var frame: u64
            builtin const mouse_down: bool
            builtin const sample_index: u32

            fn get_zoom() -> f64 { return zoom }
        "#;
        let (compiled, state_layout) = compile_only_with_builtins(source, None, codegen::PIPELINE_BUILTINS).expect("compile failed");
        let mut builtins = [200.0, 200.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let mut scene = SceneBuilder::new();
        let mut state_block = state_layout.alloc();
        let mut ctx = PdcContext {
            builtins: builtins.as_mut_ptr(),
            scene: &mut scene as *mut _,
            state: state_block.as_mut_ptr(),
            host: std::ptr::null_mut(),
        };

        unsafe { (compiled.fn_ptr)(&mut ctx); }

        // Host reads zoom = 1.0
        let z = unsafe { compiled.call_fn("get_zoom", &mut ctx, &[]).unwrap() };
        assert_eq!(z, PdcValue::F64(1.0));

        // Host updates zoom between frames
        builtins[7] = 5.0;

        // PDC reads the updated value
        let z = unsafe { compiled.call_fn("get_zoom", &mut ctx, &[]).unwrap() };
        assert_eq!(z, PdcValue::F64(5.0));
    }

    // ---- Pipeline host function tests (Phase 4) ----

    #[test]
    fn pipeline_host_basic_operations() {
        let source = r#"
            builtin const width: f32
            builtin const height: f32

            var grid: i32 = 0
            var grid_next: i32 = 0
            var sim_kernel: i32 = 0

            fn init() {
                grid = create_buffer("gpu_f32", 0.0)
                grid_next = create_buffer("gpu_f32", 0.0)
                sim_kernel = load_kernel("step", "step.wgsl", 1)
            }

            fn frame() {
                bind_buffer("input", grid, 0)
                bind_buffer("output", grid_next, 1)
                run_kernel(sim_kernel)
                swap_buffers(grid, grid_next)
                display()
            }
        "#;
        let (compiled, state_layout) =
            compile_only_with_builtins(source, None, codegen::PIPELINE_BUILTINS)
                .expect("compile failed");
        let mut builtins = [200.0f64; 12];
        let mut scene = SceneBuilder::new();
        let mut state_block = state_layout.alloc();
        // Use a shared log via Rc<RefCell> so we can read it after the mock is borrowed
        use std::cell::RefCell;
        use std::rc::Rc;
        let log: Rc<RefCell<Vec<String>>> = Rc::new(RefCell::new(Vec::new()));

        struct LoggingHost { log: Rc<RefCell<Vec<String>>>, next_handle: i32 }
        impl runtime::PipelineHost for LoggingHost {
            fn create_buffer(&mut self, t: &str, v: f64) -> i32 {
                let h = self.next_handle; self.next_handle += 1;
                self.log.borrow_mut().push(format!("create_buffer({t}, {v})")); h
            }
            fn swap_buffers(&mut self, a: i32, b: i32) { self.log.borrow_mut().push(format!("swap_buffers({a}, {b})")); }
            fn load_kernel(&mut self, n: &str, p: &str, k: i32) -> i32 {
                let h = self.next_handle; self.next_handle += 1;
                self.log.borrow_mut().push(format!("load_kernel({n}, {p}, {k})")); h
            }
            fn bind_buffer(&mut self, p: &str, h: i32, o: bool) { self.log.borrow_mut().push(format!("bind_buffer({p}, {h}, {o})")); }
            fn set_kernel_arg_f64(&mut self, n: &str, v: f64) { self.log.borrow_mut().push(format!("set_kernel_arg_f64({n}, {v})")); }
            fn set_kernel_arg_f32(&mut self, n: &str, v: f32) { self.log.borrow_mut().push(format!("set_kernel_arg_f32({n}, {v})")); }
            fn run_kernel(&mut self, h: i32) { self.log.borrow_mut().push(format!("run_kernel({h})")); }
            fn display(&mut self) { self.log.borrow_mut().push("display()".into()); }
            fn display_buffer(&mut self, h: i32) { self.log.borrow_mut().push(format!("display_buffer({h})")); }
            fn load_texture(&mut self, n: &str, p: &str) -> i32 {
                let h = self.next_handle; self.next_handle += 1;
                self.log.borrow_mut().push(format!("load_texture({n}, {p})")); h
            }
        }

        let mut host_box: Box<dyn runtime::PipelineHost> = Box::new(LoggingHost {
            log: log.clone(), next_handle: 1,
        });
        let host_ptr: *mut Box<dyn runtime::PipelineHost> = &mut host_box;
        let mut ctx = PdcContext {
            builtins: builtins.as_mut_ptr(),
            scene: &mut scene as *mut _,
            state: state_block.as_mut_ptr(),
            host: host_ptr as *mut u8,
        };

        // Initialize state vars
        unsafe { (compiled.fn_ptr)(&mut ctx); }

        // Run init
        unsafe { compiled.call_init(&mut ctx).unwrap(); }

        // Run one frame
        unsafe { compiled.call_frame(&mut ctx).unwrap(); }

        let log = log.borrow();
        assert_eq!(*log, vec![
            "create_buffer(gpu_f32, 0)",
            "create_buffer(gpu_f32, 0)",
            "load_kernel(step, step.wgsl, 1)",
            "bind_buffer(input, 1, false)",
            "bind_buffer(output, 2, true)",
            "run_kernel(3)",
            "swap_buffers(1, 2)",
            "display()",
        ]);
    }

    // ---- Event handler tests (Phase 5) ----

    #[test]
    fn event_handler_discovery() {
        let (compiled, _, _) = compile_with_state(r#"
            var paused: bool = false
            var zoom: f64 = 1.0

            fn on_keypress_space() { paused = !paused }
            fn on_keydown_left() { zoom = zoom - 0.1 }
            fn on_keydown_right() { zoom = zoom + 0.1 }
            fn on_mousedown() { paused = false }
            fn on_click() { zoom = 1.0 }
            fn frame() {}
            fn helper() {}
        "#);

        let mut handlers = compiled.event_handlers();
        handlers.sort();
        assert_eq!(handlers, vec![
            "on_click",
            "on_keydown_left",
            "on_keydown_right",
            "on_keypress_space",
            "on_mousedown",
        ]);

        let mut kb = compiled.keyboard_handlers();
        kb.sort();
        assert_eq!(kb, vec![
            ("on_keydown", "left"),
            ("on_keydown", "right"),
            ("on_keypress", "space"),
        ]);
    }

    #[test]
    fn event_handler_modifies_state() {
        let (compiled, _layout, mut state_block) = compile_with_state(r#"
            var paused: bool = false
            var zoom: f64 = 1.0

            fn on_keypress_space() { paused = !paused }
            fn on_keydown_plus() { zoom = zoom * 1.1 }
            fn on_mousedown() { zoom = 1.0 }
            fn get_paused() -> bool { return paused }
            fn get_zoom() -> f64 { return zoom }
        "#);
        let mut builtins = [200.0f64, 200.0f64];
        let mut scene = SceneBuilder::new();
        let mut ctx = PdcContext {
            builtins: builtins.as_mut_ptr(),
            scene: &mut scene as *mut _,
            state: state_block.as_mut_ptr(),
            host: std::ptr::null_mut(),
        };

        // Init state
        unsafe { (compiled.fn_ptr)(&mut ctx); }

        // Toggle pause
        let handled = unsafe { compiled.call_event("on_keypress_space", &mut ctx).unwrap() };
        assert!(handled);
        assert_eq!(unsafe { compiled.call_fn("get_paused", &mut ctx, &[]).unwrap() }, PdcValue::Bool(true));

        // Toggle again
        unsafe { compiled.call_event("on_keypress_space", &mut ctx).unwrap(); }
        assert_eq!(unsafe { compiled.call_fn("get_paused", &mut ctx, &[]).unwrap() }, PdcValue::Bool(false));

        // Zoom in
        unsafe { compiled.call_event("on_keydown_plus", &mut ctx).unwrap(); }
        let zoom = unsafe { compiled.call_fn("get_zoom", &mut ctx, &[]).unwrap() };
        assert_eq!(zoom, PdcValue::F64(1.1));

        // Mouse resets zoom
        unsafe { compiled.call_event("on_mousedown", &mut ctx).unwrap(); }
        let zoom = unsafe { compiled.call_fn("get_zoom", &mut ctx, &[]).unwrap() };
        assert_eq!(zoom, PdcValue::F64(1.0));

        // Non-existent handler returns false
        let handled = unsafe { compiled.call_event("on_keypress_q", &mut ctx).unwrap() };
        assert!(!handled);
    }
}
