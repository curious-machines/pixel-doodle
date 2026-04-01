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

/// Compile a PDC source without executing. Returns the compiled program
/// for inspection and individual function calls from test code.
///
/// Unlike `compile_and_run`, this skips dead code elimination so that all
/// user-defined functions remain available for testing via `call_fn`.
pub fn compile_only(
    source: &str,
    source_path: Option<&FsPath>,
) -> Result<codegen::CompiledProgram, PdcError> {
    let base_dir = source_path.and_then(|p| p.parent());
    let program = load_and_parse(source, base_dir)?;
    // No DCE — we want all functions available for testing
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

/// Compile and evaluate a PDC expression, returning its value.
///
/// Wraps the expression in a synthetic function, compiles, and calls it.
/// The `expected_type` must be a valid PDC type name (e.g., "f64", "i32", "bool").
pub fn eval_expr(expr: &str, expected_type: &str) -> Result<codegen::PdcValue, PdcError> {
    let source = format!(
        "builtin const width: f32\nbuiltin const height: f32\n\
         fn __eval__() -> {expected_type} {{ return {expr} }}\n"
    );
    let compiled = compile_only(&source, None)?;
    let builtins = [200.0f64, 200.0f64];
    let mut scene_builder = SceneBuilder::new();
    let mut ctx = PdcContext {
        builtins: builtins.as_ptr(),
        scene: &mut scene_builder as *mut _,
    };
    unsafe { compiled.call_fn("__eval__", &mut ctx, &[]) }
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
    let compiled = codegen::compile(&program, &checker.types, &builtins_layout, &checker.user_fns, &checker.structs, &checker.enums, &checker.fn_aliases, &checker.op_overloads)?;

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
        let compiled = compile_only(&full, None).expect("compile_only failed");
        let builtins = [200.0f64, 200.0f64];
        let mut scene = SceneBuilder::new();
        let mut ctx = PdcContext {
            builtins: builtins.as_ptr(),
            scene: &mut scene as *mut _,
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
        let compiled = compile_only(&full, None).unwrap();
        let builtins = [200.0f64, 200.0f64];
        let mut scene = SceneBuilder::new();
        let mut ctx = PdcContext {
            builtins: builtins.as_ptr(),
            scene: &mut scene as *mut _,
        };
        let err = unsafe { compiled.call_fn("nonexistent", &mut ctx, &[]) };
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("no compiled function"));
    }

    #[test]
    fn call_fn_wrong_arg_count() {
        let full = format!("{}fn add(a: f64, b: f64) -> f64 {{ return a + b }}", pdc_header());
        let compiled = compile_only(&full, None).unwrap();
        let builtins = [200.0f64, 200.0f64];
        let mut scene = SceneBuilder::new();
        let mut ctx = PdcContext {
            builtins: builtins.as_ptr(),
            scene: &mut scene as *mut _,
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
}
