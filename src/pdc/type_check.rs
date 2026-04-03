use std::collections::{HashMap, HashSet};

use super::ast::*;
use super::error::PdcError;
use super::span::Spanned;

/// A built-in function signature.
#[allow(dead_code)]
struct BuiltinFn {
    params: Vec<PdcType>,
    ret: PdcType,
    takes_ctx: bool,
}

/// A user-defined function signature (discovered during type checking).
#[derive(Clone, Debug)]
pub struct UserFnSig {
    pub params: Vec<PdcType>,
    pub ret: PdcType,
    /// Number of required (non-defaulted) parameters.
    pub required: usize,
    /// Default value expressions for trailing optional parameters.
    /// Length = params.len() - required. Index 0 = first optional param.
    pub defaults: Vec<Spanned<Expr>>,
}

/// Overload set: one or more signatures for the same function name.
#[derive(Clone, Debug)]
pub struct OverloadSet {
    pub sigs: Vec<UserFnSig>,
}

/// Struct definition info for type checking.
#[derive(Clone)]
pub struct StructInfo {
    pub fields: Vec<(String, PdcType)>,
}

/// Enum variant info for type checking.
#[derive(Clone)]
pub struct EnumVariantInfo {
    pub name: String,
    pub field_names: Vec<String>,
    pub field_types: Vec<PdcType>,
}

/// Enum definition info for type checking.
#[derive(Clone)]
pub struct EnumInfo {
    pub variants: Vec<EnumVariantInfo>,
}

/// Tracks which names a module exports (pub items).
#[derive(Clone, Default)]
pub struct ModuleExports {
    pub names: HashSet<String>,
}

pub struct TypeChecker {
    scopes: Vec<HashMap<String, PdcType>>,
    /// Variables declared as const — assignments to these are rejected.
    const_vars: HashSet<String>,
    builtins: HashMap<String, BuiltinFn>,
    /// User-defined function signatures (supports overloading).
    /// Module functions are stored with qualified keys: "math::lerp".
    pub user_fns: HashMap<String, OverloadSet>,
    /// User-defined struct definitions. Module structs: "module::Name".
    pub structs: HashMap<String, StructInfo>,
    /// User-defined enum definitions. Module enums: "module::Name".
    pub enums: HashMap<String, EnumInfo>,
    /// Type aliases: `type Name = ExistingType`
    pub type_aliases: HashMap<String, PdcType>,
    /// Type assigned to each AST node, indexed by node ID.
    pub types: Vec<PdcType>,
    /// Per-module export info.
    module_exports: HashMap<String, ModuleExports>,
    /// Alias map: unqualified name → qualified "module::name".
    /// Created by import statements.
    pub fn_aliases: HashMap<String, String>,
    /// Operator overloads: key = mangled op name (e.g. "__op_add__"),
    /// value = overload set for different type combinations.
    pub op_overloads: HashMap<String, OverloadSet>,
}

impl TypeChecker {
    pub fn new() -> Self {
        let mut tc = Self {
            scopes: vec![HashMap::new()],
            const_vars: HashSet::new(),
            builtins: HashMap::new(),
            user_fns: HashMap::new(),
            structs: HashMap::new(),
            enums: HashMap::new(),
            type_aliases: HashMap::new(),
            types: Vec::new(),
            module_exports: HashMap::new(),
            fn_aliases: HashMap::new(),
            op_overloads: HashMap::new(),
        };
        tc.register_builtins();
        tc
    }

    fn register_builtins(&mut self) {
        self.builtins.insert("Path".into(), BuiltinFn {
            params: vec![],
            ret: PdcType::PathHandle,
            takes_ctx: true,
        });

        for name in &["move_to", "line_to"] {
            self.builtins.insert(name.to_string(), BuiltinFn {
                params: vec![PdcType::PathHandle, PdcType::F64, PdcType::F64],
                ret: PdcType::Void,
                takes_ctx: true,
            });
        }
        self.builtins.insert("quad_to".into(), BuiltinFn {
            params: vec![PdcType::PathHandle, PdcType::F64, PdcType::F64, PdcType::F64, PdcType::F64],
            ret: PdcType::Void,
            takes_ctx: true,
        });
        self.builtins.insert("cubic_to".into(), BuiltinFn {
            params: vec![PdcType::PathHandle, PdcType::F64, PdcType::F64, PdcType::F64, PdcType::F64, PdcType::F64, PdcType::F64],
            ret: PdcType::Void,
            takes_ctx: true,
        });
        self.builtins.insert("close".into(), BuiltinFn {
            params: vec![PdcType::PathHandle],
            ret: PdcType::Void,
            takes_ctx: true,
        });

        self.builtins.insert("fill".into(), BuiltinFn {
            params: vec![PdcType::PathHandle, PdcType::U32],
            ret: PdcType::Void,
            takes_ctx: true,
        });
        self.builtins.insert("stroke".into(), BuiltinFn {
            params: vec![PdcType::PathHandle, PdcType::F32, PdcType::U32],
            ret: PdcType::Void,
            takes_ctx: true,
        });

        // Array methods also registered as builtins so UFCS-as-function works:
        // push(arr, val) as well as arr.push(val)
        // Note: MethodCall path has type-aware handling for generic arrays.
        let arr_ty = PdcType::Array(Box::new(PdcType::F64));
        self.builtins.insert("push".into(), BuiltinFn {
            params: vec![arr_ty.clone(), PdcType::F64],
            ret: PdcType::Void,
            takes_ctx: true,
        });
        self.builtins.insert("len".into(), BuiltinFn {
            params: vec![arr_ty.clone()],
            ret: PdcType::I32,
            takes_ctx: true,
        });
        self.builtins.insert("get".into(), BuiltinFn {
            params: vec![arr_ty.clone(), PdcType::I32],
            ret: PdcType::F64,
            takes_ctx: true,
        });
        self.builtins.insert("set".into(), BuiltinFn {
            params: vec![arr_ty.clone(), PdcType::I32, PdcType::F64],
            ret: PdcType::Void,
            takes_ctx: true,
        });

        // Styled draw overloads
        self.builtins.insert("fill_styled".into(), BuiltinFn {
            params: vec![PdcType::PathHandle, PdcType::U32, PdcType::I32],
            ret: PdcType::Void,
            takes_ctx: true,
        });
        self.builtins.insert("stroke_styled".into(), BuiltinFn {
            params: vec![PdcType::PathHandle, PdcType::F32, PdcType::U32, PdcType::I32, PdcType::I32],
            ret: PdcType::Void,
            takes_ctx: true,
        });

        // Built-in enums: Bind, FillRule, LineCap, LineJoin
        // Buffer and Kernel are type namespaces with factory constructors
        self.define_var("Buffer", PdcType::Module("Buffer".into()));
        self.define_var("Kernel", PdcType::Module("Kernel".into()));

        self.enums.insert("Bind".into(), EnumInfo {
            variants: vec![
                EnumVariantInfo { name: "In".into(), field_names: vec!["buffer".into()], field_types: vec![PdcType::BufferHandle] },
                EnumVariantInfo { name: "Out".into(), field_names: vec!["buffer".into()], field_types: vec![PdcType::BufferHandle] },
            ],
        });
        self.define_var("Bind", PdcType::Enum("Bind".into()));

        self.enums.insert("FillRule".into(), EnumInfo {
            variants: vec![
                EnumVariantInfo { name: "EvenOdd".into(), field_names: vec![], field_types: vec![] },
                EnumVariantInfo { name: "NonZero".into(), field_names: vec![], field_types: vec![] },
            ],
        });
        self.define_var("FillRule", PdcType::Enum("FillRule".into()));

        self.enums.insert("LineCap".into(), EnumInfo {
            variants: vec![
                EnumVariantInfo { name: "Butt".into(), field_names: vec![], field_types: vec![] },
                EnumVariantInfo { name: "Round".into(), field_names: vec![], field_types: vec![] },
                EnumVariantInfo { name: "Square".into(), field_names: vec![], field_types: vec![] },
            ],
        });
        self.define_var("LineCap", PdcType::Enum("LineCap".into()));

        self.enums.insert("LineJoin".into(), EnumInfo {
            variants: vec![
                EnumVariantInfo { name: "Miter".into(), field_names: vec![], field_types: vec![] },
                EnumVariantInfo { name: "Round".into(), field_names: vec![], field_types: vec![] },
                EnumVariantInfo { name: "Bevel".into(), field_names: vec![], field_types: vec![] },
            ],
        });
        self.define_var("LineJoin", PdcType::Enum("LineJoin".into()));

        for name in &["sin", "cos", "tan", "asin", "acos", "atan", "sqrt", "abs", "floor", "ceil", "round", "exp", "ln", "log2", "log10", "fract", "exp2"] {
            self.builtins.insert(name.to_string(), BuiltinFn {
                params: vec![PdcType::F64],
                ret: PdcType::F64,
                takes_ctx: false,
            });
        }
        for name in &["min", "max", "atan2", "fmod", "pow"] {
            self.builtins.insert(name.to_string(), BuiltinFn {
                params: vec![PdcType::F64, PdcType::F64],
                ret: PdcType::F64,
                takes_ctx: false,
            });
        }

        // Pipeline host functions
        // Buffer methods
        self.builtins.insert("display_buffer".into(), BuiltinFn {
            params: vec![PdcType::BufferHandle],
            ret: PdcType::Void,
            takes_ctx: true,
        });
        self.builtins.insert("swap".into(), BuiltinFn {
            params: vec![PdcType::BufferHandle, PdcType::BufferHandle],
            ret: PdcType::Void,
            takes_ctx: true,
        });
        // Kernel methods
        self.builtins.insert("run".into(), BuiltinFn {
            params: vec![PdcType::KernelHandle],
            ret: PdcType::Void,
            takes_ctx: true,
        });
        // display()
        self.builtins.insert("display".into(), BuiltinFn {
            params: vec![],
            ret: PdcType::Void,
            takes_ctx: true,
        });
        // load_texture(name: string, path: string) -> i32
        self.builtins.insert("load_texture".into(), BuiltinFn {
            params: vec![PdcType::Str, PdcType::Str],
            ret: PdcType::I32,
            takes_ctx: true,
        });

        // Scene kernels
        self.builtins.insert("load_scene".into(), BuiltinFn {
            params: vec![PdcType::Str, PdcType::Str],
            ret: PdcType::I32,
            takes_ctx: true,
        });
        self.builtins.insert("run_scene".into(), BuiltinFn {
            params: vec![PdcType::I32],
            ret: PdcType::Void,
            takes_ctx: true,
        });
        self.builtins.insert("scene_tiles_x".into(), BuiltinFn {
            params: vec![PdcType::I32],
            ret: PdcType::F64,
            takes_ctx: true,
        });
        self.builtins.insert("scene_num_paths".into(), BuiltinFn {
            params: vec![PdcType::I32],
            ret: PdcType::F64,
            takes_ctx: true,
        });
        self.builtins.insert("scene_buffer".into(), BuiltinFn {
            params: vec![PdcType::I32, PdcType::Str],
            ret: PdcType::BufferHandle,
            takes_ctx: true,
        });

        // Frame control
        self.builtins.insert("request_redraw".into(), BuiltinFn {
            params: vec![],
            ret: PdcType::Void,
            takes_ctx: true,
        });

        // Progressive rendering
        self.builtins.insert("set_max_samples".into(), BuiltinFn {
            params: vec![PdcType::I32],
            ret: PdcType::Void,
            takes_ctx: true,
        });
        self.builtins.insert("is_converged".into(), BuiltinFn {
            params: vec![],
            ret: PdcType::Bool,
            takes_ctx: true,
        });
        self.builtins.insert("accumulate_sample".into(), BuiltinFn {
            params: vec![],
            ret: PdcType::Void,
            takes_ctx: true,
        });
        self.builtins.insert("display_accumulated".into(), BuiltinFn {
            params: vec![],
            ret: PdcType::Void,
            takes_ctx: true,
        });
        self.builtins.insert("reset_accumulation".into(), BuiltinFn {
            params: vec![],
            ret: PdcType::Void,
            takes_ctx: true,
        });
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn define_var(&mut self, name: &str, ty: PdcType) {
        self.scopes.last_mut().unwrap().insert(name.to_string(), ty);
    }

    fn lookup_var(&self, name: &str) -> Option<&PdcType> {
        for scope in self.scopes.iter().rev() {
            if let Some(ty) = scope.get(name) {
                return Some(ty);
            }
        }
        None
    }

    /// Resolve type aliases. If the type is a Struct(name) that matches a type alias, return the aliased type.
    fn resolve_type(&self, ty: &PdcType) -> PdcType {
        match ty {
            PdcType::Struct(name) => {
                if let Some(aliased) = self.type_aliases.get(name) {
                    self.resolve_type(aliased)
                } else {
                    ty.clone()
                }
            }
            PdcType::Array(elem) => {
                PdcType::Array(Box::new(self.resolve_type(elem)))
            }
            _ => ty.clone(),
        }
    }

    fn set_type(&mut self, id: u32, ty: PdcType) {
        let id = id as usize;
        if id >= self.types.len() {
            self.types.resize(id + 1, PdcType::Unknown);
        }
        self.types[id] = ty;
    }

    pub fn check_program(&mut self, program: &Program) -> Result<(), PdcError> {
        // ── Phase A: Register all module definitions with qualified names ──
        for module in &program.modules {
            let mod_name = &module.name;
            let mut exports = ModuleExports::default();

            // Sub-pass 1: type aliases
            for stmt in &module.stmts {
                if let Stmt::TypeAlias { vis, name, ty } = &stmt.node {
                    let qualified = format!("{mod_name}::{name}");
                    let resolved = self.resolve_type(ty);
                    self.type_aliases.insert(qualified, resolved);
                    if *vis == Visibility::Public {
                        exports.names.insert(name.clone());
                    }
                }
            }

            // Sub-pass 2: functions, structs, enums
            for stmt in &module.stmts {
                match &stmt.node {
                    Stmt::FnDef(fndef) => {
                        let qualified = format!("{mod_name}::{}", fndef.name);
                        let sig = self.sig_from_fndef(fndef);
                        if fndef.name.starts_with("__op_") {
                            self.op_overloads.entry(fndef.name.clone())
                                .or_insert_with(|| OverloadSet { sigs: Vec::new() })
                                .sigs.push(sig.clone());
                        }
                        self.user_fns.entry(qualified)
                            .or_insert_with(|| OverloadSet { sigs: Vec::new() })
                            .sigs.push(sig);
                        if fndef.vis == Visibility::Public {
                            exports.names.insert(fndef.name.clone());
                        }
                    }
                    Stmt::StructDef(sdef) => {
                        let qualified = format!("{mod_name}::{}", sdef.name);
                        self.structs.insert(qualified, StructInfo {
                            fields: sdef.fields.iter().map(|f| (f.name.clone(), self.resolve_type(&f.ty))).collect(),
                        });
                        if sdef.vis == Visibility::Public {
                            exports.names.insert(sdef.name.clone());
                        }
                    }
                    Stmt::EnumDef(edef) => {
                        let qualified = format!("{mod_name}::{}", edef.name);
                        self.enums.insert(qualified.clone(), EnumInfo {
                            variants: edef.variants.iter().map(|v| EnumVariantInfo {
                                name: v.name.clone(),
                                field_names: v.fields.iter().map(|f| f.name.clone()).collect(),
                                field_types: v.fields.iter().map(|f| f.ty.clone()).collect(),
                            }).collect(),
                        });
                        if edef.vis == Visibility::Public {
                            exports.names.insert(edef.name.clone());
                        }
                    }
                    Stmt::ConstDecl { vis, name, .. } | Stmt::VarDecl { vis, name, .. } => {
                        if *vis == Visibility::Public {
                            exports.names.insert(name.clone());
                        }
                    }
                    _ => {}
                }
            }

            self.module_exports.insert(mod_name.clone(), exports);
        }

        // ── Phase B: Type-check module bodies ──
        // Each module gets a scope mapping unqualified names → qualified entries
        for module in &program.modules {
            let mod_name = &module.name;
            self.push_scope();

            // Map all module-internal names (qualified) into this scope as unqualified
            for stmt in &module.stmts {
                match &stmt.node {
                    Stmt::FnDef(fndef) => {
                        // Make the qualified overload set accessible by unqualified name too
                        let qualified = format!("{mod_name}::{}", fndef.name);
                        if let Some(overloads) = self.user_fns.get(&qualified).cloned() {
                            self.user_fns.entry(fndef.name.clone())
                                .or_insert_with(|| OverloadSet { sigs: Vec::new() })
                                .sigs.extend(overloads.sigs);
                        }
                    }
                    Stmt::StructDef(sdef) => {
                        let qualified = format!("{mod_name}::{}", sdef.name);
                        if let Some(info) = self.structs.get(&qualified).cloned() {
                            self.structs.insert(sdef.name.clone(), info);
                        }
                    }
                    Stmt::EnumDef(edef) => {
                        let qualified = format!("{mod_name}::{}", edef.name);
                        if let Some(info) = self.enums.get(&qualified).cloned() {
                            self.enums.insert(edef.name.clone(), info);
                        }
                        self.define_var(&edef.name, PdcType::Enum(edef.name.clone()));
                    }
                    _ => {}
                }
            }

            // Type-check the module's statements
            for stmt in &module.stmts {
                self.check_stmt(stmt)?;
            }

            // Persist module const/var types with qualified names in the global scope
            for stmt in &module.stmts {
                match &stmt.node {
                    Stmt::ConstDecl { vis: _, name, .. } | Stmt::VarDecl { vis: _, name, .. } => {
                        if let Some(ty) = self.lookup_var(name).cloned() {
                            let qualified = format!("{mod_name}::{name}");
                            self.scopes[0].insert(qualified, ty);
                        }
                    }
                    _ => {}
                }
            }

            // Clean up unqualified aliases from this module
            for stmt in &module.stmts {
                match &stmt.node {
                    Stmt::FnDef(fndef) => { self.user_fns.remove(&fndef.name); }
                    Stmt::StructDef(sdef) => { self.structs.remove(&sdef.name); }
                    Stmt::EnumDef(edef) => { self.enums.remove(&edef.name); }
                    _ => {}
                }
            }

            self.pop_scope();
        }

        // ── Phase C: Process main program imports ──
        for stmt in &program.stmts {
            if let Stmt::Import { module, names } = &stmt.node {
                let exports = self.module_exports.get(module).cloned().unwrap_or_default();

                if names.is_empty() {
                    // Namespaced import: `import math`
                    self.define_var(module, PdcType::Module(module.clone()));
                } else {
                    // Direct import: `import { lerp, PI } from math`
                    for name in names {
                        if !exports.names.contains(name) {
                            return Err(PdcError::Type {
                                span: stmt.span,
                                message: format!("'{name}' is not exported from module '{module}'"),
                            });
                        }
                        let qualified = format!("{module}::{name}");

                        // Check for collision with existing names
                        if self.user_fns.contains_key(name) || self.lookup_var(name).is_some() {
                            return Err(PdcError::Type {
                                span: stmt.span,
                                message: format!("'{name}' is already defined; conflicts with import from '{module}'"),
                            });
                        }

                        // Create aliases for functions
                        if self.user_fns.contains_key(&qualified) {
                            let overloads = self.user_fns.get(&qualified).cloned().unwrap();
                            self.user_fns.insert(name.clone(), overloads);
                            self.fn_aliases.insert(name.clone(), qualified.clone());
                        }

                        // Create aliases for structs
                        if let Some(info) = self.structs.get(&qualified).cloned() {
                            self.structs.insert(name.clone(), info);
                        }

                        // Create aliases for enums
                        if let Some(info) = self.enums.get(&qualified).cloned() {
                            self.enums.insert(name.clone(), info);
                            self.define_var(name, PdcType::Enum(name.clone()));
                        }

                        // Create aliases for type aliases
                        if let Some(ty) = self.type_aliases.get(&qualified).cloned() {
                            self.type_aliases.insert(name.clone(), ty);
                        }

                        // Create aliases for const/var
                        if let Some(ty) = self.lookup_var(&qualified).cloned() {
                            self.define_var(name, ty);
                        }
                    }
                }
            }
        }

        // ── Phase D: Register and check main program ──

        // Pass 1: type aliases
        for stmt in &program.stmts {
            if let Stmt::TypeAlias { vis: _, name, ty } = &stmt.node {
                let resolved = self.resolve_type(ty);
                self.type_aliases.insert(name.clone(), resolved);
            }
        }

        // Pass 2: functions, structs, enums
        for stmt in &program.stmts {
            match &stmt.node {
                Stmt::FnDef(fndef) => {
                    let sig = self.sig_from_fndef(fndef);
                    if fndef.name.starts_with("__op_") {
                        self.op_overloads.entry(fndef.name.clone())
                            .or_insert_with(|| OverloadSet { sigs: Vec::new() })
                            .sigs.push(sig.clone());
                    }
                    self.user_fns.entry(fndef.name.clone())
                        .or_insert_with(|| OverloadSet { sigs: Vec::new() })
                        .sigs.push(sig);
                }
                Stmt::StructDef(sdef) => {
                    self.structs.insert(sdef.name.clone(), StructInfo {
                        fields: sdef.fields.iter().map(|f| (f.name.clone(), self.resolve_type(&f.ty))).collect(),
                    });
                }
                Stmt::EnumDef(edef) => {
                    self.enums.insert(edef.name.clone(), EnumInfo {
                        variants: edef.variants.iter().map(|v| EnumVariantInfo {
                            name: v.name.clone(),
                            field_names: v.fields.iter().map(|f| f.name.clone()).collect(),
                            field_types: v.fields.iter().map(|f| f.ty.clone()).collect(),
                        }).collect(),
                    });
                    self.define_var(&edef.name, PdcType::Enum(edef.name.clone()));
                }
                _ => {}
            }
        }

        // Pass 3: type check main statements
        for stmt in &program.stmts {
            self.check_stmt(stmt)?;
        }
        Ok(())
    }

    fn check_stmt(&mut self, stmt: &Spanned<Stmt>) -> Result<(), PdcError> {
        match &stmt.node {
            Stmt::BuiltinDecl { name, ty, mutable } => {
                let resolved = self.resolve_type(ty);
                self.define_var(name, resolved);
                if !mutable {
                    self.const_vars.insert(name.clone());
                }
            }
            Stmt::ConstDecl { vis: _, name, ty, value } => {
                let val_ty = self.check_expr(value)?;
                let final_ty = if let Some(declared) = ty {
                    let resolved = self.resolve_type(declared);
                    self.check_compatible(&val_ty, &resolved, value.span)?;
                    resolved
                } else {
                    val_ty
                };
                self.define_var(name, final_ty);
                self.const_vars.insert(name.clone());
            }
            Stmt::VarDecl { vis: _, name, ty, value } => {
                let val_ty = self.check_expr(value)?;
                let final_ty = if let Some(declared) = ty {
                    let resolved = self.resolve_type(declared);
                    self.check_compatible(&val_ty, &resolved, value.span)?;
                    resolved
                } else {
                    val_ty
                };
                self.define_var(name, final_ty);
            }
            Stmt::TupleDestructure { names, value, is_const } => {
                let val_ty = self.check_expr(value)?;
                if let PdcType::Tuple(elems) = &val_ty {
                    if names.len() != elems.len() {
                        return Err(PdcError::Type {
                            span: value.span,
                            message: format!(
                                "tuple destructure: expected {} names, got {} (tuple has {} elements)",
                                elems.len(), names.len(), elems.len(),
                            ),
                        });
                    }
                    for (i, name) in names.iter().enumerate() {
                        if name != "_" {
                            self.define_var(name, elems[i].clone());
                            if *is_const {
                                self.const_vars.insert(name.clone());
                            }
                        }
                    }
                } else {
                    return Err(PdcError::Type {
                        span: value.span,
                        message: format!("cannot destructure non-tuple type {val_ty}"),
                    });
                }
            }
            Stmt::IndexAssign { object, index, value } => {
                let obj_ty = self.check_expr(object)?;
                self.check_expr(index)?;
                let val_ty = self.check_expr(value)?;
                if let PdcType::Array(ref elem_ty) = obj_ty {
                    self.check_compatible(&val_ty, elem_ty, value.span)?;
                } else {
                    return Err(PdcError::Type {
                        span: object.span,
                        message: format!("cannot index-assign non-array type {obj_ty}"),
                    });
                }
            }
            Stmt::FieldAssign { object, field: _, value } => {
                let obj_ty = self.check_expr(object)?;
                if obj_ty != PdcType::KernelHandle {
                    return Err(PdcError::Type {
                        span: object.span,
                        message: format!("field assignment is only supported on Kernel handles, got {obj_ty}"),
                    });
                }
                // Virtual property: type is inferred from RHS, always valid
                self.check_expr(value)?;
            }
            Stmt::Assign { name, value } => {
                if self.const_vars.contains(name) {
                    return Err(PdcError::Type {
                        span: stmt.span,
                        message: format!("cannot assign to const variable '{name}'"),
                    });
                }
                let expected = self.lookup_var(name).cloned().ok_or_else(|| PdcError::Type {
                    span: stmt.span,
                    message: format!("undefined variable '{name}'"),
                })?;
                let val_ty = self.check_expr(value)?;
                self.check_compatible(&val_ty, &expected, value.span)?;
            }
            Stmt::ExprStmt(expr) => {
                self.check_expr(expr)?;
            }
            Stmt::If {
                condition,
                then_body,
                elsif_clauses,
                else_body,
            } => {
                let cond_ty = self.check_expr(condition)?;
                if cond_ty != PdcType::Bool {
                    return Err(PdcError::Type {
                        span: condition.span,
                        message: format!("if condition must be bool, got {cond_ty}"),
                    });
                }
                self.check_block(then_body)?;
                for (cond, body) in elsif_clauses {
                    let ct = self.check_expr(cond)?;
                    if ct != PdcType::Bool {
                        return Err(PdcError::Type {
                            span: cond.span,
                            message: format!("elsif condition must be bool, got {ct}"),
                        });
                    }
                    self.check_block(body)?;
                }
                if let Some(else_b) = else_body {
                    self.check_block(else_b)?;
                }
            }
            Stmt::While { condition, body } => {
                let cond_ty = self.check_expr(condition)?;
                if cond_ty != PdcType::Bool {
                    return Err(PdcError::Type {
                        span: condition.span,
                        message: format!("while condition must be bool, got {cond_ty}"),
                    });
                }
                self.check_block(body)?;
            }
            Stmt::For { mutable,
                var_name,
                start,
                end,
                inclusive: _,
                body,
            } => {
                let st = self.check_expr(start)?;
                let et = self.check_expr(end)?;
                if !st.is_int() {
                    return Err(PdcError::Type {
                        span: start.span,
                        message: format!("for range start must be integer, got {st}"),
                    });
                }
                if !et.is_int() {
                    return Err(PdcError::Type {
                        span: end.span,
                        message: format!("for range end must be integer, got {et}"),
                    });
                }
                self.push_scope();
                self.define_var(var_name, PdcType::I32);
                if !mutable {
                    self.const_vars.insert(var_name.clone());
                }
                for s in &body.stmts {
                    self.check_stmt(s)?;
                }
                if !mutable {
                    self.const_vars.remove(var_name);
                }
                self.pop_scope();
            }
            Stmt::ForEach { mutable,
                var_name,
                destructure_names,
                collection,
                body,
            } => {
                let coll_ty = self.check_expr(collection)?;
                let elem_ty = match &coll_ty {
                    PdcType::Array(et) => *et.clone(),
                    _ => {
                        return Err(PdcError::Type {
                            span: collection.span,
                            message: format!("cannot iterate over type {coll_ty}, expected Array"),
                        });
                    }
                };
                self.push_scope();
                if !destructure_names.is_empty() {
                    // Destructuring: element must be a tuple
                    if let PdcType::Tuple(ref elems) = elem_ty {
                        if destructure_names.len() != elems.len() {
                            return Err(PdcError::Type {
                                span: collection.span,
                                message: format!(
                                    "for-each destructure: expected {} names, got {} (tuple has {} elements)",
                                    elems.len(), destructure_names.len(), elems.len(),
                                ),
                            });
                        }
                        for (i, name) in destructure_names.iter().enumerate() {
                            if name != "_" {
                                self.define_var(name, elems[i].clone());
                                if !mutable {
                                    self.const_vars.insert(name.clone());
                                }
                            }
                        }
                    } else {
                        return Err(PdcError::Type {
                            span: collection.span,
                            message: format!("cannot destructure non-tuple element type {elem_ty}"),
                        });
                    }
                } else {
                    self.define_var(var_name, elem_ty);
                    if !mutable {
                        self.const_vars.insert(var_name.clone());
                    }
                }
                for s in &body.stmts {
                    self.check_stmt(s)?;
                }
                if !mutable {
                    if !destructure_names.is_empty() {
                        for name in destructure_names {
                            self.const_vars.remove(name);
                        }
                    } else {
                        self.const_vars.remove(var_name);
                    }
                }
                self.pop_scope();
            }
            Stmt::Loop { body } => {
                self.check_block(body)?;
            }
            Stmt::Break | Stmt::Continue => {}
            Stmt::Return(value) => {
                if let Some(expr) = value {
                    self.check_expr(expr)?;
                }
            }
            Stmt::Match { scrutinee, arms } => {
                let scr_ty = self.check_expr(scrutinee)?;

                let enum_name = match &scr_ty {
                    PdcType::Enum(name) => name.clone(),
                    _ => return Err(PdcError::Type {
                        span: scrutinee.span,
                        message: format!("cannot match non-enum type {scr_ty}"),
                    }),
                };

                let info = self.enums.get(&enum_name).cloned().ok_or_else(|| PdcError::Type {
                    span: scrutinee.span,
                    message: format!("undefined enum '{enum_name}'"),
                })?;

                let mut covered_variants: Vec<String> = Vec::new();
                let mut has_wildcard = false;

                for arm in arms {
                    self.push_scope();

                    match &arm.pattern {
                        MatchPattern::EnumVariant { enum_name: pat_enum, variant, bindings } => {
                            // Verify enum name matches (or is empty for dot-shorthand)
                            if !pat_enum.is_empty() && *pat_enum != enum_name {
                                return Err(PdcError::Type {
                                    span: scrutinee.span,
                                    message: format!("match pattern {pat_enum}.{variant} doesn't match scrutinee type {enum_name}"),
                                });
                            }

                            let vi = info.variants.iter().find(|v| v.name == *variant)
                                .ok_or_else(|| PdcError::Type {
                                    span: scrutinee.span,
                                    message: format!("enum '{enum_name}' has no variant '{variant}'"),
                                })?;

                            // Define destructured bindings
                            for (bi, bname) in bindings.iter().enumerate() {
                                if bname != "_" && bi < vi.field_types.len() {
                                    self.define_var(bname, vi.field_types[bi].clone());
                                }
                            }

                            covered_variants.push(variant.clone());
                        }
                        MatchPattern::Wildcard => {
                            has_wildcard = true;
                        }
                    }

                    for s in &arm.body.stmts {
                        self.check_stmt(s)?;
                    }
                    self.pop_scope();
                }

                // Exhaustiveness check
                if !has_wildcard {
                    for v in &info.variants {
                        if !covered_variants.contains(&v.name) {
                            return Err(PdcError::Type {
                                span: scrutinee.span,
                                message: format!("non-exhaustive match: missing variant '{}.{}'", enum_name, v.name),
                            });
                        }
                    }
                }
            }
            Stmt::Import { module, names } => {
                if names.is_empty() {
                    // Namespaced import: `import math` → define `math` as a module namespace
                    self.define_var(module, PdcType::Module(module.clone()));
                } else {
                    // Selective import: `import { PI, lerp } from math`
                    for name in names {
                        let qualified = format!("{module}::{name}");

                        if self.user_fns.contains_key(&qualified) {
                            let overloads = self.user_fns.get(&qualified).cloned().unwrap();
                            self.user_fns.entry(name.clone())
                                .or_insert_with(|| OverloadSet { sigs: Vec::new() })
                                .sigs.extend(overloads.sigs);
                            self.fn_aliases.insert(name.clone(), qualified.clone());
                        }

                        if let Some(info) = self.structs.get(&qualified).cloned() {
                            self.structs.insert(name.clone(), info);
                        }

                        if let Some(info) = self.enums.get(&qualified).cloned() {
                            self.enums.insert(name.clone(), info);
                            self.define_var(name, PdcType::Enum(name.clone()));
                        }

                        if let Some(ty) = self.type_aliases.get(&qualified).cloned() {
                            self.type_aliases.insert(name.clone(), ty);
                        }

                        if let Some(ty) = self.lookup_var(&qualified).cloned() {
                            self.define_var(name, ty);
                        }
                    }
                }
            }
            Stmt::TestDef { body, .. } => {
                self.push_scope();
                for s in &body.stmts {
                    self.check_stmt(s)?;
                }
                self.pop_scope();
            }
            Stmt::StructDef(_) | Stmt::EnumDef(_) | Stmt::TypeAlias { .. } => {
                // Already registered in first pass
            }
            Stmt::FnDef(fndef) => {
                // Type-check default value expressions in the outer scope
                for param in &fndef.params {
                    if let Some(ref default_expr) = param.default {
                        let default_ty = self.check_expr(default_expr)?;
                        self.check_compatible(&default_ty, &param.ty, default_expr.span)?;
                    }
                }
                // Function bodies see all outer scope variables (top-level consts, builtins)
                self.push_scope();
                for param in &fndef.params {
                    self.define_var(&param.name, param.ty.clone());
                }
                for s in &fndef.body.stmts {
                    self.check_stmt(s)?;
                }
                self.pop_scope();
            }
        }
        Ok(())
    }

    fn check_block(&mut self, block: &Block) -> Result<(), PdcError> {
        self.push_scope();
        for stmt in &block.stmts {
            self.check_stmt(stmt)?;
        }
        self.pop_scope();
        Ok(())
    }

    fn check_expr(&mut self, expr: &Spanned<Expr>) -> Result<PdcType, PdcError> {
        let ty = match &expr.node {
            Expr::Literal(lit) => match lit {
                Literal::Int(v) => {
                    if *v < 0 || *v <= i32::MAX as i64 {
                        PdcType::I32
                    } else if *v <= u32::MAX as i64 {
                        PdcType::U32
                    } else {
                        PdcType::I64
                    }
                }
                Literal::Float(_) => PdcType::F64,
                Literal::Bool(_) => PdcType::Bool,
                Literal::String(_) => PdcType::Str,
            },
            Expr::Variable(name) => {
                if let Some(ty) = self.lookup_var(name).cloned() {
                    ty
                } else if let Some(builtin) = self.builtins.get(name.as_str()) {
                    // Function reference to a builtin
                    PdcType::FnRef {
                        params: builtin.params.clone(),
                        ret: Box::new(builtin.ret.clone()),
                    }
                } else if let Some(overloads) = self.user_fns.get(name.as_str()) {
                    // Function reference to a user function (use first overload)
                    let sig = &overloads.sigs[0];
                    PdcType::FnRef {
                        params: sig.params.clone(),
                        ret: Box::new(sig.ret.clone()),
                    }
                } else {
                    return Err(PdcError::Type {
                        span: expr.span,
                        message: format!("undefined variable '{name}'"),
                    });
                }
            }
            Expr::BinaryOp { op, left, right } => {
                let lt = self.check_expr(left)?;
                let rt = self.check_expr(right)?;
                self.check_binary_op(*op, &lt, &rt, expr.span)?
            }
            Expr::UnaryOp { op, operand } => {
                let t = self.check_expr(operand)?;
                // Check for user-defined unary operator overload
                let uop_name = unaryop_to_op_name(*op);
                if let Some(overloads) = self.op_overloads.get(uop_name) {
                    if let Some(sig) = self.resolve_overload(overloads, &[t.clone()]) {
                        self.set_type(expr.id, sig.ret.clone());
                        return Ok(sig.ret);
                    }
                }
                match op {
                    UnaryOp::Neg => {
                        if !t.is_numeric() {
                            return Err(PdcError::Type {
                                span: expr.span,
                                message: format!("cannot negate type {t}"),
                            });
                        }
                        t
                    }
                    UnaryOp::Not => {
                        if t != PdcType::Bool {
                            return Err(PdcError::Type {
                                span: expr.span,
                                message: format!("cannot apply ! to type {t}"),
                            });
                        }
                        PdcType::Bool
                    }
                    UnaryOp::BitNot => {
                        if !t.is_int() {
                            return Err(PdcError::Type {
                                span: expr.span,
                                message: format!("cannot apply ~ to type {t}, expected integer"),
                            });
                        }
                        t
                    }
                }
            }
            Expr::Call { name, args, arg_names } => {
                // Struct construction: named args + known struct name
                let has_named_args = arg_names.iter().any(|n| n.is_some());
                if has_named_args {
                    if let Some(info) = self.structs.get(name).cloned() {
                        // Struct construction
                        for (i, arg) in args.iter().enumerate() {
                            let val_ty = self.check_expr(arg)?;
                            let fname = arg_names[i].as_ref().unwrap();
                            let expected = info.fields.iter()
                                .find(|(n, _)| n == fname)
                                .map(|(_, t)| t.clone())
                                .ok_or_else(|| PdcError::Type {
                                    span: arg.span,
                                    message: format!("struct '{name}' has no field '{fname}'"),
                                })?;
                            self.check_compatible(&val_ty, &expected, arg.span)?;
                        }
                        self.set_type(expr.id, PdcType::Struct(name.clone()));
                        return Ok(PdcType::Struct(name.clone()));
                    }
                    // Not a struct — check as named function call below
                }

                // Array<type>() constructor
                if name.starts_with("Array<") {
                    let elem_type_str = &name[6..name.len()-1]; // extract type between < >
                    let elem_ty = match elem_type_str {
                        "f32" => PdcType::F32,
                        "f64" => PdcType::F64,
                        "i32" => PdcType::I32,
                        "u32" => PdcType::U32,
                        "bool" => PdcType::Bool,
                        _ => PdcType::F64, // default
                    };
                    let arr_ty = PdcType::Array(Box::new(elem_ty));
                    self.set_type(expr.id, arr_ty.clone());
                    return Ok(arr_ty);
                }

                // Overloaded builtins: fill/stroke with style parameters
                if name == "fill" && args.len() == 3 {
                    // fill(path, color, rule) → fill_styled
                    for arg in args.iter() { self.check_expr(arg)?; }
                    self.set_type(expr.id, PdcType::Void);
                    return Ok(PdcType::Void);
                }
                if name == "stroke" && args.len() == 5 {
                    // stroke(path, width, color, cap, join) → stroke_styled
                    for arg in args.iter() { self.check_expr(arg)?; }
                    self.set_type(expr.id, PdcType::Void);
                    return Ok(PdcType::Void);
                }

                // Test assertion intrinsics
                if name == "assert_eq" {
                    if args.len() != 2 {
                        return Err(PdcError::Type {
                            span: expr.span,
                            message: format!("assert_eq expects 2 arguments, got {}", args.len()),
                        });
                    }
                    let ty_a = self.check_expr(&args[0])?;
                    let ty_b = self.check_expr(&args[1])?;
                    self.check_compatible(&ty_a, &ty_b, args[1].span)?;
                    self.set_type(expr.id, PdcType::Void);
                    return Ok(PdcType::Void);
                }
                if name == "assert_near" {
                    if args.len() != 3 {
                        return Err(PdcError::Type {
                            span: expr.span,
                            message: format!("assert_near expects 3 arguments, got {}", args.len()),
                        });
                    }
                    for arg in args.iter() {
                        let ty = self.check_expr(arg)?;
                        self.check_compatible(&ty, &PdcType::F64, arg.span)?;
                    }
                    self.set_type(expr.id, PdcType::Void);
                    return Ok(PdcType::Void);
                }
                if name == "assert_true" {
                    if args.len() != 1 {
                        return Err(PdcError::Type {
                            span: expr.span,
                            message: format!("assert_true expects 1 argument, got {}", args.len()),
                        });
                    }
                    let ty = self.check_expr(&args[0])?;
                    self.check_compatible(&ty, &PdcType::Bool, args[0].span)?;
                    self.set_type(expr.id, PdcType::Void);
                    return Ok(PdcType::Void);
                }

                if let Some(cast_ty) = self.is_type_cast(name) {
                    if args.len() != 1 {
                        return Err(PdcError::Type {
                            span: expr.span,
                            message: format!("type cast {name}() expects 1 argument, got {}", args.len()),
                        });
                    }
                    self.check_expr(&args[0])?;
                    cast_ty
                } else if let Some(builtin) = self.builtins.get(name.as_str()) {
                    let expected_params = builtin.params.clone();
                    let ret = builtin.ret.clone();
                    if args.len() != expected_params.len() {
                        return Err(PdcError::Type {
                            span: expr.span,
                            message: format!(
                                "function '{name}' expects {} arguments, got {}",
                                expected_params.len(),
                                args.len()
                            ),
                        });
                    }
                    for (i, arg) in args.iter().enumerate() {
                        let arg_ty = self.check_expr_with_hint(arg, Some(&expected_params[i]))?;
                        self.check_compatible(&arg_ty, &expected_params[i], arg.span)?;
                    }
                    ret
                } else if let Some(overloads) = self.user_fns.get(name.as_str()).cloned() {
                    // Type-check args first to get their types for overload resolution.
                    // If there's exactly one overload, use its param types as hints
                    // to resolve dot-shorthand enum values.
                    let hints: Option<Vec<PdcType>> = if overloads.sigs.len() == 1 {
                        Some(overloads.sigs[0].params.clone())
                    } else {
                        None
                    };
                    let mut arg_types = Vec::new();
                    for (i, arg) in args.iter().enumerate() {
                        let hint = hints.as_ref().and_then(|h| h.get(i));
                        arg_types.push(self.check_expr_with_hint(arg, hint)?);
                    }
                    let sig = self.resolve_overload(&overloads, &arg_types)
                        .ok_or_else(|| PdcError::Type {
                            span: expr.span,
                            message: format!(
                                "no matching overload for '{name}' with argument types ({})",
                                arg_types.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(", "),
                            ),
                        })?;
                    for (i, arg) in args.iter().enumerate() {
                        self.check_compatible(&arg_types[i], &sig.params[i], arg.span)?;
                    }
                    sig.ret
                } else {
                    return Err(PdcError::Type {
                        span: expr.span,
                        message: format!("undefined function '{name}'"),
                    });
                }
            }
            Expr::MethodCall { object, method, args } => {
                let obj_ty = self.check_expr(object)?;

                // Module namespaced call: math.sin(x) → math::sin(x)
                if let PdcType::Module(mod_name) = &obj_ty {
                    // Buffer factory constructors: Buffer.I32(), Buffer.Vec4F32(), etc.
                    if mod_name == "Buffer" {
                        let valid = ["F32", "I32", "U32", "Vec2F32", "Vec3F32", "Vec4F32"];
                        if !valid.contains(&method.as_str()) {
                            return Err(PdcError::Type {
                                span: expr.span,
                                message: format!("Buffer has no variant '{method}'. Valid: {}", valid.join(", ")),
                            });
                        }
                        if !args.is_empty() {
                            return Err(PdcError::Type {
                                span: expr.span,
                                message: format!("Buffer.{method}() takes no arguments, got {}", args.len()),
                            });
                        }
                        self.set_type(expr.id, PdcType::BufferHandle);
                        return Ok(PdcType::BufferHandle);
                    }
                    // Kernel factory constructors: Kernel.Sim("name", "path"), Kernel.Pixel(...)
                    if mod_name == "Kernel" {
                        let valid = ["Pixel", "Sim"];
                        if !valid.contains(&method.as_str()) {
                            return Err(PdcError::Type {
                                span: expr.span,
                                message: format!("Kernel has no variant '{method}'. Valid: {}", valid.join(", ")),
                            });
                        }
                        if args.len() != 2 {
                            return Err(PdcError::Type {
                                span: expr.span,
                                message: format!("Kernel.{method}() expects 2 arguments (name, path), got {}", args.len()),
                            });
                        }
                        for arg in args.iter() {
                            let arg_ty = self.check_expr(arg)?;
                            self.check_compatible(&arg_ty, &PdcType::Str, arg.span)?;
                        }
                        self.set_type(expr.id, PdcType::KernelHandle);
                        return Ok(PdcType::KernelHandle);
                    }

                    let qualified = format!("{mod_name}::{method}");

                    // Check pub visibility
                    if let Some(exports) = self.module_exports.get(mod_name.as_str()) {
                        if !exports.names.contains(method.as_str()) {
                            return Err(PdcError::Type {
                                span: expr.span,
                                message: format!("'{method}' is not exported from module '{mod_name}'"),
                            });
                        }
                    }

                    // Check args
                    let mut arg_types = Vec::new();
                    for arg in args.iter() {
                        arg_types.push(self.check_expr(arg)?);
                    }
                    // Try user function with qualified name
                    if let Some(overloads) = self.user_fns.get(&qualified).cloned() {
                        let sig = self.resolve_overload(&overloads, &arg_types)
                            .ok_or_else(|| PdcError::Type {
                                span: expr.span,
                                message: format!("no matching overload for '{mod_name}.{method}'"),
                            })?;
                        for (i, arg) in args.iter().enumerate() {
                            self.check_compatible(&arg_types[i], &sig.params[i], arg.span)?;
                        }
                        self.set_type(expr.id, sig.ret.clone());
                        return Ok(sig.ret);
                    }
                    // Try builtin (builtins are unqualified — sin, cos, etc.)
                    if let Some(builtin) = self.builtins.get(method.as_str()) {
                        let ret = builtin.ret.clone();
                        self.set_type(expr.id, ret.clone());
                        return Ok(ret);
                    }
                    return Err(PdcError::Type {
                        span: expr.span,
                        message: format!("module '{mod_name}' has no function '{method}'"),
                    });
                }

                // Tuple methods: len
                if let PdcType::Tuple(_) = &obj_ty {
                    match method.as_str() {
                        "len" => PdcType::I32,
                        _ => {
                            return Err(PdcError::Type {
                                span: expr.span,
                                message: format!("tuple has no method '{method}'"),
                            });
                        }
                    }
                } else
                // Slice methods: len, get
                if let PdcType::Slice(ref elem_ty) = obj_ty {
                    match method.as_str() {
                        "len" => PdcType::I32,
                        "get" => {
                            if args.len() == 1 {
                                self.check_expr(&args[0])?;
                            }
                            *elem_ty.clone()
                        }
                        _ => {
                            return Err(PdcError::Type {
                                span: expr.span,
                                message: format!("slice has no method '{method}'"),
                            });
                        }
                    }
                } else
                // String methods
                if obj_ty == PdcType::Str {
                    match method.as_str() {
                        "len" => PdcType::I32,
                        "slice" => {
                            if args.len() == 2 {
                                self.check_expr(&args[0])?;
                                self.check_expr(&args[1])?;
                            }
                            PdcType::Str
                        }
                        "concat" => {
                            if args.len() == 1 {
                                let arg_ty = self.check_expr(&args[0])?;
                                if arg_ty != PdcType::Str {
                                    return Err(PdcError::Type {
                                        span: args[0].span,
                                        message: format!("concat expects string, got {arg_ty}"),
                                    });
                                }
                            }
                            PdcType::Str
                        }
                        "char_at" => {
                            if args.len() == 1 {
                                self.check_expr(&args[0])?;
                            }
                            PdcType::Str
                        }
                        _ => {
                            return Err(PdcError::Type {
                                span: expr.span,
                                message: format!("string has no method '{method}'"),
                            });
                        }
                    }
                } else
                // Array methods: push, len, get, set
                if let PdcType::Array(ref elem_ty) = obj_ty {
                    match method.as_str() {
                        "push" => {
                            if args.len() == 1 {
                                let arg_ty = self.check_expr(&args[0])?;
                                self.check_compatible(&arg_ty, elem_ty, args[0].span)?;
                            }
                            PdcType::Void
                        }
                        "len" => PdcType::I32,
                        "get" => {
                            if args.len() == 1 {
                                self.check_expr(&args[0])?;
                            }
                            *elem_ty.clone()
                        }
                        "set" => {
                            if args.len() == 2 {
                                self.check_expr(&args[0])?;
                                let val_ty = self.check_expr(&args[1])?;
                                self.check_compatible(&val_ty, elem_ty, args[1].span)?;
                            }
                            PdcType::Void
                        }
                        "map" => {
                            if args.len() != 1 {
                                return Err(PdcError::Type {
                                    span: expr.span,
                                    message: "map() expects exactly 1 argument (a function reference)".into(),
                                });
                            }
                            let fn_ty = self.check_expr(&args[0])?;
                            match &fn_ty {
                                PdcType::FnRef { params, ret } => {
                                    if params.len() != 1 {
                                        return Err(PdcError::Type {
                                            span: args[0].span,
                                            message: format!("map function must take 1 parameter, got {}", params.len()),
                                        });
                                    }
                                    self.check_compatible(elem_ty, &params[0], args[0].span)?;
                                    PdcType::Array(Box::new(*ret.clone()))
                                }
                                _ => {
                                    return Err(PdcError::Type {
                                        span: args[0].span,
                                        message: format!("map() argument must be a function reference, got {fn_ty}"),
                                    });
                                }
                            }
                        }
                        "slice" => {
                            if args.len() == 2 {
                                self.check_expr(&args[0])?;
                                self.check_expr(&args[1])?;
                            }
                            PdcType::Slice(elem_ty.clone())
                        }
                        _ => {
                            return Err(PdcError::Type {
                                span: expr.span,
                                message: format!("array has no method '{method}'"),
                            });
                        }
                    }
                } else
                // Enum variant construction: EnumName.Variant(args)
                if let PdcType::Enum(ref ename) = obj_ty {
                    let info = self.enums.get(ename).cloned().ok_or_else(|| PdcError::Type {
                        span: expr.span,
                        message: format!("undefined enum '{ename}'"),
                    })?;
                    let var_info = info.variants.iter().find(|v| v.name == *method)
                        .ok_or_else(|| PdcError::Type {
                            span: expr.span,
                            message: format!("enum '{ename}' has no variant '{method}'"),
                        })?.clone();
                    if args.len() != var_info.field_types.len() {
                        return Err(PdcError::Type {
                            span: expr.span,
                            message: format!(
                                "variant '{}.{}' expects {} arguments, got {}",
                                ename, method, var_info.field_types.len(), args.len(),
                            ),
                        });
                    }
                    for (i, arg) in args.iter().enumerate() {
                        let arg_ty = self.check_expr(arg)?;
                        self.check_compatible(&arg_ty, &var_info.field_types[i], arg.span)?;
                    }
                    PdcType::Enum(ename.clone())
                } else if let Some(builtin) = self.builtins.get(method.as_str()) {
                    let expected_params = builtin.params.clone();
                    let ret = builtin.ret.clone();
                    let total_args = 1 + args.len();
                    if total_args != expected_params.len() {
                        return Err(PdcError::Type {
                            span: expr.span,
                            message: format!(
                                "method '{method}' expects {} arguments (including self), got {}",
                                expected_params.len(),
                                total_args,
                            ),
                        });
                    }
                    self.check_compatible(&obj_ty, &expected_params[0], object.span)?;
                    for (i, arg) in args.iter().enumerate() {
                        let arg_ty = self.check_expr_with_hint(arg, Some(&expected_params[i + 1]))?;
                        self.check_compatible(&arg_ty, &expected_params[i + 1], arg.span)?;
                    }
                    ret
                } else if let Some(overloads) = self.user_fns.get(method.as_str()).cloned() {
                    // Build full arg types: [obj_ty, arg0_ty, arg1_ty, ...]
                    let mut arg_types = vec![obj_ty.clone()];
                    let hints: Option<Vec<PdcType>> = if overloads.sigs.len() == 1 {
                        Some(overloads.sigs[0].params.clone())
                    } else {
                        None
                    };
                    for (i, arg) in args.iter().enumerate() {
                        let hint = hints.as_ref().and_then(|h| h.get(i + 1));
                        arg_types.push(self.check_expr_with_hint(arg, hint)?);
                    }
                    let sig = self.resolve_overload(&overloads, &arg_types)
                        .ok_or_else(|| PdcError::Type {
                            span: expr.span,
                            message: format!(
                                "no matching overload for method '{method}' with argument types ({})",
                                arg_types.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(", "),
                            ),
                        })?;
                    self.check_compatible(&obj_ty, &sig.params[0], object.span)?;
                    for (i, arg) in args.iter().enumerate() {
                        self.check_compatible(&arg_types[i + 1], &sig.params[i + 1], arg.span)?;
                    }
                    sig.ret
                } else {
                    return Err(PdcError::Type {
                        span: expr.span,
                        message: format!("undefined method '{method}'"),
                    });
                }
            }
            Expr::Index { object, index } => {
                let obj_ty = self.check_expr(object)?;
                self.check_expr(index)?;
                if let PdcType::Array(elem_ty) = &obj_ty {
                    *elem_ty.clone()
                } else if let PdcType::Slice(elem_ty) = &obj_ty {
                    *elem_ty.clone()
                } else {
                    return Err(PdcError::Type {
                        span: expr.span,
                        message: format!("cannot index type {obj_ty}"),
                    });
                }
            }
            Expr::TupleConstruct { elements } => {
                let elem_types: Vec<PdcType> = elements.iter()
                    .map(|e| self.check_expr(e))
                    .collect::<Result<_, _>>()?;
                PdcType::Tuple(elem_types)
            }
            Expr::TupleIndex { object, index } => {
                let obj_ty = self.check_expr(object)?;
                if let PdcType::Tuple(elems) = &obj_ty {
                    if *index < elems.len() {
                        elems[*index].clone()
                    } else {
                        return Err(PdcError::Type {
                            span: expr.span,
                            message: format!("tuple index {} out of range (tuple has {} elements)", index, elems.len()),
                        });
                    }
                } else {
                    return Err(PdcError::Type {
                        span: expr.span,
                        message: format!("cannot index non-tuple type {obj_ty}"),
                    });
                }
            }
            Expr::FieldAccess { object, field } => {
                let obj_ty = self.check_expr(object)?;
                match &obj_ty {
                    PdcType::Module(mod_name) => {
                        // Module namespaced field: math.PI → math::PI
                        if let Some(exports) = self.module_exports.get(mod_name.as_str()) {
                            if !exports.names.contains(field.as_str()) {
                                return Err(PdcError::Type {
                                    span: expr.span,
                                    message: format!("'{field}' is not exported from module '{mod_name}'"),
                                });
                            }
                        }
                        let qualified = format!("{mod_name}::{field}");
                        self.lookup_var(&qualified).cloned()
                            .ok_or_else(|| PdcError::Type {
                                span: expr.span,
                                message: format!("module '{mod_name}' has no member '{field}'"),
                            })?
                    }
                    PdcType::Struct(name) => {
                        let info = self.structs.get(name).cloned().ok_or_else(|| PdcError::Type {
                            span: expr.span,
                            message: format!("undefined struct '{name}'"),
                        })?;
                        info.fields.iter()
                            .find(|(n, _)| n == field)
                            .map(|(_, t)| t.clone())
                            .ok_or_else(|| PdcError::Type {
                                span: expr.span,
                                message: format!("struct '{name}' has no field '{field}'"),
                            })?
                    }
                    PdcType::Enum(name) => {
                        let info = self.enums.get(name).cloned().ok_or_else(|| PdcError::Type {
                            span: expr.span,
                            message: format!("undefined enum '{name}'"),
                        })?;
                        if !info.variants.iter().any(|v| v.name == *field) {
                            return Err(PdcError::Type {
                                span: expr.span,
                                message: format!("enum '{name}' has no variant '{field}'"),
                            });
                        }
                        PdcType::Enum(name.clone())
                    }
                    _ => {
                        return Err(PdcError::Type {
                            span: expr.span,
                            message: format!("cannot access field '{field}' on type {obj_ty}"),
                        });
                    }
                }
            }
            Expr::DotShorthand(variant) => {
                return Err(PdcError::Type {
                    span: expr.span,
                    message: format!("'.{variant}' enum shorthand can only be used where the enum type is known (e.g., function arguments)"),
                });
            }
            Expr::Ternary { condition, then_expr, else_expr } => {
                let cond_ty = self.check_expr(condition)?;
                if cond_ty != PdcType::Bool {
                    return Err(PdcError::Type {
                        span: condition.span,
                        message: format!("ternary condition must be bool, got {cond_ty}"),
                    });
                }
                let then_ty = self.check_expr(then_expr)?;
                let else_ty = self.check_expr(else_expr)?;
                if then_ty == else_ty {
                    then_ty
                } else if then_ty.is_numeric() && else_ty.is_numeric() {
                    self.unify_numeric(&then_ty, &else_ty, expr.span)?
                } else {
                    return Err(PdcError::Type {
                        span: expr.span,
                        message: format!("ternary branches must have compatible types, got {then_ty} and {else_ty}"),
                    });
                }
            }
            Expr::StructConstruct { name, fields } => {
                let info = self.structs.get(name).cloned().ok_or_else(|| PdcError::Type {
                    span: expr.span,
                    message: format!("undefined struct '{name}'"),
                })?;
                for (fname, fexpr) in fields {
                    let val_ty = self.check_expr(fexpr)?;
                    let expected = info.fields.iter()
                        .find(|(n, _)| n == fname)
                        .map(|(_, t)| t.clone())
                        .ok_or_else(|| PdcError::Type {
                            span: fexpr.span,
                            message: format!("struct '{name}' has no field '{fname}'"),
                        })?;
                    self.check_compatible(&val_ty, &expected, fexpr.span)?;
                }
                PdcType::Struct(name.clone())
            }
        };

        self.set_type(expr.id, ty.clone());
        Ok(ty)
    }

    /// Check an expression with an optional type hint. When the expression is a
    /// `.Variant` dot-shorthand and the hint is an enum type, resolve the variant
    /// against that enum. Otherwise falls back to `check_expr`.
    fn check_expr_with_hint(
        &mut self,
        expr: &Spanned<Expr>,
        hint: Option<&PdcType>,
    ) -> Result<PdcType, PdcError> {
        if let Expr::DotShorthand(variant) = &expr.node {
            if let Some(PdcType::Enum(enum_name)) = hint {
                let info = self.enums.get(enum_name).cloned().ok_or_else(|| PdcError::Type {
                    span: expr.span,
                    message: format!("undefined enum '{enum_name}'"),
                })?;
                if !info.variants.iter().any(|v| v.name == *variant) {
                    return Err(PdcError::Type {
                        span: expr.span,
                        message: format!("enum '{enum_name}' has no variant '{variant}'"),
                    });
                }
                let ty = PdcType::Enum(enum_name.clone());
                self.set_type(expr.id, ty.clone());
                return Ok(ty);
            }
        }
        self.check_expr(expr)
    }

    fn is_type_cast(&self, name: &str) -> Option<PdcType> {
        match name {
            "f32" => Some(PdcType::F32),
            "f64" => Some(PdcType::F64),
            "i8" => Some(PdcType::I8),
            "i16" => Some(PdcType::I16),
            "i32" => Some(PdcType::I32),
            "i64" => Some(PdcType::I64),
            "u8" => Some(PdcType::U8),
            "u16" => Some(PdcType::U16),
            "u32" => Some(PdcType::U32),
            "u64" => Some(PdcType::U64),
            "bool" => Some(PdcType::Bool),
            _ => {
                // Check type aliases that resolve to numeric types
                if let Some(aliased) = self.type_aliases.get(name) {
                    if aliased.is_numeric() || *aliased == PdcType::Bool {
                        return Some(aliased.clone());
                    }
                }
                None
            }
        }
    }

    fn check_binary_op(
        &self,
        op: BinOp,
        left: &PdcType,
        right: &PdcType,
        span: super::span::Span,
    ) -> Result<PdcType, PdcError> {
        // Check for user-defined operator overload
        let op_name = binop_to_op_name(op);
        if let Some(overloads) = self.op_overloads.get(op_name) {
            let arg_types = [left.clone(), right.clone()];
            if let Some(sig) = self.resolve_overload(overloads, &arg_types) {
                return Ok(sig.ret);
            }
        }

        // Array broadcasting: array op array, array op scalar, scalar op array
        let is_arithmetic_or_cmp = matches!(op,
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod | BinOp::Pow |
            BinOp::Eq | BinOp::NotEq | BinOp::Lt | BinOp::LtEq | BinOp::Gt | BinOp::GtEq |
            BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor | BinOp::Shl | BinOp::Shr
        );
        if is_arithmetic_or_cmp {
            match (left, right) {
                (PdcType::Array(elem_l), PdcType::Array(elem_r)) => {
                    // array op array: element-wise
                    let result_elem = self.check_binary_op(op, elem_l, elem_r, span)?;
                    return if matches!(op, BinOp::Eq | BinOp::NotEq | BinOp::Lt | BinOp::LtEq | BinOp::Gt | BinOp::GtEq) {
                        Ok(PdcType::Array(Box::new(PdcType::Bool)))
                    } else {
                        Ok(PdcType::Array(Box::new(result_elem)))
                    };
                }
                (PdcType::Array(elem), scalar) if scalar.is_numeric() => {
                    let result_elem = self.check_binary_op(op, elem, scalar, span)?;
                    return if matches!(op, BinOp::Eq | BinOp::NotEq | BinOp::Lt | BinOp::LtEq | BinOp::Gt | BinOp::GtEq) {
                        Ok(PdcType::Array(Box::new(PdcType::Bool)))
                    } else {
                        Ok(PdcType::Array(Box::new(result_elem)))
                    };
                }
                (scalar, PdcType::Array(elem)) if scalar.is_numeric() => {
                    let result_elem = self.check_binary_op(op, scalar, elem, span)?;
                    return if matches!(op, BinOp::Eq | BinOp::NotEq | BinOp::Lt | BinOp::LtEq | BinOp::Gt | BinOp::GtEq) {
                        Ok(PdcType::Array(Box::new(PdcType::Bool)))
                    } else {
                        Ok(PdcType::Array(Box::new(result_elem)))
                    };
                }
                _ => {} // fall through to normal checking
            }
        }

        // Logical operators: support array<bool> element-wise
        if matches!(op, BinOp::And | BinOp::Or) {
            match (left, right) {
                (PdcType::Array(elem_l), PdcType::Array(elem_r)) => {
                    self.check_binary_op(op, elem_l, elem_r, span)?;
                    return Ok(PdcType::Array(Box::new(PdcType::Bool)));
                }
                (PdcType::Array(elem), PdcType::Bool) | (PdcType::Bool, PdcType::Array(elem)) => {
                    self.check_binary_op(op, elem, &PdcType::Bool, span)?;
                    return Ok(PdcType::Array(Box::new(PdcType::Bool)));
                }
                _ => {} // fall through to scalar check
            }
        }

        // String operations
        if *left == PdcType::Str || *right == PdcType::Str {
            match op {
                BinOp::Add => {
                    if *left == PdcType::Str && *right == PdcType::Str {
                        return Ok(PdcType::Str);
                    }
                    return Err(PdcError::Type {
                        span,
                        message: format!("cannot add {left} and {right}"),
                    });
                }
                BinOp::Eq | BinOp::NotEq => {
                    if *left == PdcType::Str && *right == PdcType::Str {
                        return Ok(PdcType::Bool);
                    }
                    return Err(PdcError::Type {
                        span,
                        message: format!("cannot compare {left} and {right}"),
                    });
                }
                _ => {
                    return Err(PdcError::Type {
                        span,
                        message: format!("operator not supported for string type"),
                    });
                }
            }
        }

        match op {
            BinOp::And | BinOp::Or => {
                if *left != PdcType::Bool || *right != PdcType::Bool {
                    return Err(PdcError::Type {
                        span,
                        message: format!("logical operator requires bool, got {left} and {right}"),
                    });
                }
                Ok(PdcType::Bool)
            }
            BinOp::Eq | BinOp::NotEq | BinOp::Lt | BinOp::LtEq | BinOp::Gt | BinOp::GtEq => {
                // Enum comparison: same enum type
                if let (PdcType::Enum(a), PdcType::Enum(b)) = (left, right) {
                    if a != b {
                        return Err(PdcError::Type {
                            span,
                            message: format!("cannot compare different enum types {a} and {b}"),
                        });
                    }
                    return Ok(PdcType::Bool);
                }
                let _ = self.unify_numeric(left, right, span)?;
                Ok(PdcType::Bool)
            }
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod | BinOp::Pow => {
                self.unify_numeric(left, right, span)
            }
            BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor | BinOp::Shl | BinOp::Shr => {
                if !left.is_int() || !right.is_int() {
                    return Err(PdcError::Type {
                        span,
                        message: format!("bitwise operator requires integer types, got {left} and {right}"),
                    });
                }
                self.unify_numeric(left, right, span)
            }
        }
    }

    fn unify_numeric(
        &self,
        a: &PdcType,
        b: &PdcType,
        span: super::span::Span,
    ) -> Result<PdcType, PdcError> {
        if !a.is_numeric() || !b.is_numeric() {
            return Err(PdcError::Type {
                span,
                message: format!("expected numeric types, got {a} and {b}"),
            });
        }
        if a == b {
            return Ok(a.clone());
        }
        // Widening: float wins over int, larger wins over smaller
        match (a, b) {
            (PdcType::F64, _) | (_, PdcType::F64) => Ok(PdcType::F64),
            (PdcType::F32, _) | (_, PdcType::F32) => Ok(PdcType::F32),
            (PdcType::I64, _) | (_, PdcType::I64) => Ok(PdcType::I64),
            (PdcType::U64, _) | (_, PdcType::U64) => Ok(PdcType::I64), // mixed sign → signed
            (PdcType::I32, _) | (_, PdcType::I32) => Ok(PdcType::I32),
            (PdcType::U32, _) | (_, PdcType::U32) => Ok(PdcType::I32),
            (PdcType::I16, _) | (_, PdcType::I16) => Ok(PdcType::I16),
            (PdcType::U16, _) | (_, PdcType::U16) => Ok(PdcType::I16),
            _ => Ok(PdcType::I32), // i8/u8 → i32
        }
    }

    /// Build a UserFnSig from a FnDef, capturing default expressions.
    fn sig_from_fndef(&self, fndef: &FnDef) -> UserFnSig {
        let required = fndef.params.iter().take_while(|p| p.default.is_none()).count();
        let defaults: Vec<Spanned<Expr>> = fndef.params[required..]
            .iter()
            .map(|p| p.default.clone().unwrap())
            .collect();
        UserFnSig {
            params: fndef.params.iter().map(|p| self.resolve_type(&p.ty)).collect(),
            ret: self.resolve_type(&fndef.return_type),
            required,
            defaults,
        }
    }

    /// Resolve the best-matching overload for given argument types.
    fn resolve_overload(&self, overloads: &OverloadSet, arg_types: &[PdcType]) -> Option<UserFnSig> {
        // Check if arg count is in range [required..=total] and types match
        let matches = |sig: &UserFnSig, exact: bool| -> bool {
            let n = arg_types.len();
            if n < sig.required || n > sig.params.len() {
                return false;
            }
            sig.params[..n].iter().zip(arg_types).all(|(p, a)| {
                if exact {
                    p == a
                } else {
                    p == a || (p.is_numeric() && a.is_numeric()) ||
                    matches!((p, a), (PdcType::Array(_), PdcType::Array(_)))
                }
            })
        };

        // Exact match first
        for sig in &overloads.sigs {
            if matches(sig, true) {
                return Some(sig.clone());
            }
        }
        // Compatible match (numeric coercion)
        for sig in &overloads.sigs {
            if matches(sig, false) {
                return Some(sig.clone());
            }
        }
        None
    }

    fn check_compatible(
        &self,
        from: &PdcType,
        to: &PdcType,
        span: super::span::Span,
    ) -> Result<(), PdcError> {
        if from == to {
            return Ok(());
        }
        if from.is_numeric() && to.is_numeric() {
            return Ok(());
        }
        // Arrays with compatible element types
        if let (PdcType::Array(_), PdcType::Array(_)) = (from, to) {
            return Ok(());
        }
        Err(PdcError::Type {
            span,
            message: format!("type mismatch: expected {to}, got {from}"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::lexer;
    use super::super::parser;
    use super::super::span::IdAlloc;
    use super::super::ast::Program;

    const PRELUDE: &str = "builtin const width: f32\nbuiltin const height: f32\n";

    fn check_ok(source: &str) {
        let full = format!("{PRELUDE}{source}");
        let tokens = lexer::lex(&full).expect("lex failed");
        let mut ids = IdAlloc::new();
        let stmts = parser::parse(tokens, &mut ids).expect("parse failed");
        let program = Program { modules: vec![], stmts };
        let mut checker = TypeChecker::new();
        checker.check_program(&program).expect("type check failed");
    }

    fn check_err(source: &str, expected_substr: &str) {
        let full = format!("{PRELUDE}{source}");
        let tokens = lexer::lex(&full).expect("lex failed");
        let mut ids = IdAlloc::new();
        let stmts = parser::parse(tokens, &mut ids).expect("parse failed");
        let program = Program { modules: vec![], stmts };
        let mut checker = TypeChecker::new();
        match checker.check_program(&program) {
            Err(e) => {
                let msg = e.to_string();
                assert!(
                    msg.contains(expected_substr),
                    "error '{}' does not contain '{}'",
                    msg,
                    expected_substr,
                );
            }
            Ok(_) => panic!("expected error containing '{}'", expected_substr),
        }
    }

    // ── Happy path: variable declarations ──

    #[test]
    fn var_decl_explicit_type() {
        check_ok("var x: f64 = 1.0");
    }

    #[test]
    fn var_decl_inferred_type() {
        check_ok("var x = 42");
    }

    #[test]
    fn const_decl_explicit_type() {
        check_ok("const x: i32 = 10");
    }

    #[test]
    fn const_decl_inferred_type() {
        check_ok("const x = 3.14");
    }

    #[test]
    fn const_decl_bool() {
        check_ok("const flag = true");
    }

    // ── Happy path: function definitions and calls ──

    #[test]
    fn fn_def_and_call() {
        check_ok(
            r#"
            fn add(a: f64, b: f64) -> f64 {
                return a + b
            }
            var result = add(1.0, 2.0)
            "#,
        );
    }

    #[test]
    fn fn_void_return() {
        check_ok(
            r#"
            fn do_nothing() -> void {
                var x = 1
            }
            do_nothing()
            "#,
        );
    }

    #[test]
    fn fn_empty_body() {
        check_ok(
            r#"
            fn noop() -> void {
            }
            noop()
            "#,
        );
    }

    #[test]
    fn fn_recursive() {
        check_ok(
            r#"
            fn countdown(n: i32) -> i32 {
                if n <= 0 {
                    return 0
                }
                return countdown(n - 1)
            }
            var result = countdown(10)
            "#,
        );
    }

    #[test]
    fn fn_default_params() {
        check_ok(
            r#"
            fn greet(x: f64, y: f64 = 10.0) -> f64 {
                return x + y
            }
            var a = greet(1.0)
            var b = greet(1.0, 2.0)
            "#,
        );
    }

    // ── Happy path: struct definitions and construction ──

    #[test]
    fn struct_def_and_construct() {
        check_ok(
            r#"
            struct Point {
                x: f64,
                y: f64,
            }
            var p = Point(x: 1.0, y: 2.0)
            "#,
        );
    }

    #[test]
    fn struct_field_access() {
        check_ok(
            r#"
            struct Point {
                x: f64,
                y: f64,
            }
            var p = Point(x: 1.0, y: 2.0)
            var xval = p.x
            "#,
        );
    }

    // ── Happy path: enum definitions ──

    #[test]
    fn enum_def() {
        check_ok(
            r#"
            enum Color {
                Red,
                Green,
                Blue,
            }
            var c = Color.Red
            "#,
        );
    }

    #[test]
    fn enum_match_exhaustive() {
        check_ok(
            r#"
            enum Dir {
                Up,
                Down,
            }
            var d = Dir.Up
            match d {
                .Up => { var x = 1 }
                .Down => { var x = 2 }
            }
            "#,
        );
    }

    // ── Happy path: type aliases ──

    #[test]
    fn type_alias() {
        check_ok(
            r#"
            type Real = f64
            var x: Real = 1.0
            "#,
        );
    }

    #[test]
    fn type_alias_cast() {
        check_ok(
            r#"
            type Real = f64
            var x = Real(42)
            "#,
        );
    }

    // ── Happy path: const vs var semantics ──

    #[test]
    fn var_can_be_reassigned() {
        check_ok(
            r#"
            var x: i32 = 1
            x = 2
            "#,
        );
    }

    // ── Happy path: binary operations ──

    #[test]
    fn binary_op_matching_types() {
        check_ok("var x = 1 + 2");
    }

    #[test]
    fn binary_op_float() {
        check_ok("var x = 1.0 * 2.0");
    }

    #[test]
    fn binary_comparison() {
        check_ok("var x = 1 < 2");
    }

    #[test]
    fn binary_logical() {
        check_ok("var x = true && false");
    }

    #[test]
    fn binary_numeric_widening() {
        // i32 + f64 should unify to f64
        check_ok("var x = 1 + 2.0");
    }

    // ── Happy path: nested scopes ──

    #[test]
    fn nested_scope_no_leak() {
        // Variable defined in inner block should not be visible in outer scope.
        // The inner block is an if-body, which creates a scope.
        check_ok(
            r#"
            var x: i32 = 1
            if true {
                var y: i32 = 2
            }
            x = 3
            "#,
        );
    }

    // ── Happy path: for loop ──

    #[test]
    fn for_loop() {
        check_ok(
            r#"
            var sum: i32 = 0
            for i in 0..10 {
                sum = sum + i
            }
            "#,
        );
    }

    // ── Happy path: while loop ──

    #[test]
    fn while_loop() {
        check_ok(
            r#"
            var i: i32 = 0
            while i < 10 {
                i = i + 1
            }
            "#,
        );
    }

    // ── Happy path: ternary expression ──

    #[test]
    fn ternary_expr() {
        check_ok("var x = true ? 1 : 2");
    }

    // ── Happy path: string operations ──

    #[test]
    fn string_concat() {
        check_ok(r#"var s = "hello" + " world""#);
    }

    // ── Happy path: tuple ──

    #[test]
    fn tuple_construct_and_index() {
        check_ok(
            r#"
            var t = (1.0, 2.0, 3.0)
            var x = t.0
            "#,
        );
    }

    // ── Error cases: undefined variable ──

    #[test]
    fn err_undefined_variable() {
        check_err("var x = y", "undefined variable 'y'");
    }

    // ── Error cases: const reassignment ──

    #[test]
    fn err_const_reassign() {
        check_err(
            r#"
            const x: i32 = 1
            x = 2
            "#,
            "cannot assign to const variable 'x'",
        );
    }

    // ── Error cases: builtin const reassignment ──

    #[test]
    fn err_builtin_const_reassign() {
        check_err("width = 100.0", "cannot assign to const variable 'width'");
    }

    // ── Error cases: wrong number of arguments ──

    #[test]
    fn err_wrong_arg_count_builtin() {
        check_err("sin(1.0, 2.0)", "expects 1 arguments, got 2");
    }

    #[test]
    fn err_wrong_arg_count_user_fn() {
        check_err(
            r#"
            fn add(a: f64, b: f64) -> f64 {
                return a + b
            }
            var x = add(1.0)
            "#,
            "no matching overload for 'add'",
        );
    }

    // ── Error cases: undefined function ──

    #[test]
    fn err_undefined_function() {
        check_err("var x = nonexistent(1)", "undefined function 'nonexistent'");
    }

    // ── Error cases: non-existent struct field ──

    #[test]
    fn err_struct_no_such_field_access() {
        check_err(
            r#"
            struct Point {
                x: f64,
                y: f64,
            }
            var p = Point(x: 1.0, y: 2.0)
            var z = p.z
            "#,
            "has no field 'z'",
        );
    }

    #[test]
    fn err_struct_no_such_field_construct() {
        check_err(
            r#"
            struct Point {
                x: f64,
                y: f64,
            }
            var p = Point(x: 1.0, z: 2.0)
            "#,
            "has no field 'z'",
        );
    }

    // ── Error cases: type mismatch in binary ops ──

    #[test]
    fn err_logical_op_non_bool() {
        check_err("var x = 1 && 2", "logical operator requires bool, got i32 and i32");
    }

    #[test]
    fn err_bitwise_on_float() {
        check_err(
            "var x = 1.0 & 2.0",
            "bitwise operator requires integer types",
        );
    }

    // ── Error cases: if condition not bool ──

    #[test]
    fn err_if_condition_not_bool() {
        check_err(
            r#"
            if 42 {
                var x = 1
            }
            "#,
            "if condition must be bool",
        );
    }

    // ── Error cases: while condition not bool ──

    #[test]
    fn err_while_condition_not_bool() {
        check_err(
            r#"
            while 1 {
                var x = 1
            }
            "#,
            "while condition must be bool",
        );
    }

    // ── Error cases: for range not integer ──

    #[test]
    fn err_for_range_not_integer() {
        check_err(
            r#"
            for i in 0.0..10.0 {
                var x = i
            }
            "#,
            "for range start must be integer",
        );
    }

    // ── Error cases: assign to undefined variable ──

    #[test]
    fn err_assign_undefined_variable() {
        check_err("x = 5", "undefined variable 'x'");
    }

    // ── Error cases: type mismatch on assignment ──

    #[test]
    fn err_assign_type_mismatch() {
        check_err(
            r#"
            var x: bool = true
            x = 42
            "#,
            "type mismatch",
        );
    }

    // ── Error cases: negate non-numeric ──

    #[test]
    fn err_negate_bool() {
        check_err("var x = -true", "cannot negate type bool");
    }

    // ── Error cases: not on non-bool ──

    #[test]
    fn err_not_on_int() {
        check_err("var x = !42", "cannot apply ! to type i32");
    }

    // ── Error cases: field access on non-struct ──

    #[test]
    fn err_field_access_on_int() {
        check_err(
            r#"
            var x: i32 = 1
            var y = x.foo
            "#,
            "cannot access field 'foo' on type i32",
        );
    }

    // ── Error cases: index non-array ──

    #[test]
    fn err_index_non_array() {
        check_err(
            r#"
            var x: i32 = 1
            var y = x[0]
            "#,
            "cannot index type i32",
        );
    }

    // ── Error cases: enum non-exhaustive match ──

    #[test]
    fn err_match_non_exhaustive() {
        check_err(
            r#"
            enum Dir {
                Up,
                Down,
            }
            var d = Dir.Up
            match d {
                .Up => { var x = 1 }
            }
            "#,
            "non-exhaustive match",
        );
    }

    // ── Edge cases: shadowing ──

    #[test]
    fn shadowing_inner_scope() {
        check_ok(
            r#"
            var x: i32 = 1
            if true {
                var x: f64 = 2.0
            }
            x = 3
            "#,
        );
    }

    // ── Edge cases: inner scope variable not visible outside ──

    #[test]
    fn err_inner_scope_var_not_visible() {
        check_err(
            r#"
            if true {
                var inner: i32 = 1
            }
            inner = 2
            "#,
            "undefined variable 'inner'",
        );
    }

    // ── Edge cases: for loop variable is const by default ──

    #[test]
    fn err_for_loop_var_const() {
        check_err(
            r#"
            for i in 0..10 {
                i = 5
            }
            "#,
            "cannot assign to const variable 'i'",
        );
    }

    // ── Edge cases: type cast ──

    #[test]
    fn type_cast() {
        check_ok("var x = f64(42)");
    }

    #[test]
    fn err_type_cast_wrong_arg_count() {
        check_err("var x = f64(1, 2)", "type cast f64() expects 1 argument, got 2");
    }

    // ── Edge cases: multiple functions (overloading) ──

    #[test]
    fn fn_overloading() {
        check_ok(
            r#"
            fn double(x: f64) -> f64 {
                return x * 2.0
            }
            fn double(x: i32) -> i32 {
                return x * 2
            }
            var a = double(1.0)
            var b = double(1)
            "#,
        );
    }

    // ── Edge cases: enum variant with fields ──

    #[test]
    fn enum_variant_with_fields() {
        check_ok(
            r#"
            enum Shape {
                Circle(r: f64),
                Rect(w: f64, h: f64),
            }
            var s = Shape.Circle(10.0)
            "#,
        );
    }

    // ── Edge cases: match with wildcard ──

    #[test]
    fn match_wildcard() {
        check_ok(
            r#"
            enum Dir {
                Up,
                Down,
                Left,
                Right,
            }
            var d = Dir.Up
            match d {
                .Up => { var x = 1 }
                _ => { var x = 0 }
            }
            "#,
        );
    }

    // ── Edge cases: string comparison ──

    #[test]
    fn string_equality() {
        check_ok(r#"var x = "a" == "b""#);
    }

    // ── Edge cases: ternary type mismatch ──

    #[test]
    fn err_ternary_type_mismatch() {
        check_err(
            r#"var x = true ? 1 : "hello""#,
            "ternary branches must have compatible types",
        );
    }

    // ── Edge cases: ternary condition not bool ──

    #[test]
    fn err_ternary_condition_not_bool() {
        check_err(
            "var x = 42 ? 1 : 2",
            "ternary condition must be bool",
        );
    }

    // ── Edge cases: tuple destructure ──

    #[test]
    fn tuple_destructure() {
        check_ok(
            r#"
            const (a, b) = (1.0, 2.0)
            "#,
        );
    }

    #[test]
    fn err_tuple_destructure_wrong_count() {
        check_err(
            r#"
            const (a, b, c) = (1.0, 2.0)
            "#,
            "tuple destructure",
        );
    }

    // ── Edge cases: array operations ──

    #[test]
    fn array_construct_and_methods() {
        check_ok(
            r#"
            var arr = Array<f64>()
            arr.push(1.0)
            var n = arr.len()
            "#,
        );
    }

    // ── Edge cases: match on non-enum ──

    #[test]
    fn err_match_non_enum() {
        check_err(
            r#"
            var x: i32 = 1
            match x {
                _ => { var y = 1 }
            }
            "#,
            "cannot match non-enum type",
        );
    }
}
