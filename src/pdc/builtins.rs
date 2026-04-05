use super::ast::PdcType;

/// How a builtin maps to a runtime function call in codegen.
#[derive(Debug, Clone)]
pub enum CodegenAction {
    /// Simple runtime call with a fixed symbol name.
    RuntimeCall {
        symbol: &'static str,
        takes_ctx: bool,
        /// Per-overload symbol overrides. If `sig_symbols[i]` is `Some(s)`,
        /// overload `i` uses symbol `s` instead of the default.
        sig_symbols: Vec<Option<&'static str>>,
    },
    /// Inline math intrinsic: emit native hardware instruction when possible,
    /// falling back to a runtime call for non-float types.
    MathIntrinsic {
        fallback_symbol: &'static str,
    },
    /// String-argument constructor: args are string literals passed as (ptr, len) pairs.
    StringArgCall {
        symbol: &'static str,
    },
    /// Assert intrinsic: type-dispatched runtime call.
    AssertIntrinsic {
        variant: AssertVariant,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssertVariant {
    Eq,
    Near,
    True,
}

/// Specification for a function parameter — concrete type or generic constraint.
#[derive(Debug, Clone)]
pub enum ParamSpec {
    /// A specific concrete type.
    Concrete(PdcType),
    /// Must match the receiver's element type (for generic methods like array.push).
    SelfElement,
    /// A function reference with a specific parameter count.
    FnRefShape { param_count: usize },
}

/// Specification for a function's return type.
#[derive(Debug, Clone)]
pub enum ReturnSpec {
    /// A specific concrete type.
    Concrete(PdcType),
    /// Same as the receiver's element type (e.g. array.get → T).
    SelfElement,
    /// Array of the fn_ref argument's return type (e.g. array.map(fn(T)->U) → Array<U>).
    MappedElement,
    /// BufferHandle wrapping the kernel fn's return type (render(fn) → Buffer<T>).
    BufferOfKernelReturn,
    /// Slice of the receiver's element type (array.slice → Slice<T>).
    SliceOfSelfElement,
    /// Same as the second argument's type (e.g. render(kernel, buffer) → buffer's type).
    SameAsArg(usize),
}

/// Whether a builtin supports method-call syntax (UFCS).
#[derive(Debug, Clone)]
pub enum ReceiverKind {
    /// First param is the receiver; any call site with a matching first-arg type
    /// can use `obj.method(rest...)` syntax.
    FirstParam,
}

/// A single overload signature for a builtin function.
#[derive(Debug, Clone)]
pub struct BuiltinSig {
    pub param_names: Vec<&'static str>,
    pub params: Vec<ParamSpec>,
    pub ret: ReturnSpec,
}

/// Full definition of a builtin function.
#[derive(Debug, Clone)]
pub struct BuiltinDef {
    /// PDC-facing name (e.g. "fill", "sin", "render").
    pub name: &'static str,
    /// Overload signatures, tried in order. Most builtins have exactly one.
    pub sigs: Vec<BuiltinSig>,
    /// If set, this builtin can be called with method syntax on the first param.
    pub method_receiver: Option<ReceiverKind>,
    /// How codegen should emit this call.
    pub codegen: CodegenAction,
}

impl CodegenAction {
    /// Whether this action requires passing the context pointer as the first argument.
    pub fn takes_ctx(&self) -> bool {
        match self {
            CodegenAction::RuntimeCall { takes_ctx, .. } => *takes_ctx,
            CodegenAction::MathIntrinsic { .. } => false,
            CodegenAction::StringArgCall { .. } => true,
            CodegenAction::AssertIntrinsic { .. } => true,
        }
    }

    /// The default runtime symbol for this action (if applicable).
    pub fn symbol(&self) -> Option<&'static str> {
        match self {
            CodegenAction::RuntimeCall { symbol, .. } => Some(symbol),
            CodegenAction::MathIntrinsic { fallback_symbol } => Some(fallback_symbol),
            CodegenAction::StringArgCall { symbol } => Some(symbol),
            CodegenAction::AssertIntrinsic { .. } => None,
        }
    }

    /// The runtime symbol to use for a specific overload index.
    pub fn symbol_for_sig(&self, sig_index: usize) -> Option<&'static str> {
        match self {
            CodegenAction::RuntimeCall { symbol, sig_symbols, .. } => {
                sig_symbols.get(sig_index)
                    .and_then(|o| *o)
                    .or(Some(*symbol))
            }
            _ => self.symbol(),
        }
    }
}

// ---------------------------------------------------------------------------
// Convenience helpers for building signatures
// ---------------------------------------------------------------------------

fn sig(names: &[&'static str], params: &[PdcType], ret: PdcType) -> BuiltinSig {
    BuiltinSig {
        param_names: names.to_vec(),
        params: params.iter().map(|t| ParamSpec::Concrete(t.clone())).collect(),
        ret: ReturnSpec::Concrete(ret),
    }
}

fn runtime(symbol: &'static str, takes_ctx: bool) -> CodegenAction {
    CodegenAction::RuntimeCall {
        symbol,
        takes_ctx,
        sig_symbols: vec![],
    }
}

fn runtime_overloaded(symbol: &'static str, takes_ctx: bool, sig_symbols: Vec<Option<&'static str>>) -> CodegenAction {
    CodegenAction::RuntimeCall {
        symbol,
        takes_ctx,
        sig_symbols,
    }
}

fn ufcs() -> Option<ReceiverKind> {
    Some(ReceiverKind::FirstParam)
}

// ---------------------------------------------------------------------------
// The registry
// ---------------------------------------------------------------------------

/// Returns the complete list of builtin function definitions.
pub fn builtin_registry() -> Vec<BuiltinDef> {
    let path = PdcType::PathHandle;
    let f32_ = PdcType::F32;
    let f64_ = PdcType::F64;
    let i32_ = PdcType::I32;
    let u32_ = PdcType::U32;
    let bool_ = PdcType::Bool;
    let void = PdcType::Void;
    let str_ = PdcType::Str;
    let buf_any = PdcType::BufferHandle(Box::new(PdcType::Unknown));
    let kernel = PdcType::KernelHandle;
    let scene = PdcType::SceneHandle;
    let key_ty = PdcType::Enum("Key".into());
    let handler_ty = PdcType::FnRef { params: vec![], ret: Box::new(PdcType::Void) };

    vec![
        // -----------------------------------------------------------------
        // Path construction
        // -----------------------------------------------------------------
        BuiltinDef {
            name: "Path",
            sigs: vec![sig(&[], &[], path.clone())],
            method_receiver: None,
            codegen: runtime("pdc_path", true),
        },
        BuiltinDef {
            name: "move_to",
            sigs: vec![sig(&["path", "x", "y"], &[path.clone(), f64_.clone(), f64_.clone()], void.clone())],
            method_receiver: ufcs(),
            codegen: runtime("pdc_move_to", true),
        },
        BuiltinDef {
            name: "line_to",
            sigs: vec![sig(&["path", "x", "y"], &[path.clone(), f64_.clone(), f64_.clone()], void.clone())],
            method_receiver: ufcs(),
            codegen: runtime("pdc_line_to", true),
        },
        BuiltinDef {
            name: "quad_to",
            sigs: vec![sig(&["path", "cx", "cy", "x", "y"], &[path.clone(), f64_.clone(), f64_.clone(), f64_.clone(), f64_.clone()], void.clone())],
            method_receiver: ufcs(),
            codegen: runtime("pdc_quad_to", true),
        },
        BuiltinDef {
            name: "cubic_to",
            sigs: vec![sig(&["path", "c1x", "c1y", "c2x", "c2y", "x", "y"], &[path.clone(), f64_.clone(), f64_.clone(), f64_.clone(), f64_.clone(), f64_.clone(), f64_.clone()], void.clone())],
            method_receiver: ufcs(),
            codegen: runtime("pdc_cubic_to", true),
        },
        BuiltinDef {
            name: "close",
            sigs: vec![sig(&["path"], &[path.clone()], void.clone())],
            method_receiver: ufcs(),
            codegen: runtime("pdc_close", true),
        },

        // -----------------------------------------------------------------
        // Path rendering (with overloads for styled variants)
        // -----------------------------------------------------------------
        BuiltinDef {
            name: "fill",
            sigs: vec![
                sig(&["path", "color"], &[path.clone(), u32_.clone()], void.clone()),
                sig(&["path", "color", "rule"], &[path.clone(), u32_.clone(), i32_.clone()], void.clone()),
            ],
            method_receiver: ufcs(),
            codegen: runtime_overloaded("pdc_fill", true, vec![None, Some("pdc_fill_styled")]),
        },
        BuiltinDef {
            name: "stroke",
            sigs: vec![
                sig(&["path", "width", "color"], &[path.clone(), f32_.clone(), u32_.clone()], void.clone()),
                sig(&["path", "width", "color", "cap", "join"], &[path.clone(), f32_.clone(), u32_.clone(), i32_.clone(), i32_.clone()], void.clone()),
            ],
            method_receiver: ufcs(),
            codegen: runtime_overloaded("pdc_stroke", true, vec![None, Some("pdc_stroke_styled")]),
        },

        // -----------------------------------------------------------------
        // Array methods (callable as both functions and methods)
        // -----------------------------------------------------------------
        BuiltinDef {
            name: "push",
            sigs: vec![BuiltinSig {
                param_names: vec!["arr", "val"],
                params: vec![ParamSpec::Concrete(PdcType::Array(Box::new(PdcType::F64))), ParamSpec::SelfElement],
                ret: ReturnSpec::Concrete(void.clone()),
            }],
            method_receiver: ufcs(),
            codegen: runtime("pdc_array_push", true),
        },
        BuiltinDef {
            name: "len",
            sigs: vec![BuiltinSig {
                param_names: vec!["arr"],
                params: vec![ParamSpec::Concrete(PdcType::Array(Box::new(PdcType::F64)))],
                ret: ReturnSpec::Concrete(i32_.clone()),
            }],
            method_receiver: ufcs(),
            codegen: runtime("pdc_array_len", true),
        },
        BuiltinDef {
            name: "get",
            sigs: vec![BuiltinSig {
                param_names: vec!["arr", "index"],
                params: vec![ParamSpec::Concrete(PdcType::Array(Box::new(PdcType::F64))), ParamSpec::Concrete(i32_.clone())],
                ret: ReturnSpec::SelfElement,
            }],
            method_receiver: ufcs(),
            codegen: runtime("pdc_array_get", true),
        },
        BuiltinDef {
            name: "set",
            sigs: vec![BuiltinSig {
                param_names: vec!["arr", "index", "val"],
                params: vec![ParamSpec::Concrete(PdcType::Array(Box::new(PdcType::F64))), ParamSpec::Concrete(i32_.clone()), ParamSpec::SelfElement],
                ret: ReturnSpec::Concrete(void.clone()),
            }],
            method_receiver: ufcs(),
            codegen: runtime("pdc_array_set", true),
        },
        BuiltinDef {
            name: "map",
            sigs: vec![BuiltinSig {
                param_names: vec!["arr", "fn_ref"],
                params: vec![ParamSpec::Concrete(PdcType::Array(Box::new(PdcType::F64))), ParamSpec::FnRefShape { param_count: 1 }],
                ret: ReturnSpec::MappedElement,
            }],
            method_receiver: ufcs(),
            codegen: runtime("pdc_array_map", true),
        },

        // -----------------------------------------------------------------
        // String methods (method-only in practice)
        // -----------------------------------------------------------------
        BuiltinDef {
            name: "str_len",
            sigs: vec![sig(&["s"], &[str_.clone()], i32_.clone())],
            method_receiver: ufcs(),
            codegen: runtime("pdc_string_len", true),
        },
        BuiltinDef {
            name: "str_slice",
            sigs: vec![sig(&["s", "start", "end"], &[str_.clone(), i32_.clone(), i32_.clone()], str_.clone())],
            method_receiver: ufcs(),
            codegen: runtime("pdc_string_slice", true),
        },
        BuiltinDef {
            name: "str_concat",
            sigs: vec![sig(&["s", "other"], &[str_.clone(), str_.clone()], str_.clone())],
            method_receiver: ufcs(),
            codegen: runtime("pdc_string_concat", true),
        },
        BuiltinDef {
            name: "str_char_at",
            sigs: vec![sig(&["s", "index"], &[str_.clone(), i32_.clone()], str_.clone())],
            method_receiver: ufcs(),
            codegen: runtime("pdc_string_char_at", true),
        },

        // -----------------------------------------------------------------
        // Buffer / Kernel / Pipeline operations
        // -----------------------------------------------------------------
        BuiltinDef {
            name: "swap",
            sigs: vec![sig(&["a", "b"], &[buf_any.clone(), buf_any.clone()], void.clone())],
            method_receiver: None,
            codegen: runtime("pdc_swap_buffers", true),
        },
        BuiltinDef {
            name: "run",
            sigs: vec![sig(&["kernel"], &[kernel.clone()], void.clone())],
            method_receiver: ufcs(),
            codegen: runtime("pdc_run_kernel", true),
        },
        BuiltinDef {
            name: "render",
            sigs: vec![
                // Pipeline: render(kernel, buffer, clear) -> bool
                sig(&["kernel", "buffer", "clear"], &[kernel.clone(), buf_any.clone(), bool_.clone()], bool_.clone()),
                // PDC pixel kernel: render(kernel_fn) -> Buffer<T>
                BuiltinSig {
                    param_names: vec!["kernel"],
                    params: vec![ParamSpec::FnRefShape { param_count: 4 }],
                    ret: ReturnSpec::BufferOfKernelReturn,
                },
                // PDC pixel kernel with target buffer: render(kernel_fn, buffer) -> Buffer<T>
                BuiltinSig {
                    param_names: vec!["kernel", "buffer"],
                    params: vec![ParamSpec::FnRefShape { param_count: 4 }, ParamSpec::Concrete(buf_any.clone())],
                    ret: ReturnSpec::SameAsArg(1),
                },
            ],
            method_receiver: ufcs(),
            codegen: runtime_overloaded("pdc_render_kernel", true, vec![
                None,
                Some("pdc_render_pdc_kernel"),
                Some("pdc_render_pdc_kernel_buf"),
            ]),
        },
        BuiltinDef {
            name: "display",
            sigs: vec![
                // display() — display current buffer
                sig(&[], &[], void.clone()),
                // display(buffer) — display specific buffer (method syntax: buf.display())
                sig(&["buffer"], &[buf_any.clone()], void.clone()),
            ],
            method_receiver: ufcs(),
            codegen: runtime_overloaded("pdc_display", true, vec![None, Some("pdc_display_buffer")]),
        },

        // -----------------------------------------------------------------
        // Resource loading (string-argument constructors)
        // -----------------------------------------------------------------
        BuiltinDef {
            name: "Texture",
            sigs: vec![sig(&["name", "path"], &[str_.clone(), str_.clone()], PdcType::TextureHandle)],
            method_receiver: None,
            codegen: CodegenAction::StringArgCall { symbol: "pdc_load_texture" },
        },
        BuiltinDef {
            name: "Scene",
            sigs: vec![sig(&["name", "path"], &[str_.clone(), str_.clone()], scene.clone())],
            method_receiver: None,
            codegen: CodegenAction::StringArgCall { symbol: "pdc_load_scene" },
        },

        // -----------------------------------------------------------------
        // Scene methods
        // -----------------------------------------------------------------
        BuiltinDef {
            name: "run_scene",
            sigs: vec![sig(&["scene"], &[scene.clone()], void.clone())],
            method_receiver: ufcs(),
            codegen: runtime("pdc_run_scene", true),
        },

        // -----------------------------------------------------------------
        // Math — single argument (all f64 -> f64, no context)
        // -----------------------------------------------------------------
        BuiltinDef { name: "sin", sigs: vec![sig(&["x"], &[f64_.clone()], f64_.clone())], method_receiver: None, codegen: runtime("pdc_sin", false) },
        BuiltinDef { name: "cos", sigs: vec![sig(&["x"], &[f64_.clone()], f64_.clone())], method_receiver: None, codegen: runtime("pdc_cos", false) },
        BuiltinDef { name: "tan", sigs: vec![sig(&["x"], &[f64_.clone()], f64_.clone())], method_receiver: None, codegen: runtime("pdc_tan", false) },
        BuiltinDef { name: "asin", sigs: vec![sig(&["x"], &[f64_.clone()], f64_.clone())], method_receiver: None, codegen: runtime("pdc_asin", false) },
        BuiltinDef { name: "acos", sigs: vec![sig(&["x"], &[f64_.clone()], f64_.clone())], method_receiver: None, codegen: runtime("pdc_acos", false) },
        BuiltinDef { name: "atan", sigs: vec![sig(&["x"], &[f64_.clone()], f64_.clone())], method_receiver: None, codegen: runtime("pdc_atan", false) },
        BuiltinDef { name: "sqrt", sigs: vec![sig(&["x"], &[f64_.clone()], f64_.clone())], method_receiver: None, codegen: runtime("pdc_sqrt", false) },
        BuiltinDef { name: "abs", sigs: vec![sig(&["x"], &[f64_.clone()], f64_.clone())], method_receiver: None, codegen: runtime("pdc_abs", false) },
        BuiltinDef { name: "floor", sigs: vec![sig(&["x"], &[f64_.clone()], f64_.clone())], method_receiver: None, codegen: runtime("pdc_floor", false) },
        BuiltinDef { name: "ceil", sigs: vec![sig(&["x"], &[f64_.clone()], f64_.clone())], method_receiver: None, codegen: runtime("pdc_ceil", false) },
        BuiltinDef { name: "round", sigs: vec![sig(&["x"], &[f64_.clone()], f64_.clone())], method_receiver: None, codegen: runtime("pdc_round", false) },
        BuiltinDef { name: "exp", sigs: vec![sig(&["x"], &[f64_.clone()], f64_.clone())], method_receiver: None, codegen: runtime("pdc_exp", false) },
        BuiltinDef { name: "ln", sigs: vec![sig(&["x"], &[f64_.clone()], f64_.clone())], method_receiver: None, codegen: runtime("pdc_ln", false) },
        BuiltinDef { name: "log2", sigs: vec![sig(&["x"], &[f64_.clone()], f64_.clone())], method_receiver: None, codegen: runtime("pdc_log2", false) },
        BuiltinDef { name: "log10", sigs: vec![sig(&["x"], &[f64_.clone()], f64_.clone())], method_receiver: None, codegen: runtime("pdc_log10", false) },
        BuiltinDef { name: "fract", sigs: vec![sig(&["x"], &[f64_.clone()], f64_.clone())], method_receiver: None, codegen: runtime("pdc_fract", false) },
        BuiltinDef { name: "exp2", sigs: vec![sig(&["x"], &[f64_.clone()], f64_.clone())], method_receiver: None, codegen: runtime("pdc_exp2", false) },

        // -----------------------------------------------------------------
        // Math — two arguments
        // -----------------------------------------------------------------
        BuiltinDef { name: "min", sigs: vec![sig(&["a", "b"], &[f64_.clone(), f64_.clone()], f64_.clone())], method_receiver: None, codegen: CodegenAction::MathIntrinsic { fallback_symbol: "pdc_min" } },
        BuiltinDef { name: "max", sigs: vec![sig(&["a", "b"], &[f64_.clone(), f64_.clone()], f64_.clone())], method_receiver: None, codegen: CodegenAction::MathIntrinsic { fallback_symbol: "pdc_max" } },
        BuiltinDef { name: "pow", sigs: vec![sig(&["a", "b"], &[f64_.clone(), f64_.clone()], f64_.clone())], method_receiver: None, codegen: CodegenAction::MathIntrinsic { fallback_symbol: "pdc_pow" } },
        BuiltinDef { name: "atan2", sigs: vec![sig(&["a", "b"], &[f64_.clone(), f64_.clone()], f64_.clone())], method_receiver: None, codegen: runtime("pdc_atan2", false) },
        BuiltinDef { name: "fmod", sigs: vec![sig(&["a", "b"], &[f64_.clone(), f64_.clone()], f64_.clone())], method_receiver: None, codegen: runtime("pdc_fmod", false) },

        // -----------------------------------------------------------------
        // Event handlers — keyboard
        // -----------------------------------------------------------------
        BuiltinDef { name: "set_keypress", sigs: vec![sig(&["key", "handler"], &[key_ty.clone(), handler_ty.clone()], void.clone())], method_receiver: None, codegen: runtime("pdc_set_keypress", true) },
        BuiltinDef { name: "set_keydown", sigs: vec![sig(&["key", "handler"], &[key_ty.clone(), handler_ty.clone()], void.clone())], method_receiver: None, codegen: runtime("pdc_set_keydown", true) },
        BuiltinDef { name: "set_keyup", sigs: vec![sig(&["key", "handler"], &[key_ty.clone(), handler_ty.clone()], void.clone())], method_receiver: None, codegen: runtime("pdc_set_keyup", true) },
        BuiltinDef { name: "clear_keypress", sigs: vec![sig(&["key"], &[key_ty.clone()], void.clone())], method_receiver: None, codegen: runtime("pdc_clear_keypress", true) },
        BuiltinDef { name: "clear_keydown", sigs: vec![sig(&["key"], &[key_ty.clone()], void.clone())], method_receiver: None, codegen: runtime("pdc_clear_keydown", true) },
        BuiltinDef { name: "clear_keyup", sigs: vec![sig(&["key"], &[key_ty.clone()], void.clone())], method_receiver: None, codegen: runtime("pdc_clear_keyup", true) },

        // -----------------------------------------------------------------
        // Event handlers — mouse
        // -----------------------------------------------------------------
        BuiltinDef { name: "set_mousedown", sigs: vec![sig(&["handler"], &[handler_ty.clone()], void.clone())], method_receiver: None, codegen: runtime("pdc_set_mousedown", true) },
        BuiltinDef { name: "set_mouseup", sigs: vec![sig(&["handler"], &[handler_ty.clone()], void.clone())], method_receiver: None, codegen: runtime("pdc_set_mouseup", true) },
        BuiltinDef { name: "set_click", sigs: vec![sig(&["handler"], &[handler_ty.clone()], void.clone())], method_receiver: None, codegen: runtime("pdc_set_click", true) },
        BuiltinDef { name: "clear_mousedown", sigs: vec![sig(&[], &[], void.clone())], method_receiver: None, codegen: runtime("pdc_clear_mousedown", true) },
        BuiltinDef { name: "clear_mouseup", sigs: vec![sig(&[], &[], void.clone())], method_receiver: None, codegen: runtime("pdc_clear_mouseup", true) },
        BuiltinDef { name: "clear_click", sigs: vec![sig(&[], &[], void.clone())], method_receiver: None, codegen: runtime("pdc_clear_click", true) },

        // -----------------------------------------------------------------
        // Assertions (test intrinsics)
        // -----------------------------------------------------------------
        BuiltinDef {
            name: "assert_eq",
            sigs: vec![sig(&["a", "b"], &[f64_.clone(), f64_.clone()], void.clone())],
            method_receiver: None,
            codegen: CodegenAction::AssertIntrinsic { variant: AssertVariant::Eq },
        },
        BuiltinDef {
            name: "assert_near",
            sigs: vec![sig(&["a", "b", "epsilon"], &[f64_.clone(), f64_.clone(), f64_.clone()], void.clone())],
            method_receiver: None,
            codegen: CodegenAction::AssertIntrinsic { variant: AssertVariant::Near },
        },
        BuiltinDef {
            name: "assert_true",
            sigs: vec![sig(&["val"], &[bool_.clone()], void.clone())],
            method_receiver: None,
            codegen: CodegenAction::AssertIntrinsic { variant: AssertVariant::True },
        },
    ]
}

/// Build a lookup map from builtin name to its definition.
pub fn builtin_map() -> std::collections::HashMap<String, BuiltinDef> {
    builtin_registry().into_iter().map(|def| (def.name.to_string(), def)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_has_no_duplicate_names() {
        let registry = builtin_registry();
        let mut seen = std::collections::HashSet::new();
        for def in &registry {
            assert!(seen.insert(def.name), "duplicate builtin name: {}", def.name);
        }
    }

    #[test]
    fn all_sigs_have_matching_param_names_and_params() {
        for def in builtin_registry() {
            for (i, sig) in def.sigs.iter().enumerate() {
                assert_eq!(
                    sig.param_names.len(), sig.params.len(),
                    "builtin '{}' sig[{}]: param_names.len() ({}) != params.len() ({})",
                    def.name, i, sig.param_names.len(), sig.params.len(),
                );
            }
        }
    }

    #[test]
    fn sig_symbols_len_matches_sigs_when_present() {
        for def in builtin_registry() {
            if let CodegenAction::RuntimeCall { ref sig_symbols, .. } = def.codegen {
                if !sig_symbols.is_empty() {
                    assert_eq!(
                        sig_symbols.len(), def.sigs.len(),
                        "builtin '{}': sig_symbols.len() ({}) != sigs.len() ({})",
                        def.name, sig_symbols.len(), def.sigs.len(),
                    );
                }
            }
        }
    }

    #[test]
    fn no_two_sigs_share_arg_count() {
        for def in builtin_registry() {
            let mut counts = std::collections::HashSet::new();
            for sig in &def.sigs {
                assert!(
                    counts.insert(sig.params.len()),
                    "builtin '{}' has multiple sigs with {} params",
                    def.name, sig.params.len(),
                );
            }
        }
    }

    #[test]
    fn registry_covers_existing_builtins() {
        let registry = builtin_registry();
        let names: std::collections::HashSet<&str> = registry.iter().map(|d| d.name).collect();

        // All names from the current register_builtins() HashMap
        let expected = [
            "Path", "move_to", "line_to", "quad_to", "cubic_to", "close",
            "fill", "stroke", "push", "len", "get", "set",
            "fill_styled", "stroke_styled",
            "swap", "run", "render", "display",
            "Texture", "Scene",
            "sin", "cos", "tan", "asin", "acos", "atan",
            "sqrt", "abs", "floor", "ceil", "round",
            "exp", "ln", "log2", "log10", "fract", "exp2",
            "min", "max", "atan2", "fmod", "pow",
            "set_keypress", "set_keydown", "set_keyup",
            "clear_keypress", "clear_keydown", "clear_keyup",
            "set_mousedown", "set_mouseup", "set_click",
            "clear_mousedown", "clear_mouseup", "clear_click",
        ];

        for name in &expected {
            // fill_styled and stroke_styled are folded into fill/stroke overloads
            if *name == "fill_styled" || *name == "stroke_styled" {
                continue;
            }
            assert!(names.contains(name), "registry missing builtin: {}", name);
        }

        // Special-case builtins that were previously only in if-blocks
        let special_cases = ["assert_eq", "assert_near", "assert_true"];
        for name in &special_cases {
            assert!(names.contains(name), "registry missing special-case builtin: {}", name);
        }

        // Method-only builtins that were previously only in MethodCall arm
        assert!(names.contains("map"), "registry missing method-only builtin: map");
        assert!(names.contains("run_scene"), "registry missing method-only builtin: run_scene");
    }

    #[test]
    fn all_runtime_symbols_exist() {
        use crate::pdc::runtime;
        let runtime_syms: std::collections::HashSet<&str> = runtime::runtime_symbols()
            .iter().map(|(name, _)| *name).collect();

        // Method-only builtins whose symbols are dispatched through dedicated
        // codegen handlers (emit_array_method, etc.) rather than emit_call.
        // Their registry symbols are informational, not directly looked up.
        let method_only: std::collections::HashSet<&str> = [
            "push", "len", "get", "set", "map",
            "str_len", "str_slice", "str_concat", "str_char_at",
            "run_scene",
        ].into_iter().collect();

        for def in builtin_registry() {
            if method_only.contains(def.name) {
                continue;
            }
            // Collect all symbols this builtin declares
            let symbols: Vec<&str> = match &def.codegen {
                CodegenAction::RuntimeCall { symbol, sig_symbols, .. } => {
                    let mut syms = vec![*symbol];
                    for override_sym in sig_symbols.iter().flatten() {
                        syms.push(override_sym);
                    }
                    syms
                }
                CodegenAction::MathIntrinsic { fallback_symbol, .. } => vec![*fallback_symbol],
                CodegenAction::StringArgCall { symbol, .. } => vec![*symbol],
                CodegenAction::AssertIntrinsic { .. } => continue,
            };

            for sym in symbols {
                assert!(
                    runtime_syms.contains(sym),
                    "builtin '{}' references runtime symbol '{}' which is not in runtime_symbols()",
                    def.name, sym,
                );
            }
        }
    }
}
