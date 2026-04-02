use std::collections::HashMap;

use super::ast::PdcType;
use super::error::PdcError;
use super::runtime;

/// Compiled PDC function type.
pub type PdcSceneFn = unsafe extern "C" fn(*mut runtime::PdcContext);

/// A value returned from or passed to a PDC function call.
#[derive(Debug, Clone, PartialEq)]
pub enum PdcValue {
    F64(f64),
    F32(f32),
    I32(i32),
    I64(i64),
    U32(u32),
    U64(u64),
    Bool(bool),
    Void,
}

impl PdcValue {
    pub(crate) fn as_f32(&self) -> f32 {
        match self {
            PdcValue::F32(x) => *x,
            PdcValue::F64(x) => *x as f32,
            _ => panic!("expected float PdcValue, got {:?}", self),
        }
    }
}

/// JIT-compiled PDC program.
pub struct CompiledProgram {
    pub fn_ptr: PdcSceneFn,
    /// User-defined function pointers, keyed by qualified name (e.g., "add", "math::lerp").
    /// Each entry is (fn_pointer, param_types, return_type).
    pub user_fns: HashMap<String, (*const u8, Vec<PdcType>, PdcType)>,
    /// Test function pointers: (test_name, fn_pointer).
    /// Each test is a void function taking only a PdcContext pointer.
    pub test_fns: Vec<(String, PdcSceneFn)>,
    _jit_handle: Box<dyn Send + Sync>,
}

// Safety: The function pointers in user_fns point to JIT'd code owned by _jit_handle.
// They are valid for the lifetime of CompiledProgram and safe to call from any thread
// (PDC functions are pure or only access the provided PdcContext).
unsafe impl Send for CompiledProgram {}
unsafe impl Sync for CompiledProgram {}

/// Classify a PdcType into its ABI category for calling convention dispatch.
/// 'd' = f64 (xmm, 64-bit), 's' = f32 (xmm, 32-bit), 'i' = integer (GPR).
/// Integer types of all widths are safe to pass as i64 since smaller values
/// occupy the lower bits of a 64-bit register. Float types must be distinguished
/// because f32 and f64 have different XMM register encodings.
pub fn abi_class(ty: &PdcType) -> char {
    match ty {
        PdcType::F64 => 'd',
        PdcType::F32 => 's',
        _ => 'i',
    }
}

impl CompiledProgram {
    /// Create a new `CompiledProgram` from its components.
    pub fn new(
        fn_ptr: PdcSceneFn,
        user_fns: HashMap<String, (*const u8, Vec<PdcType>, PdcType)>,
        test_fns: Vec<(String, PdcSceneFn)>,
        jit_handle: Box<dyn Send + Sync>,
    ) -> Self {
        Self {
            fn_ptr,
            user_fns,
            test_fns,
            _jit_handle: jit_handle,
        }
    }

    /// Call a user-defined function by qualified name with typed arguments.
    ///
    /// Supports functions with 0-3 parameters using common type combinations.
    /// Float args (f32/f64) and integer args (i32/i64/bool/u32/u64) are dispatched
    /// through type-correct function signatures to respect the C calling convention
    /// (float args in XMM registers, integer args in GPRs).
    ///
    /// # Safety
    /// Arguments must match the function's declared parameter types.
    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn call_fn(
        &self,
        name: &str,
        ctx: &mut runtime::PdcContext,
        args: &[PdcValue],
    ) -> Result<PdcValue, PdcError> {
        let (ptr, param_types, ret_type) = self.user_fns.get(name)
            .ok_or_else(|| PdcError::Codegen {
                message: format!("no compiled function named '{name}'"),
            })?;

        if args.len() != param_types.len() {
            return Err(PdcError::Codegen {
                message: format!(
                    "function '{name}' expects {} args, got {}",
                    param_types.len(), args.len()
                ),
            });
        }

        let ctx_ptr: *mut runtime::PdcContext = ctx;

        // Build ABI signature string for dispatch: e.g. "ff" = two float args
        let sig: String = param_types.iter().map(abi_class).collect();

        // Helper: extract f64 from a PdcValue (promotes f32)
        fn as_f64(v: &PdcValue) -> f64 {
            match v {
                PdcValue::F64(x) => *x,
                PdcValue::F32(x) => *x as f64,
                _ => panic!("expected float PdcValue"),
            }
        }

        // Helper: extract i64 from a PdcValue (promotes smaller int types)
        fn as_i64(v: &PdcValue) -> i64 {
            match v {
                PdcValue::I32(x) => *x as i64,
                PdcValue::I64(x) => *x,
                PdcValue::U32(x) => *x as i64,
                PdcValue::U64(x) => *x as i64,
                PdcValue::Bool(x) => *x as i64,
                _ => panic!("expected integer PdcValue"),
            }
        }

        // Wrap the raw return value into a PdcValue based on the declared return type.
        macro_rules! wrap_ret_f64 {
            ($val:expr, $ret_type:expr) => {
                match $ret_type {
                    PdcType::F64 => Ok(PdcValue::F64($val)),
                    PdcType::F32 => Ok(PdcValue::F32($val as f32)),
                    _ => unreachable!(),
                }
            }
        }
        macro_rules! wrap_ret_i64 {
            ($val:expr, $ret_type:expr) => {
                match $ret_type {
                    PdcType::I32 => Ok(PdcValue::I32($val as i32)),
                    PdcType::I64 => Ok(PdcValue::I64($val)),
                    PdcType::U32 => Ok(PdcValue::U32($val as u32)),
                    PdcType::U64 => Ok(PdcValue::U64($val as u64)),
                    PdcType::Bool => Ok(PdcValue::Bool($val != 0)),
                    _ => unreachable!(),
                }
            }
        }

        let ret_class = abi_class(ret_type);

        match (ret_class, sig.as_str()) {
            // --- f64 return ---
            ('d', "") => {
                let f: unsafe extern "C" fn(*mut runtime::PdcContext) -> f64 = std::mem::transmute(*ptr);
                wrap_ret_f64!(f(ctx_ptr), ret_type)
            }
            ('d', "d") => {
                let f: unsafe extern "C" fn(*mut runtime::PdcContext, f64) -> f64 = std::mem::transmute(*ptr);
                wrap_ret_f64!(f(ctx_ptr, as_f64(&args[0])), ret_type)
            }
            ('d', "dd") => {
                let f: unsafe extern "C" fn(*mut runtime::PdcContext, f64, f64) -> f64 = std::mem::transmute(*ptr);
                wrap_ret_f64!(f(ctx_ptr, as_f64(&args[0]), as_f64(&args[1])), ret_type)
            }
            ('d', "ddd") => {
                let f: unsafe extern "C" fn(*mut runtime::PdcContext, f64, f64, f64) -> f64 = std::mem::transmute(*ptr);
                wrap_ret_f64!(f(ctx_ptr, as_f64(&args[0]), as_f64(&args[1]), as_f64(&args[2])), ret_type)
            }
            ('d', "i") => {
                let f: unsafe extern "C" fn(*mut runtime::PdcContext, i64) -> f64 = std::mem::transmute(*ptr);
                wrap_ret_f64!(f(ctx_ptr, as_i64(&args[0])), ret_type)
            }
            ('d', "ii") => {
                let f: unsafe extern "C" fn(*mut runtime::PdcContext, i64, i64) -> f64 = std::mem::transmute(*ptr);
                wrap_ret_f64!(f(ctx_ptr, as_i64(&args[0]), as_i64(&args[1])), ret_type)
            }
            ('d', "di") => {
                let f: unsafe extern "C" fn(*mut runtime::PdcContext, f64, i64) -> f64 = std::mem::transmute(*ptr);
                wrap_ret_f64!(f(ctx_ptr, as_f64(&args[0]), as_i64(&args[1])), ret_type)
            }
            ('d', "id") => {
                let f: unsafe extern "C" fn(*mut runtime::PdcContext, i64, f64) -> f64 = std::mem::transmute(*ptr);
                wrap_ret_f64!(f(ctx_ptr, as_i64(&args[0]), as_f64(&args[1])), ret_type)
            }
            // --- f32 return ---
            ('s', "") => {
                let f: unsafe extern "C" fn(*mut runtime::PdcContext) -> f32 = std::mem::transmute(*ptr);
                Ok(PdcValue::F32(f(ctx_ptr)))
            }
            ('s', "s") => {
                let f: unsafe extern "C" fn(*mut runtime::PdcContext, f32) -> f32 = std::mem::transmute(*ptr);
                Ok(PdcValue::F32(f(ctx_ptr, args[0].as_f32())))
            }
            ('s', "ss") => {
                let f: unsafe extern "C" fn(*mut runtime::PdcContext, f32, f32) -> f32 = std::mem::transmute(*ptr);
                Ok(PdcValue::F32(f(ctx_ptr, args[0].as_f32(), args[1].as_f32())))
            }
            // --- Int return ---
            ('i', "") => {
                let f: unsafe extern "C" fn(*mut runtime::PdcContext) -> i64 = std::mem::transmute(*ptr);
                wrap_ret_i64!(f(ctx_ptr), ret_type)
            }
            ('i', "i") => {
                let f: unsafe extern "C" fn(*mut runtime::PdcContext, i64) -> i64 = std::mem::transmute(*ptr);
                wrap_ret_i64!(f(ctx_ptr, as_i64(&args[0])), ret_type)
            }
            ('i', "ii") => {
                let f: unsafe extern "C" fn(*mut runtime::PdcContext, i64, i64) -> i64 = std::mem::transmute(*ptr);
                wrap_ret_i64!(f(ctx_ptr, as_i64(&args[0]), as_i64(&args[1])), ret_type)
            }
            ('i', "iii") => {
                let f: unsafe extern "C" fn(*mut runtime::PdcContext, i64, i64, i64) -> i64 = std::mem::transmute(*ptr);
                wrap_ret_i64!(f(ctx_ptr, as_i64(&args[0]), as_i64(&args[1]), as_i64(&args[2])), ret_type)
            }
            ('i', "d") => {
                let f: unsafe extern "C" fn(*mut runtime::PdcContext, f64) -> i64 = std::mem::transmute(*ptr);
                wrap_ret_i64!(f(ctx_ptr, as_f64(&args[0])), ret_type)
            }
            ('i', "dd") => {
                let f: unsafe extern "C" fn(*mut runtime::PdcContext, f64, f64) -> i64 = std::mem::transmute(*ptr);
                wrap_ret_i64!(f(ctx_ptr, as_f64(&args[0]), as_f64(&args[1])), ret_type)
            }
            // --- Void return ---
            (_, "") if *ret_type == PdcType::Void => {
                let f: unsafe extern "C" fn(*mut runtime::PdcContext) = std::mem::transmute(*ptr);
                f(ctx_ptr);
                Ok(PdcValue::Void)
            }
            (_, "d") if *ret_type == PdcType::Void => {
                let f: unsafe extern "C" fn(*mut runtime::PdcContext, f64) = std::mem::transmute(*ptr);
                f(ctx_ptr, as_f64(&args[0]));
                Ok(PdcValue::Void)
            }
            (_, "i") if *ret_type == PdcType::Void => {
                let f: unsafe extern "C" fn(*mut runtime::PdcContext, i64) = std::mem::transmute(*ptr);
                f(ctx_ptr, as_i64(&args[0]));
                Ok(PdcValue::Void)
            }
            _ => Err(PdcError::Codegen {
                message: format!(
                    "unsupported call signature for '{name}': ret={:?}, params={:?}",
                    ret_type, param_types
                ),
            }),
        }
    }
}

pub struct BuiltinInfo {
    pub offset: usize,
    pub ty: PdcType,
}

/// Mangle a qualified name for JIT symbols: "math::lerp" -> "math__lerp"
pub fn mangle_name(qualified: &str) -> String {
    qualified.replace("::", "__")
}

/// Build a map from builtin name to offset + type for the builtins array.
pub fn build_builtin_map(layout: &[(&str, PdcType)]) -> HashMap<String, BuiltinInfo> {
    layout
        .iter()
        .enumerate()
        .map(|(i, (name, ty))| {
            (
                name.to_string(),
                BuiltinInfo {
                    offset: i,
                    ty: ty.clone(),
                },
            )
        })
        .collect()
}
