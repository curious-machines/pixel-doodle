use crate::vector::flatten::{CubicBezier, Curve, Point, QuadBezier};

/// Context passed to JIT'd PDC functions.
#[repr(C)]
pub struct PdcContext {
    /// Pointer to builtin values array (f64). Mutable builtins write back to this array.
    pub builtins: *mut f64,
    /// Pointer to SceneBuilder.
    pub scene: *mut SceneBuilder,
    /// Pointer to persistent state block for module-level mutable variables.
    /// Null when no state block is needed (e.g., stateless scene scripts).
    pub state: *mut u8,
}

/// Fill rule for path filling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FillRule {
    EvenOdd = 0,
    NonZero = 1,
}

/// Line cap style for stroke endpoints.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LineCap {
    Butt = 0,
    Round = 1,
    Square = 2,
}

/// Line join style for stroke corners.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LineJoin {
    Miter = 0,
    Round = 1,
    Bevel = 2,
}

/// Draw command types.
#[derive(Debug, Clone)]
pub enum DrawCommand {
    Fill { path_handle: u32, color: u32, rule: FillRule },
    Stroke { path_handle: u32, width: f32, color: u32, cap: LineCap, join: LineJoin },
}

/// Runtime array with contiguous byte storage.
pub struct ArrayData {
    pub bytes: Vec<u8>,
    pub element_size: usize,
}

impl ArrayData {
    fn new(element_size: usize) -> Self {
        Self {
            bytes: Vec::new(),
            element_size,
        }
    }

    fn len(&self) -> usize {
        if self.element_size == 0 { 0 } else { self.bytes.len() / self.element_size }
    }
}

/// Accumulates paths, draw commands, and arrays during PDC execution.
pub struct SceneBuilder {
    /// Curves per path handle.
    pub paths: Vec<PathData>,
    /// Draw commands in submission order.
    pub draws: Vec<DrawCommand>,
    /// Runtime arrays (handle-based). Contiguous byte storage with known element size.
    pub arrays: Vec<ArrayData>,
    /// Runtime strings (handle-based).
    pub strings: Vec<String>,
}

pub struct PathData {
    pub curves: Vec<Curve>,
    pub current: Option<Point>,
    pub subpath_start: Option<Point>,
}

impl SceneBuilder {
    pub fn new() -> Self {
        Self {
            paths: Vec::new(),
            draws: Vec::new(),
            arrays: Vec::new(),
            strings: Vec::new(),
        }
    }

    fn new_path(&mut self) -> u32 {
        let handle = self.paths.len() as u32;
        self.paths.push(PathData {
            curves: Vec::new(),
            current: None,
            subpath_start: None,
        });
        handle
    }

    fn path_mut(&mut self, handle: u32) -> &mut PathData {
        &mut self.paths[handle as usize]
    }
}

// ── Extern "C" runtime functions ──

#[unsafe(no_mangle)]
pub extern "C" fn pdc_path(ctx: *mut PdcContext) -> u32 {
    let scene = unsafe { &mut *(*ctx).scene };
    scene.new_path()
}

#[unsafe(no_mangle)]
pub extern "C" fn pdc_move_to(ctx: *mut PdcContext, handle: u32, x: f64, y: f64) {
    let scene = unsafe { &mut *(*ctx).scene };
    let path = scene.path_mut(handle);
    let pt = Point::new(x, y);
    path.current = Some(pt);
    path.subpath_start = Some(pt);
}

#[unsafe(no_mangle)]
pub extern "C" fn pdc_line_to(ctx: *mut PdcContext, handle: u32, x: f64, y: f64) {
    let scene = unsafe { &mut *(*ctx).scene };
    let path = scene.path_mut(handle);
    let to = Point::new(x, y);
    if let Some(from) = path.current {
        path.curves.push(Curve::Line(from, to));
    }
    path.current = Some(to);
}

#[unsafe(no_mangle)]
pub extern "C" fn pdc_quad_to(
    ctx: *mut PdcContext,
    handle: u32,
    cx: f64,
    cy: f64,
    x: f64,
    y: f64,
) {
    let scene = unsafe { &mut *(*ctx).scene };
    let path = scene.path_mut(handle);
    let to = Point::new(x, y);
    if let Some(from) = path.current {
        path.curves.push(Curve::Quad(QuadBezier {
            p0: from,
            p1: Point::new(cx, cy),
            p2: to,
        }));
    }
    path.current = Some(to);
}

#[unsafe(no_mangle)]
pub extern "C" fn pdc_cubic_to(
    ctx: *mut PdcContext,
    handle: u32,
    c1x: f64,
    c1y: f64,
    c2x: f64,
    c2y: f64,
    x: f64,
    y: f64,
) {
    let scene = unsafe { &mut *(*ctx).scene };
    let path = scene.path_mut(handle);
    let to = Point::new(x, y);
    if let Some(from) = path.current {
        path.curves.push(Curve::Cubic(CubicBezier {
            p0: from,
            p1: Point::new(c1x, c1y),
            p2: Point::new(c2x, c2y),
            p3: to,
        }));
    }
    path.current = Some(to);
}

#[unsafe(no_mangle)]
pub extern "C" fn pdc_close(ctx: *mut PdcContext, handle: u32) {
    let scene = unsafe { &mut *(*ctx).scene };
    let path = scene.path_mut(handle);
    if let (Some(current), Some(start)) = (path.current, path.subpath_start) {
        if (current.x - start.x).abs() > 1e-6 || (current.y - start.y).abs() > 1e-6 {
            path.curves.push(Curve::Line(current, start));
        }
        path.current = Some(start);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn pdc_fill(ctx: *mut PdcContext, handle: u32, color: u32) {
    let scene = unsafe { &mut *(*ctx).scene };
    scene.draws.push(DrawCommand::Fill {
        path_handle: handle,
        color,
        rule: FillRule::EvenOdd,
    });
}

#[unsafe(no_mangle)]
pub extern "C" fn pdc_fill_styled(ctx: *mut PdcContext, handle: u32, color: u32, rule: i32) {
    let scene = unsafe { &mut *(*ctx).scene };
    let rule = match rule {
        1 => FillRule::NonZero,
        _ => FillRule::EvenOdd,
    };
    scene.draws.push(DrawCommand::Fill {
        path_handle: handle,
        color,
        rule,
    });
}

#[unsafe(no_mangle)]
pub extern "C" fn pdc_stroke(ctx: *mut PdcContext, handle: u32, width: f32, color: u32) {
    let scene = unsafe { &mut *(*ctx).scene };
    scene.draws.push(DrawCommand::Stroke {
        path_handle: handle,
        width,
        color,
        cap: LineCap::Butt,
        join: LineJoin::Miter,
    });
}

#[unsafe(no_mangle)]
pub extern "C" fn pdc_stroke_styled(ctx: *mut PdcContext, handle: u32, width: f32, color: u32, cap: i32, join: i32) {
    let scene = unsafe { &mut *(*ctx).scene };
    let cap = match cap {
        1 => LineCap::Round,
        2 => LineCap::Square,
        _ => LineCap::Butt,
    };
    let join = match join {
        1 => LineJoin::Round,
        2 => LineJoin::Bevel,
        _ => LineJoin::Miter,
    };
    scene.draws.push(DrawCommand::Stroke {
        path_handle: handle,
        width,
        color,
        cap,
        join,
    });
}

// ── Math runtime functions ──

#[unsafe(no_mangle)]
pub extern "C" fn pdc_sin(x: f64) -> f64 { x.sin() }
#[unsafe(no_mangle)]
pub extern "C" fn pdc_cos(x: f64) -> f64 { x.cos() }
#[unsafe(no_mangle)]
pub extern "C" fn pdc_tan(x: f64) -> f64 { x.tan() }
#[unsafe(no_mangle)]
pub extern "C" fn pdc_asin(x: f64) -> f64 { x.asin() }
#[unsafe(no_mangle)]
pub extern "C" fn pdc_acos(x: f64) -> f64 { x.acos() }
#[unsafe(no_mangle)]
pub extern "C" fn pdc_atan(x: f64) -> f64 { x.atan() }
#[unsafe(no_mangle)]
pub extern "C" fn pdc_sqrt(x: f64) -> f64 { x.sqrt() }
#[unsafe(no_mangle)]
pub extern "C" fn pdc_abs(x: f64) -> f64 { x.abs() }
#[unsafe(no_mangle)]
pub extern "C" fn pdc_floor(x: f64) -> f64 { x.floor() }
#[unsafe(no_mangle)]
pub extern "C" fn pdc_ceil(x: f64) -> f64 { x.ceil() }
#[unsafe(no_mangle)]
pub extern "C" fn pdc_round(x: f64) -> f64 { x.round() }
#[unsafe(no_mangle)]
pub extern "C" fn pdc_exp(x: f64) -> f64 { x.exp() }
#[unsafe(no_mangle)]
pub extern "C" fn pdc_ln(x: f64) -> f64 { x.ln() }
#[unsafe(no_mangle)]
pub extern "C" fn pdc_log2(x: f64) -> f64 { x.log2() }
#[unsafe(no_mangle)]
pub extern "C" fn pdc_log10(x: f64) -> f64 { x.log10() }
#[unsafe(no_mangle)]
pub extern "C" fn pdc_fract(x: f64) -> f64 { x.fract() }
#[unsafe(no_mangle)]
pub extern "C" fn pdc_exp2(x: f64) -> f64 { x.exp2() }
#[unsafe(no_mangle)]
pub extern "C" fn pdc_min(a: f64, b: f64) -> f64 { a.min(b) }
#[unsafe(no_mangle)]
pub extern "C" fn pdc_max(a: f64, b: f64) -> f64 { a.max(b) }
#[unsafe(no_mangle)]
pub extern "C" fn pdc_atan2(y: f64, x: f64) -> f64 { y.atan2(x) }
#[unsafe(no_mangle)]
pub extern "C" fn pdc_fmod(a: f64, b: f64) -> f64 { a % b }

#[unsafe(no_mangle)]
pub extern "C" fn pdc_pow(a: f64, b: f64) -> f64 { a.powf(b) }

// ── Array runtime functions ──

// ── Array runtime functions ──
// Size-specific functions: elements stored at their natural size.
// The byte buffer is contiguous and GPU-uploadable.

// Array creation — element_size passed from codegen
#[unsafe(no_mangle)]
pub extern "C" fn pdc_array_new(ctx: *mut PdcContext, element_size: u32) -> u32 {
    let scene = unsafe { &mut *(*ctx).scene };
    let handle = scene.arrays.len() as u32;
    scene.arrays.push(ArrayData::new(element_size as usize));
    handle
}

#[unsafe(no_mangle)]
pub extern "C" fn pdc_array_len(ctx: *mut PdcContext, handle: u32) -> i32 {
    let scene = unsafe { &mut *(*ctx).scene };
    scene.arrays[handle as usize].len() as i32
}

// Push/get/set by element size — JIT picks the right one.

#[unsafe(no_mangle)]
pub extern "C" fn pdc_array_push_1(ctx: *mut PdcContext, handle: u32, value: u8) {
    let scene = unsafe { &mut *(*ctx).scene };
    scene.arrays[handle as usize].bytes.extend_from_slice(&value.to_le_bytes());
}

#[unsafe(no_mangle)]
pub extern "C" fn pdc_array_push_2(ctx: *mut PdcContext, handle: u32, value: u16) {
    let scene = unsafe { &mut *(*ctx).scene };
    scene.arrays[handle as usize].bytes.extend_from_slice(&value.to_le_bytes());
}

#[unsafe(no_mangle)]
pub extern "C" fn pdc_array_push_4(ctx: *mut PdcContext, handle: u32, value: u32) {
    let scene = unsafe { &mut *(*ctx).scene };
    scene.arrays[handle as usize].bytes.extend_from_slice(&value.to_le_bytes());
}

#[unsafe(no_mangle)]
pub extern "C" fn pdc_array_push_8(ctx: *mut PdcContext, handle: u32, value: u64) {
    let scene = unsafe { &mut *(*ctx).scene };
    scene.arrays[handle as usize].bytes.extend_from_slice(&value.to_le_bytes());
}

#[unsafe(no_mangle)]
pub extern "C" fn pdc_array_get_1(ctx: *mut PdcContext, handle: u32, index: i32) -> u8 {
    let scene = unsafe { &mut *(*ctx).scene };
    let arr = &scene.arrays[handle as usize];
    let offset = index as usize * arr.element_size;
    if offset < arr.bytes.len() { arr.bytes[offset] } else { 0 }
}

#[unsafe(no_mangle)]
pub extern "C" fn pdc_array_get_2(ctx: *mut PdcContext, handle: u32, index: i32) -> u16 {
    let scene = unsafe { &mut *(*ctx).scene };
    let arr = &scene.arrays[handle as usize];
    let offset = index as usize * arr.element_size;
    if offset + 1 < arr.bytes.len() {
        u16::from_le_bytes([arr.bytes[offset], arr.bytes[offset + 1]])
    } else { 0 }
}

#[unsafe(no_mangle)]
pub extern "C" fn pdc_array_get_4(ctx: *mut PdcContext, handle: u32, index: i32) -> u32 {
    let scene = unsafe { &mut *(*ctx).scene };
    let arr = &scene.arrays[handle as usize];
    let offset = index as usize * arr.element_size;
    if offset + 3 < arr.bytes.len() {
        u32::from_le_bytes(arr.bytes[offset..offset + 4].try_into().unwrap())
    } else { 0 }
}

#[unsafe(no_mangle)]
pub extern "C" fn pdc_array_get_8(ctx: *mut PdcContext, handle: u32, index: i32) -> u64 {
    let scene = unsafe { &mut *(*ctx).scene };
    let arr = &scene.arrays[handle as usize];
    let offset = index as usize * arr.element_size;
    if offset + 7 < arr.bytes.len() {
        u64::from_le_bytes(arr.bytes[offset..offset + 8].try_into().unwrap())
    } else { 0 }
}

#[unsafe(no_mangle)]
pub extern "C" fn pdc_array_set_1(ctx: *mut PdcContext, handle: u32, index: i32, value: u8) {
    let scene = unsafe { &mut *(*ctx).scene };
    let arr = &mut scene.arrays[handle as usize];
    let offset = index as usize * arr.element_size;
    if offset < arr.bytes.len() { arr.bytes[offset] = value; }
}

#[unsafe(no_mangle)]
pub extern "C" fn pdc_array_set_2(ctx: *mut PdcContext, handle: u32, index: i32, value: u16) {
    let scene = unsafe { &mut *(*ctx).scene };
    let arr = &mut scene.arrays[handle as usize];
    let offset = index as usize * arr.element_size;
    if offset + 1 < arr.bytes.len() {
        arr.bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn pdc_array_set_4(ctx: *mut PdcContext, handle: u32, index: i32, value: u32) {
    let scene = unsafe { &mut *(*ctx).scene };
    let arr = &mut scene.arrays[handle as usize];
    let offset = index as usize * arr.element_size;
    if offset + 3 < arr.bytes.len() {
        arr.bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn pdc_array_set_8(ctx: *mut PdcContext, handle: u32, index: i32, value: u64) {
    let scene = unsafe { &mut *(*ctx).scene };
    let arr = &mut scene.arrays[handle as usize];
    let offset = index as usize * arr.element_size;
    if offset + 7 < arr.bytes.len() {
        arr.bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }
}

// ── Slice runtime functions ──

/// Get an element from a slice (array_handle, start, index).
/// Internally just does array_get at (start + index).
#[unsafe(no_mangle)]
pub extern "C" fn pdc_slice_get_1(ctx: *mut PdcContext, arr_handle: i32, start: i32, index: i32) -> u8 {
    pdc_array_get_1(ctx as *mut PdcContext, arr_handle as u32, start + index)
}

#[unsafe(no_mangle)]
pub extern "C" fn pdc_slice_get_2(ctx: *mut PdcContext, arr_handle: i32, start: i32, index: i32) -> u16 {
    pdc_array_get_2(ctx as *mut PdcContext, arr_handle as u32, start + index)
}

#[unsafe(no_mangle)]
pub extern "C" fn pdc_slice_get_4(ctx: *mut PdcContext, arr_handle: i32, start: i32, index: i32) -> u32 {
    pdc_array_get_4(ctx as *mut PdcContext, arr_handle as u32, start + index)
}

#[unsafe(no_mangle)]
pub extern "C" fn pdc_slice_get_8(ctx: *mut PdcContext, arr_handle: i32, start: i32, index: i32) -> u64 {
    pdc_array_get_8(ctx as *mut PdcContext, arr_handle as u32, start + index)
}

// ── String runtime functions ──

/// Create a new string from a pointer and length, returns handle.
#[unsafe(no_mangle)]
pub extern "C" fn pdc_string_new(ctx: *mut PdcContext, ptr: *const u8, len: i32) -> i32 {
    let scene = unsafe { &mut *(*ctx).scene };
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len as usize) };
    let s = std::str::from_utf8(bytes).unwrap_or("").to_string();
    let handle = scene.strings.len();
    scene.strings.push(s);
    handle as i32
}

/// Get string length in bytes.
#[unsafe(no_mangle)]
pub extern "C" fn pdc_string_len(ctx: *mut PdcContext, handle: i32) -> i32 {
    let scene = unsafe { &*(*ctx).scene };
    scene.strings[handle as usize].len() as i32
}

/// Concatenate two strings, returns new handle.
#[unsafe(no_mangle)]
pub extern "C" fn pdc_string_concat(ctx: *mut PdcContext, a: i32, b: i32) -> i32 {
    let scene = unsafe { &mut *(*ctx).scene };
    let result = format!("{}{}", scene.strings[a as usize], scene.strings[b as usize]);
    let handle = scene.strings.len();
    scene.strings.push(result);
    handle as i32
}

/// Compare two strings for equality.
#[unsafe(no_mangle)]
pub extern "C" fn pdc_string_eq(ctx: *mut PdcContext, a: i32, b: i32) -> i8 {
    let scene = unsafe { &*(*ctx).scene };
    (scene.strings[a as usize] == scene.strings[b as usize]) as i8
}

/// Get a substring (slice), returns new handle.
#[unsafe(no_mangle)]
pub extern "C" fn pdc_string_slice(ctx: *mut PdcContext, handle: i32, start: i32, end: i32) -> i32 {
    let scene = unsafe { &mut *(*ctx).scene };
    let s = &scene.strings[handle as usize];
    let start = (start as usize).min(s.len());
    let end = (end as usize).min(s.len());
    let sliced = s[start..end].to_string();
    let new_handle = scene.strings.len();
    scene.strings.push(sliced);
    new_handle as i32
}

/// Get character at index as a single-char string, returns new handle.
#[unsafe(no_mangle)]
pub extern "C" fn pdc_string_char_at(ctx: *mut PdcContext, handle: i32, index: i32) -> i32 {
    let scene = unsafe { &mut *(*ctx).scene };
    let s = &scene.strings[handle as usize];
    let ch = s.chars().nth(index as usize).unwrap_or('\0');
    let new_handle = scene.strings.len();
    scene.strings.push(ch.to_string());
    new_handle as i32
}

/// Result of running a single PDC test.
#[derive(Debug)]
pub struct PdcTestResult {
    pub name: String,
    pub passed: bool,
    pub message: String,
}

/// A collected assertion failure from a PDC test.
#[derive(Debug)]
pub struct AssertFailure {
    pub message: String,
}

use std::cell::RefCell;

thread_local! {
    /// Accumulates assertion failures for the currently running PDC test.
    static ASSERT_FAILURES: RefCell<Vec<AssertFailure>> = RefCell::new(Vec::new());
}

/// Clear accumulated assertion failures and return them.
pub fn take_assert_failures() -> Vec<AssertFailure> {
    ASSERT_FAILURES.with(|f| f.borrow_mut().drain(..).collect())
}

/// Push an assertion failure message.
fn push_assert_failure(message: String) {
    ASSERT_FAILURES.with(|f| f.borrow_mut().push(AssertFailure { message }));
}

// ── Assert runtime functions (called from JIT'd test code) ──

/// assert_eq for f64 values.
unsafe extern "C" fn pdc_assert_eq_f64(_ctx: *mut PdcContext, a: f64, b: f64) {
    if a != b {
        push_assert_failure(format!("assert_eq failed: left = {a}, right = {b}"));
    }
}

/// assert_eq for i64 values (covers i32, u32, bool promoted to i64).
unsafe extern "C" fn pdc_assert_eq_i64(_ctx: *mut PdcContext, a: i64, b: i64) {
    if a != b {
        push_assert_failure(format!("assert_eq failed: left = {a}, right = {b}"));
    }
}

/// assert_eq for f32 values.
unsafe extern "C" fn pdc_assert_eq_f32(_ctx: *mut PdcContext, a: f32, b: f32) {
    if a != b {
        push_assert_failure(format!("assert_eq failed: left = {a}, right = {b}"));
    }
}

/// assert_near for f64 values with epsilon tolerance.
unsafe extern "C" fn pdc_assert_near(_ctx: *mut PdcContext, a: f64, b: f64, epsilon: f64) {
    if (a - b).abs() > epsilon {
        push_assert_failure(format!(
            "assert_near failed: left = {a}, right = {b}, diff = {}, epsilon = {epsilon}",
            (a - b).abs()
        ));
    }
}

/// assert_true for boolean values (passed as i64).
unsafe extern "C" fn pdc_assert_true(_ctx: *mut PdcContext, cond: i64) {
    if cond == 0 {
        push_assert_failure("assert_true failed: condition was false".to_string());
    }
}

/// Return all runtime symbols for JIT registration.
pub fn runtime_symbols() -> Vec<(&'static str, *const u8)> {
    vec![
        // Path primitives
        ("pdc_path", pdc_path as *const u8),
        ("pdc_move_to", pdc_move_to as *const u8),
        ("pdc_line_to", pdc_line_to as *const u8),
        ("pdc_quad_to", pdc_quad_to as *const u8),
        ("pdc_cubic_to", pdc_cubic_to as *const u8),
        ("pdc_close", pdc_close as *const u8),
        // Draw commands
        ("pdc_fill", pdc_fill as *const u8),
        ("pdc_stroke", pdc_stroke as *const u8),
        ("pdc_fill_styled", pdc_fill_styled as *const u8),
        ("pdc_stroke_styled", pdc_stroke_styled as *const u8),
        // Math 1-arg
        ("pdc_sin", pdc_sin as *const u8),
        ("pdc_cos", pdc_cos as *const u8),
        ("pdc_tan", pdc_tan as *const u8),
        ("pdc_asin", pdc_asin as *const u8),
        ("pdc_acos", pdc_acos as *const u8),
        ("pdc_atan", pdc_atan as *const u8),
        ("pdc_sqrt", pdc_sqrt as *const u8),
        ("pdc_abs", pdc_abs as *const u8),
        ("pdc_floor", pdc_floor as *const u8),
        ("pdc_ceil", pdc_ceil as *const u8),
        ("pdc_round", pdc_round as *const u8),
        ("pdc_exp", pdc_exp as *const u8),
        ("pdc_ln", pdc_ln as *const u8),
        ("pdc_log2", pdc_log2 as *const u8),
        ("pdc_log10", pdc_log10 as *const u8),
        ("pdc_fract", pdc_fract as *const u8),
        ("pdc_exp2", pdc_exp2 as *const u8),
        // Math 2-arg
        ("pdc_min", pdc_min as *const u8),
        ("pdc_max", pdc_max as *const u8),
        ("pdc_atan2", pdc_atan2 as *const u8),
        ("pdc_fmod", pdc_fmod as *const u8),
        ("pdc_pow", pdc_pow as *const u8),
        // Arrays
        ("pdc_array_new", pdc_array_new as *const u8),
        ("pdc_array_len", pdc_array_len as *const u8),
        ("pdc_array_push_1", pdc_array_push_1 as *const u8),
        ("pdc_array_push_2", pdc_array_push_2 as *const u8),
        ("pdc_array_push_4", pdc_array_push_4 as *const u8),
        ("pdc_array_push_8", pdc_array_push_8 as *const u8),
        ("pdc_array_get_1", pdc_array_get_1 as *const u8),
        ("pdc_array_get_2", pdc_array_get_2 as *const u8),
        ("pdc_array_get_4", pdc_array_get_4 as *const u8),
        ("pdc_array_get_8", pdc_array_get_8 as *const u8),
        ("pdc_array_set_1", pdc_array_set_1 as *const u8),
        ("pdc_array_set_2", pdc_array_set_2 as *const u8),
        ("pdc_array_set_4", pdc_array_set_4 as *const u8),
        ("pdc_array_set_8", pdc_array_set_8 as *const u8),
        // Slices
        ("pdc_slice_get_1", pdc_slice_get_1 as *const u8),
        ("pdc_slice_get_2", pdc_slice_get_2 as *const u8),
        ("pdc_slice_get_4", pdc_slice_get_4 as *const u8),
        ("pdc_slice_get_8", pdc_slice_get_8 as *const u8),
        // Strings
        ("pdc_string_new", pdc_string_new as *const u8),
        ("pdc_string_len", pdc_string_len as *const u8),
        ("pdc_string_concat", pdc_string_concat as *const u8),
        ("pdc_string_eq", pdc_string_eq as *const u8),
        ("pdc_string_slice", pdc_string_slice as *const u8),
        ("pdc_string_char_at", pdc_string_char_at as *const u8),
        // Test assertions
        ("pdc_assert_eq_f64", pdc_assert_eq_f64 as *const u8),
        ("pdc_assert_eq_i64", pdc_assert_eq_i64 as *const u8),
        ("pdc_assert_eq_f32", pdc_assert_eq_f32 as *const u8),
        ("pdc_assert_near", pdc_assert_near as *const u8),
        ("pdc_assert_true", pdc_assert_true as *const u8),
    ]
}
