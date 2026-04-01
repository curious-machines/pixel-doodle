use crate::vector::flatten::{CubicBezier, Curve, Point, QuadBezier};

/// Context passed to JIT'd PDC functions.
#[repr(C)]
pub struct PdcContext {
    /// Pointer to builtin values array (f64).
    pub builtins: *const f64,
    /// Pointer to SceneBuilder.
    pub scene: *mut SceneBuilder,
}

/// Draw command types.
#[derive(Debug, Clone)]
pub enum DrawCommand {
    Fill { path_handle: u32, color: u32 },
    Stroke { path_handle: u32, width: f32, color: u32 },
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
    });
}

#[unsafe(no_mangle)]
pub extern "C" fn pdc_stroke(ctx: *mut PdcContext, handle: u32, width: f32, color: u32) {
    let scene = unsafe { &mut *(*ctx).scene };
    scene.draws.push(DrawCommand::Stroke {
        path_handle: handle,
        width,
        color,
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
pub extern "C" fn pdc_min(a: f64, b: f64) -> f64 { a.min(b) }
#[unsafe(no_mangle)]
pub extern "C" fn pdc_max(a: f64, b: f64) -> f64 { a.max(b) }
#[unsafe(no_mangle)]
pub extern "C" fn pdc_atan2(y: f64, x: f64) -> f64 { y.atan2(x) }
#[unsafe(no_mangle)]
pub extern "C" fn pdc_fmod(a: f64, b: f64) -> f64 { a % b }

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
        // Math 2-arg
        ("pdc_min", pdc_min as *const u8),
        ("pdc_max", pdc_max as *const u8),
        ("pdc_atan2", pdc_atan2 as *const u8),
        ("pdc_fmod", pdc_fmod as *const u8),
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
    ]
}
