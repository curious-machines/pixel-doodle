// Gradient: maps x to red, y to green, constant blue.
// WGSL equivalent of gradient.pdl

struct Params {
    width: u32,
    height: u32,
    max_iter: u32,
    stride: u32,
    x_min: f32,
    y_min: f32,
    x_step: f32,
    y_step: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    if col >= params.width || row >= params.height {
        return;
    }
    let x = params.x_min + f32(col) * params.x_step;
    let y = params.y_min + f32(row) * params.y_step;

    // Use signed conversion + bitcast to match CPU backends (fcvt_to_sint).
    // u32(negative_float) clamps to 0 in WGSL, but the CPU JIT produces
    // negative i32 bit patterns that wrap through pack_argb shifts.
    let r_u = bitcast<u32>(i32(x * 255.0));
    let g_u = bitcast<u32>(i32(y * 255.0));
    let b = 128u;
    // No clamping — match CPU pack_argb which lets values wrap via shift+or.
    let color = 0xFF000000u | (r_u << 16u) | (g_u << 8u) | b;

    let idx = row * params.stride + col;
    output[idx] = color;
}
