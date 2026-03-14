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

    let r_u = u32(x * 255.0);
    let g_u = u32(y * 255.0);
    let b = 128u;
    let color = 0xFF000000u | (min(r_u, 255u) << 16u) | (min(g_u, 255u) << 8u) | min(b, 255u);

    let idx = row * params.stride + col;
    output[idx] = color;
}
