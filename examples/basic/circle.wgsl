// Distance field: circle centered at origin, radius 1.
// Inside = blue, outside = black.
// WGSL equivalent of circle.pdl

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

    let dist2 = x * x + y * y;
    var b: u32;
    if dist2 <= 1.0 {
        b = 200u;
    } else {
        b = 0u;
    }
    let color = 0xFF000000u | b;

    let idx = row * params.stride + col;
    output[idx] = color;
}
