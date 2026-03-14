// Checkerboard pattern using floor + modulo to create alternating squares.
// WGSL equivalent of checkerboard.pdl

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

    let scale = 4.0;
    let ix = u32(floor(x * scale));
    let iy = u32(floor(y * scale));
    let parity = (ix + iy) % 2u;
    var c: u32;
    if parity == 0u {
        c = 255u;
    } else {
        c = 0u;
    }
    let color = 0xFF000000u | (c << 16u) | (c << 8u) | c;

    let idx = row * params.stride + col;
    output[idx] = color;
}
