// Solid red — simplest possible kernel.
// WGSL equivalent of solid_red.pdir

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

    // 0xFF000000 | (255 << 16) | (0 << 8) | 0 = 0xFFFF0000
    let color = 0xFFFF0000u;

    let idx = row * params.stride + col;
    output[idx] = color;
}
