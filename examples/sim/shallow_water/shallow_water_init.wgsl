// Initialize shallow water field: rest height 1.0 with a few bumps.
// Field layout: vec4<f32> per cell = (h, vx, vy, unused)

struct Params {
    width: u32,
    height: u32,
    stride: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> field_out: array<vec4<f32>>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= params.width || y >= params.height {
        return;
    }

    let fx = f32(x) / f32(params.width);
    let fy = f32(y) / f32(params.height);

    // Central bump
    let dx1 = fx - 0.5;
    let dy1 = fy - 0.5;
    let d1 = dx1 * dx1 + dy1 * dy1;
    var bump1 = 0.0;
    if d1 < 0.01 {
        bump1 = 0.3 * (1.0 - d1 / 0.01);
    }

    // Off-center bump
    let dx2 = fx - 0.3;
    let dy2 = fy - 0.3;
    let d2 = dx2 * dx2 + dy2 * dy2;
    var bump2 = 0.0;
    if d2 < 0.008 {
        bump2 = 0.2 * (1.0 - d2 / 0.008);
    }

    let h = 1.0 + bump1 + bump2;
    let idx = y * params.width + x;
    field_out[idx] = vec4<f32>(h, 0.0, 0.0, 0.0);
}
