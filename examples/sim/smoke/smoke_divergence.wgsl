struct Params {
    width: u32,
    height: u32,
    stride: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read>       field_in: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> div_out:  array<f32>;

fn idx(x: u32, y: u32, w: u32) -> u32 {
    return y * w + x;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let w = params.width;
    let h = params.height;
    if x >= w || y >= h { return; }

    let i = idx(x, y, w);

    // Boundary: divergence = 0
    if x == 0u || x == w - 1u || y == 0u || y == h - 1u {
        div_out[i] = 0.0;
        return;
    }

    let vx_r = field_in[idx(x + 1u, y, w)].x;
    let vx_l = field_in[idx(x - 1u, y, w)].x;
    let vy_d = field_in[idx(x, y + 1u, w)].y;
    let vy_u = field_in[idx(x, y - 1u, w)].y;

    div_out[i] = (vx_r - vx_l + vy_d - vy_u) * 0.5;
}
