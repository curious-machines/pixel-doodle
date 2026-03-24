struct Params {
    width: u32,
    height: u32,
    stride: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read>       div_in:    array<f32>;
@group(0) @binding(2) var<storage, read>       press_in:  array<f32>;
@group(0) @binding(3) var<storage, read_write> press_out: array<f32>;

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

    // Boundary: pressure = 0
    if x == 0u || x == w - 1u || y == 0u || y == h - 1u {
        press_out[i] = 0.0;
        return;
    }

    let p_l = press_in[idx(x - 1u, y, w)];
    let p_r = press_in[idx(x + 1u, y, w)];
    let p_u = press_in[idx(x, y - 1u, w)];
    let p_d = press_in[idx(x, y + 1u, w)];
    let d = div_in[i];

    press_out[i] = (p_l + p_r + p_u + p_d - d) * 0.25;
}
