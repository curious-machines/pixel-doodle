struct Params {
    width: u32,
    height: u32,
    stride: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read>       pressure:  array<f32>;
@group(0) @binding(2) var<storage, read>       field_in:  array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> field_out: array<vec4<f32>>;

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
    var cell = field_in[i];

    // Interior: subtract pressure gradient
    if x > 0u && x < w - 1u && y > 0u && y < h - 1u {
        let p_r = pressure[idx(x + 1u, y, w)];
        let p_l = pressure[idx(x - 1u, y, w)];
        let p_d = pressure[idx(x, y + 1u, w)];
        let p_u = pressure[idx(x, y - 1u, w)];

        cell.x -= 0.5 * (p_r - p_l);
        cell.y -= 0.5 * (p_d - p_u);
    } else {
        // Boundary: zero velocity
        cell.x = 0.0;
        cell.y = 0.0;
    }

    field_out[i] = cell;
}
