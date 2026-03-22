struct Params {
    width: u32,
    height: u32,
    stride: u32,
    _pad0: u32,
    feed: f32,
    kill: f32,
    du: f32,
    dv: f32,
    dt: f32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read>       field_in:  array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> field_out: array<vec2<f32>>;

fn idx(x: u32, y: u32) -> u32 {
    return y * params.width + x;
}

fn wrap_x(x: i32) -> u32 {
    return u32((x + i32(params.width)) % i32(params.width));
}

fn wrap_y(y: i32) -> u32 {
    return u32((y + i32(params.height)) % i32(params.height));
}

@compute @workgroup_size(16, 16)
fn step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= params.width || y >= params.height {
        return;
    }

    let ix = i32(x);
    let iy = i32(y);

    let c  = field_in[idx(x, y)];
    let l  = field_in[idx(wrap_x(ix - 1), y)];
    let r  = field_in[idx(wrap_x(ix + 1), y)];
    let u  = field_in[idx(x, wrap_y(iy - 1))];
    let d  = field_in[idx(x, wrap_y(iy + 1))];

    let lap = l + r + u + d - 4.0 * c;

    let uvv = c.x * c.y * c.y;

    let new_u = c.x + params.dt * (params.du * lap.x - uvv + params.feed * (1.0 - c.x));
    let new_v = c.y + params.dt * (params.dv * lap.y + uvv - (params.feed + params.kill) * c.y);

    field_out[idx(x, y)] = vec2<f32>(new_u, new_v);
}

// Visualization: convert field_in (u, v) to ARGB pixels
@group(0) @binding(0) var<uniform> vis_params: Params;
@group(0) @binding(1) var<storage, read>       vis_field: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> pixels:    array<u32>;

@compute @workgroup_size(16, 16)
fn visualize(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= vis_params.width || y >= vis_params.height {
        return;
    }

    let field_idx = y * vis_params.width + x;
    let pixel_idx = y * vis_params.stride + x;
    let v = clamp(vis_field[field_idx].y, 0.0, 1.0);

    // Color ramp: black -> blue -> white -> orange
    var r: f32;
    var g: f32;
    var b: f32;

    if v < 0.25 {
        let t = v / 0.25;
        r = 0.0; g = 0.0; b = t;
    } else if v < 0.5 {
        let t = (v - 0.25) / 0.25;
        r = t; g = t; b = 1.0;
    } else if v < 0.75 {
        let t = (v - 0.5) / 0.25;
        r = 1.0; g = 1.0 - 0.35 * t; b = 1.0 - t;
    } else {
        let t = (v - 0.75) / 0.25;
        r = 1.0; g = 0.65 - 0.25 * t; b = t * 0.1;
    }

    let rb = u32(r * 255.0);
    let gb = u32(g * 255.0);
    let bb = u32(b * 255.0);
    pixels[pixel_idx] = 0xFF000000u | (rb << 16u) | (gb << 8u) | bb;
}
