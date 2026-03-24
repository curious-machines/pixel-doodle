// Gray-Scott reaction-diffusion step kernel
// Physics constants (matching CPU kernel defaults)
const FEED: f32 = 0.037;
const KILL: f32 = 0.06;
const DU: f32 = 0.21;
const DV: f32 = 0.105;
const DT: f32 = 1.0;

struct Params {
    width: u32,
    height: u32,
    stride: u32,
    _pad: u32,
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
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= params.width || y >= params.height {
        return;
    }

    let ix = i32(x);
    let iy = i32(y);

    let c = field_in[idx(x, y)];
    let l = field_in[idx(wrap_x(ix - 1), y)];
    let r = field_in[idx(wrap_x(ix + 1), y)];
    let u = field_in[idx(x, wrap_y(iy - 1))];
    let d = field_in[idx(x, wrap_y(iy + 1))];

    let lap = l + r + u + d - 4.0 * c;
    let uvv = c.x * c.y * c.y;

    let new_u = c.x + DT * (DU * lap.x - uvv + FEED * (1.0 - c.x));
    let new_v = c.y + DT * (DV * lap.y + uvv - (FEED + KILL) * c.y);

    field_out[idx(x, y)] = vec2<f32>(new_u, new_v);
}
