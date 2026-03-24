const GRAVITY: f32 = 9.81;
const DAMPING: f32 = 0.001;
const DT: f32 = 0.016;

struct Params {
    width: u32,
    height: u32,
    stride: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
// Field: vec4<f32> per cell = (h, vx, vy, unused)
@group(0) @binding(1) var<storage, read>       field_in:  array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> field_out: array<vec4<f32>>;

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

    // Lax-Friedrichs: replace center with neighbor average for stability
    let h_avg = (l.x + r.x + u.x + d.x) * 0.25;
    let vx_avg = (l.y + r.y + u.y + d.y) * 0.25;
    let vy_avg = (l.z + r.z + u.z + d.z) * 0.25;

    // Central differences for height gradient
    let dh_dx = (r.x - l.x) * 0.5;
    let dh_dy = (d.x - u.x) * 0.5;

    // Update velocity (Lax-Friedrichs): pressure gradient + damping
    let new_vx = vx_avg - DT * GRAVITY * dh_dx - DT * DAMPING * vx_avg;
    let new_vy = vy_avg - DT * GRAVITY * dh_dy - DT * DAMPING * vy_avg;

    // Flux divergence for height update (Lax-Friedrichs)
    let flux_x = (r.x * r.y - l.x * l.y) * 0.5;
    let flux_y = (d.x * d.z - u.x * u.z) * 0.5;
    let new_h = h_avg - DT * (flux_x + flux_y);

    field_out[idx(x, y)] = vec4<f32>(new_h, new_vx, new_vy, 0.0);
}
