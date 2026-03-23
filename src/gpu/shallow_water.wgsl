struct Params {
    width: u32,
    height: u32,
    stride: u32,
    _pad0: u32,
    gravity: f32,
    damping: f32,
    dt: f32,
    _pad1: u32,
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
fn step(@builtin(global_invocation_id) gid: vec3<u32>) {
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
    let new_vx = vx_avg - params.dt * params.gravity * dh_dx - params.dt * params.damping * vx_avg;
    let new_vy = vy_avg - params.dt * params.gravity * dh_dy - params.dt * params.damping * vy_avg;

    // Flux divergence for height update (Lax-Friedrichs)
    let flux_x = (r.x * r.y - l.x * l.y) * 0.5;
    let flux_y = (d.x * d.z - u.x * u.z) * 0.5;
    let new_h = h_avg - params.dt * (flux_x + flux_y);

    field_out[idx(x, y)] = vec4<f32>(new_h, new_vx, new_vy, 0.0);
}

// Visualization: convert field to ARGB pixels
@group(0) @binding(0) var<uniform> vis_params: Params;
@group(0) @binding(1) var<storage, read>       vis_field: array<vec4<f32>>;
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
    let deviation = vis_field[field_idx].x - 1.0;

    // Color ramp: deep blue (trough) → mid blue (rest) → white (peak)
    var r: f32;
    var g: f32;
    var b: f32;

    if deviation < -0.1 {
        r = 0.0; g = 0.0; b = 0.3;
    } else if deviation < 0.0 {
        let t = (deviation + 0.1) / 0.1;
        r = 0.0; g = 0.1 * t; b = 0.3 + 0.4 * t;
    } else if deviation < 0.05 {
        let t = deviation / 0.05;
        r = 0.1 * t; g = 0.1 + 0.3 * t; b = 0.7 + 0.15 * t;
    } else if deviation < 0.15 {
        let t = (deviation - 0.05) / 0.1;
        r = 0.1 + 0.9 * t; g = 0.4 + 0.6 * t; b = 0.85 + 0.15 * t;
    } else {
        r = 1.0; g = 1.0; b = 1.0;
    }

    let rb = u32(clamp(r, 0.0, 1.0) * 255.0);
    let gb = u32(clamp(g, 0.0, 1.0) * 255.0);
    let bb = u32(clamp(b, 0.0, 1.0) * 255.0);
    pixels[pixel_idx] = 0xFF000000u | (rb << 16u) | (gb << 8u) | bb;
}
