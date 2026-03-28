// Animated wobbling circle SDF — sinusoidal position perturbation
// Ported from circle-wobble.pd

struct Params {
    width: u32,
    height: u32,
    max_iter: u32,
    stride: u32,
    x_min: f32,
    y_min: f32,
    x_step: f32,
    y_step: f32,
    sample_index: u32,
    sample_count: u32,
    time: f32,
    _pad0: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

fn sd_circle(p: vec2<f32>, r: f32) -> f32 {
    return length(p) - r;
}

fn wobble(p: vec2<f32>, freq: vec2<f32>, amt: vec2<f32>) -> vec2<f32> {
    let w = vec2<f32>(sin(p.y * freq.x) * amt.x, sin(p.x * freq.y) * amt.y);
    return p + w;
}

fn pack_rgb(rgb: vec3<f32>) -> u32 {
    let r = u32(clamp(rgb.x * 255.0, 0.0, 255.0));
    let g = u32(clamp(rgb.y * 255.0, 0.0, 255.0));
    let b = u32(clamp(rgb.z * 255.0, 0.0, 255.0));
    return 0xFF000000u | (r << 16u) | (g << 8u) | b;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    if col >= params.width || row >= params.height {
        return;
    }

    let x = params.x_min + f32(col) * params.x_step;
    let y = params.y_min + f32(row) * params.y_step;

    let pi = 3.14159265;
    let frequency = 5.0;
    let amount = 0.05;
    let offset_val = params.time % (pi * 2.0 / frequency);
    let offset = vec2<f32>(offset_val, offset_val);
    let p1 = vec2<f32>(x, y) + offset;
    let p2 = wobble(p1, vec2<f32>(frequency), vec2<f32>(amount));
    let p3 = p2 - offset;
    let d = sd_circle(p3, 0.5);

    // Coloring: warm outside, cool inside
    var col_rgb: vec3<f32>;
    if d > 0.0 {
        col_rgb = vec3<f32>(0.9, 0.6, 0.3);
    } else {
        col_rgb = vec3<f32>(0.65, 0.85, 1.0);
    }

    // Distance-based darkening + contour rings + edge highlight
    col_rgb *= 1.0 - exp(-6.0 * abs(d));
    col_rgb *= 0.8 + 0.2 * cos(150.0 * d);
    let edge = 1.0 - smoothstep(0.0, 0.01, abs(d));
    col_rgb = mix(col_rgb, vec3<f32>(1.0), edge);

    let idx = row * params.stride + col;
    output[idx] = pack_rgb(col_rgb);
}
