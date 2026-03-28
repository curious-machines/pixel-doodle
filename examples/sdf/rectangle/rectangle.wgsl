// 2D Rectangle SDF — Shadertoy-style visualization
// Ported from rectangle.pd

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

fn sd_rectangle(p: vec2<f32>, b: vec2<f32>) -> f32 {
    let d = abs(p) - b;
    return length(max(d, vec2<f32>(0.0))) + min(max(d.x, d.y), 0.0);
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
    let d = sd_rectangle(vec2<f32>(x, y), vec2<f32>(0.5, 0.25));

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
