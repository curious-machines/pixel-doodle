// Animated pulsing circle SDF — demonstrates params.time in WGSL

struct Params {
    width: u32,
    height: u32,
    max_iter: u32,
    stride: u32,
    x_min: f32,
    y_min: f32,
    x_step: f32,
    y_step: f32,
    _pad0: u32,
    _pad1: u32,
    time: f32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

fn sd_circle(p: vec2<f32>, r: f32) -> f32 {
    return length(p) - r;
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

    let idx = row * params.stride + col;
    let x = params.x_min + f32(col) * params.x_step;
    let y = params.y_min + f32(row) * params.y_step;
    let p = vec2<f32>(x, y);

    let r = 0.4 + 0.15 * sin(params.time * 3.0);
    let d = sd_circle(p, r);

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

    output[idx] = pack_rgb(col_rgb);
}
