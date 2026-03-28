// Animated 2D SDF demo — rounded cross with pulsing height/radius
// Ported from sdf2d_demo.pd (sd_rounded_cross inlined from sdf2d_library)

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

fn sd_rounded_cross(p: vec2<f32>, h: f32) -> f32 {
    let k = 0.5 * (h + 1.0 / h);
    let ap = abs(p);
    let va = vec2<f32>(ap.x - 1.0, ap.y - k);
    let vb = vec2<f32>(ap.x, ap.y - h);
    let vc = vec2<f32>(ap.x - 1.0, ap.y);
    if ap.x < 1.0 && ap.y < ap.x * (k - h) + h {
        return k - length(va);
    }
    return sqrt(min(dot(vb, vb), dot(vc, vc)));
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
    let p = vec2<f32>(x, y);

    let height = 0.501 - 0.499 * cos(params.time * 1.1);
    let radius = 0.100 + 0.100 * sin(params.time * 1.7 + 2.0);
    let d = sd_rounded_cross(p, height) - radius;

    // Shadertoy-style visualization
    var col_rgb: vec3<f32>;
    if d > 0.0 {
        col_rgb = vec3<f32>(0.9, 0.6, 0.3);
    } else {
        col_rgb = vec3<f32>(0.65, 0.85, 1.0);
    }

    col_rgb *= 1.0 - exp(-6.0 * abs(d));
    col_rgb *= 0.8 + 0.2 * cos(150.0 * d);
    let edge = 1.0 - smoothstep(0.0, 0.01, abs(d));
    col_rgb = mix(col_rgb, vec3<f32>(1.0), edge);

    let idx = row * params.stride + col;
    output[idx] = pack_rgb(col_rgb);
}
