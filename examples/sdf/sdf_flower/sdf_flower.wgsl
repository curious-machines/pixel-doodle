// SDF scene: six-petal flower + concentric ring, blended with smooth min.
// Uses polar coords (atan2, sqrt), sinusoidal modulation (sin, cos),
// exponential smooth min (exp, log), glow falloff (exp), gamma (pow).
// WGSL equivalent of sdf_flower.pdir

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

// Exponential smooth minimum: blends two distances with soft transition.
fn smin(a: f32, b: f32, k: f32) -> f32 {
    let ea = exp(-k * a);
    let eb = exp(-k * b);
    return -log(ea + eb) / k;
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

    // Polar coordinates
    let r = sqrt(x * x + y * y);
    let angle = atan2(y, x);

    // Flower SDF: circle modulated by sin(6θ)
    let flower_r = 0.45 + 0.12 * sin(angle * 6.0);
    let d_flower = r - flower_r;

    // Ring SDF: |r - 0.85| - 0.04
    let d_ring = abs(r - 0.85) - 0.04;

    // Smooth min blend (k = 12)
    let d = smin(d_flower, d_ring, 12.0);

    // Brightness: sharp fill inside + soft glow halo outside
    let fill = clamp(-d * 40.0, 0.0, 1.0);
    let glow = exp(-abs(d) * 6.0) * 0.4;
    let brightness = min(fill + glow, 1.0);

    // Angular rainbow: hue = 0.5 + 0.5·cos(θ + phase), 120° apart
    let r_base = 0.5 + 0.5 * cos(angle);
    let g_base = 0.5 + 0.5 * cos(angle + 2.094);
    let b_base = 0.5 + 0.5 * cos(angle + 4.189);

    // Gamma correction to lift midtones
    let r_gamma = pow(r_base, 0.8);
    let g_gamma = pow(g_base, 0.8);
    let b_gamma = pow(b_base, 0.8);

    // Final color
    let rc = u32(r_gamma * brightness * 255.0);
    let gc = u32(g_gamma * brightness * 255.0);
    let bc = u32(b_gamma * brightness * 255.0);
    let color = 0xFF000000u | (rc << 16u) | (gc << 8u) | bc;

    let idx = row * params.stride + col;
    output[idx] = color;
}
