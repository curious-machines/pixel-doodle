// SDF scene: circle + axis-aligned box, union'd.
// Distance-based coloring with glow/falloff.
// WGSL equivalent of sdf.pdl

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

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    if col >= params.width || row >= params.height {
        return;
    }
    let x = params.x_min + f32(col) * params.x_step;
    let y = params.y_min + f32(row) * params.y_step;

    // Circle SDF: centered at (-0.5, 0.0), radius 0.4
    let cx = x + 0.5;
    let d_circle = sqrt(cx * cx + y * y) - 0.4;

    // Box SDF: centered at (0.5, 0.0), half-size 0.3 x 0.25
    let bx = abs(x - 0.5) - 0.3;
    let by = abs(y) - 0.25;
    let d_box = max(bx, by);

    // Union
    let d = min(d_circle, d_box);

    // Glow falloff: intensity = clamp(1.0 - d * 4.0, 0.0, 1.0)
    let intensity = clamp(1.0 - d * 4.0, 0.0, 1.0);

    let r = u32(intensity * 100.0);
    let g = u32(intensity * 200.0);
    let b = u32(intensity * 255.0);
    let color = 0xFF000000u | (r << 16u) | (g << 8u) | b;

    let idx = row * params.stride + col;
    output[idx] = color;
}
