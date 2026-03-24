struct Params {
    width: u32,
    height: u32,
    stride: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read>       field_in: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> pixels:   array<u32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= params.width || y >= params.height {
        return;
    }

    let field_idx = y * params.width + x;
    let pixel_idx = y * params.stride + x;
    let deviation = field_in[field_idx].x - 1.0;

    // Color ramp: deep blue (trough) -> mid blue (rest) -> white (peak)
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
