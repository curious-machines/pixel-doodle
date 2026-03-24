// Gray-Scott visualization: convert field (u, v) to ARGB pixels

struct Params {
    width: u32,
    height: u32,
    stride: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read>       field_in: array<vec2<f32>>;
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
    let v = clamp(field_in[field_idx].y, 0.0, 1.0);

    // Color ramp: black -> blue -> white -> orange
    var r: f32;
    var g: f32;
    var b: f32;

    if v < 0.25 {
        let t = v / 0.25;
        r = 0.0; g = 0.0; b = t;
    } else if v < 0.5 {
        let t = (v - 0.25) / 0.25;
        r = t; g = t; b = 1.0;
    } else if v < 0.75 {
        let t = (v - 0.5) / 0.25;
        r = 1.0; g = 1.0 - 0.35 * t; b = 1.0 - t;
    } else {
        let t = (v - 0.75) / 0.25;
        r = 1.0; g = 0.65 - 0.25 * t; b = t * 0.1;
    }

    let rb = u32(r * 255.0);
    let gb = u32(g * 255.0);
    let bb = u32(b * 255.0);
    pixels[pixel_idx] = 0xFF000000u | (rb << 16u) | (gb << 8u) | bb;
}
