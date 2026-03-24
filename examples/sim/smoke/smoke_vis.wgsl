struct Params {
    width: u32,
    height: u32,
    stride: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read>       field_in: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> pixels:   array<u32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= params.width || y >= params.height { return; }

    let field_idx = y * params.width + x;
    let pixel_idx = y * params.stride + x;

    let density = field_in[field_idx].z;
    let intensity = clamp(density, 0.0, 1.0);

    let v = u32(intensity * 255.0);
    pixels[pixel_idx] = 0xFF000000u | (v << 16u) | (v << 8u) | v;
}
