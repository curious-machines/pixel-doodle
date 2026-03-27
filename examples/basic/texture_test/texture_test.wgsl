// Texture sampling test — displays a texture mapped to the viewport with
// aspect-ratio-preserving letterbox/pillarbox.

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
@group(0) @binding(3) var tex_sampler: sampler;
@group(0) @binding(4) var img: texture_2d<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    if col >= params.width || row >= params.height {
        return;
    }

    let tex_dims = textureDimensions(img);
    let tw = f32(tex_dims.x);
    let th = f32(tex_dims.y);
    let tex_aspect = tw / th;
    let scr_aspect = f32(params.width) / f32(params.height);

    let u_raw = f32(col) / f32(params.width);
    let v_raw = f32(row) / f32(params.height);

    // Fit texture to screen (letterbox/pillarbox)
    var u_scale = 1.0;
    var v_scale = 1.0;
    if scr_aspect > tex_aspect {
        u_scale = tex_aspect / scr_aspect;
    } else {
        v_scale = scr_aspect / tex_aspect;
    }
    let u = (u_raw - 0.5) / u_scale + 0.5;
    let v = (v_raw - 0.5) / v_scale + 0.5;

    let in_bounds = u >= 0.0 && u <= 1.0 && v >= 0.0 && v <= 1.0;

    var color = 0xFF000000u; // black
    if in_bounds {
        let rgba = textureSampleLevel(img, tex_sampler, vec2<f32>(u, v), 0.0);
        let r = u32(rgba.x * 255.0);
        let g = u32(rgba.y * 255.0);
        let b = u32(rgba.z * 255.0);
        color = 0xFF000000u | (r << 16u) | (g << 8u) | b;
    }

    let idx = row * params.stride + col;
    output[idx] = color;
}
