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
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<storage, read_write> accum: array<vec4<f32>>;

fn pixel_hash(col: u32, row: u32, sample_index: u32) -> u32 {
    let k = 0x45d9f3bu;
    var h = col * k + row;
    h = h * k + sample_index;
    h = h ^ (h >> 16u);
    h = h * k;
    h = h ^ (h >> 16u);
    return h;
}

fn unpack_rgb(color: u32) -> vec3<f32> {
    let r = f32((color >> 16u) & 0xFFu);
    let g = f32((color >> 8u) & 0xFFu);
    let b = f32(color & 0xFFu);
    return vec3<f32>(r, g, b);
}

fn pack_rgb(rgb: vec3<f32>) -> u32 {
    let r = u32(clamp(rgb.x, 0.0, 255.0));
    let g = u32(clamp(rgb.y, 0.0, 255.0));
    let b = u32(clamp(rgb.z, 0.0, 255.0));
    return 0xFF000000u | (r << 16u) | (g << 8u) | b;
}

fn eval_kernel(cx: f32, cy: f32) -> u32 {
    let r = sqrt(cx * cx + cy * cy);

    let spacing: f32 = 0.1;
    let thickness: f32 = 0.005;

    let fr = fract(r / spacing);
    let centered = abs(fr - 0.5);
    let ring_sdf = centered * spacing - thickness;

    let inside = ring_sdf <= 0.0;

    var out_r: u32;
    var out_g: u32;
    var out_b: u32;
    if inside {
        out_r = 240u;
        out_g = 240u;
        out_b = 240u;
    } else {
        out_r = 30u;
        out_g = 30u;
        out_b = 40u;
    }

    return 0xFF000000u | (out_r << 16u) | (out_g << 8u) | out_b;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    if col >= params.width || row >= params.height {
        return;
    }

    let idx = row * params.stride + col;

    var jx: f32 = 0.0;
    var jy: f32 = 0.0;
    if params.sample_index != 0xFFFFFFFFu {
        let h = pixel_hash(col, row, params.sample_index);
        jx = f32(h & 0xFFFFu) / 65536.0;
        jy = f32(h >> 16u) / 65536.0;
    }
    let cx = params.x_min + (f32(col) + jx) * params.x_step;
    let cy = params.y_min + (f32(row) + jy) * params.y_step;

    let color = eval_kernel(cx, cy);

    if params.sample_count > 0u {
        let rgb = unpack_rgb(color);
        let prev = accum[idx];
        let updated = vec4<f32>(prev.x + rgb.x, prev.y + rgb.y, prev.z + rgb.z, 0.0);
        accum[idx] = updated;
        let inv = 1.0 / f32(params.sample_count);
        let resolved = vec3<f32>(updated.x * inv, updated.y * inv, updated.z * inv);
        output[idx] = pack_rgb(resolved);
    } else {
        output[idx] = color;
    }
}
