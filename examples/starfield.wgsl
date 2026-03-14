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

fn hash1(cx: f32, cy: f32) -> f32 {
    return fract(sin(cx * 127.1 + cy * 311.7) * 43758.5453);
}

fn hash2(cx: f32, cy: f32) -> f32 {
    return fract(sin(cx * 269.5 + cy * 183.3) * 43758.5453);
}

fn hash3(cx: f32, cy: f32) -> f32 {
    return fract(sin(cx * 419.2 + cy * 371.9) * 43758.5453);
}

fn hash4(cx: f32, cy: f32) -> f32 {
    return fract(sin(cx * 547.3 + cy * 251.7) * 43758.5453);
}

fn star_contrib(px: f32, py: f32, cell_x: f32, cell_y: f32) -> f32 {
    let sx = hash1(cell_x, cell_y);
    let sy = hash2(cell_x, cell_y);
    let br = hash3(cell_x, cell_y);

    if br <= 0.7 {
        return 0.0;
    }

    let bright = (br - 0.7) / 0.3;
    let star_gx = cell_x + sx;
    let star_gy = cell_y + sy;
    let dx = px - star_gx;
    let dy = py - star_gy;
    let dist = sqrt(dx * dx + dy * dy);

    let star_r = 0.08 + bright * 0.12;
    let glow = exp(-dist / star_r * 3.0);
    return glow * bright;
}

fn eval_kernel(cx: f32, cy: f32) -> u32 {
    let scale: f32 = 10.0;
    let px = cx * scale;
    let py = cy * scale;

    let cell_x = floor(px);
    let cell_y = floor(py);

    // Check 3x3 neighborhood
    var intensity: f32 = 0.0;
    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
            let nx = cell_x + f32(dx);
            let ny = cell_y + f32(dy);
            intensity = intensity + star_contrib(px, py, nx, ny);
        }
    }

    // Color temperature from center cell
    let temp = hash4(cell_x, cell_y);
    let inv_temp = 1.0 - temp;
    let star_r = 180.0 * inv_temp + 255.0 * temp;
    let star_g = 200.0 * inv_temp + 220.0 * temp;
    let star_b = 255.0 * inv_temp + 170.0 * temp;

    let r = clamp(3.0 + star_r * intensity, 0.0, 255.0);
    let g = clamp(3.0 + star_g * intensity, 0.0, 255.0);
    let b = clamp(8.0 + star_b * intensity, 0.0, 255.0);

    let ri = u32(r);
    let gi = u32(g);
    let bi = u32(b);
    return 0xFF000000u | (ri << 16u) | (gi << 8u) | bi;
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
