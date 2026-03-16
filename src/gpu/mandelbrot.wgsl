struct Params {
    width: u32,
    height: u32,
    max_iter: u32,
    stride: u32,  // pixels per row in storage buffer (aligned)
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

fn mandelbrot(cx: f32, cy: f32, max_iter: u32) -> u32 {
    var zx: f32 = 0.0;
    var zy: f32 = 0.0;
    var i: u32 = 0u;
    loop {
        if zx * zx + zy * zy > 4.0 || i >= max_iter {
            break;
        }
        let tmp = zx * zx - zy * zy + cx;
        zy = 2.0 * zx * zy + cy;
        zx = tmp;
        i = i + 1u;
    }
    return i;
}

fn iter_to_color(iter: u32, max_iter: u32) -> u32 {
    if iter == max_iter {
        return 0x00000000u;
    }
    let t = f32(iter) / f32(max_iter);
    let one_minus_t = 1.0 - t;
    let r = u32(9.0 * one_minus_t * t * t * t * 255.0);
    let g = u32(15.0 * one_minus_t * one_minus_t * t * t * 255.0);
    let b = u32(8.5 * one_minus_t * one_minus_t * one_minus_t * t * 255.0);
    // Pack as BGRA for Bgra8Unorm texture: byte layout [B, G, R, A]
    // As a u32 in little-endian: A<<24 | R<<16 | G<<8 | B
    return 0xFF000000u | (min(r, 255u) << 16u) | (min(g, 255u) << 8u) | min(b, 255u);
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

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    if col >= params.width || row >= params.height {
        return;
    }

    let idx = row * params.stride + col;

    // Compute jittered coordinates (or center if non-progressive)
    var jx: f32 = 0.0;
    var jy: f32 = 0.0;
    if params.sample_index != 0xFFFFFFFFu {
        let h = pixel_hash(col, row, params.sample_index);
        jx = f32(h & 0xFFFFu) / 65536.0;
        jy = f32(h >> 16u) / 65536.0;
    }
    let cx = params.x_min + (f32(col) + jx) * params.x_step;
    let cy = params.y_min + (f32(row) + jy) * params.y_step;

    let iter = mandelbrot(cx, cy, params.max_iter);
    let color = iter_to_color(iter, params.max_iter);

    if params.sample_count > 0u {
        // Progressive mode: accumulate and resolve
        let rgb = unpack_rgb(color);
        let prev = accum[idx];
        let updated = vec4<f32>(prev.x + rgb.x, prev.y + rgb.y, prev.z + rgb.z, 0.0);
        accum[idx] = updated;
        let inv = 1.0 / f32(params.sample_count);
        let resolved = vec3<f32>(updated.x * inv, updated.y * inv, updated.z * inv);
        output[idx] = pack_rgb(resolved);
    } else {
        // Non-progressive: direct write
        output[idx] = color;
    }
}
