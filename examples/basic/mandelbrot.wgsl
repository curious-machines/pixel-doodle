struct Params {
    width: u32,
    height: u32,
    max_iter: u32,
    stride: u32,  // pixels per row in storage buffer (aligned)
    x_min: f32,
    y_min: f32,
    x_step: f32,
    y_step: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

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

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    if col >= params.width || row >= params.height {
        return;
    }
    let cx = params.x_min + f32(col) * params.x_step;
    let cy = params.y_min + f32(row) * params.y_step;
    let iter = mandelbrot(cx, cy, params.max_iter);
    let color = iter_to_color(iter, params.max_iter);
    let idx = row * params.stride + col;
    output[idx] = color;
}
