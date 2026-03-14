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

fn gear_sdf(px: f32, py: f32, gcx: f32, gcy: f32, r: f32, teeth: f32, depth: f32) -> f32 {
    let dx = px - gcx;
    let dy = py - gcy;
    let dist = sqrt(dx * dx + dy * dy);
    let angle = atan2(dy, dx);

    // Near-square wave via clamped sin
    let wave = clamp(sin(teeth * angle) * 8.0, -1.0, 1.0);
    let gear_r = r + wave * depth;
    return dist - gear_r;
}

fn circle_sdf(px: f32, py: f32, ccx: f32, ccy: f32, r: f32) -> f32 {
    let dx = px - ccx;
    let dy = py - ccy;
    return sqrt(dx * dx + dy * dy) - r;
}

fn eval_kernel(cx: f32, cy: f32) -> u32 {
    // Three interlocking gears
    let d_g1 = gear_sdf(cx, cy, -0.45, 0.0, 0.5, 12.0, 0.06);
    let d_g2 = gear_sdf(cx, cy, 0.45, 0.0, 0.35, 8.0, 0.06);
    let d_g3 = gear_sdf(cx, cy, 0.0, -0.65, 0.25, 6.0, 0.05);

    // Hub holes
    let d_hub1 = circle_sdf(cx, cy, -0.45, 0.0, 0.15);
    let d_hub2 = circle_sdf(cx, cy, 0.45, 0.0, 0.10);
    let d_hub3 = circle_sdf(cx, cy, 0.0, -0.65, 0.08);

    // Spoke holes in large gear
    let d_sp1 = circle_sdf(cx, cy, -0.72, 0.0, 0.06);
    let d_sp2 = circle_sdf(cx, cy, -0.18, 0.0, 0.06);
    let d_sp3 = circle_sdf(cx, cy, -0.45, 0.27, 0.06);
    let d_sp4 = circle_sdf(cx, cy, -0.45, -0.27, 0.06);

    // Union gears, union holes, subtract
    let d_gears = min(min(d_g1, d_g2), d_g3);
    let d_holes = min(min(min(d_hub1, d_hub2), d_hub3), min(min(d_sp1, d_sp2), min(d_sp3, d_sp4)));
    let d_final = max(d_gears, -d_holes);

    let in_gear = d_final <= 0.0;
    let in_g1 = d_g1 <= 0.0;
    let in_g2 = d_g2 <= 0.0;

    // Colors per gear
    var base_r: f32;
    var base_g: f32;
    var base_b: f32;

    if in_g1 {
        // Copper
        base_r = 180.0; base_g = 120.0; base_b = 80.0;
    } else if in_g2 {
        // Silver
        base_r = 160.0; base_g = 170.0; base_b = 180.0;
    } else {
        // Gold
        base_r = 200.0; base_g = 180.0; base_b = 80.0;
    }

    // Edge highlight
    let edge_glow = exp(-abs(d_final) * 60.0) * 40.0;
    let lit_r = base_r + edge_glow;
    let lit_g = base_g + edge_glow;
    let lit_b = base_b + edge_glow;

    var r: f32;
    var g: f32;
    var b: f32;
    if in_gear {
        r = lit_r; g = lit_g; b = lit_b;
    } else {
        r = 20.0; g = 22.0; b = 28.0;
    }

    let ri = u32(clamp(r, 0.0, 255.0));
    let gi = u32(clamp(g, 0.0, 255.0));
    let bi = u32(clamp(b, 0.0, 255.0));
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
