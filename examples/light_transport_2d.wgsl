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

fn pixel_hash(a: u32, b: u32) -> u32 {
    let k = 0x45d9f3bu;
    var h = a * k + b;
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
    let wave = clamp(sin(teeth * angle) * 8.0, -1.0, 1.0);
    let gear_r = r + wave * depth;
    return dist - gear_r;
}

fn circle_sdf(px: f32, py: f32, ccx: f32, ccy: f32, r: f32) -> f32 {
    let dx = px - ccx;
    let dy = py - ccy;
    return sqrt(dx * dx + dy * dy) - r;
}

fn scene_sdf(px: f32, py: f32) -> f32 {
    let d_g1 = gear_sdf(px, py, -0.45, 0.0, 0.5, 12.0, 0.06);
    let d_g2 = gear_sdf(px, py, 0.45, 0.0, 0.35, 8.0, 0.06);
    let d_g3 = gear_sdf(px, py, 0.0, -0.65, 0.25, 6.0, 0.05);

    let d_hub1 = circle_sdf(px, py, -0.45, 0.0, 0.15);
    let d_hub2 = circle_sdf(px, py, 0.45, 0.0, 0.10);
    let d_hub3 = circle_sdf(px, py, 0.0, -0.65, 0.08);

    let d_sp1 = circle_sdf(px, py, -0.72, 0.0, 0.06);
    let d_sp2 = circle_sdf(px, py, -0.18, 0.0, 0.06);
    let d_sp3 = circle_sdf(px, py, -0.45, 0.27, 0.06);
    let d_sp4 = circle_sdf(px, py, -0.45, -0.27, 0.06);

    let d_gears = min(min(d_g1, d_g2), d_g3);
    let d_holes = min(min(min(d_hub1, d_hub2), d_hub3), min(min(d_sp1, d_sp2), min(d_sp3, d_sp4)));
    return max(d_gears, -d_holes);
}

fn light_sdf(px: f32, py: f32) -> f32 {
    let d_l1 = circle_sdf(px, py, 0.0, 0.8, 0.12);
    let d_l2 = circle_sdf(px, py, -0.8, -0.3, 0.08);
    return min(d_l1, d_l2);
}

fn full_scene_sdf(px: f32, py: f32) -> f32 {
    return min(scene_sdf(px, py), light_sdf(px, py));
}

fn compute_normal(px: f32, py: f32) -> vec2<f32> {
    let eps: f32 = 0.002;
    let nx = scene_sdf(px + eps, py) - scene_sdf(px - eps, py);
    let ny = scene_sdf(px, py + eps) - scene_sdf(px, py - eps);
    let len = max(sqrt(nx * nx + ny * ny), 0.0001);
    return vec2<f32>(nx / len, ny / len);
}

fn eval_kernel(cx: f32, cy: f32, col: u32, row: u32, sample_idx: u32) -> u32 {
    let d_pixel = scene_sdf(cx, cy);
    let in_gear = d_pixel <= 0.0;

    // Compute surface normal at pixel for first bounce setup
    let n0 = compute_normal(cx, cy);
    let nudge_dist: f32 = 0.005;

    // Project to surface then nudge outward:
    // Start = pixel + normal * (|d_pixel| + nudge_dist)
    let proj_dist = max(-d_pixel, 0.0) + nudge_dist;
    var ray_x = cx + n0.x * proj_dist;
    var ray_y = cy + n0.y * proj_dist;

    // Initial random direction in outward normal half-plane
    let seed = pixel_hash(col, row);
    let seed2 = pixel_hash(seed, sample_idx);
    let rand_angle = f32(seed2) / 4294967296.0;
    let n0_angle = atan2(n0.y, n0.x);
    let theta0 = n0_angle + (rand_angle * 3.141592653589793 - 1.5707963267948966);
    var ray_dx = cos(theta0);
    var ray_dy = sin(theta0);
    var rng = pixel_hash(seed2, sample_idx);

    // Per-gear albedo colors
    let d_g1 = gear_sdf(cx, cy, -0.45, 0.0, 0.5, 12.0, 0.06);
    let d_g2 = gear_sdf(cx, cy, 0.45, 0.0, 0.35, 8.0, 0.06);
    var alb: vec3<f32>;
    if d_g1 <= 0.0 {
        alb = vec3<f32>(0.75, 0.5, 0.35);   // copper
    } else if d_g2 <= 0.0 {
        alb = vec3<f32>(0.65, 0.68, 0.72);  // silver
    } else {
        alb = vec3<f32>(0.8, 0.72, 0.35);   // gold
    }

    // Path tracing state
    // Ambient base: ensures interior pixels aren't pure black
    let ambient = alb * 0.45;
    var rad = ambient;
    var throughput = alb;
    var alive = in_gear;

    let max_dist: f32 = 2.0;
    let min_d: f32 = 0.001;
    let emission = vec3<f32>(6.0, 5.5, 4.0);
    let sky = vec3<f32>(0.5, 0.5, 0.55);

    // Bounce loop (8 bounces for better indirect illumination)
    for (var bounce: u32 = 0u; bounce < 8u; bounce = bounce + 1u) {
        if !alive {
            break;
        }

        // Ray march
        var mrx = ray_x;
        var mry = ray_y;
        var total_d: f32 = 0.0;
        for (var step: u32 = 0u; step < 64u; step = step + 1u) {
            let d = abs(full_scene_sdf(mrx, mry));
            total_d = total_d + d;
            if d < min_d || total_d > max_dist {
                break;
            }
            mrx = mrx + ray_dx * d;
            mry = mry + ray_dy * d;
        }

        // Classify hit by re-evaluating SDF at endpoint
        let d_end = abs(full_scene_sdf(mrx, mry));
        let escaped = d_end > min_d;
        let hit_light = light_sdf(mrx, mry) <= min_d;

        if escaped {
            rad = rad + throughput * sky;
            alive = false;
        } else if hit_light {
            rad = rad + throughput * emission;
            alive = false;
        } else {
            // Compute normal via central differences
            let n = compute_normal(mrx, mry);

            // Lambertian: random direction in normal half-plane
            rng = pixel_hash(rng, bounce);
            let rand2 = f32(rng) / 4294967296.0;
            let n_angle = atan2(n.y, n.x);
            let new_theta = n_angle + (rand2 * 3.141592653589793 - 1.5707963267948966);
            ray_dx = cos(new_theta);
            ray_dy = sin(new_theta);

            // Cosine weight
            let cos_w = max(ray_dx * n.x + ray_dy * n.y, 0.0);
            throughput = throughput * alb * cos_w;

            // Nudge origin along normal
            ray_x = mrx + n.x * nudge_dist;
            ray_y = mry + n.y * nudge_dist;
        }
    }

    // Reinhard tone map per channel, then scale to 255
    var r: f32;
    var g: f32;
    var b: f32;
    if in_gear {
        r = rad.x / (1.0 + rad.x) * 255.0;
        g = rad.y / (1.0 + rad.y) * 255.0;
        b = rad.z / (1.0 + rad.z) * 255.0;
    } else {
        r = 20.0; g = 22.0; b = 28.0;
    }

    return 0xFF000000u | (u32(clamp(r, 0.0, 255.0)) << 16u) | (u32(clamp(g, 0.0, 255.0)) << 8u) | u32(clamp(b, 0.0, 255.0));
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
        let h = pixel_hash(col * 0x45d9f3bu + row, params.sample_index);
        jx = f32(h & 0xFFFFu) / 65536.0;
        jy = f32(h >> 16u) / 65536.0;
    }
    let cx = params.x_min + (f32(col) + jx) * params.x_step;
    let cy = params.y_min + (f32(row) + jy) * params.y_step;

    let color = eval_kernel(cx, cy, col, row, params.sample_index);

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
