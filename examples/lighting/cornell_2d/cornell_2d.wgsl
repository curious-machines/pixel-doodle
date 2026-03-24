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

const LIGHT_X: f32 = -0.7;
const LIGHT_Y: f32 = -0.5;
const LIGHT_R: f32 = 0.06;
const EMISSION: f32 = 60.0;
const AMBIENT: f32 = 0.08;

fn box_sdf(px: f32, py: f32) -> f32 {
    // Box centered at (0.05, 0.0), half-widths (0.3, 0.3)
    let dx = abs(px - 0.05) - 0.3;
    let dy = abs(py) - 0.3;
    return length(vec2(max(dx, 0.0), max(dy, 0.0))) + min(max(dx, dy), 0.0);
}

fn scene_sdf(px: f32, py: f32) -> f32 {
    return box_sdf(px, py);
}

fn light_sdf(px: f32, py: f32) -> f32 {
    let dx = px - LIGHT_X;
    let dy = py - LIGHT_Y;
    return sqrt(dx * dx + dy * dy) - LIGHT_R;
}

fn full_sdf(px: f32, py: f32) -> f32 {
    return min(scene_sdf(px, py), light_sdf(px, py));
}

fn compute_normal(px: f32, py: f32) -> vec2<f32> {
    let eps: f32 = 0.002;
    let nx = scene_sdf(px + eps, py) - scene_sdf(px - eps, py);
    let ny = scene_sdf(px, py + eps) - scene_sdf(px, py - eps);
    let len = max(sqrt(nx * nx + ny * ny), 0.0001);
    return vec2<f32>(nx / len, ny / len);
}

// Direct illumination: march toward a random point on the light, check visibility
fn direct_illumination(px: f32, py: f32, seed: u32) -> f32 {
    // Random point on light circle for soft shadows
    let light_angle = f32(seed) / 4294967296.0 * 6.283185307;
    let target_x = LIGHT_X + cos(light_angle) * LIGHT_R;
    let target_y = LIGHT_Y + sin(light_angle) * LIGHT_R;

    let to_x = target_x - px;
    let to_y = target_y - py;
    let dist = sqrt(to_x * to_x + to_y * to_y);
    let dx = to_x / dist;
    let dy = to_y / dist;

    // March toward light
    var mrx = px;
    var mry = py;
    var total: f32 = 0.0;
    let min_d: f32 = 0.001;

    for (var step: u32 = 0u; step < 64u; step = step + 1u) {
        let d = abs(full_sdf(mrx, mry));
        total = total + d;
        if d < min_d || total > dist + 0.1 { break; }
        mrx = mrx + dx * d;
        mry = mry + dy * d;
    }

    // Check if we reached the light
    if light_sdf(mrx, mry) <= 0.002 {
        // Visible: brightness = emission * light_radius / (π * distance)
        return EMISSION * LIGHT_R / (3.14159 * dist) + AMBIENT;
    }
    return AMBIENT;
}

fn eval_kernel(cx: f32, cy: f32, col: u32, row: u32, sample_idx: u32) -> u32 {
    // Inside box → black
    if box_sdf(cx, cy) <= 0.0 {
        return 0xFF000000u;
    }

    // All non-box pixels: direct illumination with soft shadows
    let seed = pixel_hash(col, row);
    let seed2 = pixel_hash(seed, sample_idx);
    let rng = pixel_hash(seed2, sample_idx);
    let brightness = direct_illumination(cx, cy, rng);
    let mapped = brightness / (1.0 + brightness) * 255.0;
    let v = u32(clamp(mapped, 0.0, 255.0));
    return 0xFF000000u | (v << 16u) | (v << 8u) | v;
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
