// 3D Raymarched SDF demo — rounded box + sphere + torus with smooth union
// Ported from sdf3d_demo.pd (library functions inlined)

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
    time: f32,
    _pad0: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

// Smooth union boolean
fn opSmoothUnion(d1: f32, d2: f32, k: f32) -> f32 {
    let h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return mix(d2, d1, h) - k * h * (1.0 - h);
}

// Rounded box SDF
fn sdRoundBox(p: vec3<f32>, b: vec3<f32>, r: f32) -> f32 {
    let q = abs(p) - b + vec3<f32>(r);
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}

// Sphere SDF
fn sdSphere(p: vec3<f32>, r: f32) -> f32 {
    return length(p) - r;
}

// Torus SDF
fn sdTorus(p: vec3<f32>, tx: f32, ty: f32) -> f32 {
    let q = vec2<f32>(length(vec2<f32>(p.x, p.z)) - tx, p.y);
    return length(q) - ty;
}

// Scene distance
fn map(p: vec3<f32>) -> f32 {
    let d_box = sdRoundBox(p - vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.3), 0.05);
    let d_sphere = sdSphere(p - vec3<f32>(0.5, 0.2, 0.0), 0.3);
    let d_torus = sdTorus(p - vec3<f32>(-0.3, 0.4, 0.3), 0.35, 0.1);
    let d1 = opSmoothUnion(d_box, d_sphere, 0.2);
    return opSmoothUnion(d1, d_torus, 0.2);
}

// Tetrahedron technique for SDF normals
fn calc_normal(p: vec3<f32>) -> vec3<f32> {
    let eps = 0.0005;
    let k = 0.5773;
    let e1 = vec3<f32>( k, -k, -k);
    let e2 = vec3<f32>(-k, -k,  k);
    let e3 = vec3<f32>(-k,  k, -k);
    let e4 = vec3<f32>( k,  k,  k);
    let n = e1 * map(p + e1 * eps)
          + e2 * map(p + e2 * eps)
          + e3 * map(p + e3 * eps)
          + e4 * map(p + e4 * eps);
    return normalize(n);
}

fn pack_rgb(rgb: vec3<f32>) -> u32 {
    let r = u32(clamp(rgb.x * 255.0, 0.0, 255.0));
    let g = u32(clamp(rgb.y * 255.0, 0.0, 255.0));
    let b = u32(clamp(rgb.z * 255.0, 0.0, 255.0));
    return 0xFF000000u | (r << 16u) | (g << 8u) | b;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    if col >= params.width || row >= params.height {
        return;
    }

    let x = params.x_min + f32(col) * params.x_step;
    let y = params.y_min + f32(row) * params.y_step;

    // Camera orbit
    let an = 0.5 * params.time;
    let ro = vec3<f32>(1.5 * cos(an), 0.8, 1.5 * sin(an));
    let ta = vec3<f32>(0.0, 0.1, 0.0);

    // Camera matrix
    let ww = normalize(ta - ro);
    let uu = normalize(cross(ww, vec3<f32>(0.0, 1.0, 0.0)));
    let vv = cross(uu, ww);

    // View ray
    let rd = normalize(x * uu + y * vv + 1.5 * ww);

    // Raymarch
    var t = 0.0f;
    var i = 0u;
    let tmax = 5.0;
    loop {
        let pos = ro + t * rd;
        let h = map(pos);
        if h < 0.0001 || t > tmax || i >= 256u {
            break;
        }
        t += h;
        i += 1u;
    }

    // Shading
    let hit_pos = ro + t * rd;
    let nor = calc_normal(hit_pos);
    let sun_dir = vec3<f32>(0.57703, 0.57703, 0.57703);
    let dif = clamp(dot(nor, sun_dir), 0.0, 1.0);
    let amb = 0.5 + 0.5 * dot(nor, vec3<f32>(0.0, 1.0, 0.0));
    let lit = vec3<f32>(0.2, 0.3, 0.4) * amb + vec3<f32>(0.8, 0.7, 0.5) * dif;

    var final_col: vec3<f32>;
    if t < tmax {
        final_col = lit;
    } else {
        final_col = vec3<f32>(0.05, 0.05, 0.1);
    }

    // Gamma
    final_col = sqrt(final_col);

    let idx = row * params.stride + col;
    output[idx] = pack_rgb(final_col);
}
