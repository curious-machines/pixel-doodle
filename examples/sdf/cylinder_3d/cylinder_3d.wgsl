// 3D Raymarched Cylinder — ported from https://www.shadertoy.com/view/wdXGDr
// Iq's exact SDF for a cylinder between two arbitrary endpoints.

struct Params {
    width: u32,
    height: u32,
    max_iter: u32,
    stride: u32,
    x_min: f32,
    y_min: f32,
    x_step: f32,
    y_step: f32,
    _pad0: u32,
    _pad1: u32,
    time: f32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

// Exact SDF for a cylinder between endpoints a and b with radius r
fn sd_cylinder(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, r: f32) -> f32 {
    let ba = b - a;
    let pa = p - a;
    let baba = dot(ba, ba);
    let paba = dot(pa, ba);
    let x = length(pa * baba - ba * paba) - r * baba;
    let y = abs(paba - baba * 0.5) - baba * 0.5;
    let x2 = x * x;
    let y2 = y * y * baba;

    var d: f32;
    if max(x, y) < 0.0 {
        d = -min(x2, y2);
    } else {
        var dx: f32 = 0.0;
        if x > 0.0 { dx = x2; }
        var dy: f32 = 0.0;
        if y > 0.0 { dy = y2; }
        d = dx + dy;
    }

    return sign(d) * sqrt(abs(d)) / baba;
}

fn map(p: vec3<f32>) -> f32 {
    return sd_cylinder(p,
        vec3<f32>(-0.2, -0.3, -0.1),
        vec3<f32>(0.3, 0.3, 0.4),
        0.2);
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

fn eval_kernel(cx: f32, cy: f32) -> u32 {
    // Camera orbit
    let an = 0.5 * (params.time - 10.0);
    let ro = vec3<f32>(cos(an), 0.4, sin(an));
    let ta = vec3<f32>(0.0, 0.0, 0.0);

    // Camera matrix
    let ww = normalize(ta - ro);
    let uu = normalize(cross(ww, vec3<f32>(0.0, 1.0, 0.0)));
    let vv = cross(uu, ww);

    // View ray
    let rd = normalize(cx * uu + cy * vv + 1.5 * ww);

    // Raymarch
    let tmax = 3.0;
    var t: f32 = 0.0;
    for (var i: u32 = 0u; i < 256u; i = i + 1u) {
        let pos = ro + t * rd;
        let h = map(pos);
        if h < 0.0001 || t > tmax {
            break;
        }
        t = t + h;
    }

    // Shading
    var col = vec3<f32>(0.0, 0.0, 0.0);
    if t < tmax {
        let pos = ro + t * rd;
        let nor = calc_normal(pos);
        let dif = clamp(dot(nor, vec3<f32>(0.57703, 0.57703, 0.57703)), 0.0, 1.0);
        let amb = 0.5 + 0.5 * dot(nor, vec3<f32>(0.0, 1.0, 0.0));
        col = vec3<f32>(0.2, 0.3, 0.4) * amb + vec3<f32>(0.8, 0.7, 0.5) * dif;
    }

    // Gamma
    let final_col = sqrt(col);

    let r = u32(clamp(final_col.x * 255.0, 0.0, 255.0));
    let g = u32(clamp(final_col.y * 255.0, 0.0, 255.0));
    let b = u32(clamp(final_col.z * 255.0, 0.0, 255.0));
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
    let cx = params.x_min + f32(col) * params.x_step;
    let cy = params.y_min + f32(row) * params.y_step;

    output[idx] = eval_kernel(cx, cy);
}
