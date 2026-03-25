// Initialize Gray-Scott field: u=1.0 everywhere, v=0.0 everywhere,
// with 8 circular blobs where u=0.5 and v=0.25 (radius 5 pixels).

struct Params {
    width: u32,
    height: u32,
    stride: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> field_out: array<vec2<f32>>;

fn hash_u32(n: u32) -> u32 {
    let a = n ^ (n >> 13u);
    let b = a * 1274126177u;
    return b ^ (b >> 16u);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= params.width || y >= params.height {
        return;
    }

    let radius: f32 = 5.0;
    let r2 = radius * radius;

    var u: f32 = 1.0;
    var v: f32 = 0.0;

    // Check 8 random blob centers
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        let hx = hash_u32(i * 2u);
        let hy = hash_u32(i * 2u + 1u);
        let cx = f32(hx & 0xFFFFu) / 65536.0 * f32(params.width);
        let cy = f32(hy & 0xFFFFu) / 65536.0 * f32(params.height);
        let dx = f32(x) - cx;
        let dy = f32(y) - cy;
        let d2 = dx * dx + dy * dy;
        if d2 < r2 {
            u = 0.5;
            v = 0.25;
        }
    }

    let idx = y * params.width + x;
    field_out[idx] = vec2<f32>(u, v);
}
