// Physics constants
const DT: f32 = 4.0;
const DISSIPATION: f32 = 0.998;
const BUOYANCY: f32 = 0.08;

struct Params {
    width: u32,
    height: u32,
    stride: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read>       field_in:  array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> field_out: array<vec4<f32>>;

fn idx(x: u32, y: u32, w: u32) -> u32 {
    return y * w + x;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let w = params.width;
    let h = params.height;
    if x >= w || y >= h { return; }

    let i = idx(x, y, w);
    let cell = field_in[i];

    // Semi-Lagrangian: trace backward
    let src_x = f32(x) - DT * cell.x;
    let src_y = f32(y) - DT * cell.y;

    // Bilinear interpolation (inlined — WGSL can't pass storage ptrs to functions)
    let cx = clamp(src_x, 0.0, f32(w) - 1.001);
    let cy = clamp(src_y, 0.0, f32(h) - 1.001);
    let x0 = u32(floor(cx));
    let y0 = u32(floor(cy));
    let x1 = min(x0 + 1u, w - 1u);
    let y1 = min(y0 + 1u, h - 1u);
    let sx = cx - floor(cx);
    let sy = cy - floor(cy);
    let v00 = field_in[idx(x0, y0, w)];
    let v10 = field_in[idx(x1, y0, w)];
    let v01 = field_in[idx(x0, y1, w)];
    let v11 = field_in[idx(x1, y1, w)];
    let top = mix(v00, v10, sx);
    let bot = mix(v01, v11, sx);
    var result = mix(top, bot, sy);

    // Buoyancy: density pushes upward (negative y direction in screen coords)
    result.y -= BUOYANCY * result.z * DT;

    // Density dissipation
    result.z *= DISSIPATION;

    // Open boundaries: zero velocity at edges
    if x == 0u || x == w - 1u || y == 0u || y == h - 1u {
        result.x = 0.0;
        result.y = 0.0;
    }

    field_out[i] = result;
}
