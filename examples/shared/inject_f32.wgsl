// Inject a value into an f32 buffer within a circular radius.
// User args (appended after standard SimParams in uniform buffer):
//   inject_x, inject_y: center point (mouse position)
//   radius: injection radius in pixels
//   value: value to write (flat) or add (quadratic)
//   falloff_quadratic: 0 = flat (overwrite), 1 = quadratic (additive falloff)

struct Params {
    width: u32,
    height: u32,
    stride: u32,
    _pad: u32,
    // User args
    inject_x: f32,
    inject_y: f32,
    radius: f32,
    value: f32,
    falloff_quadratic: f32,
    _pad2: f32,
    _pad3: f32,
    _pad4: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read>       buf_in:     array<f32>;
@group(0) @binding(2) var<storage, read_write> buf_out: array<f32>;

fn idx(x: u32, y: u32) -> u32 {
    return y * params.width + x;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= params.width || y >= params.height { return; }

    let fx = f32(x);
    let fy = f32(y);
    let dx = fx - params.inject_x;
    let dy = fy - params.inject_y;
    let d2 = dx * dx + dy * dy;
    let r2 = params.radius * params.radius;
    let i = idx(x, y);
    let current = buf_in[i];

    var result = current;
    if d2 <= r2 {
        if params.falloff_quadratic >= 0.5 {
            result = current + params.value * (1.0 - d2 / r2);
        } else {
            result = params.value;
        }
    }
    buf_out[i] = result;
}
