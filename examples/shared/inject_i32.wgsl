// Inject a value into an i32 buffer within a circular radius.
// Flat mode only (overwrite). For i32 buffers like Game of Life grids.

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
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read>       target:     array<i32>;
@group(0) @binding(2) var<storage, read_write> target_out: array<i32>;

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
    let current = target[i];

    var result = current;
    if d2 <= r2 {
        result = i32(params.value);
    }
    target_out[i] = result;
}
