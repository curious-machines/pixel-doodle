// Inject a value into a single component of a vec4<f32> buffer.
// Reads the current value, modifies one component, writes back.
// Works for vec2<f32> buffers too (just use component 0 or 1).
//
// User args:
//   inject_x, inject_y: center point (mouse position)
//   radius: injection radius in pixels
//   value: value to write (flat) or add (quadratic)
//   falloff_quadratic: 0 = flat (overwrite), 1 = quadratic (additive falloff)
//   component: which component to modify (0=x, 1=y, 2=z, 3=w)

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
    falloff_quadratic: u32,
    component: u32,
    _pad2: u32,
    _pad3: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read>       target:     array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> target_out: array<vec4<f32>>;

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
        var inject_val = params.value;
        if params.falloff_quadratic >= 1u {
            let factor = 1.0 - d2 / r2;
            switch params.component {
                case 0u: { result.x = current.x + inject_val * factor; }
                case 1u: { result.y = current.y + inject_val * factor; }
                case 2u: { result.z = current.z + inject_val * factor; }
                case 3u: { result.w = current.w + inject_val * factor; }
                default: {}
            }
        } else {
            switch params.component {
                case 0u: { result.x = inject_val; }
                case 1u: { result.y = inject_val; }
                case 2u: { result.z = inject_val; }
                case 3u: { result.w = inject_val; }
                default: {}
            }
        }
    }
    target_out[i] = result;
}
