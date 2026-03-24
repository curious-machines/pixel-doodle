// Game of Life visualization: convert grid age to ARGB pixels (1:1 mapping)

struct Params {
    width: u32,
    height: u32,
    stride: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read>       grid_in: array<i32>;
@group(0) @binding(2) var<storage, read_write> pixels:  array<u32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= params.width || y >= params.height {
        return;
    }

    let pixel_idx = y * params.stride + x;
    let grid_idx = y * params.width + x;
    let age = grid_in[grid_idx];

    var r: f32;
    var g: f32;
    var b: f32;

    if age > 0 {
        // Alive: green -> cyan -> blue over 100 generations
        let t = clamp(f32(age) / 100.0, 0.0, 1.0);
        r = (1.0 - t) * 50.0 / 255.0;
        g = (1.0 - t * 0.5);
        b = t;
    } else if age < 0 {
        // Dead, fading: red dimming to black
        let fade = f32(age + 20) / 20.0;  // 1.0 at -1, 0.0 at -20
        r = clamp(fade, 0.0, 1.0) * 180.0 / 255.0;
        g = 0.0;
        b = 0.0;
    } else {
        // Fully dead
        r = 0.0;
        g = 0.0;
        b = 0.0;
    }

    let rb = u32(r * 255.0);
    let gb = u32(g * 255.0);
    let bb = u32(b * 255.0);
    pixels[pixel_idx] = 0xFF000000u | (rb << 16u) | (gb << 8u) | bb;
}
