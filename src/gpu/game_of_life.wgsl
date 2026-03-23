// Game of Life (B3/S23) with age-based coloring — GPU compute shader
//
// Grid cells store i32 age:
//   positive = alive for N generations
//   negative = dead, fading out (clamped at -20)
//   0 = dead (fully faded)

struct Params {
    width: u32,
    height: u32,
    stride: u32,
    _pad0: u32,
    center_x: f32,
    center_y: f32,
    zoom: f32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read>       grid_in:  array<i32>;
@group(0) @binding(2) var<storage, read_write> grid_out: array<i32>;

fn idx(x: u32, y: u32) -> u32 {
    return y * params.width + x;
}

fn wrap_x(x: i32) -> u32 {
    return u32((x + i32(params.width)) % i32(params.width));
}

fn wrap_y(y: i32) -> u32 {
    return u32((y + i32(params.height)) % i32(params.height));
}

@compute @workgroup_size(16, 16)
fn step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= params.width || y >= params.height {
        return;
    }

    let ix = i32(x);
    let iy = i32(y);

    // Count alive neighbors (alive = age > 0)
    var count = 0u;
    count += u32(grid_in[idx(wrap_x(ix - 1), wrap_y(iy - 1))] > 0);
    count += u32(grid_in[idx(x,            wrap_y(iy - 1))] > 0);
    count += u32(grid_in[idx(wrap_x(ix + 1), wrap_y(iy - 1))] > 0);
    count += u32(grid_in[idx(wrap_x(ix - 1), y)] > 0);
    count += u32(grid_in[idx(wrap_x(ix + 1), y)] > 0);
    count += u32(grid_in[idx(wrap_x(ix - 1), wrap_y(iy + 1))] > 0);
    count += u32(grid_in[idx(x,            wrap_y(iy + 1))] > 0);
    count += u32(grid_in[idx(wrap_x(ix + 1), wrap_y(iy + 1))] > 0);

    let old_age = grid_in[idx(x, y)];
    let alive = old_age > 0;

    // B3/S23 rules
    let new_alive = (alive && (count == 2u || count == 3u)) || (!alive && count == 3u);

    var new_age: i32;
    if new_alive {
        new_age = select(1, old_age + 1, old_age > 0);
    } else {
        if old_age > 0 {
            new_age = -1;  // just died
        } else {
            new_age = max(old_age - 1, -20);  // continue fading
        }
    }

    grid_out[idx(x, y)] = new_age;
}

// Visualization: convert grid age to ARGB pixels
@group(0) @binding(0) var<uniform> vis_params: Params;
@group(0) @binding(1) var<storage, read>       vis_grid: array<i32>;
@group(0) @binding(2) var<storage, read_write> pixels:   array<u32>;

@compute @workgroup_size(16, 16)
fn visualize(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= vis_params.width || y >= vis_params.height {
        return;
    }

    let pixel_idx = y * vis_params.stride + x;

    // Map screen pixel to grid cell through viewport
    let fw = f32(vis_params.width);
    let fh = f32(vis_params.height);
    let grid_fx = (f32(x) - fw * 0.5) / vis_params.zoom + vis_params.center_x * fw;
    let grid_fy = (f32(y) - fh * 0.5) / vis_params.zoom + vis_params.center_y * fh;

    // Nearest-neighbor sampling with toroidal wrap
    let gx = u32(((i32(floor(grid_fx)) % i32(vis_params.width)) + i32(vis_params.width)) % i32(vis_params.width));
    let gy = u32(((i32(floor(grid_fy)) % i32(vis_params.height)) + i32(vis_params.height)) % i32(vis_params.height));
    let grid_idx = gy * vis_params.width + gx;
    let age = vis_grid[grid_idx];

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
