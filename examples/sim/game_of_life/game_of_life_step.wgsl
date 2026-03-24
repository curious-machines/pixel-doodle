// Game of Life (B3/S23) step kernel
//
// Grid cells store i32 age:
//   positive = alive for N generations
//   negative = dead, fading out (clamped at -20)
//   0 = dead (fully faded)

struct Params {
    width: u32,
    height: u32,
    stride: u32,
    _pad: u32,
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
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
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
