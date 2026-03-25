// Initialize Game of Life grid with ~30% random live cells

struct Params {
    width: u32,
    height: u32,
    stride: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> grid: array<i32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= params.width || y >= params.height {
        return;
    }

    // Hash position for deterministic random (same as random_binary.pd)
    let h1 = x * 374761393u + y * 668265263u + 12345u;
    let h2 = h1 ^ (h1 >> 13u);
    let h3 = h2 * 1274126177u;
    let h4 = h3 ^ (h3 >> 16u);
    let hf = f32(h4 & 65535u) / 65536.0;

    grid[y * params.width + x] = select(0, 1, hf < 0.3);
}
