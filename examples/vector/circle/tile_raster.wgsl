// Tile-based vector rasterization kernel.
//
// One workgroup per tile. Each thread handles one pixel within its tile.
// Computes per-path winding numbers by counting segment crossings to the
// left of the pixel. Composites paths back-to-front (painter's algorithm).
// Supports per-path fill rules: even-odd (0) or nonzero (1).

struct Params {
    width: u32,
    height: u32,
    stride: u32,
    tile_size: u32,
    tiles_x: u32,
    num_paths: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> segments: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> seg_path_ids: array<u32>;
@group(0) @binding(3) var<storage, read> tile_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> tile_counts: array<u32>;
@group(0) @binding(5) var<storage, read> tile_indices: array<u32>;
@group(0) @binding(6) var<storage, read> path_colors: array<u32>;
@group(0) @binding(7) var<storage, read_write> pixels: array<u32>;
@group(0) @binding(8) var<storage, read> path_fill_rules: array<u32>;

const TILE_SIZE: u32 = 16u;
const MAX_PATHS: u32 = 32u;
const BG_COLOR: u32 = 0xFF000000u;

const FILL_EVEN_ODD: u32 = 0u;
const FILL_NONZERO: u32 = 1u;

@compute @workgroup_size(TILE_SIZE, TILE_SIZE)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let tile_id = wg_id.y * params.tiles_x + wg_id.x;
    let pixel_x = wg_id.x * params.tile_size + local_id.x;
    let pixel_y = wg_id.y * params.tile_size + local_id.y;

    if pixel_x >= params.width || pixel_y >= params.height {
        return;
    }

    // Sample point at pixel center
    let px = f32(pixel_x) + 0.5;
    let py = f32(pixel_y) + 0.5;

    let count = tile_counts[tile_id];
    let offset = tile_offsets[tile_id];

    // Per-path winding numbers
    var winding: array<i32, MAX_PATHS>;
    let num_paths = min(params.num_paths, MAX_PATHS);

    for (var i: u32 = 0u; i < count; i++) {
        let seg_idx = tile_indices[offset + i];
        let seg = segments[seg_idx];
        let path_id = seg_path_ids[seg_idx];

        if path_id >= num_paths {
            continue;
        }

        let x0 = seg.x;
        let y0 = seg.y;
        let x1 = seg.z;
        let y1 = seg.w;

        // Ray crossing test: horizontal ray from -inf to (px, py)
        if (y0 <= py && y1 > py) || (y1 <= py && y0 > py) {
            let t = (py - y0) / (y1 - y0);
            let x_cross = x0 + t * (x1 - x0);

            if x_cross < px {
                if y1 > y0 {
                    winding[path_id] += 1;
                } else {
                    winding[path_id] -= 1;
                }
            }
        }
    }

    // Composite back-to-front (painter's algorithm)
    var color = BG_COLOR;
    for (var p: u32 = 0u; p < num_paths; p++) {
        let w = winding[p];
        let rule = path_fill_rules[p];
        var filled = false;
        if rule == FILL_NONZERO {
            filled = w != 0;
        } else {
            filled = (w & 1) != 0;
        }
        if filled {
            color = path_colors[p];
        }
    }

    let pixel_idx = pixel_y * params.stride + pixel_x;
    pixels[pixel_idx] = color;
}
