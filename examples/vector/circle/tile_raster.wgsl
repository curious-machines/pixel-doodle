// Tile-based vector rasterization kernel.
//
// One workgroup per tile. Each thread handles one pixel within its tile.
// Segments within each tile are sorted by path_id, so the kernel walks
// them in a single pass, resolving and compositing each path as it
// encounters a new path_id. No path count limit.
//
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
const BG_COLOR: u32 = 0xFF000000u;

const FILL_EVEN_ODD: u32 = 0u;
const FILL_NONZERO: u32 = 1u;

// Sentinel value for "no current path"
const NO_PATH: u32 = 0xFFFFFFFFu;

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

    let px = f32(pixel_x) + 0.5;
    let py = f32(pixel_y) + 0.5;

    let count = tile_counts[tile_id];
    let offset = tile_offsets[tile_id];

    var color = BG_COLOR;
    var winding: i32 = 0;
    var current_path: u32 = NO_PATH;

    // Walk segments in path_id order (sorted by CPU).
    // When the path changes, resolve the previous path and composite.
    for (var i: u32 = 0u; i < count; i++) {
        let seg_idx = tile_indices[offset + i];
        let path_id = seg_path_ids[seg_idx];

        // Path boundary: resolve previous path
        if path_id != current_path {
            if current_path != NO_PATH {
                color = resolve_path(color, winding, current_path);
            }
            current_path = path_id;
            winding = 0;
        }

        let seg = segments[seg_idx];
        let x0 = seg.x;
        let y0 = seg.y;
        let x1 = seg.z;
        let y1 = seg.w;

        if (y0 <= py && y1 > py) || (y1 <= py && y0 > py) {
            let t = (py - y0) / (y1 - y0);
            let x_cross = x0 + t * (x1 - x0);

            if x_cross < px {
                if y1 > y0 {
                    winding += 1;
                } else {
                    winding -= 1;
                }
            }
        }
    }

    // Resolve the last path
    if current_path != NO_PATH {
        color = resolve_path(color, winding, current_path);
    }

    let pixel_idx = pixel_y * params.stride + pixel_x;
    pixels[pixel_idx] = color;
}

/// Apply fill rule and composite over the current color with alpha blending.
fn resolve_path(bg: u32, winding: i32, path_id: u32) -> u32 {
    let rule = path_fill_rules[path_id];
    var filled = false;
    if rule == FILL_NONZERO {
        filled = winding != 0;
    } else {
        filled = (winding & 1) != 0;
    }
    if filled {
        let src = path_colors[path_id];
        let sa = f32((src >> 24u) & 0xFFu) / 255.0;

        // Fully opaque: skip blending
        if sa >= 1.0 {
            return src;
        }

        // Alpha blend: src over dst
        let sr = f32((src >> 16u) & 0xFFu);
        let sg = f32((src >> 8u) & 0xFFu);
        let sb = f32(src & 0xFFu);

        let dr = f32((bg >> 16u) & 0xFFu);
        let dg = f32((bg >> 8u) & 0xFFu);
        let db = f32(bg & 0xFFu);

        let inv_sa = 1.0 - sa;
        let or = sr * sa + dr * inv_sa;
        let og = sg * sa + dg * inv_sa;
        let ob = sb * sa + db * inv_sa;

        let rb = u32(clamp(or, 0.0, 255.0));
        let gb = u32(clamp(og, 0.0, 255.0));
        let bb = u32(clamp(ob, 0.0, 255.0));
        return 0xFF000000u | (rb << 16u) | (gb << 8u) | bb;
    }
    return bg;
}
