pub mod flatten;
pub mod tile_renderer;

use flatten::{CubicBezier, Curve, Path, Point};

/// Pre-binned vector scene data ready for GPU upload.
///
/// Segments are line segments from flattened curves.
/// Tile bins map each screen tile to the segments that affect it.
pub struct VectorScene {
    /// Line segments as [x0, y0, x1, y1] per segment.
    pub segments: Vec<[f32; 4]>,
    /// Path ID per segment.
    pub seg_path_ids: Vec<u32>,
    /// Per-tile start offset into `tile_indices`.
    pub tile_offsets: Vec<u32>,
    /// Per-tile segment count.
    pub tile_counts: Vec<u32>,
    /// Concatenated segment indices per tile.
    pub tile_indices: Vec<u32>,
    /// Packed ARGB color per path.
    pub path_colors: Vec<u32>,
    /// Tile size in pixels (e.g., 16).
    pub tile_size: u32,
}

/// Kappa constant for approximating a quarter-circle with a cubic bezier.
/// k = (4/3) * (sqrt(2) - 1) ≈ 0.5522847498
const KAPPA: f32 = 0.5522847498;

/// Generate a multi-shape test scene.
///
/// Back to front:
/// - Path 0: blue rectangle (partially behind donut, visible through hole)
/// - Path 1: orange donut (outer CCW + inner CW for hole)
/// - Path 2: green diamond overlapping the donut's edge
pub fn test_scene(
    tolerance: f32,
    tile_size: u32,
    width: u32,
    height: u32,
) -> VectorScene {
    let cx = width as f32 / 2.0;
    let cy = height as f32 / 2.0;

    // Path 0: blue rectangle behind the donut, offset right so it's partially visible
    let rect = rect_path(cx - 30.0, cy - 80.0, 200.0, 160.0, 0);

    // Path 1: orange donut (outer + inner share path_id 1)
    let outer_r = height as f32 * 0.35;
    let inner_r = height as f32 * 0.18;
    let donut_outer = circle_path(cx, cy, outer_r, false, 1);
    let donut_inner = circle_path(cx, cy, inner_r, true, 1);

    // Path 2: green diamond overlapping the donut's right edge
    let diamond = diamond_path(cx + outer_r * 0.7, cy, outer_r * 0.4, 2);

    let paths = vec![rect, donut_outer, donut_inner, diamond];
    let (segments, seg_path_ids) = flatten::flatten_paths(&paths, tolerance);

    let path_colors = vec![
        0xFF4488FF, // path 0: blue
        0xFFFF8800, // path 1: orange
        0xFF44CC44, // path 2: green
    ];

    bin_tiles(&segments, &seg_path_ids, path_colors, tile_size, width, height)
}

/// Create a rectangle path from line segments.
fn rect_path(x: f32, y: f32, w: f32, h: f32, path_id: u32) -> Path {
    let tl = Point::new(x, y);
    let tr = Point::new(x + w, y);
    let br = Point::new(x + w, y + h);
    let bl = Point::new(x, y + h);

    Path {
        curves: vec![
            Curve::Line(tl, tr),
            Curve::Line(tr, br),
            Curve::Line(br, bl),
            Curve::Line(bl, tl),
        ],
        path_id,
    }
}

/// Create a diamond (rotated square) path from line segments.
fn diamond_path(cx: f32, cy: f32, radius: f32, path_id: u32) -> Path {
    let top = Point::new(cx, cy - radius);
    let right = Point::new(cx + radius, cy);
    let bottom = Point::new(cx, cy + radius);
    let left = Point::new(cx - radius, cy);

    Path {
        curves: vec![
            Curve::Line(top, right),
            Curve::Line(right, bottom),
            Curve::Line(bottom, left),
            Curve::Line(left, top),
        ],
        path_id,
    }
}

/// Create a circle path from 4 cubic beziers.
///
/// If `clockwise` is false, the circle winds counterclockwise (standard).
/// If `clockwise` is true, the circle winds clockwise (for holes).
fn circle_path(cx: f32, cy: f32, r: f32, clockwise: bool, path_id: u32) -> Path {
    let k = r * KAPPA;

    // Quarter-arc control points, counterclockwise from right (3 o'clock)
    // Arc 1: right → top
    // Arc 2: top → left
    // Arc 3: left → bottom
    // Arc 4: bottom → right
    let right = Point::new(cx + r, cy);
    let top = Point::new(cx, cy - r);
    let left = Point::new(cx - r, cy);
    let bottom = Point::new(cx, cy + r);

    let mut curves = vec![
        Curve::Cubic(CubicBezier {
            p0: right,
            p1: Point::new(cx + r, cy - k),
            p2: Point::new(cx + k, cy - r),
            p3: top,
        }),
        Curve::Cubic(CubicBezier {
            p0: top,
            p1: Point::new(cx - k, cy - r),
            p2: Point::new(cx - r, cy - k),
            p3: left,
        }),
        Curve::Cubic(CubicBezier {
            p0: left,
            p1: Point::new(cx - r, cy + k),
            p2: Point::new(cx - k, cy + r),
            p3: bottom,
        }),
        Curve::Cubic(CubicBezier {
            p0: bottom,
            p1: Point::new(cx + k, cy + r),
            p2: Point::new(cx + r, cy + k),
            p3: right,
        }),
    ];

    if clockwise {
        // Reverse each curve's direction and reverse the order
        curves = curves
            .into_iter()
            .rev()
            .map(|c| match c {
                Curve::Cubic(cb) => Curve::Cubic(CubicBezier {
                    p0: cb.p3,
                    p1: cb.p2,
                    p2: cb.p1,
                    p3: cb.p0,
                }),
                other => other,
            })
            .collect();
    }

    Path { curves, path_id }
}

/// Compute tile bins for a set of segments.
///
/// Conservative binning: a segment is added to every tile in rows where the
/// segment's y-range overlaps the tile row's y-range. This ensures winding
/// number correctness (all crossings to the left of a pixel are counted).
///
/// TODO: Optimize — this is O(segments × tiles_per_row) and won't scale to
/// complex scenes. Future options: x-range filtering, prefix-sum of crossings
/// across tile columns, or coarse rasterization pass.
fn bin_tiles(
    segments: &[[f32; 4]],
    seg_path_ids: &[u32],
    path_colors: Vec<u32>,
    tile_size: u32,
    width: u32,
    height: u32,
) -> VectorScene {
    let tiles_x = (width + tile_size - 1) / tile_size;
    let tiles_y = (height + tile_size - 1) / tile_size;
    let num_tiles = (tiles_x * tiles_y) as usize;

    // Build per-tile segment lists
    let mut tile_lists: Vec<Vec<u32>> = vec![Vec::new(); num_tiles];

    for (seg_idx, seg) in segments.iter().enumerate() {
        let y0 = seg[1];
        let y1 = seg[3];
        let seg_y_min = y0.min(y1);
        let seg_y_max = y0.max(y1);

        // Skip horizontal segments (they don't contribute to winding)
        if seg_y_min == seg_y_max {
            continue;
        }

        // Find tile rows that overlap this segment's y-range
        let ty_min = ((seg_y_min.max(0.0) as u32) / tile_size).min(tiles_y - 1);
        let ty_max = ((seg_y_max.max(0.0) as u32) / tile_size).min(tiles_y - 1);

        for ty in ty_min..=ty_max {
            let tile_y_min = (ty * tile_size) as f32;
            let tile_y_max = ((ty + 1) * tile_size) as f32;

            // Check y-range overlap
            if seg_y_max > tile_y_min && seg_y_min < tile_y_max {
                // Add to all tiles in this row (conservative, correct for winding)
                for tx in 0..tiles_x {
                    let tile_id = ty * tiles_x + tx;
                    tile_lists[tile_id as usize].push(seg_idx as u32);
                }
            }
        }
    }

    // Flatten tile lists into offset/count/indices arrays
    let mut tile_offsets = Vec::with_capacity(num_tiles);
    let mut tile_counts = Vec::with_capacity(num_tiles);
    let mut tile_indices = Vec::new();

    for list in &tile_lists {
        tile_offsets.push(tile_indices.len() as u32);
        tile_counts.push(list.len() as u32);
        tile_indices.extend_from_slice(list);
    }

    VectorScene {
        segments: segments.to_vec(),
        seg_path_ids: seg_path_ids.to_vec(),
        tile_offsets,
        tile_counts,
        tile_indices,
        path_colors,
        tile_size,
    }
}
