pub mod flatten;
pub mod stroke;
pub mod tile_renderer;

use flatten::{CubicBezier, Curve, Path, Point};
use stroke::StrokeStyle;

/// Pre-binned vector scene data ready for GPU upload.
///
/// Segments are line segments from flattened curves.
/// Tile bins map each screen tile to the segments that affect it.
/// Fill rule constants.
pub const FILL_EVEN_ODD: u32 = 0;
pub const FILL_NONZERO: u32 = 1;

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
    /// Fill rule per path (0 = even-odd, 1 = nonzero).
    pub path_fill_rules: Vec<u32>,
    /// Tile size in pixels (e.g., 16).
    pub tile_size: u32,
}

/// Kappa constant for approximating a quarter-circle with a cubic bezier.
/// k = (4/3) * (sqrt(2) - 1) ≈ 0.5522847498
const KAPPA: f32 = 0.5522847498;

/// Flatten a path's curves, then expand into a stroke outline.
///
/// Returns a filled `Path` representing the stroke, with the given `stroke_path_id`.
pub fn stroke_flattened(
    path: &Path,
    tolerance: f32,
    style: &StrokeStyle,
    stroke_path_id: u32,
) -> Path {
    // Flatten the source path into line segments
    let paths = [Path {
        curves: path.curves.clone(),
        path_id: 0, // temporary, not used
    }];
    let (segments, _) = flatten::flatten_paths(&paths, tolerance);

    // Expand into stroke outline
    stroke::stroke_path(&segments, style, stroke_path_id)
}

/// Generate a multi-shape test scene.
///
/// Back to front:
/// - Path 0: blue rectangle (partially behind donut, visible through hole)
/// - Path 1: orange donut fill (outer CCW + inner CW for hole)
/// - Path 2: green diamond overlapping the donut's edge
/// - Path 3: white stroke on the diamond (90° corners — miter join)
/// - Path 4: yellow stroke on a triangle (acute angles — tests sharp miter)
/// - Path 5: cyan stroke on a chevron (very acute angle — tests bevel fallback)
/// - Path 6: red stroke on the circle (smooth curve — many gentle joins)
pub fn test_scene(
    tolerance: f32,
    tile_size: u32,
    width: u32,
    height: u32,
) -> VectorScene {
    let cx = width as f32 / 2.0;
    let cy = height as f32 / 2.0;
    let stroke_w = 6.0;
    let stroke_style = StrokeStyle {
        width: stroke_w,
        miter_limit: 4.0,
    };

    // Path 0: blue rectangle behind the donut
    let rect = rect_path(cx - 30.0, cy - 80.0, 200.0, 160.0, 0);

    // Path 1: orange donut fill
    let outer_r = height as f32 * 0.35;
    let inner_r = height as f32 * 0.18;
    let donut_outer = circle_path(cx, cy, outer_r, false, 1);
    let donut_inner = circle_path(cx, cy, inner_r, true, 1);

    // Path 2: green diamond (90° corners)
    let diamond_shape = diamond_path(cx + outer_r * 0.7, cy, outer_r * 0.4, 2);

    // Path 3: white stroke on diamond (tests 90° miter joins)
    let diamond_stroke = stroke_flattened(&diamond_shape, tolerance, &stroke_style, 3);

    // Path 4: yellow stroke on a triangle (acute ~60° angles — miter)
    let tri = triangle_path(100.0, height as f32 - 50.0, 80.0, 4);
    let tri_stroke = stroke_flattened(&tri, tolerance, &stroke_style, 4);

    // Path 5: cyan stroke on a chevron (very acute ~20° angle — should trigger bevel fallback)
    let chevron = chevron_path(250.0, height as f32 - 80.0, 5);
    let chevron_stroke = stroke_flattened(&chevron, tolerance, &stroke_style, 5);

    // Path 6: red stroke on the outer circle (tests many gentle-angle joins from flattened curve)
    let circle_for_stroke = circle_path(cx, cy, outer_r + 20.0, false, 6);
    let circle_stroke = stroke_flattened(&circle_for_stroke, tolerance, &stroke_style, 6);

    let paths = vec![
        rect,
        donut_outer,
        donut_inner,
        diamond_shape,
        diamond_stroke,
        tri_stroke,
        chevron_stroke,
        circle_stroke,
    ];
    let (segments, seg_path_ids) = flatten::flatten_paths(&paths, tolerance);

    let path_colors = vec![
        0xFF4488FF, // path 0: blue rect
        0xFFFF8800, // path 1: orange donut
        0xFF44CC44, // path 2: green diamond
        0xFFFFFFFF, // path 3: white diamond stroke
        0xFFFFFF00, // path 4: yellow triangle stroke
        0xFF00FFFF, // path 5: cyan chevron stroke
        0xFFFF4444, // path 6: red circle stroke
    ];

    let path_fill_rules = vec![
        FILL_EVEN_ODD, // path 0: blue rect (fill)
        FILL_EVEN_ODD, // path 1: orange donut (fill with hole)
        FILL_EVEN_ODD, // path 2: green diamond (fill)
        FILL_NONZERO,  // path 3: white diamond stroke
        FILL_NONZERO,  // path 4: yellow triangle stroke
        FILL_NONZERO,  // path 5: cyan chevron stroke
        FILL_NONZERO,  // path 6: red circle stroke
    ];

    bin_tiles(&segments, &seg_path_ids, path_colors, path_fill_rules, tile_size, width, height)
}

/// Create an equilateral triangle path (acute ~60° angles).
fn triangle_path(cx: f32, cy: f32, size: f32, path_id: u32) -> Path {
    let h = size * (3.0_f32).sqrt() / 2.0;
    let top = Point::new(cx, cy - h * 2.0 / 3.0);
    let bl = Point::new(cx - size / 2.0, cy + h / 3.0);
    let br = Point::new(cx + size / 2.0, cy + h / 3.0);

    Path {
        curves: vec![
            Curve::Line(top, br),
            Curve::Line(br, bl),
            Curve::Line(bl, top),
        ],
        path_id,
    }
}

/// Create a chevron (V shape) with a very acute angle at the bottom.
/// Open path (not closed) — tests butt caps and an acute miter/bevel.
/// The angle is ~14° which guarantees bevel fallback at miter_limit=4.0.
fn chevron_path(cx: f32, cy: f32, path_id: u32) -> Path {
    let top_left = Point::new(cx - 10.0, cy - 60.0);
    let bottom = Point::new(cx, cy + 40.0);
    let top_right = Point::new(cx + 10.0, cy - 60.0);

    Path {
        curves: vec![
            Curve::Line(top_left, bottom),
            Curve::Line(bottom, top_right),
        ],
        path_id,
    }
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

/// Generate a stress test scene with many filled and stroked circles.
///
/// Each circle gets a fill path and a stroke path, for `2 * count` total paths.
/// Circles are placed pseudo-randomly using a deterministic hash.
pub fn test_stress(
    count: u32,
    tolerance: f32,
    tile_size: u32,
    width: u32,
    height: u32,
) -> VectorScene {
    use rayon::prelude::*;

    let stroke_style = StrokeStyle {
        width: 2.0,
        miter_limit: 4.0,
    };

    // Generate paths in parallel: each circle produces a fill path + stroke path
    let t_gen = std::time::Instant::now();
    let path_data: Vec<(Path, Path, u32, u32)> = (0..count)
        .into_par_iter()
        .map(|i| {
            // Deterministic pseudo-random placement
            let hash = hash_u32(i);
            let cx = (hash % width) as f32;
            let cy = ((hash / 7) % height) as f32;
            let r = 5.0 + (hash % 30) as f32;

            let fill_id = i * 2;
            let stroke_id = i * 2 + 1;

            let fill = circle_path(cx, cy, r, false, fill_id);
            let stroke_src = circle_path(cx, cy, r, false, 0); // temp id
            let stroke = stroke_flattened(&stroke_src, tolerance, &stroke_style, stroke_id);

            let fill_color = color_from_hash(hash);
            let stroke_color = 0xFF000000; // black strokes

            (fill, stroke, fill_color, stroke_color)
        })
        .collect();
    let gen_ms = t_gen.elapsed().as_secs_f64() * 1000.0;

    // Collect all paths and colors in order
    let mut paths = Vec::with_capacity(count as usize * 2);
    let mut path_colors = Vec::with_capacity(count as usize * 2);
    let mut path_fill_rules = Vec::with_capacity(count as usize * 2);

    for (fill, stroke, fill_color, stroke_color) in path_data {
        paths.push(fill);
        path_colors.push(fill_color);
        path_fill_rules.push(FILL_EVEN_ODD);

        paths.push(stroke);
        path_colors.push(stroke_color);
        path_fill_rules.push(FILL_NONZERO);
    }

    let t_flatten = std::time::Instant::now();
    let (segments, seg_path_ids) = flatten::flatten_paths(&paths, tolerance);
    let flatten_ms = t_flatten.elapsed().as_secs_f64() * 1000.0;

    let t_bin = std::time::Instant::now();
    let scene = bin_tiles(
        &segments,
        &seg_path_ids,
        path_colors,
        path_fill_rules,
        tile_size,
        width,
        height,
    );
    let bin_ms = t_bin.elapsed().as_secs_f64() * 1000.0;

    eprintln!(
        "[stress] gen+stroke: {gen_ms:.1}ms, flatten: {flatten_ms:.1}ms, bin: {bin_ms:.1}ms"
    );

    scene
}

/// Simple deterministic hash for pseudo-random values.
fn hash_u32(mut x: u32) -> u32 {
    x = x.wrapping_mul(0x9E3779B9);
    x ^= x >> 16;
    x = x.wrapping_mul(0x21F0AAAD);
    x ^= x >> 15;
    x = x.wrapping_mul(0x735A2D97);
    x ^= x >> 15;
    x
}

/// Generate a color from a hash value.
fn color_from_hash(hash: u32) -> u32 {
    let r = 80 + (hash % 176);
    let g = 80 + ((hash >> 8) % 176);
    let b = 80 + ((hash >> 16) % 176);
    0xFF000000 | (r << 16) | (g << 8) | b
}

/// Compute tile bins for a set of segments.
///
/// A segment is added to a tile if:
/// 1. The segment's y-range overlaps the tile row's y-range, AND
/// 2. The segment's maximum x is >= the tile's left edge (x-range filtering).
///
/// Condition 2 is the key optimization: segments entirely to the right of a
/// tile can't contribute crossings to pixels in that tile (the ray goes left
/// from the pixel). Segments to the left CAN contribute crossings.
///
/// TODO: Further optimization with sorted segments (Option 3) for GPU-side
/// path skipping.
/// Public wrapper for bin_tiles, used by PDC integration.
pub fn bin_tiles_pub(
    segments: &[[f32; 4]],
    seg_path_ids: &[u32],
    path_colors: Vec<u32>,
    path_fill_rules: Vec<u32>,
    tile_size: u32,
    width: u32,
    height: u32,
) -> VectorScene {
    bin_tiles(segments, seg_path_ids, path_colors, path_fill_rules, tile_size, width, height)
}

fn bin_tiles(
    segments: &[[f32; 4]],
    seg_path_ids: &[u32],
    path_colors: Vec<u32>,
    path_fill_rules: Vec<u32>,
    tile_size: u32,
    width: u32,
    height: u32,
) -> VectorScene {
    use rayon::prelude::*;

    let tiles_x = (width + tile_size - 1) / tile_size;
    let tiles_y = (height + tile_size - 1) / tile_size;

    // Pre-compute segment metadata to avoid recomputing per tile row
    struct SegInfo {
        seg_idx: u32,
        path_id: u32,
        y_min: f32,
        y_max: f32,
        tx_min: u32,
    }

    let seg_infos: Vec<SegInfo> = segments
        .iter()
        .enumerate()
        .filter_map(|(seg_idx, seg)| {
            let y0 = seg[1];
            let y1 = seg[3];
            let y_min = y0.min(y1);
            let y_max = y0.max(y1);

            // Skip horizontal segments
            if y_min == y_max {
                return None;
            }

            let x_min = seg[0].min(seg[2]);

            // Skip segments entirely to the right of the screen
            if x_min >= width as f32 {
                return None;
            }

            let tx_min = if x_min < 0.0 {
                0
            } else {
                ((x_min as u32) / tile_size).min(tiles_x - 1)
            };

            Some(SegInfo {
                seg_idx: seg_idx as u32,
                path_id: seg_path_ids[seg_idx],
                y_min,
                y_max,
                tx_min,
            })
        })
        .collect();

    // Bin and sort per tile row in parallel
    let row_results: Vec<Vec<Vec<u32>>> = (0..tiles_y)
        .into_par_iter()
        .map(|ty| {
            let tile_y_min = (ty * tile_size) as f32;
            let tile_y_max = ((ty + 1) * tile_size) as f32;

            // Build lists for each tile in this row
            let mut row_tiles: Vec<Vec<(u32, u32)>> = vec![Vec::new(); tiles_x as usize];

            for info in &seg_infos {
                if info.y_max > tile_y_min && info.y_min < tile_y_max {
                    for tx in info.tx_min..tiles_x {
                        row_tiles[tx as usize].push((info.path_id, info.seg_idx));
                    }
                }
            }

            // Sort each tile by path_id and extract segment indices
            row_tiles
                .into_iter()
                .map(|mut list| {
                    list.sort_unstable_by_key(|(path_id, _)| *path_id);
                    list.into_iter().map(|(_, seg_idx)| seg_idx).collect()
                })
                .collect()
        })
        .collect();

    // Flatten row results into offset/count/indices arrays
    let num_tiles = (tiles_x * tiles_y) as usize;
    let mut tile_offsets = Vec::with_capacity(num_tiles);
    let mut tile_counts = Vec::with_capacity(num_tiles);
    let mut tile_indices = Vec::new();

    for row in &row_results {
        for list in row {
            tile_offsets.push(tile_indices.len() as u32);
            tile_counts.push(list.len() as u32);
            tile_indices.extend_from_slice(list);
        }
    }

    VectorScene {
        segments: segments.to_vec(),
        seg_path_ids: seg_path_ids.to_vec(),
        tile_offsets,
        tile_counts,
        tile_indices,
        path_colors,
        path_fill_rules,
        tile_size,
    }
}
