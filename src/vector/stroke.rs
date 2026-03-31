use super::flatten::{Curve, Path, Point};

/// Stroke expansion options.
pub struct StrokeStyle {
    /// Stroke width in pixels.
    pub width: f32,
    /// Maximum miter ratio (miter_length / half_width).
    /// When exceeded, falls back to bevel join. SVG default is 4.0.
    pub miter_limit: f32,
}

impl Default for StrokeStyle {
    fn default() -> Self {
        Self {
            width: 1.0,
            miter_limit: 4.0,
        }
    }
}

/// Expand a flattened path (line segments only) into a filled stroke outline.
///
/// The resulting path should be rendered with **nonzero** fill rule to handle
/// self-intersecting outlines at acute angle bevels correctly.
pub fn stroke_path(segments: &[[f32; 4]], style: &StrokeStyle, path_id: u32) -> Path {
    if segments.is_empty() {
        return Path {
            curves: Vec::new(),
            path_id,
        };
    }

    let half_w = style.width / 2.0;
    let n = segments.len();

    // Detect closed path
    let first_start = [segments[0][0], segments[0][1]];
    let last_end = [segments[n - 1][2], segments[n - 1][3]];
    let closed = (first_start[0] - last_end[0]).abs() < 1e-4
        && (first_start[1] - last_end[1]).abs() < 1e-4;

    // Compute unit normals for each segment
    let normals: Vec<[f32; 2]> = segments.iter().map(|s| seg_normal(s)).collect();

    if closed {
        stroke_closed(segments, &normals, style, half_w, path_id)
    } else {
        stroke_open(segments, &normals, style, half_w, path_id)
    }
}

/// Stroke a closed path. Outputs two separate closed contours.
fn stroke_closed(
    segments: &[[f32; 4]],
    normals: &[[f32; 2]],
    style: &StrokeStyle,
    half_w: f32,
    path_id: u32,
) -> Path {
    let n = segments.len();
    let mut left_points: Vec<Point> = Vec::new();
    let mut right_points: Vec<Point> = Vec::new();

    for i in 0..n {
        let prev_i = if i == 0 { n - 1 } else { i - 1 };
        let vertex = Point::new(segments[i][0], segments[i][1]);
        let (l, r) = miter_offset(vertex, normals[prev_i], normals[i], half_w, style);
        left_points.extend(l);
        right_points.extend(r);
    }

    let mut curves = Vec::new();

    // Left contour (forward)
    for i in 0..left_points.len() {
        let next = (i + 1) % left_points.len();
        curves.push(Curve::Line(left_points[i], left_points[next]));
    }

    // Right contour (reversed winding)
    for i in (0..right_points.len()).rev() {
        let prev = if i == 0 { right_points.len() - 1 } else { i - 1 };
        curves.push(Curve::Line(right_points[i], right_points[prev]));
    }

    Path { curves, path_id }
}

/// Stroke an open path. Single closed contour with butt caps.
fn stroke_open(
    segments: &[[f32; 4]],
    normals: &[[f32; 2]],
    style: &StrokeStyle,
    half_w: f32,
    path_id: u32,
) -> Path {
    let n = segments.len();
    let mut left_points: Vec<Point> = Vec::new();
    let mut right_points: Vec<Point> = Vec::new();

    // Start point
    let n_first = normals[0];
    let start = Point::new(segments[0][0], segments[0][1]);
    left_points.push(Point::new(
        start.x + n_first[0] * half_w,
        start.y + n_first[1] * half_w,
    ));
    right_points.push(Point::new(
        start.x - n_first[0] * half_w,
        start.y - n_first[1] * half_w,
    ));

    // Interior joints
    for i in 1..n {
        let vertex = Point::new(segments[i][0], segments[i][1]);
        let (l, r) = miter_offset(vertex, normals[i - 1], normals[i], half_w, style);
        left_points.extend(l);
        right_points.extend(r);
    }

    // End point
    let n_last = normals[n - 1];
    let end = Point::new(segments[n - 1][2], segments[n - 1][3]);
    left_points.push(Point::new(
        end.x + n_last[0] * half_w,
        end.y + n_last[1] * half_w,
    ));
    right_points.push(Point::new(
        end.x - n_last[0] * half_w,
        end.y - n_last[1] * half_w,
    ));

    // Build outline: left forward → end cap → right backward → start cap
    let mut curves = Vec::new();

    for i in 0..left_points.len() - 1 {
        curves.push(Curve::Line(left_points[i], left_points[i + 1]));
    }
    curves.push(Curve::Line(
        *left_points.last().unwrap(),
        *right_points.last().unwrap(),
    ));
    for i in (0..right_points.len() - 1).rev() {
        curves.push(Curve::Line(right_points[i + 1], right_points[i]));
    }
    curves.push(Curve::Line(right_points[0], left_points[0]));

    Path { curves, path_id }
}

/// Compute miter offset at a joint. Returns (left_points, right_points).
/// Uses bisector of normals. Falls back to bevel when miter limit exceeded.
fn miter_offset(
    p: Point,
    n0: [f32; 2],
    n1: [f32; 2],
    half_w: f32,
    style: &StrokeStyle,
) -> (Vec<Point>, Vec<Point>) {
    let mx = n0[0] + n1[0];
    let my = n0[1] + n1[1];
    let mlen = (mx * mx + my * my).sqrt();

    if mlen < 1e-6 {
        // Near-parallel (180° turn)
        return (
            vec![Point::new(p.x + n0[0] * half_w, p.y + n0[1] * half_w)],
            vec![Point::new(p.x - n0[0] * half_w, p.y - n0[1] * half_w)],
        );
    }

    let nmx = mx / mlen;
    let nmy = my / mlen;
    let cos_half = n0[0] * nmx + n0[1] * nmy;
    let miter_len = if cos_half.abs() > 1e-6 {
        half_w / cos_half
    } else {
        half_w
    };

    if miter_len.abs() / half_w > style.miter_limit {
        // Bevel: two points per side
        bevel_points(p, n0, n1, half_w)
    } else {
        // Miter: single point per side
        (
            vec![Point::new(p.x + nmx * miter_len, p.y + nmy * miter_len)],
            vec![Point::new(p.x - nmx * miter_len, p.y - nmy * miter_len)],
        )
    }
}

/// Bevel: two points per side, connecting the gap.
/// Self-intersection on inner side is handled by nonzero fill rule.
fn bevel_points(
    p: Point,
    n0: [f32; 2],
    n1: [f32; 2],
    half_w: f32,
) -> (Vec<Point>, Vec<Point>) {
    (
        vec![
            Point::new(p.x + n0[0] * half_w, p.y + n0[1] * half_w),
            Point::new(p.x + n1[0] * half_w, p.y + n1[1] * half_w),
        ],
        vec![
            Point::new(p.x - n0[0] * half_w, p.y - n0[1] * half_w),
            Point::new(p.x - n1[0] * half_w, p.y - n1[1] * half_w),
        ],
    )
}

/// Compute unit normal for a segment (points left of travel direction).
fn seg_normal(seg: &[f32; 4]) -> [f32; 2] {
    let dx = seg[2] - seg[0];
    let dy = seg[3] - seg[1];
    let len = (dx * dx + dy * dy).sqrt();
    if len < 1e-10 {
        [0.0, 1.0]
    } else {
        [-dy / len, dx / len]
    }
}
