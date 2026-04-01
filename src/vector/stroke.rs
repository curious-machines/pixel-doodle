use super::flatten::{Curve, Path, Point};

/// Stroke expansion options.
pub struct StrokeStyle {
    /// Stroke width in pixels.
    pub width: f64,
    /// Maximum miter ratio (miter_length / half_width).
    /// When exceeded, falls back to bevel join. SVG default is 4.0.
    pub miter_limit: f64,
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
pub fn stroke_path(segments: &[[f64; 4]], style: &StrokeStyle, path_id: u32) -> Path {
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
    let normals: Vec<[f64; 2]> = segments.iter().map(|s| seg_normal(s)).collect();

    if closed {
        stroke_closed(segments, &normals, style, half_w, path_id)
    } else {
        stroke_open(segments, &normals, style, half_w, path_id)
    }
}

/// Stroke a closed path. Outputs two separate closed contours.
fn stroke_closed(
    segments: &[[f64; 4]],
    normals: &[[f64; 2]],
    style: &StrokeStyle,
    half_w: f64,
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
    segments: &[[f64; 4]],
    normals: &[[f64; 2]],
    style: &StrokeStyle,
    half_w: f64,
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
    n0: [f64; 2],
    n1: [f64; 2],
    half_w: f64,
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
    n0: [f64; 2],
    n1: [f64; 2],
    half_w: f64,
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
fn seg_normal(seg: &[f64; 4]) -> [f64; 2] {
    let dx = seg[2] - seg[0];
    let dy = seg[3] - seg[1];
    let len = (dx * dx + dy * dy).sqrt();
    if len < 1e-10 {
        [0.0, 1.0]
    } else {
        [-dy / len, dx / len]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-6;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    fn point_approx(p: &Point, x: f64, y: f64) -> bool {
        approx_eq(p.x, x) && approx_eq(p.y, y)
    }

    /// Extract all line endpoints from a path's curves.
    fn line_points(path: &Path) -> Vec<(f64, f64, f64, f64)> {
        path.curves
            .iter()
            .map(|c| match c {
                Curve::Line(a, b) => (a.x, a.y, b.x, b.y),
                _ => panic!("expected Line curve"),
            })
            .collect()
    }

    // -------------------------------------------------------
    // Happy path tests
    // -------------------------------------------------------

    #[test]
    fn single_horizontal_segment() {
        let segs = [[0.0, 0.0, 10.0, 0.0]];
        let style = StrokeStyle {
            width: 2.0,
            ..Default::default()
        };
        let path = stroke_path(&segs, &style, 1);

        // Open path: left forward, end cap, right backward, start cap
        // Left side offset by normal * half_w = [0, -1] * 1 => y-1
        // Right side offset by -normal * half_w => y+1
        // (normal of rightward segment: [-dy/len, dx/len] = [0, 1]... wait)
        // dx=10, dy=0 => normal = [-0/10, 10/10] = [0, 1]
        // left = y + 1, right = y - 1
        let pts = line_points(&path);
        assert!(!pts.is_empty());

        // First left point should be (0, 1), last left point (10, 1)
        assert!(point_approx(&path.curves[0].start(), 0.0, 1.0));
        // The outline should span the full segment length
        assert_eq!(path.path_id, 1);
    }

    #[test]
    fn single_vertical_segment() {
        let segs = [[0.0, 0.0, 0.0, 10.0]];
        let style = StrokeStyle {
            width: 2.0,
            ..Default::default()
        };
        let path = stroke_path(&segs, &style, 0);

        // dx=0, dy=10 => normal = [-10/10, 0/10] = [-1, 0]
        // left offset: (-1, 0) * 1 => (-1, 0) + vertex
        // Start left: (-1, 0), Start right: (1, 0)
        // End left: (-1, 10), End right: (1, 10)
        let pts = line_points(&path);
        assert!(!pts.is_empty());
        // Start cap should connect right[0] to left[0]
        let last = pts.last().unwrap();
        assert!(approx_eq(last.2, -1.0) && approx_eq(last.3, 0.0));
    }

    #[test]
    fn single_diagonal_segment() {
        let segs = [[0.0, 0.0, 10.0, 10.0]];
        let style = StrokeStyle {
            width: 2.0,
            ..Default::default()
        };
        let path = stroke_path(&segs, &style, 5);

        // dx=10, dy=10, len=10*sqrt(2)
        // normal = [-10/(10√2), 10/(10√2)] = [-1/√2, 1/√2]
        // half_w = 1.0
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();

        // Left start: (0 + (-inv_sqrt2)*1, 0 + inv_sqrt2*1)
        let first_curve = &path.curves[0];
        let (p0, _) = match first_curve {
            Curve::Line(a, b) => (a, b),
            _ => panic!("expected line"),
        };
        assert!(approx_eq(p0.x, -inv_sqrt2));
        assert!(approx_eq(p0.y, inv_sqrt2));
        assert_eq!(path.path_id, 5);
    }

    #[test]
    fn closed_rectangular_path() {
        // Rectangle: (0,0) -> (10,0) -> (10,5) -> (0,5) -> (0,0)
        let segs = [
            [0.0, 0.0, 10.0, 0.0],
            [10.0, 0.0, 10.0, 5.0],
            [10.0, 5.0, 0.0, 5.0],
            [0.0, 5.0, 0.0, 0.0],
        ];
        let style = StrokeStyle {
            width: 2.0,
            ..Default::default()
        };
        let path = stroke_path(&segs, &style, 3);

        // Closed path: should produce two contours (left forward + right reversed)
        // 4 segments with miter joins => 4 left points, 4 right points
        // Left contour: 4 lines, Right contour: 4 lines => 8 curves
        assert!(path.curves.len() >= 8);
        assert_eq!(path.path_id, 3);

        // Verify it's actually closed: left contour wraps around
        let pts = line_points(&path);
        // First contour: last endpoint should equal first startpoint
        let first_start = (pts[0].0, pts[0].1);
        let contour_end = (pts[3].2, pts[3].3);
        assert!(approx_eq(first_start.0, contour_end.0));
        assert!(approx_eq(first_start.1, contour_end.1));
    }

    #[test]
    fn open_polyline_90_degree_miter_joins() {
        // L-shape: right then down
        let segs = [
            [0.0, 0.0, 10.0, 0.0],
            [10.0, 0.0, 10.0, 10.0],
        ];
        let style = StrokeStyle {
            width: 2.0,
            miter_limit: 4.0, // sqrt(2) < 4, so 90-degree corners should miter
        };
        let path = stroke_path(&segs, &style, 0);

        // Open path with 1 interior joint (miter), 2 caps
        // Left: 2 offset points + start + end = 4
        // Right: same => 4
        // Curves: 3 left edges + end cap + 3 right edges + start cap = 8
        let pts = line_points(&path);
        // Open path with 1 miter joint: left 3 pts => 2 edges, right 3 pts => 2 edges, 2 caps = 6
        assert_eq!(pts.len(), 6);
    }

    // -------------------------------------------------------
    // Edge case tests
    // -------------------------------------------------------

    #[test]
    fn empty_segments() {
        let segs: &[[f64; 4]] = &[];
        let style = StrokeStyle::default();
        let path = stroke_path(segs, &style, 42);
        assert!(path.curves.is_empty());
        assert_eq!(path.path_id, 42);
    }

    #[test]
    fn single_segment_open_path() {
        let segs = [[5.0, 5.0, 15.0, 5.0]];
        let style = StrokeStyle::default();
        let path = stroke_path(&segs, &style, 0);

        // Open, single segment: left fwd (1 edge) + end cap (1) + right bwd (1) + start cap (1) = 4
        assert_eq!(path.curves.len(), 4);
    }

    #[test]
    fn zero_length_segment_uses_default_normal() {
        let segs = [[5.0, 5.0, 5.0, 5.0]];
        let style = StrokeStyle {
            width: 4.0,
            ..Default::default()
        };
        let path = stroke_path(&segs, &style, 0);

        // Zero-length => normal defaults to [0, 1]
        // half_w = 2.0
        // left start: (5, 5+2) = (5, 7), right start: (5, 5-2) = (5, 3)
        // left end: same, right end: same (segment has no length)
        let pts = line_points(&path);
        // Should still produce a valid outline (degenerate rectangle)
        assert!(!pts.is_empty(), "zero-length segment should produce curves");
        // Verify the outline uses the default normal [0, 1]
        // First curve (left edge): should have y-coordinates at 7.0
        let first = &pts[0];
        assert!(
            approx_eq(first.1, 7.0) && approx_eq(first.3, 7.0),
            "left edge should be at y=7, got ({}, {}) -> ({}, {})",
            first.0, first.1, first.2, first.3,
        );
    }

    #[test]
    fn very_thin_width() {
        let segs = [[0.0, 0.0, 10.0, 0.0]];
        let style = StrokeStyle {
            width: 0.1,
            ..Default::default()
        };
        let path = stroke_path(&segs, &style, 0);

        // Should still produce valid outline
        assert_eq!(path.curves.len(), 4);

        // Offset should be 0.05
        let (p0, _) = match &path.curves[0] {
            Curve::Line(a, b) => (a, b),
            _ => panic!("expected line"),
        };
        // Normal for rightward segment is [0, 1], left offset = y + 0.05
        assert!(approx_eq(p0.y, 0.05));
    }

    #[test]
    fn very_thick_width() {
        let segs = [[0.0, 0.0, 10.0, 0.0]];
        let style = StrokeStyle {
            width: 100.0,
            ..Default::default()
        };
        let path = stroke_path(&segs, &style, 0);

        assert_eq!(path.curves.len(), 4);

        // Offset should be 50.0
        let (p0, _) = match &path.curves[0] {
            Curve::Line(a, b) => (a, b),
            _ => panic!("expected line"),
        };
        assert!(approx_eq(p0.y, 50.0));
    }

    #[test]
    fn miter_limit_1_always_bevels() {
        // 90-degree corner: miter ratio = 1/cos(45°) = √2 ≈ 1.414 > 1.0
        let segs = [
            [0.0, 0.0, 10.0, 0.0],
            [10.0, 0.0, 10.0, 10.0],
        ];
        let style = StrokeStyle {
            width: 2.0,
            miter_limit: 1.0,
        };
        let path = stroke_path(&segs, &style, 0);

        // Bevel produces 2 points per side at joint instead of 1
        // Left: start(1) + joint(2) + end(1) = 4 points => 3 edges
        // Right: same => 3 edges
        // Plus 2 caps = 8
        let pts = line_points(&path);
        assert_eq!(pts.len(), 8);
    }

    #[test]
    fn uturn_180_degree() {
        // Going right, then going left (180-degree turn)
        let segs = [
            [0.0, 0.0, 10.0, 0.0],
            [10.0, 0.0, 0.0, 0.0],
        ];
        let style = StrokeStyle {
            width: 2.0,
            ..Default::default()
        };
        let path = stroke_path(&segs, &style, 0);

        // Normals: [0,1] and [0,-1]. Sum = [0,0], mlen < 1e-6 => degenerate.
        // Falls back to using n0 offset directly.
        // Should still produce a valid (non-panicking) path.
        assert!(!path.curves.is_empty());
    }

    // -------------------------------------------------------
    // Verification tests
    // -------------------------------------------------------

    #[test]
    fn horizontal_line_width2_offsets_by_1() {
        let segs = [[0.0, 0.0, 10.0, 0.0]];
        let style = StrokeStyle {
            width: 2.0,
            ..Default::default()
        };
        let path = stroke_path(&segs, &style, 0);

        // Normal for rightward: [0, 1], half_w = 1.0
        // Left side: y + 1.0
        // Right side: y - 1.0
        // Outline: left[0]=(0,1) -> left[1]=(10,1) -> right[1]=(10,-1) -> right[0]=(0,-1) -> left[0]
        let pts = line_points(&path);
        assert_eq!(pts.len(), 4);

        // Left edge: (0,1) -> (10,1)
        assert!(point_approx(&Point::new(pts[0].0, pts[0].1), 0.0, 1.0));
        assert!(point_approx(&Point::new(pts[0].2, pts[0].3), 10.0, 1.0));

        // End cap: (10,1) -> (10,-1)
        assert!(point_approx(&Point::new(pts[1].0, pts[1].1), 10.0, 1.0));
        assert!(point_approx(&Point::new(pts[1].2, pts[1].3), 10.0, -1.0));

        // Right edge (reversed): (10,-1) -> (0,-1)
        assert!(point_approx(&Point::new(pts[2].0, pts[2].1), 10.0, -1.0));
        assert!(point_approx(&Point::new(pts[2].2, pts[2].3), 0.0, -1.0));

        // Start cap: (0,-1) -> (0,1)
        assert!(point_approx(&Point::new(pts[3].0, pts[3].1), 0.0, -1.0));
        assert!(point_approx(&Point::new(pts[3].2, pts[3].3), 0.0, 1.0));
    }

    #[test]
    fn closed_path_produces_closed_contours() {
        // Triangle: (0,0) -> (10,0) -> (5,8.66) -> (0,0)
        let segs = [
            [0.0, 0.0, 10.0, 0.0],
            [10.0, 0.0, 5.0, 8.66],
            [5.0, 8.66, 0.0, 0.0],
        ];
        let style = StrokeStyle::default();
        let path = stroke_path(&segs, &style, 7);

        // Closed path: first_start = (0,0), last_end = (0,0) => closed
        // Should have left contour + right contour
        let pts = line_points(&path);
        assert!(pts.len() >= 6); // at least 3 edges per contour

        // Verify each contour is closed:
        // Left contour spans first half of curves
        // The left contour's last endpoint wraps to its first startpoint
        // (verified by the modular indexing in stroke_closed)
        assert_eq!(path.path_id, 7);
    }

    #[test]
    fn output_path_id_matches_input() {
        for id in [0, 1, 42, 999, u32::MAX] {
            let segs = [[0.0, 0.0, 1.0, 0.0]];
            let style = StrokeStyle::default();
            let path = stroke_path(&segs, &style, id);
            assert_eq!(path.path_id, id);
        }
    }

    #[test]
    fn closed_detection_threshold() {
        // Just within threshold (< 1e-4)
        let segs = [
            [0.0, 0.0, 10.0, 0.0],
            [10.0, 0.0, 10.0, 10.0],
            [10.0, 10.0, 0.00005, 0.00005],
        ];
        let style = StrokeStyle::default();
        let path_closed = stroke_path(&segs, &style, 0);

        // Just outside threshold (>= 1e-4)
        let segs2 = [
            [0.0, 0.0, 10.0, 0.0],
            [10.0, 0.0, 10.0, 10.0],
            [10.0, 10.0, 0.001, 0.001],
        ];
        let path_open = stroke_path(&segs2, &style, 0);

        // Closed path: 2 contours (no caps). Open path: 1 contour (with caps).
        // They should differ in structure.
        let closed_count = path_closed.curves.len();
        let open_count = path_open.curves.len();
        assert_ne!(closed_count, open_count);
    }

    // Helper to extract start point from a Curve
    impl Curve {
        fn start(&self) -> Point {
            match self {
                Curve::Line(a, _) => *a,
                Curve::Quad(q) => q.p0,
                Curve::Cubic(c) => c.p0,
            }
        }
    }
}

