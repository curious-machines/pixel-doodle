/// A 2D point.
#[derive(Debug, Clone, Copy)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

impl Point {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    fn midpoint(self, other: Self) -> Self {
        Self {
            x: (self.x + other.x) * 0.5,
            y: (self.y + other.y) * 0.5,
        }
    }

    fn distance_to(self, other: Self) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

/// A quadratic bezier curve (3 control points).
#[derive(Debug, Clone, Copy)]
pub struct QuadBezier {
    pub p0: Point,
    pub p1: Point,
    pub p2: Point,
}

/// A cubic bezier curve (4 control points).
#[derive(Debug, Clone, Copy)]
pub struct CubicBezier {
    pub p0: Point,
    pub p1: Point,
    pub p2: Point,
    pub p3: Point,
}

/// A bezier curve of any supported order.
#[derive(Debug, Clone, Copy)]
pub enum Curve {
    Line(Point, Point),
    Quad(QuadBezier),
    Cubic(CubicBezier),
}

/// A path is an ordered sequence of curves sharing a path ID.
pub struct Path {
    pub curves: Vec<Curve>,
    pub path_id: u32,
}

const MAX_RECURSION_DEPTH: u32 = 20;

/// Flatten a set of paths into line segments.
///
/// Returns `(segments, seg_path_ids)` where segments are `[x0, y0, x1, y1]`.
pub fn flatten_paths(paths: &[Path], tolerance: f32) -> (Vec<[f32; 4]>, Vec<u32>) {
    let mut segments = Vec::new();
    let mut seg_path_ids = Vec::new();

    for path in paths {
        for curve in &path.curves {
            flatten_curve(curve, tolerance, path.path_id, &mut segments, &mut seg_path_ids);
        }
    }

    (segments, seg_path_ids)
}

fn flatten_curve(
    curve: &Curve,
    tolerance: f32,
    path_id: u32,
    segments: &mut Vec<[f32; 4]>,
    seg_path_ids: &mut Vec<u32>,
) {
    match curve {
        Curve::Line(a, b) => {
            segments.push([a.x, a.y, b.x, b.y]);
            seg_path_ids.push(path_id);
        }
        Curve::Quad(q) => {
            flatten_quad(q, tolerance, path_id, segments, seg_path_ids, 0);
        }
        Curve::Cubic(c) => {
            flatten_cubic(c, tolerance, path_id, segments, seg_path_ids, 0);
        }
    }
}

/// Adaptive flattening of a quadratic bezier.
///
/// The maximum deviation of a quadratic bezier from its chord is
/// `0.25 * distance(p1, midpoint(p0, p2))`.
fn flatten_quad(
    q: &QuadBezier,
    tolerance: f32,
    path_id: u32,
    segments: &mut Vec<[f32; 4]>,
    seg_path_ids: &mut Vec<u32>,
    depth: u32,
) {
    let mid_chord = q.p0.midpoint(q.p2);
    let deviation = q.p1.distance_to(mid_chord) * 0.25;

    if deviation <= tolerance || depth >= MAX_RECURSION_DEPTH {
        segments.push([q.p0.x, q.p0.y, q.p2.x, q.p2.y]);
        seg_path_ids.push(path_id);
        return;
    }

    // De Casteljau split at t=0.5
    let m01 = q.p0.midpoint(q.p1);
    let m12 = q.p1.midpoint(q.p2);
    let mid = m01.midpoint(m12);

    let left = QuadBezier { p0: q.p0, p1: m01, p2: mid };
    let right = QuadBezier { p0: mid, p1: m12, p2: q.p2 };

    flatten_quad(&left, tolerance, path_id, segments, seg_path_ids, depth + 1);
    flatten_quad(&right, tolerance, path_id, segments, seg_path_ids, depth + 1);
}

/// Adaptive flattening of a cubic bezier.
///
/// Uses the maximum distance of control points from the chord as the
/// deviation metric. Specifically: max(distance(p1, chord), distance(p2, chord))
/// where chord is the line from p0 to p3.
fn flatten_cubic(
    c: &CubicBezier,
    tolerance: f32,
    path_id: u32,
    segments: &mut Vec<[f32; 4]>,
    seg_path_ids: &mut Vec<u32>,
    depth: u32,
) {
    let deviation = cubic_deviation(c);

    if deviation <= tolerance || depth >= MAX_RECURSION_DEPTH {
        segments.push([c.p0.x, c.p0.y, c.p3.x, c.p3.y]);
        seg_path_ids.push(path_id);
        return;
    }

    // De Casteljau split at t=0.5
    let m01 = c.p0.midpoint(c.p1);
    let m12 = c.p1.midpoint(c.p2);
    let m23 = c.p2.midpoint(c.p3);
    let m012 = m01.midpoint(m12);
    let m123 = m12.midpoint(m23);
    let mid = m012.midpoint(m123);

    let left = CubicBezier { p0: c.p0, p1: m01, p2: m012, p3: mid };
    let right = CubicBezier { p0: mid, p1: m123, p2: m23, p3: c.p3 };

    flatten_cubic(&left, tolerance, path_id, segments, seg_path_ids, depth + 1);
    flatten_cubic(&right, tolerance, path_id, segments, seg_path_ids, depth + 1);
}

/// Maximum distance of the control points from the chord line (p0→p3).
fn cubic_deviation(c: &CubicBezier) -> f32 {
    let d1 = point_to_line_distance(c.p1, c.p0, c.p3);
    let d2 = point_to_line_distance(c.p2, c.p0, c.p3);
    d1.max(d2)
}

/// Distance from point `p` to the line through `a` and `b`.
fn point_to_line_distance(p: Point, a: Point, b: Point) -> f32 {
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    let len_sq = dx * dx + dy * dy;

    if len_sq < 1e-10 {
        // Degenerate: a and b are the same point
        return p.distance_to(a);
    }

    let cross = (p.x - a.x) * dy - (p.y - a.y) * dx;
    cross.abs() / len_sq.sqrt()
}
