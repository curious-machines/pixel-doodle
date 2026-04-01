/// A 2D point (f64 precision for curve evaluation).
#[derive(Debug, Clone, Copy)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    fn midpoint(self, other: Self) -> Self {
        Self {
            x: (self.x + other.x) * 0.5,
            y: (self.y + other.y) * 0.5,
        }
    }

    fn distance_to(self, other: Self) -> f64 {
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

/// Flatten a set of paths into line segments, parallelized across paths.
///
/// Returns `(segments, seg_path_ids)` where segments are `[f64; 4]` for GPU.
/// All math is f64. Conversion to f32 happens at the GPU upload boundary.
pub fn flatten_paths(paths: &[Path], tolerance: f32) -> (Vec<[f64; 4]>, Vec<u32>) {
    use rayon::prelude::*;

    let tol = tolerance as f64;

    let per_path: Vec<(Vec<[f64; 4]>, Vec<u32>)> = paths
        .par_iter()
        .map(|path| {
            let mut segments = Vec::new();
            let mut seg_path_ids = Vec::new();
            for curve in &path.curves {
                flatten_curve(curve, tol, path.path_id, &mut segments, &mut seg_path_ids);
            }
            (segments, seg_path_ids)
        })
        .collect();

    let total_segs: usize = per_path.iter().map(|(s, _)| s.len()).sum();
    let mut segments = Vec::with_capacity(total_segs);
    let mut seg_path_ids = Vec::with_capacity(total_segs);
    for (segs, ids) in per_path {
        segments.extend(segs);
        seg_path_ids.extend(ids);
    }

    (segments, seg_path_ids)
}

/// Emit a line segment, converting f64 points to f32 for GPU output.
fn emit_segment(
    a: Point,
    b: Point,
    path_id: u32,
    segments: &mut Vec<[f64; 4]>,
    seg_path_ids: &mut Vec<u32>,
) {
    segments.push([a.x, a.y, b.x, b.y]);
    seg_path_ids.push(path_id);
}

fn flatten_curve(
    curve: &Curve,
    tolerance: f64,
    path_id: u32,
    segments: &mut Vec<[f64; 4]>,
    seg_path_ids: &mut Vec<u32>,
) {
    match curve {
        Curve::Line(a, b) => {
            emit_segment(*a, *b, path_id, segments, seg_path_ids);
        }
        Curve::Quad(q) => {
            flatten_quad(q, tolerance, path_id, segments, seg_path_ids, 0);
        }
        Curve::Cubic(c) => {
            flatten_cubic(c, tolerance, path_id, segments, seg_path_ids, 0);
        }
    }
}

fn flatten_quad(
    q: &QuadBezier,
    tolerance: f64,
    path_id: u32,
    segments: &mut Vec<[f64; 4]>,
    seg_path_ids: &mut Vec<u32>,
    depth: u32,
) {
    let mid_chord = q.p0.midpoint(q.p2);
    let deviation = q.p1.distance_to(mid_chord) * 0.25;

    if deviation <= tolerance || depth >= MAX_RECURSION_DEPTH {
        emit_segment(q.p0, q.p2, path_id, segments, seg_path_ids);
        return;
    }

    let m01 = q.p0.midpoint(q.p1);
    let m12 = q.p1.midpoint(q.p2);
    let mid = m01.midpoint(m12);

    let left = QuadBezier { p0: q.p0, p1: m01, p2: mid };
    let right = QuadBezier { p0: mid, p1: m12, p2: q.p2 };

    flatten_quad(&left, tolerance, path_id, segments, seg_path_ids, depth + 1);
    flatten_quad(&right, tolerance, path_id, segments, seg_path_ids, depth + 1);
}

fn flatten_cubic(
    c: &CubicBezier,
    tolerance: f64,
    path_id: u32,
    segments: &mut Vec<[f64; 4]>,
    seg_path_ids: &mut Vec<u32>,
    depth: u32,
) {
    let deviation = cubic_deviation(c);

    if deviation <= tolerance || depth >= MAX_RECURSION_DEPTH {
        emit_segment(c.p0, c.p3, path_id, segments, seg_path_ids);
        return;
    }

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

fn cubic_deviation(c: &CubicBezier) -> f64 {
    let d1 = point_to_line_distance(c.p1, c.p0, c.p3);
    let d2 = point_to_line_distance(c.p2, c.p0, c.p3);
    d1.max(d2)
}

fn point_to_line_distance(p: Point, a: Point, b: Point) -> f64 {
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    let len_sq = dx * dx + dy * dy;

    if len_sq < 1e-20 {
        return p.distance_to(a);
    }

    let cross = (p.x - a.x) * dy - (p.y - a.y) * dx;
    cross.abs() / len_sq.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pt(x: f64, y: f64) -> Point {
        Point::new(x, y)
    }

    fn line_path(p0: Point, p1: Point, path_id: u32) -> Path {
        Path {
            curves: vec![Curve::Line(p0, p1)],
            path_id,
        }
    }

    // --- Happy path tests ---

    #[test]
    fn straight_line_produces_single_segment() {
        let paths = vec![line_path(pt(0.0, 0.0), pt(10.0, 5.0), 0)];
        let (segments, ids) = flatten_paths(&paths, 0.5);
        assert_eq!(segments.len(), 1);
        assert_eq!(ids.len(), 1);
        assert_eq!(segments[0], [0.0, 0.0, 10.0, 5.0]);
        assert_eq!(ids[0], 0);
    }

    #[test]
    fn quadratic_curve_subdivides() {
        let q = QuadBezier {
            p0: pt(0.0, 0.0),
            p1: pt(50.0, 100.0),
            p2: pt(100.0, 0.0),
        };
        let paths = vec![Path {
            curves: vec![Curve::Quad(q)],
            path_id: 1,
        }];
        let (segments, ids) = flatten_paths(&paths, 0.5);
        assert!(segments.len() > 1, "curved quad should subdivide into multiple segments");
        assert_eq!(segments.len(), ids.len());
        // First segment starts at p0
        assert_eq!(segments[0][0], 0.0);
        assert_eq!(segments[0][1], 0.0);
        // Last segment ends at p2
        let last = segments.last().unwrap();
        assert_eq!(last[2], 100.0);
        assert_eq!(last[3], 0.0);
        // All path IDs should be 1
        assert!(ids.iter().all(|&id| id == 1));
    }

    #[test]
    fn cubic_curve_subdivides() {
        let c = CubicBezier {
            p0: pt(0.0, 0.0),
            p1: pt(33.0, 100.0),
            p2: pt(66.0, -100.0),
            p3: pt(100.0, 0.0),
        };
        let paths = vec![Path {
            curves: vec![Curve::Cubic(c)],
            path_id: 2,
        }];
        let (segments, ids) = flatten_paths(&paths, 0.5);
        assert!(segments.len() > 1, "S-curve cubic should subdivide into multiple segments");
        assert_eq!(segments.len(), ids.len());
        assert_eq!(segments[0][0], 0.0);
        assert_eq!(segments[0][1], 0.0);
        let last = segments.last().unwrap();
        assert_eq!(last[2], 100.0);
        assert_eq!(last[3], 0.0);
        assert!(ids.iter().all(|&id| id == 2));
    }

    #[test]
    fn circle_approximation_produces_many_segments() {
        // Approximate a unit circle with 4 cubic bezier arcs.
        // Standard kappa for quarter-circle: 4*(sqrt(2)-1)/3 ~ 0.5522847498
        let k = 0.5522847498;
        let cubics = [
            CubicBezier { p0: pt(1.0, 0.0), p1: pt(1.0, k),   p2: pt(k, 1.0),   p3: pt(0.0, 1.0) },
            CubicBezier { p0: pt(0.0, 1.0), p1: pt(-k, 1.0),  p2: pt(-1.0, k),   p3: pt(-1.0, 0.0) },
            CubicBezier { p0: pt(-1.0, 0.0), p1: pt(-1.0, -k), p2: pt(-k, -1.0),  p3: pt(0.0, -1.0) },
            CubicBezier { p0: pt(0.0, -1.0), p1: pt(k, -1.0),  p2: pt(1.0, -k),   p3: pt(1.0, 0.0) },
        ];
        let paths = vec![Path {
            curves: cubics.iter().map(|c| Curve::Cubic(*c)).collect(),
            path_id: 10,
        }];
        let (segments, ids) = flatten_paths(&paths, 0.01);
        // A circle at tight tolerance should produce many segments
        assert!(segments.len() >= 16, "circle should produce many segments, got {}", segments.len());
        assert!(ids.iter().all(|&id| id == 10));
    }

    #[test]
    fn multiple_paths_preserve_path_ids() {
        let paths = vec![
            line_path(pt(0.0, 0.0), pt(1.0, 0.0), 42),
            line_path(pt(5.0, 5.0), pt(6.0, 6.0), 99),
            line_path(pt(10.0, 10.0), pt(11.0, 11.0), 7),
        ];
        let (segments, ids) = flatten_paths(&paths, 1.0);
        assert_eq!(segments.len(), 3);
        // Due to rayon parallel iteration, order may not be preserved.
        // Collect (segment, id) pairs and sort by path_id for deterministic checks.
        let mut pairs: Vec<([f64; 4], u32)> = segments.into_iter().zip(ids).collect();
        pairs.sort_by_key(|&(_, id)| id);
        assert_eq!(pairs[0].1, 7);
        assert_eq!(pairs[1].1, 42);
        assert_eq!(pairs[2].1, 99);
        assert_eq!(pairs[1].0, [0.0, 0.0, 1.0, 0.0]);
        assert_eq!(pairs[2].0, [5.0, 5.0, 6.0, 6.0]);
    }

    #[test]
    fn multiple_curves_in_single_path() {
        let paths = vec![Path {
            curves: vec![
                Curve::Line(pt(0.0, 0.0), pt(10.0, 0.0)),
                Curve::Line(pt(10.0, 0.0), pt(10.0, 10.0)),
                Curve::Line(pt(10.0, 10.0), pt(0.0, 0.0)),
            ],
            path_id: 3,
        }];
        let (segments, ids) = flatten_paths(&paths, 1.0);
        assert_eq!(segments.len(), 3);
        assert_eq!(ids, vec![3, 3, 3]);
        assert_eq!(segments[0], [0.0, 0.0, 10.0, 0.0]);
        assert_eq!(segments[1], [10.0, 0.0, 10.0, 10.0]);
        assert_eq!(segments[2], [10.0, 10.0, 0.0, 0.0]);
    }

    // --- Edge case tests ---

    #[test]
    fn empty_paths_list_produces_empty_output() {
        let (segments, ids) = flatten_paths(&[], 0.5);
        assert!(segments.is_empty());
        assert!(ids.is_empty());
    }

    #[test]
    fn path_with_no_curves_produces_no_segments() {
        let paths = vec![Path {
            curves: vec![],
            path_id: 5,
        }];
        let (segments, ids) = flatten_paths(&paths, 0.5);
        assert!(segments.is_empty());
        assert!(ids.is_empty());
    }

    #[test]
    fn zero_length_line_produces_degenerate_segment() {
        let paths = vec![line_path(pt(7.0, 3.0), pt(7.0, 3.0), 0)];
        let (segments, ids) = flatten_paths(&paths, 0.5);
        assert_eq!(segments.len(), 1);
        assert_eq!(ids.len(), 1);
        assert_eq!(segments[0], [7.0, 3.0, 7.0, 3.0]);
    }

    #[test]
    fn very_large_tolerance_produces_minimal_subdivision() {
        // A highly curved cubic, but with a huge tolerance: should collapse to 1 segment
        let c = CubicBezier {
            p0: pt(0.0, 0.0),
            p1: pt(0.0, 100.0),
            p2: pt(100.0, -100.0),
            p3: pt(100.0, 0.0),
        };
        let paths = vec![Path {
            curves: vec![Curve::Cubic(c)],
            path_id: 0,
        }];
        let (segments, _) = flatten_paths(&paths, 1000.0);
        assert_eq!(segments.len(), 1, "huge tolerance should produce single segment");
    }

    #[test]
    fn very_small_tolerance_produces_many_segments() {
        let q = QuadBezier {
            p0: pt(0.0, 0.0),
            p1: pt(50.0, 100.0),
            p2: pt(100.0, 0.0),
        };
        let paths = vec![Path {
            curves: vec![Curve::Quad(q)],
            path_id: 0,
        }];
        let (seg_fine, _) = flatten_paths(&paths, 0.001);
        let (seg_coarse, _) = flatten_paths(&paths, 1.0);
        assert!(
            seg_fine.len() > seg_coarse.len(),
            "finer tolerance should produce more segments: fine={}, coarse={}",
            seg_fine.len(),
            seg_coarse.len()
        );
    }

    #[test]
    fn nearly_collinear_quadratic_minimal_subdivision() {
        // Control point exactly on the chord midpoint -> deviation = 0
        let q = QuadBezier {
            p0: pt(0.0, 0.0),
            p1: pt(50.0, 0.0), // on the line from p0 to p2
            p2: pt(100.0, 0.0),
        };
        let paths = vec![Path {
            curves: vec![Curve::Quad(q)],
            path_id: 0,
        }];
        let (segments, _) = flatten_paths(&paths, 0.01);
        assert_eq!(segments.len(), 1, "collinear quad should produce single segment");
    }

    // --- Tolerance verification ---

    /// Evaluate a cubic bezier at parameter t in [0, 1].
    fn eval_cubic(c: &CubicBezier, t: f64) -> Point {
        let u = 1.0 - t;
        let x = u * u * u * c.p0.x + 3.0 * u * u * t * c.p1.x + 3.0 * u * t * t * c.p2.x + t * t * t * c.p3.x;
        let y = u * u * u * c.p0.y + 3.0 * u * u * t * c.p1.y + 3.0 * u * t * t * c.p2.y + t * t * t * c.p3.y;
        pt(x, y)
    }

    /// Distance from point p to the line segment from a to b.
    fn distance_to_segment(p: Point, a: Point, b: Point) -> f64 {
        let dx = b.x - a.x;
        let dy = b.y - a.y;
        let len_sq = dx * dx + dy * dy;
        if len_sq < 1e-20 {
            return p.distance_to(a);
        }
        let t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / len_sq;
        let t = t.clamp(0.0, 1.0);
        let proj = pt(a.x + t * dx, a.y + t * dy);
        p.distance_to(proj)
    }

    #[test]
    fn cubic_flattening_stays_within_tolerance() {
        let tolerance = 0.25;
        let cubic = CubicBezier {
            p0: pt(0.0, 0.0),
            p1: pt(30.0, 80.0),
            p2: pt(70.0, -40.0),
            p3: pt(100.0, 0.0),
        };
        let paths = vec![Path {
            curves: vec![Curve::Cubic(cubic)],
            path_id: 0,
        }];
        let (segments, _) = flatten_paths(&paths, tolerance);

        // For each output segment, sample many points along the corresponding
        // portion of the original cubic and check that each sample is within
        // tolerance of the line segment.
        //
        // We don't have a direct parameter mapping, so we instead check that
        // every densely-sampled point on the entire cubic is close to at least
        // one output segment. This verifies the flattening covers the curve.
        let num_samples = 1000;
        for i in 0..=num_samples {
            let t = i as f64 / num_samples as f64;
            let p = eval_cubic(&cubic, t);

            let min_dist = segments
                .iter()
                .map(|seg| {
                    let a = pt(seg[0], seg[1]);
                    let b = pt(seg[2], seg[3]);
                    distance_to_segment(p, a, b)
                })
                .fold(f64::INFINITY, f64::min);

            assert!(
                min_dist <= tolerance as f64 + 1e-9,
                "point at t={:.4} is {:.6} from nearest segment, exceeds tolerance {}",
                t,
                min_dist,
                tolerance,
            );
        }
    }

    #[test]
    fn segments_are_contiguous() {
        // Verify that consecutive segments share endpoints (no gaps).
        let c = CubicBezier {
            p0: pt(0.0, 0.0),
            p1: pt(20.0, 60.0),
            p2: pt(80.0, -60.0),
            p3: pt(100.0, 0.0),
        };
        let paths = vec![Path {
            curves: vec![Curve::Cubic(c)],
            path_id: 0,
        }];
        let (segments, _) = flatten_paths(&paths, 0.5);
        assert!(segments.len() > 1);
        for i in 1..segments.len() {
            let prev_end = (segments[i - 1][2], segments[i - 1][3]);
            let curr_start = (segments[i][0], segments[i][1]);
            assert_eq!(
                prev_end, curr_start,
                "gap between segment {} and {}: end={:?}, start={:?}",
                i - 1, i, prev_end, curr_start
            );
        }
    }
}
