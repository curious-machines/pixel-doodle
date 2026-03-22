use rayon::prelude::*;

pub struct GrayScottParams {
    pub feed: f32,
    pub kill: f32,
    pub du: f32,
    pub dv: f32,
    pub dt: f32,
}

impl Default for GrayScottParams {
    fn default() -> Self {
        Self {
            feed: 0.037,
            kill: 0.06,
            du: 0.21,
            dv: 0.105,
            dt: 1.0,
        }
    }
}

pub struct FluidState {
    pub u: Vec<f32>,
    pub v: Vec<f32>,
    u_next: Vec<f32>,
    v_next: Vec<f32>,
    pub width: usize,
    pub height: usize,
    pub params: GrayScottParams,
    pub substeps_per_frame: u32,
}

impl FluidState {
    pub fn new(width: usize, height: usize) -> Self {
        let n = width * height;
        let mut u = vec![1.0f32; n];
        let mut v = vec![0.0f32; n];

        // Seed several small spots to break symmetry and ensure patterns emerge.
        // Use a simple LCG to scatter seeds deterministically.
        let mut rng = 12345u64;
        let mut next = || -> u64 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            rng
        };
        let num_seeds = 20;
        let radius = 6;
        for _ in 0..num_seeds {
            let sx = (next() as usize) % (width - 2 * radius) + radius;
            let sy = (next() as usize) % (height - 2 * radius) + radius;
            for dy in -(radius as isize)..=(radius as isize) {
                for dx in -(radius as isize)..=(radius as isize) {
                    if dx * dx + dy * dy > (radius * radius) as isize {
                        continue;
                    }
                    let x = (sx as isize + dx) as usize;
                    let y = (sy as isize + dy) as usize;
                    let idx = y * width + x;
                    u[idx] = 0.5;
                    v[idx] = 0.25;
                }
            }
        }

        Self {
            u,
            v,
            u_next: vec![0.0; n],
            v_next: vec![0.0; n],
            width,
            height,
            params: GrayScottParams::default(),
            substeps_per_frame: 8,
        }
    }

    pub fn step(&mut self) {
        for _ in 0..self.substeps_per_frame {
            self.step_once();
            std::mem::swap(&mut self.u, &mut self.u_next);
            std::mem::swap(&mut self.v, &mut self.v_next);
        }
    }

    fn step_once(&mut self) {
        let w = self.width;
        let h = self.height;
        let params = &self.params;
        let u_in = &self.u;
        let v_in = &self.v;

        // Split output buffers into row chunks for parallel processing.
        // We zip u_next and v_next by splitting at the same row boundaries.
        let u_out = &mut self.u_next;
        let v_out = &mut self.v_next;

        // Process rows in parallel. Each thread gets a contiguous row range.
        u_out
            .par_chunks_mut(w)
            .zip(v_out.par_chunks_mut(w))
            .enumerate()
            .for_each(|(y, (u_row, v_row))| {
                let ym = if y == 0 { h - 1 } else { y - 1 };
                let yp = if y == h - 1 { 0 } else { y + 1 };

                for x in 0..w {
                    let xm = if x == 0 { w - 1 } else { x - 1 };
                    let xp = if x == w - 1 { 0 } else { x + 1 };

                    let idx = y * w + x;
                    let u_c = u_in[idx];
                    let v_c = v_in[idx];

                    let lap_u = u_in[y * w + xm] + u_in[y * w + xp]
                        + u_in[ym * w + x] + u_in[yp * w + x]
                        - 4.0 * u_c;

                    let lap_v = v_in[y * w + xm] + v_in[y * w + xp]
                        + v_in[ym * w + x] + v_in[yp * w + x]
                        - 4.0 * v_c;

                    let uvv = u_c * v_c * v_c;

                    u_row[x] = u_c + params.dt * (params.du * lap_u - uvv + params.feed * (1.0 - u_c));
                    v_row[x] = v_c + params.dt * (params.dv * lap_v + uvv - (params.feed + params.kill) * v_c);
                }
            });
    }

    /// Inject chemical V at position (px, py) with a given radius.
    pub fn inject(&mut self, px: usize, py: usize, radius: usize) {
        let r2 = (radius * radius) as isize;
        let r = radius as isize;
        for dy in -r..=r {
            for dx in -r..=r {
                if dx * dx + dy * dy > r2 {
                    continue;
                }
                let x = (px as isize + dx).rem_euclid(self.width as isize) as usize;
                let y = (py as isize + dy).rem_euclid(self.height as isize) as usize;
                let idx = y * self.width + x;
                self.u[idx] = 0.0;
                self.v[idx] = 1.0;
            }
        }
    }

    /// Convert V concentration to ARGB pixel buffer.
    pub fn to_pixels(&self, output: &mut [u32]) {
        for (i, pixel) in output.iter_mut().enumerate() {
            let v = self.v[i].clamp(0.0, 1.0);

            // Color ramp: black → blue → white → orange
            let (r, g, b) = if v < 0.25 {
                // black → blue
                let t = v / 0.25;
                (0.0, 0.0, t)
            } else if v < 0.5 {
                // blue → white
                let t = (v - 0.25) / 0.25;
                (t, t, 1.0)
            } else if v < 0.75 {
                // white → orange
                let t = (v - 0.5) / 0.25;
                (1.0, 1.0 - 0.35 * t, 1.0 - t)
            } else {
                // orange → bright orange
                let t = (v - 0.75) / 0.25;
                (1.0, 0.65 - 0.25 * t, t * 0.1)
            };

            let rb = (r * 255.0) as u32;
            let gb = (g * 255.0) as u32;
            let bb = (b * 255.0) as u32;
            *pixel = 0xFF000000 | (rb << 16) | (gb << 8) | bb;
        }
    }
}
