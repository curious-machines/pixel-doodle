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

// ---------------------------------------------------------------------------
// Shallow Water simulation
// ---------------------------------------------------------------------------

pub struct ShallowWaterParams {
    pub gravity: f32,
    pub damping: f32,
    pub dt: f32,
}

impl Default for ShallowWaterParams {
    fn default() -> Self {
        Self {
            gravity: 9.8,
            damping: 0.001,
            dt: 0.02,
        }
    }
}

pub struct ShallowWaterState {
    pub h: Vec<f32>,
    pub vx: Vec<f32>,
    pub vy: Vec<f32>,
    h_next: Vec<f32>,
    vx_next: Vec<f32>,
    vy_next: Vec<f32>,
    pub width: usize,
    pub height: usize,
    pub params: ShallowWaterParams,
    pub substeps_per_frame: u32,
}

impl ShallowWaterState {
    pub fn new(width: usize, height: usize) -> Self {
        let n = width * height;
        let mut h = vec![1.0f32; n];
        let vx = vec![0.0f32; n];
        let vy = vec![0.0f32; n];

        // Seed a few raised bumps so there's something to see immediately
        let mut rng = 54321u64;
        let mut next = || -> u64 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            rng
        };
        let num_seeds = 5;
        let radius = 20;
        for _ in 0..num_seeds {
            let sx = (next() as usize) % (width - 2 * radius) + radius;
            let sy = (next() as usize) % (height - 2 * radius) + radius;
            for dy in -(radius as isize)..=(radius as isize) {
                for dx in -(radius as isize)..=(radius as isize) {
                    let d2 = dx * dx + dy * dy;
                    let r2 = (radius * radius) as isize;
                    if d2 > r2 {
                        continue;
                    }
                    let x = (sx as isize + dx) as usize;
                    let y = (sy as isize + dy) as usize;
                    // Gaussian-ish bump
                    let t = 1.0 - (d2 as f32 / r2 as f32);
                    h[y * width + x] += 0.3 * t * t;
                }
            }
        }

        Self {
            h,
            vx,
            vy,
            h_next: vec![0.0; n],
            vx_next: vec![0.0; n],
            vy_next: vec![0.0; n],
            width,
            height,
            params: ShallowWaterParams::default(),
            substeps_per_frame: 4,
        }
    }

    pub fn step(&mut self) {
        for _ in 0..self.substeps_per_frame {
            self.step_once();
            std::mem::swap(&mut self.h, &mut self.h_next);
            std::mem::swap(&mut self.vx, &mut self.vx_next);
            std::mem::swap(&mut self.vy, &mut self.vy_next);
        }
    }

    fn step_once(&mut self) {
        let w = self.width;
        let he = self.height;
        let g = self.params.gravity;
        let damp = self.params.damping;
        let dt = self.params.dt;
        let h_in = &self.h;
        let vx_in = &self.vx;
        let vy_in = &self.vy;

        let h_out = &mut self.h_next;
        let vx_out = &mut self.vx_next;
        let vy_out = &mut self.vy_next;

        h_out
            .par_chunks_mut(w)
            .zip(vx_out.par_chunks_mut(w))
            .zip(vy_out.par_chunks_mut(w))
            .enumerate()
            .for_each(|(y, ((h_row, vx_row), vy_row))| {
                let ym = if y == 0 { he - 1 } else { y - 1 };
                let yp = if y == he - 1 { 0 } else { y + 1 };

                for x in 0..w {
                    let xm = if x == 0 { w - 1 } else { x - 1 };
                    let xp = if x == w - 1 { 0 } else { x + 1 };

                    let h_l = h_in[y * w + xm];
                    let h_r = h_in[y * w + xp];
                    let h_u = h_in[ym * w + x];
                    let h_d = h_in[yp * w + x];

                    // Lax-Friedrichs: replace center with neighbor average for stability
                    let h_avg = (h_l + h_r + h_u + h_d) * 0.25;
                    let vx_avg = (vx_in[y * w + xm] + vx_in[y * w + xp]
                        + vx_in[ym * w + x] + vx_in[yp * w + x]) * 0.25;
                    let vy_avg = (vy_in[y * w + xm] + vy_in[y * w + xp]
                        + vy_in[ym * w + x] + vy_in[yp * w + x]) * 0.25;

                    // Central differences for height gradient
                    let dh_dx = (h_r - h_l) * 0.5;
                    let dh_dy = (h_d - h_u) * 0.5;

                    // Update velocity (Lax-Friedrichs): pressure gradient + damping
                    vx_row[x] = vx_avg - dt * g * dh_dx - dt * damp * vx_avg;
                    vy_row[x] = vy_avg - dt * g * dh_dy - dt * damp * vy_avg;

                    // Flux divergence for height update (Lax-Friedrichs)
                    let flux_x = (h_r * vx_in[y * w + xp]
                        - h_l * vx_in[y * w + xm]) * 0.5;
                    let flux_y = (h_d * vy_in[yp * w + x]
                        - h_u * vy_in[ym * w + x]) * 0.5;
                    h_row[x] = h_avg - dt * (flux_x + flux_y);
                }
            });
    }

    /// Inject a height bump at position (px, py) with a given radius.
    pub fn inject(&mut self, px: usize, py: usize, radius: usize) {
        let r2 = (radius * radius) as isize;
        let r = radius as isize;
        for dy in -r..=r {
            for dx in -r..=r {
                let d2 = dx * dx + dy * dy;
                if d2 > r2 {
                    continue;
                }
                let x = (px as isize + dx).rem_euclid(self.width as isize) as usize;
                let y = (py as isize + dy).rem_euclid(self.height as isize) as usize;
                let idx = y * self.width + x;
                let t = 1.0 - (d2 as f32 / r2 as f32);
                self.h[idx] += 0.15 * t * t;
            }
        }
    }

    /// Convert water height to ARGB pixel buffer.
    pub fn to_pixels(&self, output: &mut [u32]) {
        for (i, pixel) in output.iter_mut().enumerate() {
            // Height deviation from rest level (1.0)
            let deviation = self.h[i] - 1.0;

            // Map deviation to color: deep blue (low) → mid blue (rest) → white (peak)
            let (r, g, b) = if deviation < -0.1 {
                // Deep trough: dark blue
                (0.0_f32, 0.0, 0.3)
            } else if deviation < 0.0 {
                // Below rest: dark blue → mid blue
                let t = (deviation + 0.1) / 0.1;
                (0.0, 0.1 * t, 0.3 + 0.4 * t)
            } else if deviation < 0.05 {
                // Near rest: mid blue → light blue
                let t = deviation / 0.05;
                (0.1 * t, 0.1 + 0.3 * t, 0.7 + 0.15 * t)
            } else if deviation < 0.15 {
                // Rising: light blue → white
                let t = (deviation - 0.05) / 0.1;
                (0.1 + 0.9 * t, 0.4 + 0.6 * t, 0.85 + 0.15 * t)
            } else {
                // Peak: white
                (1.0, 1.0, 1.0)
            };

            let rb = (r.clamp(0.0, 1.0) * 255.0) as u32;
            let gb = (g.clamp(0.0, 1.0) * 255.0) as u32;
            let bb = (b.clamp(0.0, 1.0) * 255.0) as u32;
            *pixel = 0xFF000000 | (rb << 16) | (gb << 8) | bb;
        }
    }
}

// ---------------------------------------------------------------------------
// Smoke simulation (Stable Fluids — Stam 1999)
// ---------------------------------------------------------------------------

pub struct SmokeState {
    pub width: usize,
    pub height: usize,
    // Current fields
    vx: Vec<f64>,
    vy: Vec<f64>,
    density: Vec<f64>,
    // Temporary/previous fields
    vx0: Vec<f64>,
    vy0: Vec<f64>,
    density0: Vec<f64>,
    // Pressure solve workspace
    pressure: Vec<f64>,
    pressure_tmp: Vec<f64>,
    divergence: Vec<f64>,
    // Parameters
    dt: f64,
    dissipation: f64,
    buoyancy: f64,
}

impl SmokeState {
    pub fn new(width: usize, height: usize) -> Self {
        let n = width * height;
        Self {
            width,
            height,
            vx: vec![0.0; n],
            vy: vec![0.0; n],
            density: vec![0.0; n],
            vx0: vec![0.0; n],
            vy0: vec![0.0; n],
            density0: vec![0.0; n],
            pressure: vec![0.0; n],
            pressure_tmp: vec![0.0; n],
            divergence: vec![0.0; n],
            dt: 4.0,
            dissipation: 0.998,
            buoyancy: 0.08,
        }
    }

    pub fn step(&mut self) {
        let w = self.width;
        let h = self.height;

        // 1. Advect velocity and density (semi-Lagrangian)
        std::mem::swap(&mut self.vx, &mut self.vx0);
        std::mem::swap(&mut self.vy, &mut self.vy0);
        std::mem::swap(&mut self.density, &mut self.density0);

        self.vx.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            for x in 0..w {
                let i = y * w + x;
                let src_x = x as f64 - self.dt * self.vx0[i];
                let src_y = y as f64 - self.dt * self.vy0[i];
                row[x] = sample_bilinear(&self.vx0, src_x, src_y, w, h);
            }
        });

        self.vy.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            for x in 0..w {
                let i = y * w + x;
                let src_x = x as f64 - self.dt * self.vx0[i];
                let src_y = y as f64 - self.dt * self.vy0[i];
                row[x] = sample_bilinear(&self.vy0, src_x, src_y, w, h);
            }
        });

        self.density.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            for x in 0..w {
                let i = y * w + x;
                let src_x = x as f64 - self.dt * self.vx0[i];
                let src_y = y as f64 - self.dt * self.vy0[i];
                row[x] = sample_bilinear(&self.density0, src_x, src_y, w, h);
            }
        });

        // Buoyancy (uses pre-dissipation density, matching GPU)
        let buoy = self.buoyancy;
        let dt = self.dt;
        let dissipation = self.dissipation;
        self.vy.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            for x in 0..w {
                let i = y * w + x;
                row[x] -= buoy * self.density[i] * dt;
            }
        });

        // Apply density dissipation after buoyancy
        self.density.par_chunks_mut(w).for_each(|row| {
            for d in row.iter_mut() {
                *d *= dissipation;
            }
        });

        // Open boundaries: zero velocity at edges
        for x in 0..w {
            self.vx[x] = 0.0;
            self.vy[x] = 0.0;
            self.vx[(h - 1) * w + x] = 0.0;
            self.vy[(h - 1) * w + x] = 0.0;
        }
        for y in 0..h {
            self.vx[y * w] = 0.0;
            self.vy[y * w] = 0.0;
            self.vx[y * w + w - 1] = 0.0;
            self.vy[y * w + w - 1] = 0.0;
        }

        // 2. Compute divergence
        self.divergence.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            for x in 0..w {
                if x == 0 || x == w - 1 || y == 0 || y == h - 1 {
                    row[x] = 0.0;
                    continue;
                }
                let vx_r = self.vx[y * w + x + 1];
                let vx_l = self.vx[y * w + x - 1];
                let vy_d = self.vy[(y + 1) * w + x];
                let vy_u = self.vy[(y - 1) * w + x];
                row[x] = (vx_r - vx_l + vy_d - vy_u) * 0.5;
            }
        });

        // 3. Jacobi pressure solve (40 iterations, warm-started from previous frame)
        for _ in 0..40 {
            std::mem::swap(&mut self.pressure, &mut self.pressure_tmp);
            self.pressure.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
                for x in 0..w {
                    if x == 0 || x == w - 1 || y == 0 || y == h - 1 {
                        row[x] = 0.0;
                        continue;
                    }
                    let p_l = self.pressure_tmp[y * w + x - 1];
                    let p_r = self.pressure_tmp[y * w + x + 1];
                    let p_u = self.pressure_tmp[(y - 1) * w + x];
                    let p_d = self.pressure_tmp[(y + 1) * w + x];
                    let d = self.divergence[y * w + x];
                    row[x] = (p_l + p_r + p_u + p_d - d) * 0.25;
                }
            });
        }

        // 4. Project: subtract pressure gradient
        // Need to copy pressure to temp since we'll read pressure while writing velocity
        self.vx.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            for x in 0..w {
                if x == 0 || x == w - 1 || y == 0 || y == h - 1 {
                    row[x] = 0.0;
                    continue;
                }
                let p_r = self.pressure[y * w + x + 1];
                let p_l = self.pressure[y * w + x - 1];
                row[x] -= 0.5 * (p_r - p_l);
            }
        });

        self.vy.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            for x in 0..w {
                if x == 0 || x == w - 1 || y == 0 || y == h - 1 {
                    row[x] = 0.0;
                    continue;
                }
                let p_d = self.pressure[(y + 1) * w + x];
                let p_u = self.pressure[(y - 1) * w + x];
                row[x] -= 0.5 * (p_d - p_u);
            }
        });
    }

    pub fn inject(&mut self, px: usize, py: usize, radius: usize) {
        let w = self.width;
        let h = self.height;
        let r = radius as isize;

        for dy in -r..=r {
            for dx in -r..=r {
                let d2 = dx * dx + dy * dy;
                let r2 = r * r;
                if d2 > r2 {
                    continue;
                }
                let x = (px as isize + dx).clamp(0, w as isize - 1) as usize;
                let y = (py as isize + dy).clamp(0, h as isize - 1) as usize;
                let t = 1.0 - (d2 as f64 / r2 as f64);
                let i = y * w + x;
                // Overwrite (matching GPU behavior)
                self.vx[i] = 0.0;
                self.vy[i] = -3.0 * t;
                self.density[i] = 0.5 * t;
            }
        }
    }

    pub fn to_pixels(&self, pixels: &mut [u32]) {
        pixels.par_chunks_mut(self.width).enumerate().for_each(|(y, row)| {
            for x in 0..self.width {
                let d = self.density[y * self.width + x].clamp(0.0, 1.0);
                let v = (d * 255.0) as u32;
                row[x] = 0xFF000000 | (v << 16) | (v << 8) | v;
            }
        });
    }
}

fn sample_bilinear(field: &[f64], fx: f64, fy: f64, w: usize, h: usize) -> f64 {
    let cx = fx.clamp(0.0, (w - 1) as f64 - 0.001);
    let cy = fy.clamp(0.0, (h - 1) as f64 - 0.001);

    let x0 = cx.floor() as usize;
    let y0 = cy.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);

    let sx = cx - cx.floor();
    let sy = cy - cy.floor();

    let v00 = field[y0 * w + x0];
    let v10 = field[y0 * w + x1];
    let v01 = field[y1 * w + x0];
    let v11 = field[y1 * w + x1];

    let top = v00 + sx * (v10 - v00);
    let bot = v01 + sx * (v11 - v01);
    top + sy * (bot - top)
}

