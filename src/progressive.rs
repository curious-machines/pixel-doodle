/// Host-side accumulation buffer for progressive rendering.
/// Maintains per-pixel f32 channel sums that are resolved to u32 ARGB for display.
pub struct AccumulationBuffer {
    /// Per-pixel [R, G, B] sums as f32, length = width * height * 3
    channels: Vec<f32>,
    pub sample_count: u32,
    pub max_samples: u32,
    width: usize,
    height: usize,
}

impl AccumulationBuffer {
    pub fn new(width: usize, height: usize, max_samples: u32) -> Self {
        Self {
            channels: vec![0.0; width * height * 3],
            sample_count: 0,
            max_samples,
            width,
            height,
        }
    }

    /// Unpack ARGB pixels from a sample pass and add to running sums.
    pub fn accumulate(&mut self, pixel_buffer: &[u32]) {
        for (i, &pixel) in pixel_buffer.iter().enumerate() {
            let base = i * 3;
            self.channels[base] += ((pixel >> 16) & 0xFF) as f32;
            self.channels[base + 1] += ((pixel >> 8) & 0xFF) as f32;
            self.channels[base + 2] += (pixel & 0xFF) as f32;
        }
        self.sample_count += 1;
    }

    /// Divide accumulated sums by sample count and pack to u32 ARGB.
    pub fn resolve(&self, output: &mut [u32]) {
        let inv = 1.0 / self.sample_count as f32;
        for i in 0..self.width * self.height {
            let base = i * 3;
            let r = (self.channels[base] * inv) as u32;
            let g = (self.channels[base + 1] * inv) as u32;
            let b = (self.channels[base + 2] * inv) as u32;
            output[i] = 0xFF000000 | (r.min(255) << 16) | (g.min(255) << 8) | b.min(255);
        }
    }

    /// Zero the buffer and reset sample count (called on pan/zoom).
    pub fn reset(&mut self) {
        self.channels.fill(0.0);
        self.sample_count = 0;
    }

    pub fn is_converged(&self) -> bool {
        self.sample_count >= self.max_samples
    }
}
