use std::path::Path;

/// RGBA8 texture data loaded into memory.
#[derive(Debug, Clone)]
pub struct TextureData {
    /// RGBA8 pixel data, row-major, 4 bytes per pixel.
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

impl TextureData {
    /// Load a texture from an image file (PNG, JPEG, etc.).
    pub fn load(path: &Path) -> Result<Self, String> {
        let img = image::open(path)
            .map_err(|e| format!("failed to load image '{}': {}", path.display(), e))?;
        let rgba = img.to_rgba8();
        let (width, height) = rgba.dimensions();
        Ok(TextureData {
            data: rgba.into_raw(),
            width,
            height,
        })
    }

    /// Sample a pixel at integer coordinates with repeat wrapping.
    /// Returns (r, g, b, a) as f32 in [0.0, 1.0].
    #[inline]
    pub fn load_pixel_repeat(&self, x: i32, y: i32) -> [f32; 4] {
        let w = self.width as i32;
        let h = self.height as i32;
        let x = ((x % w) + w) % w;
        let y = ((y % h) + h) % h;
        self.load_pixel_unchecked(x as u32, y as u32)
    }

    /// Sample a pixel at integer coordinates with clamp-to-edge.
    /// Returns (r, g, b, a) as f32 in [0.0, 1.0].
    #[inline]
    pub fn load_pixel_clamp(&self, x: i32, y: i32) -> [f32; 4] {
        let x = x.clamp(0, self.width as i32 - 1) as u32;
        let y = y.clamp(0, self.height as i32 - 1) as u32;
        self.load_pixel_unchecked(x, y)
    }

    #[inline]
    fn load_pixel_unchecked(&self, x: u32, y: u32) -> [f32; 4] {
        let idx = (y * self.width + x) as usize * 4;
        let r = self.data[idx] as f32 / 255.0;
        let g = self.data[idx + 1] as f32 / 255.0;
        let b = self.data[idx + 2] as f32 / 255.0;
        let a = self.data[idx + 3] as f32 / 255.0;
        [r, g, b, a]
    }

    /// Bilinear sample at normalized UV coordinates with repeat wrapping.
    /// u, v in [0.0, 1.0] map to the full texture.
    #[inline]
    pub fn sample_bilinear_repeat(&self, u: f32, v: f32) -> [f32; 4] {
        let fx = u * self.width as f32 - 0.5;
        let fy = v * self.height as f32 - 0.5;
        let x0 = fx.floor() as i32;
        let y0 = fy.floor() as i32;
        let frac_x = fx - x0 as f32;
        let frac_y = fy - y0 as f32;

        let c00 = self.load_pixel_repeat(x0, y0);
        let c10 = self.load_pixel_repeat(x0 + 1, y0);
        let c01 = self.load_pixel_repeat(x0, y0 + 1);
        let c11 = self.load_pixel_repeat(x0 + 1, y0 + 1);

        lerp4(c00, c10, c01, c11, frac_x, frac_y)
    }

    /// Bilinear sample at normalized UV coordinates with clamp-to-edge.
    #[inline]
    pub fn sample_bilinear_clamp(&self, u: f32, v: f32) -> [f32; 4] {
        let fx = u * self.width as f32 - 0.5;
        let fy = v * self.height as f32 - 0.5;
        let x0 = fx.floor() as i32;
        let y0 = fy.floor() as i32;
        let frac_x = fx - x0 as f32;
        let frac_y = fy - y0 as f32;

        let c00 = self.load_pixel_clamp(x0, y0);
        let c10 = self.load_pixel_clamp(x0 + 1, y0);
        let c01 = self.load_pixel_clamp(x0, y0 + 1);
        let c11 = self.load_pixel_clamp(x0 + 1, y0 + 1);

        lerp4(c00, c10, c01, c11, frac_x, frac_y)
    }

    /// Nearest-neighbor sample at normalized UV coordinates with repeat wrapping.
    #[inline]
    pub fn sample_nearest_repeat(&self, u: f32, v: f32) -> [f32; 4] {
        let x = (u * self.width as f32).floor() as i32;
        let y = (v * self.height as f32).floor() as i32;
        self.load_pixel_repeat(x, y)
    }

    /// Nearest-neighbor sample at normalized UV coordinates with clamp-to-edge.
    #[inline]
    pub fn sample_nearest_clamp(&self, u: f32, v: f32) -> [f32; 4] {
        let x = (u * self.width as f32).floor() as i32;
        let y = (v * self.height as f32).floor() as i32;
        self.load_pixel_clamp(x, y)
    }
}

/// Bilinear interpolation of four RGBA samples.
#[inline]
fn lerp4(c00: [f32; 4], c10: [f32; 4], c01: [f32; 4], c11: [f32; 4], fx: f32, fy: f32) -> [f32; 4] {
    let mut result = [0.0f32; 4];
    for i in 0..4 {
        let top = c00[i] + (c10[i] - c00[i]) * fx;
        let bot = c01[i] + (c11[i] - c01[i]) * fx;
        result[i] = top + (bot - top) * fy;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checkerboard_2x2() -> TextureData {
        // 2x2 checkerboard: white, black, black, white
        TextureData {
            data: vec![
                255, 255, 255, 255,  0, 0, 0, 255,
                0, 0, 0, 255,  255, 255, 255, 255,
            ],
            width: 2,
            height: 2,
        }
    }

    #[test]
    fn load_pixel_repeat_wraps() {
        let tex = checkerboard_2x2();
        let p = tex.load_pixel_repeat(2, 0);
        assert_eq!(p, tex.load_pixel_repeat(0, 0));
        let p = tex.load_pixel_repeat(-1, 0);
        assert_eq!(p, tex.load_pixel_repeat(1, 0));
    }

    #[test]
    fn load_pixel_clamp_clamps() {
        let tex = checkerboard_2x2();
        let p = tex.load_pixel_clamp(10, 0);
        assert_eq!(p, tex.load_pixel_clamp(1, 0));
        let p = tex.load_pixel_clamp(-5, 0);
        assert_eq!(p, tex.load_pixel_clamp(0, 0));
    }

    #[test]
    fn sample_bilinear_center() {
        let tex = checkerboard_2x2();
        // Center of the 2x2 texture should be average of all four pixels
        let p = tex.sample_bilinear_repeat(0.5, 0.5);
        // Two white (1.0) and two black (0.0) → 0.5
        for i in 0..3 {
            assert!((p[i] - 0.5).abs() < 0.01, "channel {i}: {}", p[i]);
        }
    }
}
