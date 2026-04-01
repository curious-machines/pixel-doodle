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

    // -- Helpers --

    fn texture_1x1() -> TextureData {
        TextureData {
            data: vec![128, 64, 32, 255],
            width: 1,
            height: 1,
        }
    }

    /// 2x2 texture where each pixel has distinct RGBA channels.
    fn texture_distinct_channels() -> TextureData {
        TextureData {
            data: vec![
                10, 20, 30, 40,    50, 60, 70, 80,
                90, 100, 110, 120, 130, 140, 150, 160,
            ],
            width: 2,
            height: 2,
        }
    }

    fn assert_approx(a: [f32; 4], b: [f32; 4], tol: f32) {
        for i in 0..4 {
            assert!(
                (a[i] - b[i]).abs() < tol,
                "channel {i}: {} vs {} (diff {})",
                a[i],
                b[i],
                (a[i] - b[i]).abs()
            );
        }
    }

    // -- 1. Pixel access --

    #[test]
    fn load_pixel_repeat_negative_coords() {
        let tex = checkerboard_2x2();
        // (-1, -1) wraps to (1, 1)
        assert_eq!(tex.load_pixel_repeat(-1, -1), tex.load_pixel_repeat(1, 1));
        // (-2, -2) wraps to (0, 0)
        assert_eq!(tex.load_pixel_repeat(-2, -2), tex.load_pixel_repeat(0, 0));
        // (-3, 0) wraps to (1, 0)
        assert_eq!(tex.load_pixel_repeat(-3, 0), tex.load_pixel_repeat(1, 0));
    }

    #[test]
    fn load_pixel_repeat_large_coords() {
        let tex = checkerboard_2x2();
        assert_eq!(tex.load_pixel_repeat(100, 100), tex.load_pixel_repeat(0, 0));
        assert_eq!(tex.load_pixel_repeat(101, 101), tex.load_pixel_repeat(1, 1));
        assert_eq!(tex.load_pixel_repeat(1000, 0), tex.load_pixel_repeat(0, 0));
    }

    #[test]
    fn load_pixel_clamp_negative_coords() {
        let tex = checkerboard_2x2();
        // Negative coords clamp to 0
        assert_eq!(tex.load_pixel_clamp(-1, 0), tex.load_pixel_clamp(0, 0));
        assert_eq!(tex.load_pixel_clamp(0, -1), tex.load_pixel_clamp(0, 0));
        assert_eq!(tex.load_pixel_clamp(-100, -100), tex.load_pixel_clamp(0, 0));
    }

    #[test]
    fn load_pixel_clamp_large_coords() {
        let tex = checkerboard_2x2();
        // Large coords clamp to max (1, 1 for 2x2)
        assert_eq!(tex.load_pixel_clamp(5, 0), tex.load_pixel_clamp(1, 0));
        assert_eq!(tex.load_pixel_clamp(0, 5), tex.load_pixel_clamp(0, 1));
        assert_eq!(tex.load_pixel_clamp(1000, 1000), tex.load_pixel_clamp(1, 1));
    }

    // -- 2. Nearest sampling --

    #[test]
    fn sample_nearest_repeat_center() {
        let tex = checkerboard_2x2();
        // UV (0.25, 0.25) maps to pixel (0, 0) — white
        let p = tex.sample_nearest_repeat(0.25, 0.25);
        assert_eq!(p, tex.load_pixel_repeat(0, 0));
        // UV (0.75, 0.25) maps to pixel (1, 0) — black
        let p = tex.sample_nearest_repeat(0.75, 0.25);
        assert_eq!(p, tex.load_pixel_repeat(1, 0));
    }

    #[test]
    fn sample_nearest_repeat_beyond_boundaries() {
        let tex = checkerboard_2x2();
        // UV (1.25, 0.25) should wrap to same as (0.25, 0.25)
        assert_eq!(
            tex.sample_nearest_repeat(1.25, 0.25),
            tex.sample_nearest_repeat(0.25, 0.25)
        );
        // UV (-0.25, 0.25) wraps — floor(-0.5) = -1, repeats to 1
        assert_eq!(
            tex.sample_nearest_repeat(-0.25, 0.25),
            tex.load_pixel_repeat(-1, 0)
        );
    }

    #[test]
    fn sample_nearest_clamp_at_boundaries() {
        let tex = checkerboard_2x2();
        // UV (0.0, 0.0) → pixel (0, 0)
        assert_eq!(tex.sample_nearest_clamp(0.0, 0.0), tex.load_pixel_clamp(0, 0));
        // UV just below 1.0 → pixel (1, 1)
        assert_eq!(tex.sample_nearest_clamp(0.99, 0.99), tex.load_pixel_clamp(1, 1));
    }

    #[test]
    fn sample_nearest_clamp_beyond_boundaries() {
        let tex = checkerboard_2x2();
        // UV (2.0, 2.0) clamps to edge
        assert_eq!(tex.sample_nearest_clamp(2.0, 2.0), tex.load_pixel_clamp(1, 1));
        // UV (-1.0, -1.0) clamps to (0, 0)
        assert_eq!(tex.sample_nearest_clamp(-1.0, -1.0), tex.load_pixel_clamp(0, 0));
    }

    // -- 3. Bilinear edge cases --

    #[test]
    fn bilinear_at_exact_pixel_centers() {
        let tex = checkerboard_2x2();
        // Pixel center for (0,0) in a 2x2 is UV (0.25, 0.25)
        let p = tex.sample_bilinear_repeat(0.25, 0.25);
        assert_approx(p, tex.load_pixel_repeat(0, 0), 1e-5);
        // Pixel center for (1,0) is UV (0.75, 0.25)
        let p = tex.sample_bilinear_repeat(0.75, 0.25);
        assert_approx(p, tex.load_pixel_repeat(1, 0), 1e-5);
        // Pixel center for (1,1) is UV (0.75, 0.75)
        let p = tex.sample_bilinear_repeat(0.75, 0.75);
        assert_approx(p, tex.load_pixel_repeat(1, 1), 1e-5);
    }

    #[test]
    fn bilinear_repeat_at_uv_one() {
        let tex = checkerboard_2x2();
        // UV (1.0, 1.0) with repeat should produce the same result as UV (0.0, 0.0)
        assert_approx(
            tex.sample_bilinear_repeat(1.0, 1.0),
            tex.sample_bilinear_repeat(0.0, 0.0),
            1e-5,
        );
    }

    #[test]
    fn bilinear_clamp_at_uv_one() {
        let tex = checkerboard_2x2();
        // UV (1.0, 1.0) with clamp should return the bottom-right pixel
        // At u=1.0 v=1.0: fx = 2*1.0 - 0.5 = 1.5, floor = 1, frac = 0.5
        // All four samples clamp to (1,1), so result is pixel (1,1)
        let p = tex.sample_bilinear_clamp(1.0, 1.0);
        assert_approx(p, tex.load_pixel_clamp(1, 1), 1e-5);
    }

    #[test]
    fn bilinear_repeat_outside_zero_one() {
        let tex = checkerboard_2x2();
        // Values outside [0,1] should wrap via repeat
        assert_approx(
            tex.sample_bilinear_repeat(1.25, 1.25),
            tex.sample_bilinear_repeat(0.25, 0.25),
            1e-5,
        );
        assert_approx(
            tex.sample_bilinear_repeat(-0.75, -0.75),
            tex.sample_bilinear_repeat(0.25, 0.25),
            // Larger tolerance: wrapping with negative floats can have small rounding diffs
            1e-4,
        );
    }

    #[test]
    fn bilinear_clamp_outside_zero_one() {
        let tex = checkerboard_2x2();
        // UV far positive clamps to bottom-right pixel
        let p = tex.sample_bilinear_clamp(5.0, 5.0);
        assert_approx(p, tex.load_pixel_clamp(1, 1), 1e-5);
        // UV far negative clamps to top-left pixel
        let p = tex.sample_bilinear_clamp(-5.0, -5.0);
        assert_approx(p, tex.load_pixel_clamp(0, 0), 1e-5);
    }

    // -- 4. 1x1 texture --

    #[test]
    fn one_by_one_load_pixel_repeat() {
        let tex = texture_1x1();
        let expected = tex.load_pixel_repeat(0, 0);
        assert_eq!(tex.load_pixel_repeat(5, 5), expected);
        assert_eq!(tex.load_pixel_repeat(-3, -3), expected);
    }

    #[test]
    fn one_by_one_load_pixel_clamp() {
        let tex = texture_1x1();
        let expected = tex.load_pixel_clamp(0, 0);
        assert_eq!(tex.load_pixel_clamp(100, 100), expected);
        assert_eq!(tex.load_pixel_clamp(-100, -100), expected);
    }

    #[test]
    fn one_by_one_sample_nearest() {
        let tex = texture_1x1();
        let expected = tex.load_pixel_repeat(0, 0);
        assert_eq!(tex.sample_nearest_repeat(0.0, 0.0), expected);
        assert_eq!(tex.sample_nearest_repeat(0.5, 0.5), expected);
        assert_eq!(tex.sample_nearest_repeat(0.99, 0.99), expected);
        assert_eq!(tex.sample_nearest_clamp(0.0, 0.0), expected);
        assert_eq!(tex.sample_nearest_clamp(0.5, 0.5), expected);
        assert_eq!(tex.sample_nearest_clamp(2.0, 2.0), expected);
        assert_eq!(tex.sample_nearest_clamp(-1.0, -1.0), expected);
    }

    #[test]
    fn one_by_one_sample_bilinear() {
        let tex = texture_1x1();
        let expected = tex.load_pixel_repeat(0, 0);
        // All bilinear samples on a 1x1 texture should return the single pixel
        assert_approx(tex.sample_bilinear_repeat(0.0, 0.0), expected, 1e-5);
        assert_approx(tex.sample_bilinear_repeat(0.5, 0.5), expected, 1e-5);
        assert_approx(tex.sample_bilinear_repeat(1.0, 1.0), expected, 1e-5);
        assert_approx(tex.sample_bilinear_clamp(0.0, 0.0), expected, 1e-5);
        assert_approx(tex.sample_bilinear_clamp(0.5, 0.5), expected, 1e-5);
        assert_approx(tex.sample_bilinear_clamp(1.0, 1.0), expected, 1e-5);
    }

    // -- 5. Channel independence --

    #[test]
    fn channels_sampled_independently() {
        let tex = texture_distinct_channels();
        // Verify pixel (0,0) has distinct R, G, B, A
        let p = tex.load_pixel_repeat(0, 0);
        assert!((p[0] - 10.0 / 255.0).abs() < 1e-5);
        assert!((p[1] - 20.0 / 255.0).abs() < 1e-5);
        assert!((p[2] - 30.0 / 255.0).abs() < 1e-5);
        assert!((p[3] - 40.0 / 255.0).abs() < 1e-5);

        // Verify pixel (1,1)
        let p = tex.load_pixel_repeat(1, 1);
        assert!((p[0] - 130.0 / 255.0).abs() < 1e-5);
        assert!((p[1] - 140.0 / 255.0).abs() < 1e-5);
        assert!((p[2] - 150.0 / 255.0).abs() < 1e-5);
        assert!((p[3] - 160.0 / 255.0).abs() < 1e-5);
    }

    #[test]
    fn bilinear_channels_independent() {
        let tex = texture_distinct_channels();
        // Sample at center (0.5, 0.5) — average of all four pixels
        let p = tex.sample_bilinear_repeat(0.5, 0.5);
        // Expected per channel: average of the four pixel values
        let expected_r = (10.0 + 50.0 + 90.0 + 130.0) / 4.0 / 255.0;
        let expected_g = (20.0 + 60.0 + 100.0 + 140.0) / 4.0 / 255.0;
        let expected_b = (30.0 + 70.0 + 110.0 + 150.0) / 4.0 / 255.0;
        let expected_a = (40.0 + 80.0 + 120.0 + 160.0) / 4.0 / 255.0;
        assert_approx(p, [expected_r, expected_g, expected_b, expected_a], 1e-5);
    }
}
