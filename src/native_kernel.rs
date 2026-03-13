const MAX_ITER: u32 = 256;

fn mandelbrot(cx: f64, cy: f64) -> u32 {
    let mut zx = 0.0;
    let mut zy = 0.0;
    let mut i = 0u32;
    while zx * zx + zy * zy <= 4.0 && i < MAX_ITER {
        let tmp = zx * zx - zy * zy + cx;
        zy = 2.0 * zx * zy + cy;
        zx = tmp;
        i += 1;
    }
    i
}

fn iter_to_color(iter: u32) -> u32 {
    if iter == MAX_ITER {
        return 0x00_00_00;
    }
    let t = iter as f64 / MAX_ITER as f64;
    let r = (9.0 * (1.0 - t) * t * t * t * 255.0) as u32;
    let g = (15.0 * (1.0 - t) * (1.0 - t) * t * t * 255.0) as u32;
    let b = (8.5 * (1.0 - t) * (1.0 - t) * (1.0 - t) * t * 255.0) as u32;
    (r.min(255) << 16) | (g.min(255) << 8) | b.min(255)
}

/// Native Rust implementation of the Mandelbrot tile kernel.
/// Matches the `TileKernelFn` signature so it can be used as a reference backend.
///
/// Note: `output` points to the start of this tile's chunk (row_start's row),
/// not the full buffer. Pixel writes are relative to row_start.
pub unsafe extern "C" fn native_mandelbrot_kernel(
    output: *mut u32,
    width: u32,
    _height: u32,
    x_min: f64,
    y_min: f64,
    x_step: f64,
    y_step: f64,
    row_start: u32,
    row_end: u32,
) {
    let width = width as usize;
    for row in row_start..row_end {
        let cy = y_min + row as f64 * y_step;
        let row_offset = (row - row_start) as usize * width;
        for col in 0..width {
            let cx = x_min + col as f64 * x_step;
            let iter = mandelbrot(cx, cy);
            let color = iter_to_color(iter);
            unsafe { *output.add(row_offset + col) = color };
        }
    }
}
