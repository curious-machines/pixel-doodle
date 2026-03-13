use minifb::{Key, Window, WindowOptions};
use rayon::prelude::*;
use std::time::Instant;

const WIDTH: usize = 1200;
const HEIGHT: usize = 900;
const MAX_ITER: u32 = 256;

fn mandelbrot(cx: f64, cy: f64) -> u32 {
    let mut zx = 0.0;
    let mut zy = 0.0;
    let mut i = 0;
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
    // Smooth coloring using a simple palette
    let t = iter as f64 / MAX_ITER as f64;
    let r = (9.0 * (1.0 - t) * t * t * t * 255.0) as u32;
    let g = (15.0 * (1.0 - t) * (1.0 - t) * t * t * 255.0) as u32;
    let b = (8.5 * (1.0 - t) * (1.0 - t) * (1.0 - t) * t * 255.0) as u32;
    (r.min(255) << 16) | (g.min(255) << 8) | b.min(255)
}

fn render(buffer: &mut [u32], center_x: f64, center_y: f64, zoom: f64) {
    let aspect = WIDTH as f64 / HEIGHT as f64;
    let view_w = 3.5 / zoom;
    let view_h = view_w / aspect;
    let x_min = center_x - view_w / 2.0;
    let y_min = center_y - view_h / 2.0;

    buffer
        .par_chunks_mut(WIDTH)
        .enumerate()
        .for_each(|(row, pixels)| {
            let cy = y_min + (row as f64 / HEIGHT as f64) * view_h;
            for (col, pixel) in pixels.iter_mut().enumerate() {
                let cx = x_min + (col as f64 / WIDTH as f64) * view_w;
                *pixel = iter_to_color(mandelbrot(cx, cy));
            }
        });
}

fn main() {
    let mut buffer = vec![0u32; WIDTH * HEIGHT];

    let mut window = Window::new(
        "pixel-doodle — Mandelbrot",
        WIDTH,
        HEIGHT,
        WindowOptions::default(),
    )
    .expect("failed to create window");

    // ~60 fps cap
    window.set_target_fps(60);

    let mut center_x = -0.5;
    let mut center_y = 0.0;
    let mut zoom = 1.0;
    let mut needs_render = true;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let pan_speed = 0.05 / zoom;
        let zoom_factor = 1.05;
        let mut moved = false;

        if window.is_key_down(Key::Left) {
            center_x -= pan_speed;
            moved = true;
        }
        if window.is_key_down(Key::Right) {
            center_x += pan_speed;
            moved = true;
        }
        if window.is_key_down(Key::Up) {
            center_y -= pan_speed;
            moved = true;
        }
        if window.is_key_down(Key::Down) {
            center_y += pan_speed;
            moved = true;
        }
        if window.is_key_down(Key::Equal) || window.is_key_down(Key::NumPadPlus) {
            zoom *= zoom_factor;
            moved = true;
        }
        if window.is_key_down(Key::Minus) || window.is_key_down(Key::NumPadMinus) {
            zoom /= zoom_factor;
            moved = true;
        }

        if moved || needs_render {
            let t0 = Instant::now();
            render(&mut buffer, center_x, center_y, zoom);
            let render_ms = t0.elapsed().as_secs_f64() * 1000.0;
            window.set_title(&format!(
                "Mandelbrot | {render_ms:.1}ms | {zoom:.1}x"
            ));
            needs_render = false;
        }

        window
            .update_with_buffer(&buffer, WIDTH, HEIGHT)
            .expect("failed to update window");
    }
}
