use rayon::prelude::*;

use crate::jit::TileKernelFn;

const TILE_HEIGHT: usize = 16;

pub fn render(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    center_x: f64,
    center_y: f64,
    zoom: f64,
    kernel: TileKernelFn,
    sample_index: u32,
) {
    let aspect = width as f64 / height as f64;
    let view_w = 3.5 / zoom;
    let view_h = view_w / aspect;
    let x_min = center_x - view_w / 2.0;
    let y_min = center_y - view_h / 2.0;
    let x_step = view_w / width as f64;
    let y_step = view_h / height as f64;

    // Split buffer into tiles of TILE_HEIGHT rows each
    let rows_per_tile = TILE_HEIGHT;
    let num_tiles = (height + rows_per_tile - 1) / rows_per_tile;

    // Use par_chunks_mut on the buffer, chunked by rows_per_tile * width
    buffer
        .par_chunks_mut(rows_per_tile * width)
        .enumerate()
        .for_each(|(tile_idx, chunk)| {
            let row_start = tile_idx * rows_per_tile;
            let row_end = (row_start + rows_per_tile).min(height);
            let _ = num_tiles; // suppress unused warning

            unsafe {
                kernel(
                    chunk.as_mut_ptr(),
                    width as u32,
                    height as u32,
                    x_min,
                    y_min,
                    x_step,
                    y_step,
                    row_start as u32,
                    row_end as u32,
                    sample_index,
                );
            }
        });
}
