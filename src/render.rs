use rayon::prelude::*;

use crate::jit::{SimTileKernelFn, TextureSlot, TileKernelFn};

pub const DEFAULT_TILE_HEIGHT: usize = 1;

pub fn render(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    center_x: f64,
    center_y: f64,
    zoom: f64,
    kernel: TileKernelFn,
    sample_index: u32,
    time: f64,
    tile_height: usize,
    user_args: *const u8,
    tex_slots: &[TextureSlot],
) {
    let aspect = width as f64 / height as f64;
    let view_w = 3.5 / zoom;
    let view_h = view_w / aspect;
    let x_min = center_x - view_w / 2.0;
    let y_min = center_y - view_h / 2.0;
    let x_step = view_w / width as f64;
    let y_step = view_h / height as f64;

    let rows_per_tile = tile_height;
    let num_tiles = (height + rows_per_tile - 1) / rows_per_tile;

    // Safety: user_args points to a Vec<u8> on the caller's stack that
    // outlives this parallel region. Wrap as usize for Send.
    let args_addr = user_args as usize;
    let tex_addr = tex_slots.as_ptr() as usize;

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
                    time,
                    args_addr as *const u8,
                    tex_addr as *const TextureSlot,
                );
            }
        });
}

/// Render a simulation step using a JIT'd sim kernel.
///
/// `bufs_in`: slice of read-only f64 buffer pointers.
/// `bufs_out`: slice of writable f64 buffer pointers.
/// `pixel_output`: ARGB pixel buffer (width * height).
///
/// Safety: caller must ensure all buffer pointers are valid and that
/// output buffers have disjoint write regions per tile (guaranteed by
/// the simulation's double-buffering: we read from one set and write
/// to another). The pixel output is partitioned by par_chunks_mut.
pub fn render_sim(
    pixel_output: &mut [u32],
    width: usize,
    height: usize,
    kernel: SimTileKernelFn,
    bufs_in: &[*const f64],
    bufs_out: &[*mut f64],
    tile_height: usize,
    user_args: *const u8,
    tex_slots: &[TextureSlot],
) {
    let rows_per_tile = tile_height;

    // Stash pointer-to-pointer as usize to satisfy Send+Sync for rayon
    let in_base = bufs_in.as_ptr() as usize;
    let out_base = bufs_out.as_ptr() as usize;
    let args_addr = user_args as usize;
    let tex_addr = tex_slots.as_ptr() as usize;

    pixel_output
        .par_chunks_mut(rows_per_tile * width)
        .enumerate()
        .for_each(|(tile_idx, chunk)| {
            let row_start = tile_idx * rows_per_tile;
            let row_end = (row_start + rows_per_tile).min(height);

            unsafe {
                kernel(
                    chunk.as_mut_ptr(),
                    width as u32,
                    height as u32,
                    row_start as u32,
                    row_end as u32,
                    in_base as *const *const f64,
                    out_base as *const *mut f64,
                    args_addr as *const u8,
                    tex_addr as *const TextureSlot,
                );
            }
        });
}
