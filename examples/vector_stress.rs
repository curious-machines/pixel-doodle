//! Stress test: render 1000 filled+stroked circles via tile-based vector rasterization.
//!
//! Measures CPU flattening/binning time and reports it in the window title.
//! Flattening is parallelized across paths via rayon.
//!
//! Usage: cargo run --example vector_stress --release

use pixel_doodle::display::Display;
use pixel_doodle::vector;
use pixel_doodle::vector::tile_renderer::VectorTileRenderer;
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const WIDTH: u32 = 1200;
const HEIGHT: u32 = 900;
const TILE_SIZE: u32 = 16;
const TOLERANCE: f32 = 0.5;
const NUM_CIRCLES: u32 = 1000;

fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Wait);
    let mut app = StressApp {
        window: None,
        display: None,
        renderer: None,
    };
    event_loop.run_app(&mut app).unwrap();
}

struct StressApp {
    window: Option<Arc<Window>>,
    display: Option<Display>,
    renderer: Option<VectorTileRenderer>,
}

impl ApplicationHandler for StressApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let win_attrs = Window::default_attributes()
            .with_title("Vector Stress Test")
            .with_inner_size(winit::dpi::PhysicalSize::new(WIDTH, HEIGHT));
        let window = Arc::new(event_loop.create_window(win_attrs).unwrap());
        let display = Display::new(Arc::clone(&window), WIDTH, HEIGHT);
        let wgsl_source = include_str!("vector/circle/tile_raster.wgsl");
        let mut renderer = VectorTileRenderer::new(&display, TILE_SIZE, wgsl_source);

        // Generate and upload scene with timing
        let t0 = Instant::now();
        let scene = vector::test_stress(NUM_CIRCLES, TOLERANCE, TILE_SIZE, WIDTH, HEIGHT);
        let flatten_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let num_segments = scene.segments.len();
        let num_tile_indices = scene.tile_indices.len();

        let t1 = Instant::now();
        renderer.upload_scene(&scene);
        let upload_ms = t1.elapsed().as_secs_f64() * 1000.0;

        eprintln!(
            "[stress] {} circles → {} paths, {} segments, {} tile index entries",
            NUM_CIRCLES,
            scene.path_colors.len(),
            num_segments,
            num_tile_indices,
        );
        eprintln!(
            "[stress] flatten+stroke+bin: {flatten_ms:.1}ms, upload: {upload_ms:.1}ms"
        );

        window.set_title(&format!(
            "Vector Stress — {NUM_CIRCLES} circles, {num_segments} segs, flatten {flatten_ms:.1}ms"
        ));

        self.renderer = Some(renderer);
        self.display = Some(display);
        self.window = Some(window);

        self.window.as_ref().unwrap().request_redraw();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput {
                event: key_event, ..
            } => {
                if let PhysicalKey::Code(KeyCode::Escape) = key_event.physical_key {
                    event_loop.exit();
                }
            }
            WindowEvent::RedrawRequested => {
                if let (Some(renderer), Some(display)) =
                    (self.renderer.as_ref(), self.display.as_ref())
                {
                    let gpu_ms = renderer.render(display);
                    if let Some(window) = &self.window {
                        let title = format!("Vector Stress — {NUM_CIRCLES} circles, GPU {gpu_ms:.1}ms");
                        window.set_title(&title);
                    }
                }
            }
            _ => {}
        }
    }
}
