//! Standalone example: render a filled donut via tile-based vector rasterization.
//!
//! The donut is constructed from cubic bezier curves and adaptively flattened.
//! Tests winding number correctness (inner circle creates the hole).
//!
//! Usage: cargo run --example vector_circle

use pixel_doodle::display::Display;
use pixel_doodle::vector;
use pixel_doodle::vector::tile_renderer::VectorTileRenderer;
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;
const TILE_SIZE: u32 = 16;
const TOLERANCE: f32 = 0.5; // pixels

// Orange on black
const FILL_COLOR: u32 = 0xFFFF8800;

fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Wait);
    let mut app = VectorApp {
        window: None,
        display: None,
        renderer: None,
    };
    event_loop.run_app(&mut app).unwrap();
}

struct VectorApp {
    window: Option<Arc<Window>>,
    display: Option<Display>,
    renderer: Option<VectorTileRenderer>,
}

impl ApplicationHandler for VectorApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let win_attrs = Window::default_attributes()
            .with_title("Vector Donut — Tile Rasterization")
            .with_inner_size(winit::dpi::PhysicalSize::new(WIDTH, HEIGHT));
        let window = Arc::new(event_loop.create_window(win_attrs).unwrap());

        let display = Display::new(Arc::clone(&window), WIDTH, HEIGHT);

        // Load the WGSL shader
        let wgsl_source = include_str!("vector/circle/tile_raster.wgsl");

        // Create the tile renderer
        let mut renderer = VectorTileRenderer::new(&display, TILE_SIZE, wgsl_source);

        // Generate test scene: donut centered in the window
        let cx = WIDTH as f32 / 2.0;
        let cy = HEIGHT as f32 / 2.0;
        let outer_radius = HEIGHT as f32 * 0.4;
        let inner_radius = HEIGHT as f32 * 0.2;
        let scene = vector::test_donut(
            cx, cy, outer_radius, inner_radius, FILL_COLOR, TOLERANCE,
            TILE_SIZE, WIDTH, HEIGHT,
        );

        // Upload scene to GPU
        renderer.upload_scene(&scene);

        self.renderer = Some(renderer);
        self.display = Some(display);
        self.window = Some(window);

        // Request initial render
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
                    renderer.render(display);
                }
            }
            _ => {}
        }
    }
}
