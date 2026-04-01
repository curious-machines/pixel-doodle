//! PDC language example: compile and render a .pdc scene file.
//!
//! Usage: cargo run --example pdc_basic --release

use pixel_doodle::display::Display;
use pixel_doodle::pdc;
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
const TOLERANCE: f32 = 0.5;

fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Wait);
    let mut app = PdcApp {
        window: None,
        display: None,
        renderer: None,
    };
    event_loop.run_app(&mut app).unwrap();
}

struct PdcApp {
    window: Option<Arc<Window>>,
    display: Option<Display>,
    renderer: Option<VectorTileRenderer>,
}

impl ApplicationHandler for PdcApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let win_attrs = Window::default_attributes()
            .with_title("PDC Basic Example")
            .with_inner_size(winit::dpi::PhysicalSize::new(WIDTH, HEIGHT));
        let window = Arc::new(event_loop.create_window(win_attrs).unwrap());
        let display = Display::new(Arc::clone(&window), WIDTH, HEIGHT);

        // Load and compile PDC source
        let source = include_str!("pdc/basic.pdc");
        let scene = match pdc::compile_and_run(source, WIDTH, HEIGHT, TOLERANCE, TILE_SIZE) {
            Ok(scene) => scene,
            Err(e) => {
                eprintln!("PDC error: {}", e.format(source));
                std::process::exit(1);
            }
        };

        eprintln!(
            "[pdc] {} paths, {} segments",
            scene.path_colors.len(),
            scene.segments.len(),
        );

        // Upload and render
        let wgsl_source = include_str!("vector/circle/tile_raster.wgsl");
        let mut renderer = VectorTileRenderer::new(&display, TILE_SIZE, wgsl_source);
        renderer.upload_scene(&scene);

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
                    let _ = renderer.render(display);
                }
            }
            _ => {}
        }
    }
}
