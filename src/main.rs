mod display;
mod gpu;
mod jit;
mod kernel_ir;
mod native_kernel;
mod render;

use display::Display;
use gpu::GpuBackend;
use jit::TileKernelFn;
use kernel_ir::KernelIr;
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const WIDTH: u32 = 1200;
const HEIGHT: u32 = 900;
const MAX_ITER: u32 = 256;

fn parse_backend() -> String {
    let args: Vec<String> = std::env::args().collect();
    for i in 0..args.len() - 1 {
        if args[i] == "--backend" {
            return args[i + 1].clone();
        }
    }
    "native".to_string()
}

enum Backend {
    Cpu {
        kernel_fn: TileKernelFn,
        buffer: Vec<u32>,
    },
    Gpu {
        gpu_backend: Option<GpuBackend>,
    },
}

fn compile_cpu_kernel(backend_name: &str) -> (TileKernelFn, &'static str) {
    match backend_name {
        "native" => (native_kernel::native_mandelbrot_kernel, "native"),

        #[cfg(feature = "cranelift-backend")]
        "cranelift" => {
            let backend = jit::cranelift::CraneliftBackend;
            let ir = KernelIr::Mandelbrot { max_iter: MAX_ITER };
            let compiled = jit::JitBackend::compile(&backend, &ir);
            let kernel: &'static dyn jit::CompiledKernel = Box::leak(compiled);
            (kernel.function_ptr(), "cranelift")
        }

        #[cfg(feature = "llvm-backend")]
        "llvm" => {
            let backend = jit::llvm::LlvmBackend;
            let ir = KernelIr::Mandelbrot { max_iter: MAX_ITER };
            let compiled = jit::JitBackend::compile(&backend, &ir);
            let kernel: &'static dyn jit::CompiledKernel = Box::leak(compiled);
            (kernel.function_ptr(), "llvm")
        }

        other => {
            let mut available = vec!["native", "gpu"];
            #[cfg(feature = "cranelift-backend")]
            available.push("cranelift");
            #[cfg(feature = "llvm-backend")]
            available.push("llvm");
            eprintln!(
                "Unknown backend '{}'. Available: {}",
                other,
                available.join(", ")
            );
            std::process::exit(1);
        }
    }
}

struct App {
    backend: Backend,
    label: &'static str,
    compile_ms: f64,
    window: Option<Arc<Window>>,
    display: Option<Display>,
    center_x: f64,
    center_y: f64,
    zoom: f64,
    needs_render: bool,
    keys_down: Vec<KeyCode>,
}

impl App {
    fn new(backend: Backend, label: &'static str, compile_ms: f64) -> Self {
        Self {
            backend,
            label,
            compile_ms,
            window: None,
            display: None,
            center_x: -0.5,
            center_y: 0.0,
            zoom: 1.0,
            needs_render: true,
            keys_down: Vec::new(),
        }
    }

    fn handle_input(&mut self) -> bool {
        let pan_speed = 0.05 / self.zoom;
        let zoom_factor = 1.05;
        let mut moved = false;

        for key in &self.keys_down {
            match key {
                KeyCode::ArrowLeft => {
                    self.center_x -= pan_speed;
                    moved = true;
                }
                KeyCode::ArrowRight => {
                    self.center_x += pan_speed;
                    moved = true;
                }
                KeyCode::ArrowUp => {
                    self.center_y -= pan_speed;
                    moved = true;
                }
                KeyCode::ArrowDown => {
                    self.center_y += pan_speed;
                    moved = true;
                }
                KeyCode::Equal | KeyCode::NumpadAdd => {
                    self.zoom *= zoom_factor;
                    moved = true;
                }
                KeyCode::Minus | KeyCode::NumpadSubtract => {
                    self.zoom /= zoom_factor;
                    moved = true;
                }
                _ => {}
            }
        }
        moved
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title(format!("pixel-doodle [{}]", self.label))
            .with_inner_size(winit::dpi::PhysicalSize::new(WIDTH, HEIGHT))
            .with_resizable(false);
        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        let display = Display::new(window.clone(), WIDTH, HEIGHT);

        // Initialize GPU backend now that we have a display (and thus a device)
        if let Backend::Gpu { gpu_backend } = &mut self.backend {
            *gpu_backend = Some(GpuBackend::new(&display, WIDTH, HEIGHT, MAX_ITER));
        }

        self.display = Some(display);
        self.window = Some(window);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    if code == KeyCode::Escape && event.state == ElementState::Pressed {
                        event_loop.exit();
                        return;
                    }
                    match event.state {
                        ElementState::Pressed => {
                            if !self.keys_down.contains(&code) {
                                self.keys_down.push(code);
                            }
                        }
                        ElementState::Released => {
                            self.keys_down.retain(|&k| k != code);
                        }
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                let moved = self.handle_input();
                let display = self.display.as_ref().unwrap();

                if moved || self.needs_render {
                    let t0 = Instant::now();

                    match &mut self.backend {
                        Backend::Cpu { kernel_fn, buffer } => {
                            render::render(
                                buffer,
                                WIDTH as usize,
                                HEIGHT as usize,
                                self.center_x,
                                self.center_y,
                                self.zoom,
                                *kernel_fn,
                            );
                            let render_ms = t0.elapsed().as_secs_f64() * 1000.0;
                            if let Some(window) = &self.window {
                                window.set_title(&format!(
                                    "{} | {render_ms:.1}ms | {:.1}x | compile {:.1}ms",
                                    self.label, self.zoom, self.compile_ms
                                ));
                            }
                            display.upload_and_present(buffer);
                        }
                        Backend::Gpu { gpu_backend } => {
                            let gpu = gpu_backend.as_ref().unwrap();
                            gpu.render(display, self.center_x, self.center_y, self.zoom);
                            let render_ms = t0.elapsed().as_secs_f64() * 1000.0;
                            if let Some(window) = &self.window {
                                window.set_title(&format!(
                                    "{} | {render_ms:.1}ms | {:.1}x",
                                    self.label, self.zoom
                                ));
                            }
                        }
                    }

                    self.needs_render = false;
                } else {
                    // No changes — still need to present the existing frame
                    match &self.backend {
                        Backend::Cpu { buffer, .. } => {
                            display.upload_and_present(buffer);
                        }
                        Backend::Gpu { gpu_backend } => {
                            let gpu = gpu_backend.as_ref().unwrap();
                            gpu.render(display, self.center_x, self.center_y, self.zoom);
                        }
                    }
                }

                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

fn main() {
    let backend_name = parse_backend();

    let compile_start = Instant::now();

    let (backend, label) = if backend_name == "gpu" {
        (Backend::Gpu { gpu_backend: None }, "gpu")
    } else {
        let (kernel_fn, label) = compile_cpu_kernel(&backend_name);
        (
            Backend::Cpu {
                kernel_fn,
                buffer: vec![0u32; (WIDTH * HEIGHT) as usize],
            },
            label,
        )
    };

    let compile_ms = compile_start.elapsed().as_secs_f64() * 1000.0;
    eprintln!("[{label}] compile: {compile_ms:.1}ms");

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new(backend, label, compile_ms);
    event_loop.run_app(&mut app).unwrap();
}
