mod display;
mod gpu;
mod jit;
#[allow(dead_code)]
mod kernel_ir;
mod kernels;
#[allow(dead_code)]
mod lang;
mod native_kernel;
mod render;

use display::Display;
use gpu::GpuBackend;
use jit::TileKernelFn;
use kernel_ir::Kernel;
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const WIDTH: u32 = 1200;
const HEIGHT: u32 = 900;

struct CliArgs {
    backend: String,
    kernel_path: Option<String>,
}

fn parse_args() -> CliArgs {
    let args: Vec<String> = std::env::args().collect();
    let mut backend = "native".to_string();
    let mut kernel_path = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--backend" => {
                i += 1;
                if i < args.len() {
                    backend = args[i].clone();
                }
            }
            "--kernel" => {
                i += 1;
                if i < args.len() {
                    kernel_path = Some(args[i].clone());
                }
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                std::process::exit(1);
            }
        }
        i += 1;
    }
    CliArgs { backend, kernel_path }
}

/// What kind of kernel file was provided (or none).
enum KernelSource {
    /// A .pdl file parsed into IR (used by CPU backends).
    Pdl(Kernel),
    /// A .wgsl file loaded as source text (used by GPU backend).
    Wgsl(String),
}

fn load_kernel(kernel_path: &Option<String>, backend: &str) -> KernelSource {
    match kernel_path {
        Some(path) => {
            let src = std::fs::read_to_string(path).unwrap_or_else(|e| {
                eprintln!("Failed to read kernel file '{}': {}", path, e);
                std::process::exit(1);
            });
            if path.ends_with(".wgsl") {
                if backend != "gpu" {
                    eprintln!(".wgsl kernels require --backend gpu");
                    std::process::exit(1);
                }
                KernelSource::Wgsl(src)
            } else {
                if backend == "gpu" {
                    eprintln!("GPU backend requires a .wgsl kernel, got '{}'", path);
                    std::process::exit(1);
                }
                let kernel = lang::parser::parse(&src).unwrap_or_else(|e| {
                    eprintln!("Parse error in '{}': {}", path, e);
                    std::process::exit(1);
                });
                KernelSource::Pdl(kernel)
            }
        }
        None => {
            if backend == "gpu" {
                // No kernel specified — GPU uses built-in mandelbrot.wgsl
                KernelSource::Wgsl(String::new()) // empty signals "use default"
            } else {
                KernelSource::Pdl(kernels::gradient_kernel())
            }
        }
    }
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

fn compile_cpu_kernel(backend_name: &str, kernel: &Kernel) -> (TileKernelFn, &'static str) {
    match backend_name {
        "native" => (native_kernel::native_mandelbrot_kernel, "native"),

        #[cfg(feature = "cranelift-backend")]
        "cranelift" => {
            let backend = jit::cranelift::CraneliftBackend;
            let compiled = jit::JitBackend::compile(&backend, kernel);
            let compiled: &'static dyn jit::CompiledKernel = Box::leak(compiled);
            (compiled.function_ptr(), "cranelift")
        }

        #[cfg(feature = "llvm-backend")]
        "llvm" => {
            let backend = jit::llvm::LlvmBackend;
            let compiled = jit::JitBackend::compile(&backend, kernel);
            let compiled: &'static dyn jit::CompiledKernel = Box::leak(compiled);
            (compiled.function_ptr(), "llvm")
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
    wgsl_source: Option<String>,
    window: Option<Arc<Window>>,
    display: Option<Display>,
    center_x: f64,
    center_y: f64,
    zoom: f64,
    needs_render: bool,
    keys_down: Vec<KeyCode>,
}

impl App {
    fn new(backend: Backend, label: &'static str, compile_ms: f64, wgsl_source: Option<String>) -> Self {
        Self {
            backend,
            label,
            compile_ms,
            wgsl_source,
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
                KeyCode::Digit0 | KeyCode::Numpad0 => {
                    self.center_x = -0.5;
                    self.center_y = 0.0;
                    self.zoom = 1.0;
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

        if let Backend::Gpu { gpu_backend } = &mut self.backend {
            let wgsl = self.wgsl_source.as_deref();
            // Empty string means "use default", None also means default
            let wgsl = match wgsl {
                Some("") | None => None,
                Some(s) => Some(s),
            };
            *gpu_backend = Some(GpuBackend::new(&display, WIDTH, HEIGHT, 256, wgsl));
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
                    if (code == KeyCode::Escape || code == KeyCode::KeyQ) && event.state == ElementState::Pressed {
                        event_loop.exit();
                        return;
                    }
                    match event.state {
                        ElementState::Pressed => {
                            if !self.keys_down.contains(&code) {
                                self.keys_down.push(code);
                            }
                            // Switch to Poll for continuous rendering while keys are held
                            event_loop.set_control_flow(ControlFlow::Poll);
                            if let Some(window) = &self.window {
                                window.request_redraw();
                            }
                        }
                        ElementState::Released => {
                            self.keys_down.retain(|&k| k != code);
                            if self.keys_down.is_empty() {
                                event_loop.set_control_flow(ControlFlow::Wait);
                            }
                        }
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                let moved = self.handle_input();
                let display = self.display.as_ref().unwrap();

                match &mut self.backend {
                    Backend::Cpu { kernel_fn, buffer } => {
                        if moved || self.needs_render {
                            let t0 = Instant::now();
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
                        }
                        display.upload_and_present(buffer);
                    }
                    Backend::Gpu { gpu_backend } => {
                        let gpu = gpu_backend.as_ref().unwrap();
                        let t0 = Instant::now();
                        gpu.render(display, self.center_x, self.center_y, self.zoom);
                        if moved || self.needs_render {
                            let render_ms = t0.elapsed().as_secs_f64() * 1000.0;
                            if let Some(window) = &self.window {
                                window.set_title(&format!(
                                    "{} | {render_ms:.1}ms | {:.1}x",
                                    self.label, self.zoom
                                ));
                            }
                        }
                    }
                }

                self.needs_render = false;

                // Keep requesting redraws while keys are held for continuous pan/zoom
                if !self.keys_down.is_empty() {
                    if let Some(window) = &self.window {
                        window.request_redraw();
                    }
                }
            }
            _ => {}
        }
    }
}

fn main() {
    let args = parse_args();

    let kernel_source = load_kernel(&args.kernel_path, &args.backend);

    let compile_start = Instant::now();

    let (backend, label, wgsl_source) = match kernel_source {
        KernelSource::Wgsl(src) => {
            if src.is_empty() {
                eprintln!("[kernel] using built-in mandelbrot.wgsl");
            } else {
                eprintln!("[kernel] loaded WGSL from '{}'", args.kernel_path.as_deref().unwrap());
            }
            (Backend::Gpu { gpu_backend: None }, "gpu", Some(src))
        }
        KernelSource::Pdl(kernel) => {
            eprintln!("[kernel] loaded '{}' ({} statements)", kernel.name, kernel.body.len());
            let (kernel_fn, label) = compile_cpu_kernel(&args.backend, &kernel);
            (
                Backend::Cpu {
                    kernel_fn,
                    buffer: vec![0u32; (WIDTH * HEIGHT) as usize],
                },
                label,
                None,
            )
        }
    };

    let compile_ms = compile_start.elapsed().as_secs_f64() * 1000.0;
    eprintln!("[{label}] compile: {compile_ms:.1}ms");

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Wait);
    let mut app = App::new(backend, label, compile_ms, wgsl_source);
    event_loop.run_app(&mut app).unwrap();
}
