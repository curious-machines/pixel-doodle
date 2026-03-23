mod bench;
mod display;
mod gpu;
mod jit;
#[allow(dead_code)]
mod kernel_ir;
mod kernels;
#[allow(dead_code)]
mod lang;
mod native_kernel;
mod progressive;
mod render;
mod simulation;

use display::Display;
use gpu::GpuBackend;
use jit::TileKernelFn;
use kernel_ir::Kernel;
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const WIDTH: u32 = 1200;
const HEIGHT: u32 = 900;

struct CliArgs {
    backend: String,
    kernel_path: Option<String>,
    samples: Option<u32>,
    dump_ir: bool,
    threads: Option<usize>,
    bench: bool,
    bench_frames: u32,
    output: Option<String>,
    tile_height: usize,
    sim: Option<String>,
}

fn parse_args() -> CliArgs {
    let args: Vec<String> = std::env::args().collect();
    let mut backend = "native".to_string();
    let mut kernel_path = None;
    let mut samples = None;
    let mut dump_ir = false;
    let mut threads = None;
    let mut bench = false;
    let mut bench_frames = 100u32;
    let mut output = None;
    let mut tile_height = render::DEFAULT_TILE_HEIGHT;
    let mut sim = None;
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
            "--samples" => {
                i += 1;
                if i < args.len() {
                    samples = Some(args[i].parse::<u32>().unwrap_or_else(|_| {
                        eprintln!("--samples requires a positive integer");
                        std::process::exit(1);
                    }));
                }
            }
            "--dump-ir" => {
                dump_ir = true;
            }
            "--threads" => {
                i += 1;
                if i < args.len() {
                    threads = Some(args[i].parse::<usize>().unwrap_or_else(|_| {
                        eprintln!("--threads requires a positive integer");
                        std::process::exit(1);
                    }));
                }
            }
            "--bench" => {
                bench = true;
            }
            "--bench-frames" => {
                i += 1;
                if i < args.len() {
                    bench_frames = args[i].parse::<u32>().unwrap_or_else(|_| {
                        eprintln!("--bench-frames requires a positive integer");
                        std::process::exit(1);
                    });
                }
            }
            "--output" => {
                i += 1;
                if i < args.len() {
                    output = Some(args[i].clone());
                }
            }
            "--tile-height" => {
                i += 1;
                if i < args.len() {
                    tile_height = args[i].parse::<usize>().unwrap_or_else(|_| {
                        eprintln!("--tile-height requires a positive integer");
                        std::process::exit(1);
                    });
                }
            }
            "--sim" => {
                i += 1;
                if i < args.len() {
                    sim = Some(args[i].clone());
                }
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                std::process::exit(1);
            }
        }
        i += 1;
    }
    CliArgs { backend, kernel_path, samples, dump_ir, threads, bench, bench_frames, output, tile_height, sim }
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
            } else if path.ends_with(".pd") {
                if backend == "gpu" {
                    eprintln!("GPU backend requires a .wgsl kernel, got '{}'", path);
                    std::process::exit(1);
                }
                let kernel = lang::pd::parse(&src, Some(std::path::Path::new(path))).unwrap_or_else(|e| {
                    eprintln!("Parse error in '{}': {}", path, e);
                    std::process::exit(1);
                });
                KernelSource::Pdl(kernel)
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

/// Parse a kernel from source, auto-detecting PD vs PDL by file extension.
fn parse_kernel_src(src: &str, path: &str) -> Kernel {
    if path.ends_with(".pd") {
        lang::pd::parse(src, Some(std::path::Path::new(path))).unwrap_or_else(|e| {
            eprintln!("Parse error in '{}': {}", path, e);
            std::process::exit(1);
        })
    } else {
        lang::parser::parse(src).unwrap_or_else(|e| {
            eprintln!("Parse error in '{}': {}", path, e);
            std::process::exit(1);
        })
    }
}

enum Backend {
    Cpu {
        kernel_fn: TileKernelFn,
        buffer: Vec<u32>,
        /// Second buffer for resolved display output in progressive mode.
        display_buffer: Option<Vec<u32>>,
        accum: Option<progressive::AccumulationBuffer>,
    },
    Gpu {
        gpu_backend: Option<GpuBackend>,
        sample_count: u32,
        max_samples: Option<u32>,
    },
    Simulation {
        state: simulation::FluidState,
        pixel_buffer: Vec<u32>,
    },
    GpuSimulation {
        gpu_fluid: Option<gpu::fluid::GpuFluidBackend>,
    },
    JitSimulation {
        sim_fn: jit::SimTileKernelFn,
        /// Per-field f64 buffers: [u, v] (current) and [u_next, v_next] (next)
        u: Vec<f64>,
        v: Vec<f64>,
        u_next: Vec<f64>,
        v_next: Vec<f64>,
        pixel_buffer: Vec<u32>,
        substeps: u32,
    },
    ShallowWater {
        state: simulation::ShallowWaterState,
        pixel_buffer: Vec<u32>,
    },
    GpuShallowWater {
        gpu: Option<gpu::shallow_water_gpu::GpuShallowWaterBackend>,
    },
    Smoke {
        state: simulation::SmokeState,
        pixel_buffer: Vec<u32>,
    },
    GpuSmoke {
        gpu: Option<gpu::smoke_gpu::GpuSmokeBackend>,
    },
    JitSmoke {
        advect_fn: jit::SimTileKernelFn,
        div_fn: jit::SimTileKernelFn,
        jacobi_fn: jit::SimTileKernelFn,
        project_fn: jit::SimTileKernelFn,
        vx: Vec<f64>,
        vy: Vec<f64>,
        density: Vec<f64>,
        vx0: Vec<f64>,
        vy0: Vec<f64>,
        density0: Vec<f64>,
        pressure: Vec<f64>,
        pressure_tmp: Vec<f64>,
        divergence: Vec<f64>,
        pixel_buffer: Vec<u32>,
    },
    JitShallowWater {
        sim_fn: jit::SimTileKernelFn,
        h: Vec<f64>,
        vx: Vec<f64>,
        vy: Vec<f64>,
        h_next: Vec<f64>,
        vx_next: Vec<f64>,
        vy_next: Vec<f64>,
        pixel_buffer: Vec<u32>,
        substeps: u32,
    },
}

fn with_pool<F: FnOnce() + Send>(pool: &Option<rayon::ThreadPool>, f: F) {
    match pool {
        Some(pool) => pool.install(f),
        None => f(),
    }
}

fn compile_sim_kernel(backend_name: &str, kernel: &Kernel) -> jit::SimTileKernelFn {
    match backend_name {
        #[cfg(feature = "cranelift-backend")]
        "cranelift" => {
            let backend = jit::cranelift::CraneliftBackend;
            let compiled = jit::JitBackend::compile_sim(&backend, kernel);
            let compiled: &'static dyn jit::CompiledSimKernel = Box::leak(compiled);
            compiled.function_ptr()
        }

        #[cfg(feature = "llvm-backend")]
        "llvm" => {
            let backend = jit::llvm::LlvmBackend;
            let compiled = jit::JitBackend::compile_sim(&backend, kernel);
            let compiled: &'static dyn jit::CompiledSimKernel = Box::leak(compiled);
            compiled.function_ptr()
        }

        other => {
            eprintln!("Unknown JIT backend for sim: '{}'", other);
            std::process::exit(1);
        }
    }
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
    thread_pool: Option<rayon::ThreadPool>,
    window: Option<Arc<Window>>,
    display: Option<Display>,
    center_x: f64,
    center_y: f64,
    zoom: f64,
    needs_render: bool,
    keys_down: Vec<KeyCode>,
    start_time: Instant,
    animated: bool,
    tile_height: usize,
    mouse_pos: (f64, f64),
    mouse_down: bool,
}

impl App {
    fn new(
        backend: Backend,
        label: &'static str,
        compile_ms: f64,
        wgsl_source: Option<String>,
        thread_pool: Option<rayon::ThreadPool>,
        animated: bool,
        tile_height: usize,
    ) -> Self {
        Self {
            backend,
            label,
            compile_ms,
            wgsl_source,
            thread_pool,
            window: None,
            display: None,
            center_x: 0.0,
            center_y: 0.0,
            zoom: 1.0,
            needs_render: true,
            keys_down: Vec::new(),
            start_time: Instant::now(),
            animated,
            tile_height,
            mouse_pos: (0.0, 0.0),
            mouse_down: false,
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
                    self.center_x = 0.0;
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

        if let Backend::GpuSimulation { gpu_fluid } = &mut self.backend {
            *gpu_fluid = Some(gpu::fluid::GpuFluidBackend::new(
                &display,
                WIDTH,
                HEIGHT,
                &simulation::GrayScottParams::default(),
            ));
        }

        if let Backend::GpuShallowWater { gpu } = &mut self.backend {
            *gpu = Some(gpu::shallow_water_gpu::GpuShallowWaterBackend::new(
                &display,
                WIDTH,
                HEIGHT,
                &simulation::ShallowWaterParams::default(),
            ));
        }

        if let Backend::GpuSmoke { gpu } = &mut self.backend {
            *gpu = Some(gpu::smoke_gpu::GpuSmokeBackend::new(
                &display,
                WIDTH,
                HEIGHT,
                &gpu::smoke_gpu::SmokeParams::default(),
            ));
        }

        if let Backend::Gpu { gpu_backend, .. } = &mut self.backend {
            let wgsl = self.wgsl_source.as_deref();
            // Empty string means "use default", None also means default
            let wgsl = match wgsl {
                Some("") | None => None,
                Some(s) => Some(s),
            };
            *gpu_backend = Some(GpuBackend::new(&display, WIDTH, HEIGHT, 256, wgsl));
        }

        if self.animated {
            event_loop.set_control_flow(ControlFlow::Poll);
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
            WindowEvent::CursorMoved { position, .. } => {
                self.mouse_pos = (position.x, position.y);
            }
            WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                self.mouse_down = state == ElementState::Pressed;
            }
            WindowEvent::RedrawRequested => {
                let moved = self.handle_input();
                let display = self.display.as_ref().unwrap();
                let time = self.start_time.elapsed().as_secs_f64();
                let pool = &self.thread_pool;
                let th = self.tile_height;

                match &mut self.backend {
                    Backend::Cpu { kernel_fn, buffer, display_buffer, accum } => {
                        match accum {
                            Some(accum) => {
                                // Progressive mode
                                if moved {
                                    accum.reset();
                                }
                                if !accum.is_converged() {
                                    let t0 = Instant::now();
                                    let kfn = *kernel_fn;
                                    let (cx, cy, z) = (self.center_x, self.center_y, self.zoom);
                                    let si = accum.sample_count;
                                    with_pool(pool, || render::render(
                                        buffer,
                                        WIDTH as usize,
                                        HEIGHT as usize,
                                        cx, cy, z,
                                        kfn,
                                        si,
                                        time,
                                        th,
                                    ));
                                    accum.accumulate(buffer);
                                    let disp = display_buffer.as_mut().unwrap();
                                    accum.resolve(disp);
                                    let render_ms = t0.elapsed().as_secs_f64() * 1000.0;
                                    if let Some(window) = &self.window {
                                        window.set_title(&format!(
                                            "{} | {render_ms:.1}ms | sample {}/{} | {:.1}x | compile {:.1}ms",
                                            self.label, accum.sample_count, accum.max_samples,
                                            self.zoom, self.compile_ms
                                        ));
                                    }
                                    display.upload_and_present(disp);
                                    // Request next sample
                                    if let Some(window) = &self.window {
                                        window.request_redraw();
                                    }
                                } else {
                                    // Converged — just re-present
                                    let disp = display_buffer.as_ref().unwrap();
                                    display.upload_and_present(disp);
                                }
                            }
                            None => {
                                // Non-progressive mode (original behavior)
                                if moved || self.needs_render || self.animated {
                                    let t0 = Instant::now();
                                    let kfn = *kernel_fn;
                                    let (cx, cy, z) = (self.center_x, self.center_y, self.zoom);
                                    with_pool(pool, || render::render(
                                        buffer,
                                        WIDTH as usize,
                                        HEIGHT as usize,
                                        cx, cy, z,
                                        kfn,
                                        0xFFFFFFFF,
                                        time,
                                        th,
                                    ));
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
                        }
                    }
                    Backend::Simulation { state, pixel_buffer } => {
                        // Inject chemical on mouse drag
                        if self.mouse_down {
                            let px = self.mouse_pos.0 as usize;
                            let py = self.mouse_pos.1 as usize;
                            if px < WIDTH as usize && py < HEIGHT as usize {
                                state.inject(px, py, 5);
                            }
                        }

                        let t0 = Instant::now();
                        state.step();
                        state.to_pixels(pixel_buffer);
                        let render_ms = t0.elapsed().as_secs_f64() * 1000.0;
                        let fps = 1000.0 / render_ms;

                        if let Some(window) = &self.window {
                            window.set_title(&format!(
                                "gray-scott [{}] | {render_ms:.1}ms ({fps:.0} fps)",
                                self.label
                            ));
                        }
                        display.upload_and_present(pixel_buffer);
                    }
                    Backend::JitSimulation { sim_fn, u, v, u_next, v_next, pixel_buffer, substeps } => {
                        let t0 = Instant::now();
                        let w = WIDTH as usize;
                        let h = HEIGHT as usize;

                        for _ in 0..*substeps {
                            let bufs_in = [u.as_ptr(), v.as_ptr()];
                            let bufs_out = [u_next.as_mut_ptr(), v_next.as_mut_ptr()];
                            render::render_sim(
                                pixel_buffer, w, h, *sim_fn,
                                &bufs_in, &bufs_out, self.tile_height,
                            );
                            std::mem::swap(u, u_next);
                            std::mem::swap(v, v_next);
                        }

                        let render_ms = t0.elapsed().as_secs_f64() * 1000.0;
                        let fps = 1000.0 / render_ms;
                        if let Some(window) = &self.window {
                            window.set_title(&format!(
                                "gray-scott [{}] | {render_ms:.1}ms ({fps:.0} fps)",
                                self.label
                            ));
                        }
                        display.upload_and_present(pixel_buffer);
                    }
                    Backend::GpuSimulation { gpu_fluid } => {
                        let gpu = gpu_fluid.as_mut().unwrap();

                        if self.mouse_down {
                            let px = self.mouse_pos.0 as u32;
                            let py = self.mouse_pos.1 as u32;
                            if px < WIDTH && py < HEIGHT {
                                gpu.inject(&display.queue, px, py, 5);
                            }
                        }

                        let t0 = Instant::now();
                        gpu.step_and_render(display, 8);
                        let render_ms = t0.elapsed().as_secs_f64() * 1000.0;
                        let fps = 1000.0 / render_ms;

                        if let Some(window) = &self.window {
                            window.set_title(&format!(
                                "gray-scott [gpu] | {render_ms:.1}ms ({fps:.0} fps)",
                            ));
                        }
                    }
                    Backend::ShallowWater { state, pixel_buffer } => {
                        if self.mouse_down {
                            let px = self.mouse_pos.0 as usize;
                            let py = self.mouse_pos.1 as usize;
                            if px < WIDTH as usize && py < HEIGHT as usize {
                                state.inject(px, py, 10);
                            }
                        }

                        let t0 = Instant::now();
                        state.step();
                        state.to_pixels(pixel_buffer);
                        let render_ms = t0.elapsed().as_secs_f64() * 1000.0;
                        let fps = 1000.0 / render_ms;

                        if let Some(window) = &self.window {
                            window.set_title(&format!(
                                "shallow-water [{}] | {render_ms:.1}ms ({fps:.0} fps)",
                                self.label
                            ));
                        }
                        display.upload_and_present(pixel_buffer);
                    }
                    Backend::GpuShallowWater { gpu } => {
                        let gpu = gpu.as_mut().unwrap();

                        if self.mouse_down {
                            let px = self.mouse_pos.0 as u32;
                            let py = self.mouse_pos.1 as u32;
                            if px < WIDTH && py < HEIGHT {
                                gpu.inject(&display.queue, px, py, 10);
                            }
                        }

                        let t0 = Instant::now();
                        gpu.step_and_render(display, 4);
                        let render_ms = t0.elapsed().as_secs_f64() * 1000.0;
                        let fps = 1000.0 / render_ms;

                        if let Some(window) = &self.window {
                            window.set_title(&format!(
                                "shallow-water [gpu] | {render_ms:.1}ms ({fps:.0} fps)",
                            ));
                        }
                    }
                    Backend::JitShallowWater { sim_fn, h, vx, vy, h_next, vx_next, vy_next, pixel_buffer, substeps } => {
                        if self.mouse_down {
                            let px = self.mouse_pos.0 as isize;
                            let py = self.mouse_pos.1 as isize;
                            let w = WIDTH as isize;
                            let he = HEIGHT as isize;
                            let radius: isize = 10;
                            let r2 = radius * radius;
                            for dy in -radius..=radius {
                                for dx in -radius..=radius {
                                    let d2 = dx * dx + dy * dy;
                                    if d2 > r2 { continue; }
                                    let x = (px + dx).rem_euclid(w) as usize;
                                    let y = (py + dy).rem_euclid(he) as usize;
                                    let t = 1.0 - (d2 as f64 / r2 as f64);
                                    h[y * WIDTH as usize + x] += 0.15 * t * t;
                                }
                            }
                        }

                        let t0 = Instant::now();

                        for _ in 0..*substeps {
                            let bufs_in = [h.as_ptr(), vx.as_ptr(), vy.as_ptr()];
                            let bufs_out = [h_next.as_mut_ptr(), vx_next.as_mut_ptr(), vy_next.as_mut_ptr()];
                            render::render_sim(
                                pixel_buffer, WIDTH as usize, HEIGHT as usize, *sim_fn,
                                &bufs_in, &bufs_out, self.tile_height,
                            );
                            std::mem::swap(h, h_next);
                            std::mem::swap(vx, vx_next);
                            std::mem::swap(vy, vy_next);
                        }

                        let render_ms = t0.elapsed().as_secs_f64() * 1000.0;
                        let fps = 1000.0 / render_ms;
                        if let Some(window) = &self.window {
                            window.set_title(&format!(
                                "shallow-water [{}] | {render_ms:.1}ms ({fps:.0} fps)",
                                self.label
                            ));
                        }
                        display.upload_and_present(pixel_buffer);
                    }
                    Backend::Smoke { state, pixel_buffer } => {
                        if self.mouse_down {
                            let px = self.mouse_pos.0 as usize;
                            let py = self.mouse_pos.1 as usize;
                            if (px as u32) < WIDTH && (py as u32) < HEIGHT {
                                state.inject(px, py, 15);
                            }
                        }

                        let t0 = Instant::now();
                        state.step();
                        state.to_pixels(pixel_buffer);
                        let render_ms = t0.elapsed().as_secs_f64() * 1000.0;
                        let fps = 1000.0 / render_ms;
                        if let Some(window) = &self.window {
                            window.set_title(&format!(
                                "smoke [{}] | {render_ms:.1}ms ({fps:.0} fps)",
                                self.label
                            ));
                        }
                        display.upload_and_present(pixel_buffer);
                    }
                    Backend::GpuSmoke { gpu } => {
                        let gpu = gpu.as_mut().unwrap();

                        if self.mouse_down {
                            let px = self.mouse_pos.0 as u32;
                            let py = self.mouse_pos.1 as u32;
                            if px < WIDTH && py < HEIGHT {
                                gpu.inject(&display.queue, px, py, 15);
                            }
                        }

                        let t0 = Instant::now();
                        gpu.step_and_render(display);
                        let render_ms = t0.elapsed().as_secs_f64() * 1000.0;
                        let fps = 1000.0 / render_ms;

                        if let Some(window) = &self.window {
                            window.set_title(&format!(
                                "smoke [gpu] | {render_ms:.1}ms ({fps:.0} fps)",
                            ));
                        }
                    }
                    Backend::JitSmoke {
                        advect_fn, div_fn, jacobi_fn, project_fn,
                        vx, vy, density, vx0, vy0, density0,
                        pressure, pressure_tmp, divergence, pixel_buffer,
                    } => {
                        if self.mouse_down {
                            let px = self.mouse_pos.0 as usize;
                            let py = self.mouse_pos.1 as usize;
                            let w = WIDTH as usize;
                            let h = HEIGHT as usize;
                            if px < w && py < h {
                                let radius: isize = 15;
                                let r2 = radius * radius;
                                for idy in -radius..=radius {
                                    for idx in -radius..=radius {
                                        let d2 = idx * idx + idy * idy;
                                        if d2 > r2 { continue; }
                                        let x = (px as isize + idx).clamp(0, w as isize - 1) as usize;
                                        let y = (py as isize + idy).clamp(0, h as isize - 1) as usize;
                                        let t = 1.0 - (d2 as f64 / r2 as f64);
                                        let i = y * w + x;
                                        vx[i] = 0.0;
                                        vy[i] = -3.0 * t;
                                        density[i] = 0.5 * t;
                                    }
                                }
                            }
                        }

                        let t0 = Instant::now();
                        let w = WIDTH as usize;
                        let h = HEIGHT as usize;

                        // 1. Advect: swap then trace backward
                        std::mem::swap(vx, vx0);
                        std::mem::swap(vy, vy0);
                        std::mem::swap(density, density0);
                        render::render_sim(
                            pixel_buffer, w, h, *advect_fn,
                            &[vx0.as_ptr(), vy0.as_ptr(), density0.as_ptr()],
                            &[vx.as_mut_ptr(), vy.as_mut_ptr(), density.as_mut_ptr()],
                            self.tile_height,
                        );

                        // 2. Divergence
                        render::render_sim(
                            pixel_buffer, w, h, *div_fn,
                            &[vx.as_ptr(), vy.as_ptr()],
                            &[divergence.as_mut_ptr()],
                            self.tile_height,
                        );

                        // 3. Jacobi pressure solve (40 iterations)
                        for _ in 0..40 {
                            render::render_sim(
                                pixel_buffer, w, h, *jacobi_fn,
                                &[divergence.as_ptr(), pressure.as_ptr()],
                                &[pressure_tmp.as_mut_ptr()],
                                self.tile_height,
                            );
                            std::mem::swap(pressure, pressure_tmp);
                        }

                        // 4. Project + visualize (writes projected vx/vy to vx0/vy0 as scratch)
                        render::render_sim(
                            pixel_buffer, w, h, *project_fn,
                            &[pressure.as_ptr(), vx.as_ptr(), vy.as_ptr(), density.as_ptr()],
                            &[vx0.as_mut_ptr(), vy0.as_mut_ptr()],
                            self.tile_height,
                        );
                        std::mem::swap(vx, vx0);
                        std::mem::swap(vy, vy0);

                        let render_ms = t0.elapsed().as_secs_f64() * 1000.0;
                        let fps = 1000.0 / render_ms;
                        if let Some(window) = &self.window {
                            window.set_title(&format!(
                                "smoke [{}] | {render_ms:.1}ms ({fps:.0} fps)",
                                self.label
                            ));
                        }
                        display.upload_and_present(pixel_buffer);
                    }
                    Backend::Gpu { gpu_backend, sample_count, max_samples } => {
                        let gpu = gpu_backend.as_ref().unwrap();
                        if moved {
                            if max_samples.is_some() {
                                gpu.reset_accumulation(&display.queue);
                                *sample_count = 0;
                            }
                        }
                        match *max_samples {
                            Some(max) if *sample_count < max => {
                                let t0 = Instant::now();
                                gpu.render(
                                    display, self.center_x, self.center_y, self.zoom,
                                    *sample_count, *sample_count + 1, time,
                                );
                                *sample_count += 1;
                                let render_ms = t0.elapsed().as_secs_f64() * 1000.0;
                                if let Some(window) = &self.window {
                                    window.set_title(&format!(
                                        "{} | {render_ms:.1}ms | sample {}/{} | {:.1}x",
                                        self.label, sample_count, max, self.zoom
                                    ));
                                    window.request_redraw();
                                }
                            }
                            Some(_) => {
                                // Converged — no re-render needed
                            }
                            None => {
                                // Non-progressive: single render
                                if moved || self.needs_render || self.animated {
                                    let t0 = Instant::now();
                                    gpu.render(
                                        display, self.center_x, self.center_y, self.zoom,
                                        0xFFFFFFFF, 0, time,
                                    );
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
                    }
                }

                self.needs_render = false;

                // Keep requesting redraws for animation or while keys are held
                if self.animated || !self.keys_down.is_empty() {
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

    // Simulation mode: separate path from kernel rendering
    if let Some(ref sim_name) = args.sim {
        match sim_name.as_str() {
            "gray-scott" => {
                let label: &'static str = match args.backend.as_str() {
                    "gpu" => "gpu",
                    "native" => "native",
                    "cranelift" => "cranelift",
                    "llvm" => "llvm",
                    other => {
                        eprintln!("Unknown backend for sim: '{}'", other);
                        std::process::exit(1);
                    }
                };
                eprintln!("[sim] Gray-Scott reaction-diffusion ({}x{}) [{}]", WIDTH, HEIGHT, label);

                let backend = match args.backend.as_str() {
                    "gpu" => Backend::GpuSimulation { gpu_fluid: None },
                    "native" => {
                        let state = simulation::FluidState::new(WIDTH as usize, HEIGHT as usize);
                        let pixel_buffer = vec![0u32; (WIDTH * HEIGHT) as usize];
                        Backend::Simulation { state, pixel_buffer }
                    }
                    jit_backend => {
                        let kernel_path = args.kernel_path.as_deref().unwrap_or_else(|| {
                            eprintln!("--kernel required for JIT sim backend");
                            std::process::exit(1);
                        });
                        let src = std::fs::read_to_string(kernel_path).unwrap_or_else(|e| {
                            eprintln!("Failed to read kernel file '{}': {}", kernel_path, e);
                            std::process::exit(1);
                        });
                        let kernel = parse_kernel_src(&src, kernel_path);
                        eprintln!("[kernel] loaded '{}' ({} statements, {} buffers)",
                            kernel.name, kernel.body.len(), kernel.buffers.len());

                        let sim_fn = compile_sim_kernel(jit_backend, &kernel);

                        let n = (WIDTH * HEIGHT) as usize;
                        let mut u = vec![1.0f64; n];
                        let mut v = vec![0.0f64; n];

                        // Seed (same as native)
                        let mut rng = 12345u64;
                        let mut next = || -> u64 {
                            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                            rng
                        };
                        let w = WIDTH as usize;
                        let h = HEIGHT as usize;
                        let radius = 6usize;
                        for _ in 0..20 {
                            let sx = (next() as usize) % (w - 2 * radius) + radius;
                            let sy = (next() as usize) % (h - 2 * radius) + radius;
                            for dy in -(radius as isize)..=(radius as isize) {
                                for dx in -(radius as isize)..=(radius as isize) {
                                    if dx * dx + dy * dy > (radius * radius) as isize {
                                        continue;
                                    }
                                    let x = (sx as isize + dx) as usize;
                                    let y = (sy as isize + dy) as usize;
                                    u[y * w + x] = 0.5;
                                    v[y * w + x] = 0.25;
                                }
                            }
                        }

                        Backend::JitSimulation {
                            sim_fn,
                            u,
                            v,
                            u_next: vec![0.0; n],
                            v_next: vec![0.0; n],
                            pixel_buffer: vec![0u32; n],
                            substeps: 8,
                        }
                    }
                };

                let event_loop = EventLoop::new().unwrap();
                event_loop.set_control_flow(ControlFlow::Poll);
                let mut app = App::new(backend, label, 0.0, None, None, true, args.tile_height);
                event_loop.run_app(&mut app).unwrap();
            }
            "shallow-water" => {
                let label: &'static str = match args.backend.as_str() {
                    "gpu" => "gpu",
                    "native" => "native",
                    "cranelift" => "cranelift",
                    "llvm" => "llvm",
                    other => {
                        eprintln!("Unknown backend for shallow-water sim: '{}' (available: native, gpu, cranelift, llvm)", other);
                        std::process::exit(1);
                    }
                };
                eprintln!("[sim] Shallow water waves ({}x{}) [{}]", WIDTH, HEIGHT, label);

                let backend = match args.backend.as_str() {
                    "gpu" => Backend::GpuShallowWater { gpu: None },
                    "native" => {
                        let state = simulation::ShallowWaterState::new(WIDTH as usize, HEIGHT as usize);
                        let pixel_buffer = vec![0u32; (WIDTH * HEIGHT) as usize];
                        Backend::ShallowWater { state, pixel_buffer }
                    }
                    jit_backend => {
                        let kernel_path = args.kernel_path.as_deref().unwrap_or_else(|| {
                            eprintln!("--kernel required for JIT sim backend");
                            std::process::exit(1);
                        });
                        let src = std::fs::read_to_string(kernel_path).unwrap_or_else(|e| {
                            eprintln!("Failed to read kernel file '{}': {}", kernel_path, e);
                            std::process::exit(1);
                        });
                        let kernel = parse_kernel_src(&src, kernel_path);
                        eprintln!("[kernel] loaded '{}' ({} statements, {} buffers)",
                            kernel.name, kernel.body.len(), kernel.buffers.len());

                        let sim_fn = compile_sim_kernel(jit_backend, &kernel);

                        let n = (WIDTH * HEIGHT) as usize;
                        let mut h = vec![1.0f64; n];
                        let vx = vec![0.0f64; n];
                        let vy = vec![0.0f64; n];

                        // Seed bumps (same as native)
                        let mut rng = 54321u64;
                        let mut next = || -> u64 {
                            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                            rng
                        };
                        let w = WIDTH as usize;
                        let he = HEIGHT as usize;
                        let radius = 20usize;
                        for _ in 0..5 {
                            let sx = (next() as usize) % (w - 2 * radius) + radius;
                            let sy = (next() as usize) % (he - 2 * radius) + radius;
                            for dy in -(radius as isize)..=(radius as isize) {
                                for dx in -(radius as isize)..=(radius as isize) {
                                    let d2 = dx * dx + dy * dy;
                                    let r2 = (radius * radius) as isize;
                                    if d2 > r2 {
                                        continue;
                                    }
                                    let x = (sx as isize + dx) as usize;
                                    let y = (sy as isize + dy) as usize;
                                    let t = 1.0 - (d2 as f64 / r2 as f64);
                                    h[y * w + x] += 0.3 * t * t;
                                }
                            }
                        }

                        Backend::JitShallowWater {
                            sim_fn,
                            h,
                            vx,
                            vy,
                            h_next: vec![0.0; n],
                            vx_next: vec![0.0; n],
                            vy_next: vec![0.0; n],
                            pixel_buffer: vec![0u32; n],
                            substeps: 4,
                        }
                    }
                };

                let event_loop = EventLoop::new().unwrap();
                event_loop.set_control_flow(ControlFlow::Poll);
                let mut app = App::new(backend, label, 0.0, None, None, true, args.tile_height);
                event_loop.run_app(&mut app).unwrap();
            }
            "smoke" => {
                let label: &'static str = match args.backend.as_str() {
                    "gpu" => "gpu",
                    "native" => "native",
                    "cranelift" => "cranelift",
                    "llvm" => "llvm",
                    other => {
                        eprintln!("Unknown backend for smoke sim: '{}' (available: native, gpu, cranelift, llvm)", other);
                        std::process::exit(1);
                    }
                };
                eprintln!("[sim] Smoke ({}x{}) [{}]", WIDTH, HEIGHT, label);

                let backend = match args.backend.as_str() {
                    "gpu" => Backend::GpuSmoke { gpu: None },
                    "native" => {
                        let state = simulation::SmokeState::new(WIDTH as usize, HEIGHT as usize);
                        let pixel_buffer = vec![0u32; (WIDTH * HEIGHT) as usize];
                        Backend::Smoke { state, pixel_buffer }
                    }
                    jit_backend => {
                        let parse = |src: &str, name: &str| -> kernel_ir::Kernel {
                            lang::pd::parse(src, None).unwrap_or_else(|e| {
                                eprintln!("Parse error in smoke/{}: {}", name, e);
                                std::process::exit(1);
                            })
                        };
                        let advect_k = parse(
                            include_str!("../examples/sim/smoke/advect.pd"), "advect.pd");
                        let div_k = parse(
                            include_str!("../examples/sim/smoke/divergence.pd"), "divergence.pd");
                        let jacobi_k = parse(
                            include_str!("../examples/sim/smoke/jacobi.pd"), "jacobi.pd");
                        let project_k = parse(
                            include_str!("../examples/sim/smoke/project.pd"), "project.pd");

                        eprintln!("[kernel] smoke: advect({} stmts), divergence({} stmts), jacobi({} stmts), project({} stmts)",
                            advect_k.body.len(), div_k.body.len(), jacobi_k.body.len(), project_k.body.len());

                        let advect_fn = compile_sim_kernel(jit_backend, &advect_k);
                        let div_fn = compile_sim_kernel(jit_backend, &div_k);
                        let jacobi_fn = compile_sim_kernel(jit_backend, &jacobi_k);
                        let project_fn = compile_sim_kernel(jit_backend, &project_k);

                        let n = (WIDTH * HEIGHT) as usize;
                        Backend::JitSmoke {
                            advect_fn, div_fn, jacobi_fn, project_fn,
                            vx: vec![0.0; n],
                            vy: vec![0.0; n],
                            density: vec![0.0; n],
                            vx0: vec![0.0; n],
                            vy0: vec![0.0; n],
                            density0: vec![0.0; n],
                            pressure: vec![0.0; n],
                            pressure_tmp: vec![0.0; n],
                            divergence: vec![0.0; n],
                            pixel_buffer: vec![0u32; n],
                        }
                    }
                };

                let event_loop = EventLoop::new().unwrap();
                event_loop.set_control_flow(ControlFlow::Poll);
                let mut app = App::new(backend, label, 0.0, None, None, true, args.tile_height);
                event_loop.run_app(&mut app).unwrap();
            }
            other => {
                eprintln!("Unknown simulation: '{}'. Available: gray-scott, shallow-water, smoke", other);
                std::process::exit(1);
            }
        }
        return;
    }

    let kernel_source = load_kernel(&args.kernel_path, &args.backend);

    if args.dump_ir {
        if let KernelSource::Pdl(ref kernel) = kernel_source {
            let pdl = lang::printer::print(kernel);
            eprintln!("── Lowered IR (PDL) ──");
            eprintln!("{}", pdl);
        }
    }

    let compile_start = Instant::now();

    let animated = match &kernel_source {
        KernelSource::Pdl(kernel) => kernel.params.iter().any(|p| p.name == "time"),
        KernelSource::Wgsl(src) => src.contains("params.time"),
    };

    let (backend, label, wgsl_source) = match kernel_source {
        KernelSource::Wgsl(src) => {
            if src.is_empty() {
                eprintln!("[kernel] using built-in mandelbrot.wgsl");
            } else {
                eprintln!("[kernel] loaded WGSL from '{}'", args.kernel_path.as_deref().unwrap());
            }
            if let Some(n) = args.samples {
                eprintln!("[progressive] {} samples", n);
            }
            (
                Backend::Gpu {
                    gpu_backend: None,
                    sample_count: 0,
                    max_samples: args.samples,
                },
                "gpu",
                Some(src),
            )
        }
        KernelSource::Pdl(kernel) => {
            eprintln!("[kernel] loaded '{}' ({} statements)", kernel.name, kernel.body.len());
            let (kernel_fn, label) = compile_cpu_kernel(&args.backend, &kernel);
            let pixel_count = (WIDTH * HEIGHT) as usize;
            let (display_buffer, accum) = match args.samples {
                Some(n) => {
                    eprintln!("[progressive] {} samples", n);
                    (
                        Some(vec![0u32; pixel_count]),
                        Some(progressive::AccumulationBuffer::new(WIDTH as usize, HEIGHT as usize, n)),
                    )
                }
                None => (None, None),
            };
            (
                Backend::Cpu {
                    kernel_fn,
                    buffer: vec![0u32; pixel_count],
                    display_buffer,
                    accum,
                },
                label,
                None,
            )
        }
    };

    let compile_ms = compile_start.elapsed().as_secs_f64() * 1000.0;
    eprintln!("[{label}] compile: {compile_ms:.1}ms");

    // Build custom thread pool if --threads was specified
    let thread_pool = args.threads.map(|n| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .expect("failed to create thread pool")
    });

    if args.threads.is_some() && args.backend == "gpu" {
        eprintln!("warning: --threads is ignored for GPU backend");
    }

    let tile_height = args.tile_height;

    // Output-only mode: render a single frame and save (no window)
    if args.output.is_some() && !args.bench {
        match backend {
            Backend::Cpu { kernel_fn, .. } => {
                let pixel_count = (WIDTH * HEIGHT) as usize;
                let mut buffer = vec![0u32; pixel_count];
                with_pool(&thread_pool, || render::render(
                    &mut buffer, WIDTH as usize, HEIGHT as usize,
                    0.0, 0.0, 1.0, kernel_fn, 0xFFFFFFFF, 0.0, tile_height,
                ));
                bench::write_ppm(args.output.as_ref().unwrap(), &buffer, WIDTH, HEIGHT);
            }
            Backend::Gpu { .. } => {
                let wgsl = wgsl_source.as_deref();
                let wgsl = match wgsl {
                    Some("") | None => None,
                    Some(s) => Some(s),
                };
                let (gpu, device, queue) = GpuBackend::new_headless(WIDTH, HEIGHT, 256, wgsl);
                gpu.dispatch_compute(&device, &queue, 0.0, 0.0, 1.0, 0xFFFFFFFF, 0, 0.0);
                let buffer = gpu.readback_pixels(&device, &queue);
                bench::write_ppm(args.output.as_ref().unwrap(), &buffer, WIDTH, HEIGHT);
            }
            Backend::Simulation { .. } | Backend::GpuSimulation { .. } | Backend::JitSimulation { .. }
            | Backend::ShallowWater { .. } | Backend::GpuShallowWater { .. }
            | Backend::JitShallowWater { .. }
            | Backend::Smoke { .. } | Backend::GpuSmoke { .. } | Backend::JitSmoke { .. } => {
                unreachable!("--sim uses its own path")
            }
        }
        return;
    }

    // Bench mode: headless timing loop, no window
    if args.bench {
        let warmup = 5;
        let frames = args.bench_frames;

        match backend {
            Backend::Cpu { kernel_fn, .. } => {
                let pixel_count = (WIDTH * HEIGHT) as usize;
                let mut buffer = vec![0u32; pixel_count];

                // Warmup
                for i in 0..warmup {
                    with_pool(&thread_pool, || render::render(
                        &mut buffer, WIDTH as usize, HEIGHT as usize,
                        0.0, 0.0, 1.0, kernel_fn, i, 0.0, tile_height,
                    ));
                }

                // Timed frames
                let mut frame_times = Vec::with_capacity(frames as usize);
                for i in 0..frames {
                    let t0 = Instant::now();
                    with_pool(&thread_pool, || render::render(
                        &mut buffer, WIDTH as usize, HEIGHT as usize,
                        0.0, 0.0, 1.0, kernel_fn, i, 0.0, tile_height,
                    ));
                    frame_times.push(t0.elapsed().as_secs_f64() * 1000.0);
                }

                let thread_label = match args.threads {
                    Some(n) => format!("{} ({}t)", label, n),
                    None => format!("{} ({}t)", label, rayon::current_num_threads()),
                };
                bench::BenchResult { frame_times }.report(&thread_label, WIDTH, HEIGHT);

                if let Some(ref path) = args.output {
                    bench::write_ppm(path, &buffer, WIDTH, HEIGHT);
                }
            }
            Backend::Gpu { .. } => {
                let wgsl = wgsl_source.as_deref();
                let wgsl = match wgsl {
                    Some("") | None => None,
                    Some(s) => Some(s),
                };
                let (gpu, device, queue) = GpuBackend::new_headless(WIDTH, HEIGHT, 256, wgsl);

                // Warmup
                for i in 0..warmup {
                    gpu.dispatch_compute(&device, &queue, 0.0, 0.0, 1.0, i, 0, 0.0);
                }

                // Timed frames
                let mut frame_times = Vec::with_capacity(frames as usize);
                for i in 0..frames {
                    let t0 = Instant::now();
                    gpu.dispatch_compute(&device, &queue, 0.0, 0.0, 1.0, i, 0, 0.0);
                    frame_times.push(t0.elapsed().as_secs_f64() * 1000.0);
                }

                bench::BenchResult { frame_times }.report("gpu", WIDTH, HEIGHT);

                if let Some(ref path) = args.output {
                    let buffer = gpu.readback_pixels(&device, &queue);
                    bench::write_ppm(path, &buffer, WIDTH, HEIGHT);
                }
            }
            Backend::Simulation { .. } | Backend::GpuSimulation { .. } | Backend::JitSimulation { .. }
            | Backend::ShallowWater { .. } | Backend::GpuShallowWater { .. }
            | Backend::JitShallowWater { .. }
            | Backend::Smoke { .. } | Backend::GpuSmoke { .. } | Backend::JitSmoke { .. } => {
                unreachable!("--sim uses its own path")
            }
        }
        return;
    }

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Wait);
    let mut app = App::new(backend, label, compile_ms, wgsl_source, thread_pool, animated, tile_height);
    event_loop.run_app(&mut app).unwrap();
}
