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
}

fn with_pool<F: FnOnce() + Send>(pool: &Option<rayon::ThreadPool>, f: F) {
    match pool {
        Some(pool) => pool.install(f),
        None => f(),
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
                eprintln!("[sim] Gray-Scott reaction-diffusion ({}x{})", WIDTH, HEIGHT);
                let state = simulation::FluidState::new(WIDTH as usize, HEIGHT as usize);
                let pixel_buffer = vec![0u32; (WIDTH * HEIGHT) as usize];
                let backend = Backend::Simulation { state, pixel_buffer };

                let event_loop = EventLoop::new().unwrap();
                event_loop.set_control_flow(ControlFlow::Poll);
                let mut app = App::new(backend, "native", 0.0, None, None, true, args.tile_height);
                event_loop.run_app(&mut app).unwrap();
            }
            other => {
                eprintln!("Unknown simulation: '{}'. Available: gray-scott", other);
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
            Backend::Simulation { .. } => unreachable!("--sim uses its own path"),
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
            Backend::Simulation { .. } => unreachable!("--sim uses its own path"),
        }
        return;
    }

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Wait);
    let mut app = App::new(backend, label, compile_ms, wgsl_source, thread_pool, animated, tile_height);
    event_loop.run_app(&mut app).unwrap();
}
