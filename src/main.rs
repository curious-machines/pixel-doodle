mod bench;
#[allow(dead_code)]
mod display;
mod gpu;
mod jit;
#[allow(dead_code)]
mod kernel_ir;
#[allow(dead_code)]
mod lang;
mod pdp;
mod progressive;
mod render;

use display::Display;
use std::path::Path;
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
    config_path: Option<String>,
    threads: Option<usize>,
    bench: bool,
    bench_frames: u32,
    output: Option<String>,
    set_overrides: Vec<(String, String)>,
    settings_file: Option<String>,
}

fn resolve_dir_to_pdp(dir: &str) -> String {
    let dir_path = Path::new(dir);
    let dir_name = dir_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");

    // Try convention: <dir_name>/<dir_name>.pdp
    let convention = dir_path.join(format!("{}.pdp", dir_name));
    if convention.is_file() {
        return convention.to_string_lossy().into_owned();
    }

    // Fall back: look for any .pdp files in the directory
    let pdp_files: Vec<_> = std::fs::read_dir(dir_path)
        .unwrap_or_else(|e| {
            eprintln!("Failed to read directory '{}': {}", dir, e);
            std::process::exit(1);
        })
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry.path().extension().and_then(|e| e.to_str()) == Some("pdp")
                && entry.path().is_file()
        })
        .collect();

    match pdp_files.len() {
        0 => {
            eprintln!("No .pdp file found in '{}'", dir);
            std::process::exit(1);
        }
        1 => pdp_files[0].path().to_string_lossy().into_owned(),
        _ => {
            eprintln!("Multiple .pdp files in '{}', specify one:", dir);
            for f in &pdp_files {
                eprintln!("  {}", f.path().display());
            }
            std::process::exit(1);
        }
    }
}

fn parse_args() -> CliArgs {
    let args: Vec<String> = std::env::args().collect();
    let mut config_path = None;
    let mut threads = None;
    let mut bench = false;
    let mut bench_frames = 100u32;
    let mut output = None;
    let mut set_overrides = Vec::new();
    let mut settings_file = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
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
            "--set" => {
                i += 1;
                if i < args.len() {
                    if let Some((key, value)) = args[i].split_once('=') {
                        set_overrides.push((key.to_string(), value.to_string()));
                    } else {
                        eprintln!("--set requires key=value format");
                        std::process::exit(1);
                    }
                }
            }
            "--settings" => {
                i += 1;
                if i < args.len() {
                    settings_file = Some(args[i].clone());
                }
            }
            "--help" | "-h" => {
                eprintln!("Usage: pixel-doodle <config.pdp | directory> [options]");
                eprintln!();
                eprintln!("Options:");
                eprintln!("  --output <file.ppm>    Render one frame and save (no window)");
                eprintln!("  --bench                Headless benchmark mode");
                eprintln!("  --bench-frames <N>     Number of benchmark frames (default: 100)");
                eprintln!("  --threads <N>          Worker thread count");
                eprintln!("  --settings <file.pds>  Override settings from file");
                eprintln!("  --set key=value        Override a setting or variable");
                eprintln!("  --help                 Show this help");
                std::process::exit(0);
            }
            arg if arg.ends_with(".pdp") => {
                config_path = Some(args[i].clone());
            }
            arg if Path::new(arg).is_dir() => {
                config_path = Some(resolve_dir_to_pdp(arg));
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                eprintln!("Run with --help for usage information.");
                std::process::exit(1);
            }
        }
        i += 1;
    }
    CliArgs {
        config_path,
        threads,
        bench,
        bench_frames,
        output,
        set_overrides,
        settings_file,
    }
}

// ── PDP config runner ──

fn run_pdp(config_path: &str, args: &CliArgs) {
    let src = std::fs::read_to_string(config_path).unwrap_or_else(|e| {
        eprintln!("Failed to read config file '{}': {}", config_path, e);
        std::process::exit(1);
    });
    let base_dir = std::path::Path::new(config_path)
        .parent()
        .unwrap_or(std::path::Path::new("."));
    let config = pdp::parse(&src, base_dir).unwrap_or_else(|e| {
        eprintln!("Config error in '{}':\n{}", config_path, e);
        std::process::exit(1);
    });

    let mut runtime = pdp::runtime::Runtime::new(config, WIDTH, HEIGHT, base_dir);
    runtime.set_config_path(config_path);
    runtime.apply_settings();

    // Apply .pds file overrides
    if let Some(ref pds_path) = args.settings_file {
        let pds_src = std::fs::read_to_string(pds_path).unwrap_or_else(|e| {
            eprintln!("Failed to read settings file '{}': {}", pds_path, e);
            std::process::exit(1);
        });
        let pds_overrides = parse_pds(&pds_src);
        runtime.apply_overrides(&pds_overrides);
    }

    // Apply --set CLI overrides
    runtime.apply_overrides(&args.set_overrides);

    // Apply --threads CLI override
    if let Some(n) = args.threads {
        runtime.apply_overrides(&[("threads".to_string(), n.to_string())]);
    }

    // Compile and init
    let compile_start = Instant::now();
    runtime.compile_kernels().unwrap_or_else(|e| {
        eprintln!("Compilation error: {}", e);
        std::process::exit(1);
    });
    let compile_ms = compile_start.elapsed().as_secs_f64() * 1000.0;
    eprintln!("[pdp] compile: {compile_ms:.1}ms");

    runtime.init_buffers().unwrap_or_else(|e| {
        eprintln!("Buffer init error: {}", e);
        std::process::exit(1);
    });

    runtime.setup_progressive();

    // Build thread pool
    let thread_pool = runtime.thread_count().or(args.threads).map(|n| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .expect("failed to create thread pool")
    });

    // Output-only mode
    if let Some(ref output_path) = args.output {
        if !args.bench {
            if runtime.has_gpu_kernels {
                runtime.execute_gpu_headless();
            } else {
                runtime.execute_init_block(&thread_pool);
                runtime.execute_frame(0.0, &thread_pool);
            }
            bench::write_ppm(
                output_path,
                runtime.display_pixels(),
                runtime.width,
                runtime.height,
            );
            eprintln!("[pdp] wrote {}", output_path);
            return;
        }
    }

    // Bench mode
    if args.bench {
        let warmup = 5;
        let frames = args.bench_frames;
        eprintln!("[bench] warmup {} frames...", warmup);
        for i in 0..warmup {
            runtime.execute_frame(i as f64 * 0.016, &thread_pool);
        }
        eprintln!("[bench] timing {} frames...", frames);
        let mut times = Vec::with_capacity(frames as usize);
        for i in 0..frames {
            let t0 = Instant::now();
            runtime.execute_frame((warmup + i) as f64 * 0.016, &thread_pool);
            times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        if let Some(ref output_path) = args.output {
            bench::write_ppm(
                output_path,
                runtime.display_pixels(),
                runtime.width,
                runtime.height,
            );
        }
        let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean = times.iter().sum::<f64>() / times.len() as f64;
        let variance =
            times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / times.len() as f64;
        let stddev = variance.sqrt();
        eprintln!(
            "[bench] {frames} frames: min={min:.2}ms max={max:.2}ms mean={mean:.2}ms stddev={stddev:.2}ms"
        );
        return;
    }

    // Interactive mode
    let event_loop = EventLoop::new().unwrap();
    if runtime.needs_continuous_redraw() {
        event_loop.set_control_flow(ControlFlow::Poll);
    } else {
        event_loop.set_control_flow(ControlFlow::Wait);
    }
    let mut app = PdpApp {
        runtime,
        thread_pool,
        window: None,
        display: None,
        keys_down: Vec::new(),
        start_time: Instant::now(),
    };
    event_loop.run_app(&mut app).unwrap();
}

/// Parse a .pds settings file: flat key=value lines, # comments.
fn parse_pds(src: &str) -> Vec<(String, String)> {
    let mut overrides = Vec::new();
    for line in src.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((key, value)) = line.split_once('=') {
            let key = key.trim().to_string();
            let value = value.trim().trim_matches('"').to_string();
            overrides.push((key, value));
        }
    }
    overrides
}

// ── PDP-driven app ──

struct PdpApp {
    runtime: pdp::runtime::Runtime,
    thread_pool: Option<rayon::ThreadPool>,
    window: Option<Arc<Window>>,
    display: Option<Display>,
    keys_down: Vec<KeyCode>,
    start_time: Instant,
}

impl ApplicationHandler for PdpApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let win_attrs = Window::default_attributes()
            .with_title(self.runtime.title())
            .with_inner_size(winit::dpi::PhysicalSize::new(
                self.runtime.width,
                self.runtime.height,
            ));
        let window = Arc::new(event_loop.create_window(win_attrs).unwrap());
        let display = Display::new(
            Arc::clone(&window),
            self.runtime.width,
            self.runtime.height,
        );
        // Initialize GPU backend if needed (requires display device/queue)
        self.runtime.init_gpu(&display);
        // Execute init blocks (runs once at startup)
        self.runtime.execute_init_block(&self.thread_pool);
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
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(display) = &mut self.display {
                    display.resize(size.width, size.height);
                }
            }
            WindowEvent::KeyboardInput {
                event: key_event, ..
            } => {
                if let PhysicalKey::Code(code) = key_event.physical_key {
                    if key_event.state == ElementState::Pressed {
                        if code == KeyCode::Escape {
                            event_loop.exit();
                            return;
                        }
                        if !self.keys_down.contains(&code) {
                            self.keys_down.push(code);
                            // Fire key binding on initial press only (not OS repeats)
                            if let Some(name) = pdp::runtime::key_code_to_name(code) {
                                if self.runtime.handle_key_press(name) {
                                    event_loop.exit();
                                    return;
                                }
                                if let Some(window) = &self.window {
                                    window.request_redraw();
                                }
                            }
                        }
                    } else {
                        self.keys_down.retain(|k| *k != code);
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.runtime.mouse_x = position.x;
                self.runtime.mouse_y = position.y;
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                self.runtime.mouse_down = state == ElementState::Pressed;
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => {
                // Handle held keys for continuous pan/zoom
                for code in &self.keys_down.clone() {
                    if let Some(name) = pdp::runtime::key_code_to_name(*code) {
                        // Only re-fire for pan/zoom keys (held down behavior)
                        match name {
                            "left" | "right" | "up" | "down" | "plus" | "minus" => {
                                self.runtime.handle_key_press(name);
                            }
                            _ => {}
                        }
                    }
                }

                let time = self.start_time.elapsed().as_secs_f64();
                let t0 = Instant::now();
                let updated = self.runtime.execute_frame(time, &self.thread_pool);
                let render_ms = t0.elapsed().as_secs_f64() * 1000.0;

                if updated {
                    let display = self.display.as_ref().unwrap();

                    // GPU kernels render directly to display — don't upload CPU pixels
                    let gpu_rendered = self.runtime.render_gpu_frame(display);
                    if !gpu_rendered {
                        let pixels = self.runtime.display_pixels();
                        display.upload_and_present(pixels);
                    }

                    if let Some(window) = &self.window {
                        let title = self.runtime.title();
                        if let Some((current, max)) = self.runtime.accumulation_info() {
                            window.set_title(&format!(
                                "{title} | {render_ms:.1}ms | sample {current}/{max}"
                            ));
                        } else {
                            let fps = if render_ms > 0.0 {
                                1000.0 / render_ms
                            } else {
                                0.0
                            };
                            window.set_title(&format!(
                                "{title} | {render_ms:.1}ms ({fps:.0} fps)"
                            ));
                        }
                    }
                } else if self.display.is_some() {
                    // Re-present existing buffer even if not updated
                    let display = self.display.as_ref().unwrap();
                    display.upload_and_present(self.runtime.display_pixels());
                }

                if self.runtime.needs_continuous_redraw() || !self.keys_down.is_empty() {
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

    match &args.config_path {
        Some(path) => run_pdp(path, &args),
        None => {
            eprintln!("Usage: pixel-doodle <config.pdp> [options]");
            eprintln!();
            eprintln!("Run with --help for details.");
            std::process::exit(1);
        }
    }
}
