mod bench;
#[allow(dead_code)]
mod display;
mod jit;
#[allow(dead_code)]
mod kernel_ir;
#[allow(dead_code)]
mod lang;
mod pdc;
mod progressive;
mod render;

use display::Display;
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
                eprintln!("Usage: pixel-doodle <config.pdc> [options]");
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
            arg if arg.ends_with(".pdc") => {
                config_path = Some(args[i].clone());
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

// ── PDC config runner ──

fn run_pdc(config_path: &str, args: &CliArgs) {
    let src = std::fs::read_to_string(config_path).unwrap_or_else(|e| {
        eprintln!("Failed to read config file '{}': {}", config_path, e);
        std::process::exit(1);
    });
    let base_dir = std::path::Path::new(config_path)
        .parent()
        .unwrap_or(std::path::Path::new("."));
    let config = pdc::parse(&src, base_dir).unwrap_or_else(|e| {
        eprintln!("Config error in '{}':\n{}", config_path, e);
        std::process::exit(1);
    });

    let mut runtime = pdc::runtime::Runtime::new(config, WIDTH, HEIGHT, base_dir);
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
    eprintln!("[pdc] compile: {compile_ms:.1}ms");

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
            runtime.execute_frame(0.0, &thread_pool);
            bench::write_ppm(
                output_path,
                runtime.display_pixels(),
                runtime.width,
                runtime.height,
            );
            eprintln!("[pdc] wrote {}", output_path);
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
    let mut app = PdcApp {
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

// ── PDC-driven app ──

struct PdcApp {
    runtime: pdc::runtime::Runtime,
    thread_pool: Option<rayon::ThreadPool>,
    window: Option<Arc<Window>>,
    display: Option<Display>,
    keys_down: Vec<KeyCode>,
    start_time: Instant,
}

impl ApplicationHandler for PdcApp {
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
            WindowEvent::KeyboardInput {
                event: key_event, ..
            } => {
                if let PhysicalKey::Code(code) = key_event.physical_key {
                    if key_event.state == ElementState::Pressed {
                        if code == KeyCode::Escape || code == KeyCode::KeyQ {
                            event_loop.exit();
                            return;
                        }
                        if !self.keys_down.contains(&code) {
                            self.keys_down.push(code);
                        }
                        // Fire key binding on press
                        if let Some(name) = pdc::runtime::key_code_to_name(code) {
                            self.runtime.handle_key_press(name);
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
            }
            WindowEvent::RedrawRequested => {
                // Handle held keys for continuous pan/zoom
                for code in &self.keys_down.clone() {
                    if let Some(name) = pdc::runtime::key_code_to_name(*code) {
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
                    let pixels = self.runtime.display_pixels();
                    display.upload_and_present(pixels);

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

                if self.runtime.needs_continuous_redraw() {
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
        Some(path) => run_pdc(path, &args),
        None => {
            eprintln!("Usage: pixel-doodle <config.pdc> [options]");
            eprintln!();
            eprintln!("Run with --help for details.");
            std::process::exit(1);
        }
    }
}
