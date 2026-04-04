mod bench;

use pixel_doodle::display::Display;
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

enum ConfigSource {
    File(String),
}

struct CliArgs {
    config_source: Option<ConfigSource>,
    threads: Option<usize>,
    bench: bool,
    bench_frames: u32,
    output: Option<String>,
    set_overrides: Vec<(String, String)>,
    settings_file: Option<String>,
}

/// Find a .pdc file in `dir_path`: first by convention (`dir_name.pdc`),
/// then fall back to a single .pdc file in the directory.
fn resolve_dir(dir: &str) -> ConfigSource {
    let dir_path = Path::new(dir);
    let dir_name = dir_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");

    // Try convention name first (dir_name.pdc)
    let pdc_convention = dir_path.join(format!("{}.pdc", dir_name));
    if pdc_convention.is_file() {
        return ConfigSource::File(pdc_convention.to_string_lossy().into_owned());
    }

    // Fall back to single .pdc file in directory
    let pdc_files: Vec<_> = std::fs::read_dir(dir_path)
        .unwrap_or_else(|e| {
            eprintln!("Failed to read directory '{}': {}", dir, e);
            std::process::exit(1);
        })
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path().extension().and_then(|x| x.to_str()) == Some("pdc") && e.path().is_file()
        })
        .collect();

    match pdc_files.len() {
        1 => ConfigSource::File(pdc_files[0].path().to_string_lossy().into_owned()),
        0 => {
            eprintln!("No .pdc file found in '{}'", dir);
            std::process::exit(1);
        }
        _ => {
            eprintln!("Multiple .pdc files in '{}', specify one:", dir);
            for f in &pdc_files {
                eprintln!("  {}", f.path().display());
            }
            std::process::exit(1);
        }
    }
}

fn parse_args() -> CliArgs {
    let args: Vec<String> = std::env::args().collect();
    let mut config_source = None;
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
                    for item in args[i].split(',') {
                        let item = item.trim();
                        if let Some((key, value)) = item.split_once('=') {
                            set_overrides.push((key.trim().to_string(), value.trim().to_string()));
                        } else {
                            eprintln!("--set requires key=value format");
                            std::process::exit(1);
                        }
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
                eprintln!("Usage: pixel-doodle <config.pdc | directory> [options]");
                eprintln!();
                eprintln!("Options:");
                eprintln!("  --output <file.ppm>    Render one frame and save (no window)");
                eprintln!("  --bench                Headless benchmark mode");
                eprintln!("  --bench-frames <N>     Number of benchmark frames (default: 100)");
                eprintln!("  --threads <N>          Worker thread count");
                eprintln!("  --settings <file.pds>  Override settings from file");
                eprintln!("  --set key=val[,k=v]    Override settings (comma-delimited)");
                eprintln!("  --help                 Show this help");
                std::process::exit(0);
            }
            arg if arg.ends_with(".pdc") => {
                config_source = Some(ConfigSource::File(args[i].clone()));
            }
            arg if Path::new(arg).is_dir() => {
                config_source = Some(resolve_dir(arg));
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
        config_source,
        threads,
        bench,
        bench_frames,
        output,
        set_overrides,
        settings_file,
    }
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

// ── PDC pipeline runner ──

#[cfg(any(feature = "cranelift-backend", feature = "llvm-backend"))]
fn run_pdc(source: &ConfigSource, args: &CliArgs) {
    let ConfigSource::File(path) = source;
    let src = std::fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("Failed to read PDC file '{}': {}", path, e);
        std::process::exit(1);
    });
    let base_dir = std::path::Path::new(path.as_str())
        .parent()
        .unwrap_or(std::path::Path::new("."))
        .to_path_buf();
    let source_path = std::path::Path::new(path.as_str());

    let compile_start = Instant::now();
    let mut runtime = pixel_doodle::pdc::pipeline_runtime::PdcRuntime::new(
        &src,
        Some(source_path),
        WIDTH,
        HEIGHT,
        &base_dir,
    )
    .unwrap_or_else(|e| {
        eprintln!("PDC compile error in '{}':\n{}", path, e);
        std::process::exit(1);
    });
    let compile_ms = compile_start.elapsed().as_secs_f64() * 1000.0;
    eprintln!("[pdc] compile: {compile_ms:.1}ms");

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

    // Build thread pool and move it into the runtime for parallel kernel dispatch
    let thread_pool = args.threads.map(|n| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .expect("failed to create thread pool")
    });
    runtime.set_thread_pool(thread_pool);

    // Output-only mode
    if let Some(ref output_path) = args.output {
        if !args.bench {
            if runtime.has_gpu_kernels {
                runtime.execute_gpu_headless();
            } else {
                runtime.execute_init_block(&None);
                runtime.execute_frame(0.0, &None);
            }
            bench::write_ppm(output_path, runtime.display_pixels(), runtime.width, runtime.height);
            eprintln!("Wrote {output_path}");
            return;
        }
    }

    // Benchmark mode
    if args.bench {
        runtime.execute_init_block(&None);
        let warmup = 5;
        let frames = args.bench_frames;
        eprintln!("[bench] warmup {} frames...", warmup);
        for i in 0..warmup {
            runtime.execute_frame(i as f64 * 0.016, &None);
        }
        eprintln!("[bench] timing {} frames...", frames);
        let mut times = Vec::with_capacity(frames as usize);
        for i in 0..frames {
            let t0 = Instant::now();
            runtime.execute_frame((warmup + i) as f64 * 0.016, &None);
            times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        if let Some(ref output_path) = args.output {
            bench::write_ppm(output_path, runtime.display_pixels(), runtime.width, runtime.height);
        }
        let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean = times.iter().sum::<f64>() / times.len() as f64;
        eprintln!("[bench] {frames} frames: min={min:.2}ms max={max:.2}ms mean={mean:.2}ms");
        return;
    }

    // Interactive mode
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Wait);
    let mut app = PdcApp {
        runtime,
        window: None,
        display: None,
        keys_down: Vec::new(),
        start_time: Instant::now(),
    };
    event_loop.run_app(&mut app).unwrap();
}

#[cfg(any(feature = "cranelift-backend", feature = "llvm-backend"))]
struct PdcApp {
    runtime: pixel_doodle::pdc::pipeline_runtime::PdcRuntime,
    window: Option<Arc<Window>>,
    display: Option<Display>,
    keys_down: Vec<KeyCode>,
    start_time: Instant,
}

#[cfg(any(feature = "cranelift-backend", feature = "llvm-backend"))]
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
        self.runtime.init_gpu(&display);
        self.runtime.execute_init_block(&None);
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
                            if let Some(tag) = pixel_doodle::pdc::pipeline_runtime::key_code_to_tag(code) {
                                if self.runtime.handle_keypress(tag) {
                                    event_loop.exit();
                                    return;
                                }
                            }
                            if let Some(window) = &self.window {
                                window.request_redraw();
                            }
                        }
                    } else {
                        self.keys_down.retain(|k| *k != code);
                        if let Some(tag) = pixel_doodle::pdc::pipeline_runtime::key_code_to_tag(code) {
                            if self.runtime.handle_keyup(tag) {
                                event_loop.exit();
                                return;
                            }
                            if let Some(window) = &self.window {
                                window.request_redraw();
                            }
                        }
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
                for code in &self.keys_down.clone() {
                    if let Some(tag) = pixel_doodle::pdc::pipeline_runtime::key_code_to_tag(*code) {
                        if self.runtime.handle_keydown(tag) {
                            event_loop.exit();
                            return;
                        }
                    }
                }

                let time = self.start_time.elapsed().as_secs_f64();
                let t0 = Instant::now();
                let updated = self.runtime.execute_frame(time, &None);
                let render_ms = t0.elapsed().as_secs_f64() * 1000.0;

                if updated {
                    let display = self.display.as_ref().unwrap();
                    let gpu_rendered = self.runtime.render_gpu_frame(display);
                    if !gpu_rendered {
                        display.upload_and_present(self.runtime.display_pixels());
                    }

                    if let Some(window) = &self.window {
                        let title = self.runtime.title();
                        let fps = if render_ms > 0.0 { 1000.0 / render_ms } else { 0.0 };
                        window.set_title(&format!("{title} | {render_ms:.1}ms ({fps:.0} fps)"));
                    }
                } else if self.display.is_some() {
                    let display = self.display.as_ref().unwrap();
                    if !self.runtime.re_present_gpu_frame(display) {
                        display.upload_and_present(self.runtime.display_pixels());
                    }
                }

                if self.runtime.needs_continuous_redraw() || !self.keys_down.is_empty() || self.runtime.mouse_down {
                    let frame_time = t0.elapsed();
                    let min_frame = std::time::Duration::from_micros(8333);
                    if frame_time < min_frame {
                        std::thread::sleep(min_frame - frame_time);
                    }
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

    match &args.config_source {
        Some(source) => {
            #[cfg(any(feature = "cranelift-backend", feature = "llvm-backend"))]
            run_pdc(source, &args);
            #[cfg(not(any(feature = "cranelift-backend", feature = "llvm-backend")))]
            {
                let _ = source;
                eprintln!("PDC files require a JIT backend (cranelift or llvm)");
                std::process::exit(1);
            }
        }
        None => {
            eprintln!("Usage: pixel-doodle <config.pdc | directory> [options]");
            eprintln!();
            eprintln!("Run with --help for details.");
            std::process::exit(1);
        }
    }
}
