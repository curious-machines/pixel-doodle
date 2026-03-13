mod jit;
mod kernel_ir;
mod native_kernel;
mod render;

use jit::TileKernelFn;
use kernel_ir::KernelIr;
use minifb::{Key, Window, WindowOptions};
use std::time::Instant;

const WIDTH: usize = 1200;
const HEIGHT: usize = 900;
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

fn main() {
    let backend_name = parse_backend();

    let compile_start = Instant::now();

    let (kernel_fn, label): (TileKernelFn, &str) = match backend_name.as_str() {
        "native" => (native_kernel::native_mandelbrot_kernel, "native"),

        #[cfg(feature = "cranelift-backend")]
        "cranelift" => {
            let backend = jit::cranelift::CraneliftBackend;
            let ir = KernelIr::Mandelbrot { max_iter: MAX_ITER };
            let compiled = jit::JitBackend::compile(&backend, &ir);
            // Leak the compiled kernel so the function pointer lives forever.
            // We only compile once, so this is fine.
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
            let mut available = vec!["native"];
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
    };

    let compile_ms = compile_start.elapsed().as_secs_f64() * 1000.0;
    eprintln!("[{label}] compile: {compile_ms:.1}ms");

    let mut buffer = vec![0u32; WIDTH * HEIGHT];

    let mut window = Window::new(
        &format!("pixel-doodle [{label}]"),
        WIDTH,
        HEIGHT,
        WindowOptions::default(),
    )
    .expect("failed to create window");

    window.set_target_fps(60);

    let mut center_x = -0.5;
    let mut center_y = 0.0;
    let mut zoom = 1.0;
    let mut needs_render = true;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let pan_speed = 0.05 / zoom;
        let zoom_factor = 1.05;
        let mut moved = false;

        if window.is_key_down(Key::Left) {
            center_x -= pan_speed;
            moved = true;
        }
        if window.is_key_down(Key::Right) {
            center_x += pan_speed;
            moved = true;
        }
        if window.is_key_down(Key::Up) {
            center_y -= pan_speed;
            moved = true;
        }
        if window.is_key_down(Key::Down) {
            center_y += pan_speed;
            moved = true;
        }
        if window.is_key_down(Key::Equal) || window.is_key_down(Key::NumPadPlus) {
            zoom *= zoom_factor;
            moved = true;
        }
        if window.is_key_down(Key::Minus) || window.is_key_down(Key::NumPadMinus) {
            zoom /= zoom_factor;
            moved = true;
        }

        if moved || needs_render {
            let t0 = Instant::now();
            render::render(&mut buffer, WIDTH, HEIGHT, center_x, center_y, zoom, kernel_fn);
            let render_ms = t0.elapsed().as_secs_f64() * 1000.0;
            window.set_title(&format!(
                "{label} | {render_ms:.1}ms | {zoom:.1}x | compile {compile_ms:.1}ms"
            ));
            needs_render = false;
        }

        window
            .update_with_buffer(&buffer, WIDTH, HEIGHT)
            .expect("failed to update window");
    }
}
