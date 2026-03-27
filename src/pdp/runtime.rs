use std::collections::HashMap;
use std::path::Path;

use winit::keyboard::KeyCode;

use crate::display::Display;
use crate::gpu::GpuBackend;
use crate::gpu::sim_runner::GpuSimRunner;
use crate::jit::{self, SimTileKernelFn, TileKernelFn, UserArgSlot};
use crate::kernel_ir::Kernel;
use crate::progressive::AccumulationBuffer;
use crate::render;

use super::ast::*;

/// Resolved kernel ready for execution.
enum CompiledKernelEntry {
    Pixel {
        func: TileKernelFn,
        _compiled: Box<dyn jit::CompiledKernel>,
        user_arg_slots: Vec<UserArgSlot>,
    },
    Sim {
        func: SimTileKernelFn,
        _compiled: Box<dyn jit::CompiledSimKernel>,
        /// Read buffer slot names in order.
        read_slots: Vec<String>,
        /// Write buffer slot names in order.
        write_slots: Vec<String>,
        user_arg_slots: Vec<UserArgSlot>,
    },
    /// GPU pixel kernel — renders directly to display texture.
    Gpu {
        wgsl_source: String,
    },
    /// GPU simulation kernel — dispatched via GpuSimRunner.
    GpuSim {
        wgsl_source: String,
    },
}

/// The PDP execution engine. Holds all state for running a config-driven example.
pub struct Runtime {
    config: Config,
    kernels: HashMap<String, CompiledKernelEntry>,
    /// Simulation buffers: name -> f64 array.
    buffers: HashMap<String, Vec<f64>>,
    /// User-defined variables.
    variables: HashMap<String, f64>,
    /// Variable range constraints.
    var_ranges: HashMap<String, RangeSpec>,
    /// Pixel output buffer (ARGB).
    pub pixel_buffer: Vec<u32>,
    /// Display buffer for progressive rendering.
    display_buffer: Option<Vec<u32>>,
    /// Accumulation buffer for progressive rendering.
    accum: Option<AccumulationBuffer>,
    /// Whether accumulation state needs reset.
    accum_dirty: bool,

    // Intrinsic globals
    pub width: u32,
    pub height: u32,
    pub center_x: f64,
    pub center_y: f64,
    pub zoom: f64,
    pub mouse_x: f64,
    pub mouse_y: f64,
    pub time: f64,
    pub paused: bool,
    pub frame: u64,
    frames_executed: u64,
    pub mouse_down: bool,

    pub tile_height: usize,
    pub animated: bool,
    backend_name: String,
    /// Explicitly selected pipeline name (from settings or --set pipeline=...).
    pipeline_name: Option<String>,
    base_dir: std::path::PathBuf,
    /// GPU pixel kernel backend, initialized lazily when display is available.
    gpu_backend: Option<GpuBackend>,
    /// GPU simulation runner, initialized lazily when display is available.
    gpu_sim_runner: Option<GpuSimRunner>,
    /// Whether this config has any GPU kernels (pixel or sim).
    pub has_gpu_kernels: bool,
    /// Tracks whether GPU rendered directly to display this frame.
    gpu_rendered_this_frame: bool,
    /// Name of the GPU pixel buffer for sim display steps.
    gpu_pixel_buffer_name: Option<String>,
    /// The config file path, used in window title when no explicit title is set.
    config_path: Option<String>,
}

impl Runtime {
    pub fn new(config: Config, width: u32, height: u32, base_dir: &Path) -> Self {
        let pixel_count = (width * height) as usize;

        // Initialize variables from config
        let mut variables = HashMap::new();
        let mut var_ranges = HashMap::new();
        for v in &config.variables {
            let val = match &v.default {
                Literal::Float(f) => *f,
                Literal::Bool(b) => if *b { 1.0 } else { 0.0 },
                _ => 0.0,
            };
            variables.insert(v.name.clone(), val);
            if let Some(range) = &v.range {
                var_ranges.insert(v.name.clone(), range.clone());
            }
        }

        Runtime {
            config,
            kernels: HashMap::new(),
            buffers: HashMap::new(),
            variables,
            var_ranges,
            pixel_buffer: vec![0u32; pixel_count],
            display_buffer: None,
            accum: None,
            accum_dirty: false,
            width,
            height,
            center_x: 0.0,
            center_y: 0.0,
            zoom: 1.0,
            mouse_x: 0.0,
            mouse_y: 0.0,
            time: 0.0,
            paused: false,
            frame: 0,
            frames_executed: 0,
            mouse_down: false,
            tile_height: render::DEFAULT_TILE_HEIGHT,
            animated: false,
            backend_name: "cranelift".into(),
            pipeline_name: None,
            base_dir: base_dir.to_path_buf(),
            gpu_backend: None,
            gpu_sim_runner: None,
            has_gpu_kernels: false,
            gpu_rendered_this_frame: false,
            gpu_pixel_buffer_name: None,
            config_path: None,
        }
    }

    /// Set the config file path (used in window title fallback).
    pub fn set_config_path(&mut self, path: &str) {
        self.config_path = Some(path.to_string());
    }

    /// Apply settings from the config's settings block.
    pub fn apply_settings(&mut self) {
        for entry in &self.config.settings.entries {
            match entry.key.as_str() {
                "threads" => { /* handled externally */ }
                "backend" => {
                    if let Literal::Str(s) = &entry.value {
                        self.backend_name = s.clone();
                    }
                }
                "pipeline" => {
                    if let Literal::Str(s) = &entry.value {
                        self.pipeline_name = Some(s.clone());
                    }
                }
                "tile_height" => {
                    if let Literal::Float(v) = &entry.value {
                        self.tile_height = *v as usize;
                    }
                }
                _ => {
                    eprintln!("warning: unknown setting '{}'", entry.key);
                }
            }
        }
    }

    /// Apply overrides from --set key=value pairs.
    pub fn apply_overrides(&mut self, overrides: &[(String, String)]) {
        for (key, value) in overrides {
            match key.as_str() {
                "width" => {
                    if let Ok(v) = value.parse::<u32>() {
                        self.width = v;
                        self.pixel_buffer = vec![0u32; (self.width * self.height) as usize];
                    }
                }
                "height" => {
                    if let Ok(v) = value.parse::<u32>() {
                        self.height = v;
                        self.pixel_buffer = vec![0u32; (self.width * self.height) as usize];
                    }
                }
                "backend" => self.backend_name = value.clone(),
                "pipeline" => self.pipeline_name = Some(value.clone()),
                "tile_height" => {
                    if let Ok(v) = value.parse::<usize>() {
                        self.tile_height = v;
                    }
                }
                _ => {
                    // Try setting a variable
                    if let Ok(v) = value.parse::<f64>() {
                        self.variables.insert(key.clone(), v);
                    }
                }
            }
        }
    }

    /// Get the configured thread count from settings.
    pub fn thread_count(&self) -> Option<usize> {
        for entry in &self.config.settings.entries {
            if entry.key == "threads" {
                if let Literal::Float(v) = &entry.value {
                    return Some(*v as usize);
                }
            }
        }
        None
    }

    /// Parse and compile all declared kernels from the selected pipeline.
    pub fn compile_kernels(&mut self) -> Result<(), String> {
        // Warn once if the requested pipeline name doesn't exist
        if let Some(ref name) = self.pipeline_name {
            let pipelines = &self.config.pipelines;
            if !pipelines.iter().any(|p| p.name.as_deref() == Some(name.as_str())) {
                let available: Vec<_> = pipelines.iter().filter_map(|p| p.name.as_deref()).collect();
                eprintln!(
                    "warning: pipeline '{}' not found; available: {:?}; using first pipeline",
                    name, available
                );
            }
        }

        let mut kernel_decls: Vec<KernelDecl> = Vec::new();
        // Collect kernel declarations from the selected pipeline
        if let Some(pipeline) = self.selected_pipeline() {
            kernel_decls.extend(pipeline.kernels.iter().cloned());
        }

        let backend_name = self.backend_name.clone();
        let base_dir = self.base_dir.clone();

        for decl in &kernel_decls {
            let path = resolve_path(&decl.path, &base_dir);

            // GPU kernels (.wgsl) — defer compilation until display is available
            if path.extension().is_some_and(|e| e == "wgsl") {
                let src = std::fs::read_to_string(&path)
                    .map_err(|e| format!("failed to read kernel '{}': {}", path.display(), e))?;
                if src.contains("params.time") {
                    self.animated = true;
                }
                // Standard kernels go through GpuSimRunner, pixel kernels through GpuBackend
                let entry = if decl.kind == KernelKind::Standard {
                    CompiledKernelEntry::GpuSim { wgsl_source: src }
                } else {
                    CompiledKernelEntry::Gpu { wgsl_source: src }
                };
                self.kernels.insert(decl.name.clone(), entry);
                self.has_gpu_kernels = true;
                continue;
            }

            let src = std::fs::read_to_string(&path)
                .map_err(|e| format!("failed to read kernel '{}': {}", path.display(), e))?;

            let ir = if path.extension().is_some_and(|e| e == "pd") {
                crate::lang::pd::parse(&src, Some(&path))
                    .map_err(|e| format!("parse error in '{}': {}", path.display(), e))?
            } else {
                crate::lang::parser::parse(&src)
                    .map_err(|e| format!("parse error in '{}': {}", path.display(), e))?
            };

            // Detect animation
            if ir.params.iter().any(|p| p.name == "time") {
                self.animated = true;
            }

            match decl.kind {
                KernelKind::Pixel => {
                    let (user_arg_slots, _) =
                        jit::compute_user_arg_layout(&ir, jit::PIXEL_BUILTINS);
                    let compiled = compile_pixel_kernel(&backend_name, &ir, &user_arg_slots)?;
                    let func = compiled.function_ptr();
                    self.kernels.insert(
                        decl.name.clone(),
                        CompiledKernelEntry::Pixel {
                            func,
                            _compiled: compiled,
                            user_arg_slots,
                        },
                    );
                }
                KernelKind::Standard => {
                    let read_slots: Vec<String> = ir
                        .buffers
                        .iter()
                        .filter(|b| !b.is_output)
                        .map(|b| b.name.clone())
                        .collect();
                    let write_slots: Vec<String> = ir
                        .buffers
                        .iter()
                        .filter(|b| b.is_output)
                        .map(|b| b.name.clone())
                        .collect();
                    let (user_arg_slots, _) =
                        jit::compute_user_arg_layout(&ir, jit::SIM_BUILTINS);
                    let compiled = compile_sim_kernel(&backend_name, &ir, &user_arg_slots)?;
                    let func = compiled.function_ptr();
                    self.kernels.insert(
                        decl.name.clone(),
                        CompiledKernelEntry::Sim {
                            func,
                            _compiled: compiled,
                            read_slots,
                            write_slots,
                            user_arg_slots,
                        },
                    );
                }
            }
        }

        // Warn if backend setting has no effect on the selected pipeline
        if self.has_gpu_kernels
            && self.backend_name != "cranelift"
            && kernel_decls.iter().all(|d| {
                resolve_path(&d.path, &base_dir)
                    .extension()
                    .is_some_and(|e| e == "wgsl")
            })
        {
            eprintln!(
                "warning: backend='{}' has no effect — selected pipeline uses only WGSL kernels",
                self.backend_name
            );
        }

        Ok(())
    }

    /// Initialize all declared buffers from the selected pipeline.
    pub fn init_buffers(&mut self) -> Result<(), String> {
        let mut buf_decls: Vec<BufferDecl> = Vec::new();
        if let Some(pipeline) = self.selected_pipeline() {
            buf_decls.extend(pipeline.buffers.iter().cloned());
        }
        let size = (self.width * self.height) as usize;

        for decl in &buf_decls {
            // Skip GPU buffers — they're initialized in init_gpu()
            if decl.gpu_type.is_some() {
                continue;
            }

            let data = match &decl.init {
                BufferInit::Constant(val) => {
                    vec![*val; size]
                }
            };
            self.buffers.insert(decl.name.clone(), data);
        }
        Ok(())
    }

    /// Select which pipeline to use.
    /// Rules:
    /// - If pipeline_name is set (via settings or --set pipeline=...), select by name
    /// - Otherwise, use the first pipeline
    fn selected_pipeline(&self) -> Option<&Pipeline> {
        let pipelines = &self.config.pipelines;
        if pipelines.is_empty() {
            return None;
        }
        if let Some(ref name) = self.pipeline_name {
            pipelines
                .iter()
                .find(|p| p.name.as_deref() == Some(name.as_str()))
                .or_else(|| pipelines.first())
        } else {
            pipelines.first()
        }
    }

    /// Check if the pipeline has an accumulate step.
    fn has_accumulate(&self) -> bool {
        if let Some(pipeline) = self.selected_pipeline() {
            has_accumulate_step(&pipeline.steps)
        } else {
            false
        }
    }

    /// Set up progressive rendering if needed.
    pub fn setup_progressive(&mut self) {
        if self.has_accumulate() {
            let pipeline = self.selected_pipeline().unwrap();
            let max = find_accumulate_samples(&pipeline.steps);
            self.display_buffer = Some(vec![0u32; (self.width * self.height) as usize]);
            self.accum = Some(AccumulationBuffer::new(
                self.width as usize,
                self.height as usize,
                max,
            ));
        }
    }

    /// Initialize the GPU backend. Must be called after the display is created.
    pub fn init_gpu(&mut self, display: &Display) {
        if !self.has_gpu_kernels {
            return;
        }

        // Check for GPU pixel kernels
        if self.gpu_backend.is_none() {
            let wgsl = self.kernels.values().find_map(|entry| {
                if let CompiledKernelEntry::Gpu { wgsl_source } = entry {
                    Some(wgsl_source.clone())
                } else {
                    None
                }
            });
            if let Some(source) = wgsl {
                self.gpu_backend = Some(GpuBackend::new(
                    display, self.width, self.height, 256, &source,
                ));
            }
        }

        // Check for GPU sim kernels — create GpuSimRunner
        let has_gpu_sim = self.kernels.values().any(|e| matches!(e, CompiledKernelEntry::GpuSim { .. }));
        if has_gpu_sim && self.gpu_sim_runner.is_none() {
            let mut runner = GpuSimRunner::new(display, self.width, self.height);

            // Add GPU buffers from the selected pipeline
            let all_buffers: Vec<BufferDecl> = self
                .selected_pipeline()
                .map(|p| p.buffers.clone())
                .unwrap_or_default();

            for buf_decl in &all_buffers {
                if let Some(gpu_type) = buf_decl.gpu_type {
                    runner.add_buffer(&buf_decl.name, gpu_type);
                    match &buf_decl.init {
                        BufferInit::Constant(val) => {
                            runner.init_buffer_constant(&buf_decl.name, *val);
                        }
                    }
                }
            }

            // Compile GPU sim pipelines
            let kernel_entries: Vec<(String, String)> = self.kernels.iter().filter_map(|(name, entry)| {
                if let CompiledKernelEntry::GpuSim { wgsl_source } = entry {
                    Some((name.clone(), wgsl_source.clone()))
                } else {
                    None
                }
            }).collect();

            for (name, source) in &kernel_entries {
                if let Err(e) = runner.add_pipeline(name, source) {
                    eprintln!("warning: failed to compile GPU pipeline '{}': {}", name, e);
                }
            }

            self.gpu_sim_runner = Some(runner);
        }
    }

    /// Execute init blocks from the selected pipeline. Call once after init_gpu().
    pub fn execute_init_block(&mut self, pool: &Option<rayon::ThreadPool>) {
        let init_steps: Vec<Vec<PipelineStep>> = self
            .selected_pipeline()
            .map(|p| {
                p.steps
                    .iter()
                    .filter_map(|s| {
                        if let PipelineStep::Init { body, .. } = s {
                            Some(body.clone())
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();

        for body in &init_steps {
            self.execute_steps(body, pool);
        }
    }

    /// Execute one frame. Returns true if pixels were updated.
    /// For GPU kernels, `display` must be provided.
    pub fn execute_frame(
        &mut self,
        time: f64,
        pool: &Option<rayon::ThreadPool>,
    ) -> bool {
        self.time = time;
        self.gpu_rendered_this_frame = false;

        // Auto-increment frame when not paused
        if !self.paused {
            self.frame += 1;
        }

        // Only execute pipeline if frame has advanced
        if self.frame <= self.frames_executed && !self.animated && self.accum.is_none() {
            return false;
        }

        // Get selected pipeline's steps (clone to avoid borrow issues)
        let steps = self.selected_pipeline().map(|p| p.steps.clone());
        if let Some(steps) = steps {
            self.execute_steps(&steps, pool);
        }

        // Ensure all GPU work completes before presenting
        if let Some(runner) = &self.gpu_sim_runner {
            runner.poll_wait();
        }

        self.frames_executed = self.frame;
        true
    }

    fn execute_steps(
        &mut self,
        steps: &[PipelineStep],
        pool: &Option<rayon::ThreadPool>,
    ) {
        for step in steps {
            match step {
                PipelineStep::Display { buffer_name, .. } => {
                    self.execute_display(buffer_name.as_deref());
                }
                PipelineStep::Run {
                    kernel_name,
                    input_bindings,
                    args,
                    ..
                } => {
                    self.execute_run(kernel_name, input_bindings, args, pool);
                }
                PipelineStep::Init { body, .. } => {
                    // Init blocks are handled separately via execute_init_block()
                    // Skip during normal frame execution
                    let _ = body;
                }
                PipelineStep::Swap { pairs, .. } => {
                    for (a, b) in pairs {
                        // Try CPU buffers first, then GPU
                        if self.buffers.contains_key(a) {
                            let mut buf_a = self.buffers.remove(a).unwrap();
                            let mut buf_b = self.buffers.remove(b).unwrap();
                            std::mem::swap(&mut buf_a, &mut buf_b);
                            self.buffers.insert(a.clone(), buf_a);
                            self.buffers.insert(b.clone(), buf_b);
                        } else if let Some(runner) = &mut self.gpu_sim_runner {
                            runner.swap_buffers(a, b);
                        }
                    }
                }
                PipelineStep::Loop { iterations, body, .. } => {
                    let n = match iterations {
                        IterCount::Fixed(n) => *n,
                        IterCount::Variable(name) => {
                            self.get_variable(name) as u32
                        }
                    };
                    for _ in 0..n {
                        self.execute_steps(body, pool);
                    }
                }
                PipelineStep::Accumulate { samples, body, .. } => {
                    self.execute_accumulate(*samples, body, pool);
                }
                PipelineStep::OnClick { continuous, body, .. } => {
                    let should_run = if *continuous {
                        self.mouse_down
                    } else {
                        self.mouse_down // TODO: edge detection for single-fire
                    };
                    if should_run {
                        self.execute_steps(body, pool);
                    }
                }
            }
        }
    }

    /// Pack user-defined argument values into a byte buffer matching the compiled layout.
    fn pack_user_args(&self, slots: &[UserArgSlot], args: &[NamedArg]) -> Vec<u8> {
        use crate::kernel_ir::{ScalarType, ValType};
        if slots.is_empty() {
            // Warn if caller passed args to a kernel that has no user params
            for arg in args {
                eprintln!("warning: argument '{}' does not match any kernel parameter", arg.name);
            }
            return Vec::new();
        }
        // Check for unknown args
        for arg in args {
            if !slots.iter().any(|s| s.name == arg.name) {
                eprintln!("warning: argument '{}' does not match any kernel parameter", arg.name);
            }
        }
        // Compute total size from last slot
        let last = slots.last().unwrap();
        let last_size = match last.ty {
            ValType::Scalar(s) => s.byte_size(),
            _ => 8,
        };
        let total = ((last.offset + last_size) + 7) & !7;
        let mut buf = vec![0u8; total];
        for slot in slots {
            let value = args.iter().find(|a| a.name == slot.name);
            let f = match value {
                Some(a) => match &a.value {
                    Literal::Float(f) => *f,
                    Literal::Int(i) => *i as f64,
                    Literal::Bool(b) => if *b { 1.0 } else { 0.0 },
                    Literal::Str(_) => 0.0,
                    Literal::VarRef(name) => self.get_variable(name),
                },
                None => panic!("missing argument '{}' for kernel", slot.name),
            };
            let o = slot.offset;
            match slot.ty {
                ValType::Scalar(ScalarType::F64) => {
                    buf[o..o + 8].copy_from_slice(&f.to_le_bytes());
                }
                ValType::Scalar(ScalarType::F32) => {
                    buf[o..o + 4].copy_from_slice(&(f as f32).to_le_bytes());
                }
                ValType::Scalar(ScalarType::I8) => {
                    buf[o..o + 1].copy_from_slice(&(f as i8).to_le_bytes());
                }
                ValType::Scalar(ScalarType::U8) => {
                    buf[o..o + 1].copy_from_slice(&(f as u8).to_le_bytes());
                }
                ValType::Scalar(ScalarType::I16) => {
                    buf[o..o + 2].copy_from_slice(&(f as i16).to_le_bytes());
                }
                ValType::Scalar(ScalarType::U16) => {
                    buf[o..o + 2].copy_from_slice(&(f as u16).to_le_bytes());
                }
                ValType::Scalar(ScalarType::I32) => {
                    buf[o..o + 4].copy_from_slice(&(f as i32).to_le_bytes());
                }
                ValType::Scalar(ScalarType::U32) => {
                    buf[o..o + 4].copy_from_slice(&(f as u32).to_le_bytes());
                }
                ValType::Scalar(ScalarType::I64) => {
                    buf[o..o + 8].copy_from_slice(&(f as i64).to_le_bytes());
                }
                ValType::Scalar(ScalarType::U64) => {
                    buf[o..o + 8].copy_from_slice(&(f as u64).to_le_bytes());
                }
                _ => panic!("unsupported user-arg type {:?}", slot.ty),
            }
        }
        buf
    }

    /// Execute a kernel via `run`.
    fn execute_run(
        &mut self,
        kernel_name: &str,
        input_bindings: &[BufferBinding],
        args: &[NamedArg],
        pool: &Option<rayon::ThreadPool>,
    ) {
        let entry = self.kernels.get(kernel_name);
        match entry {
            Some(CompiledKernelEntry::Pixel { func, user_arg_slots, .. }) => {
                let kfn = *func;
                let packed = self.pack_user_args(user_arg_slots, args);
                let args_ptr = if packed.is_empty() { std::ptr::null() } else { packed.as_ptr() };
                let args_addr = args_ptr as usize;
                let (cx, cy, z, t) = (self.center_x, self.center_y, self.zoom, self.time);
                let w = self.width as usize;
                let h = self.height as usize;
                let th = self.tile_height;
                let buf = &mut self.pixel_buffer;
                let sample_index = 0xFFFFFFFF; // non-progressive by default
                with_pool(pool, || {
                    render::render(buf, w, h, cx, cy, z, kfn, sample_index, t, th, args_addr as *const u8);
                });
            }
            Some(CompiledKernelEntry::Sim {
                func,
                read_slots,
                write_slots,
                user_arg_slots,
                ..
            }) => {
                let sim_fn = *func;
                let packed = self.pack_user_args(user_arg_slots, args);
                let args_ptr = if packed.is_empty() { std::ptr::null() } else { packed.as_ptr() };
                let w = self.width as usize;
                let h = self.height as usize;
                let th = self.tile_height;

                // Build buffer pointer arrays from bindings
                let bufs_in: Vec<*const f64> = read_slots
                    .iter()
                    .map(|slot| {
                        let buf_name = find_binding(input_bindings, slot)
                            .unwrap_or_else(|| slot.clone());
                        self.buffers
                            .get(&buf_name)
                            .unwrap_or_else(|| panic!("buffer '{}' not found", buf_name))
                            .as_ptr()
                    })
                    .collect();

                let bufs_out: Vec<*mut f64> = write_slots
                    .iter()
                    .map(|slot| {
                        let buf_name = find_binding(input_bindings, slot)
                            .unwrap_or_else(|| slot.clone());
                        self.buffers
                            .get_mut(&buf_name)
                            .unwrap_or_else(|| panic!("buffer '{}' not found", buf_name))
                            .as_mut_ptr()
                    })
                    .collect();

                let pixel_buf = &mut self.pixel_buffer;
                render::render_sim(pixel_buf, w, h, sim_fn, &bufs_in, &bufs_out, th, args_ptr);
            }
            Some(CompiledKernelEntry::Gpu { .. }) => {
                // GPU pixel kernels render directly to the display texture.
                self.gpu_rendered_this_frame = true;
            }
            Some(CompiledKernelEntry::GpuSim { .. }) => {
                // GPU sim kernel — dispatch via GpuSimRunner
                if let Some(runner) = &self.gpu_sim_runner {
                    let bindings: Vec<(&str, &str)> = input_bindings
                        .iter()
                        .map(|b| (b.param_name.as_str(), b.buffer_name.as_str()))
                        .collect();
                    runner.dispatch(kernel_name, &bindings);
                }
            }
            None => {
                eprintln!("warning: kernel '{}' not found", kernel_name);
            }
        }
    }

    /// Present pixels to screen.
    fn execute_display(&mut self, buffer_name: Option<&str>) {
        match buffer_name {
            Some(name) => {
                // GPU display — present named buffer
                self.gpu_rendered_this_frame = true;
                self.gpu_pixel_buffer_name = Some(name.to_string());
            }
            None => {
                // CPU display — pixel buffer already written by run steps.
                // GPU pixel kernel — mark as rendered.
                if self.has_gpu_kernels && self.gpu_sim_runner.is_none() {
                    self.gpu_rendered_this_frame = true;
                }
            }
        }
    }

    fn execute_accumulate(
        &mut self,
        max_samples: u32,
        body: &[PipelineStep],
        pool: &Option<rayon::ThreadPool>,
    ) {
        // Initialize accumulation on first call
        if self.accum.is_none() {
            self.accum = Some(AccumulationBuffer::new(
                self.width as usize,
                self.height as usize,
                max_samples,
            ));
            self.display_buffer = Some(vec![0u32; (self.width * self.height) as usize]);
        }

        if self.accum_dirty {
            self.accum.as_mut().unwrap().reset();
            self.accum_dirty = false;
        }

        let accum = self.accum.as_mut().unwrap();
        if accum.is_converged() {
            // Already converged — just present the display buffer
            return;
        }

        let sample_index = accum.sample_count;

        // For accumulate, we need to run pixel kernels with the sample index
        // Override the pixel rendering to use progressive mode
        {
            // Find the run step for a pixel kernel inside body and run it with sample_index
            for step in body {
                match step {
                    PipelineStep::Run { kernel_name, args, .. } => {
                        if let Some(CompiledKernelEntry::Pixel { func, user_arg_slots, .. }) =
                            self.kernels.get(kernel_name)
                        {
                            let kfn = *func;
                            let packed = self.pack_user_args(user_arg_slots, args);
                            let args_ptr = if packed.is_empty() { std::ptr::null() } else { packed.as_ptr() };
                            let args_addr = args_ptr as usize;
                            let (cx, cy, z, t) =
                                (self.center_x, self.center_y, self.zoom, self.time);
                            let w = self.width as usize;
                            let h = self.height as usize;
                            let th = self.tile_height;
                            let buf = &mut self.pixel_buffer;
                            with_pool(pool, || {
                                render::render(buf, w, h, cx, cy, z, kfn, sample_index, t, th, args_addr as *const u8);
                            });
                        }
                    }
                    other => {
                        self.execute_steps(&[other.clone()], pool);
                    }
                }
            }
        }

        let accum = self.accum.as_mut().unwrap();
        accum.accumulate(&self.pixel_buffer);
        let disp = self.display_buffer.as_mut().unwrap();
        accum.resolve(disp);
    }

    /// Handle a key press event. Returns `true` if the app should quit.
    pub fn handle_key_press(&mut self, key_name: &str) -> bool {
        let bindings: Vec<KeyBinding> = self.config.key_bindings.clone();
        let mut quit = false;
        for binding in &bindings {
            if binding.key_name == key_name {
                for action in &binding.actions {
                    match action {
                        Action::Toggle(var) => {
                            let val = self.get_variable(var);
                            self.set_variable(var, if val == 0.0 { 1.0 } else { 0.0 });
                        }
                        Action::CompoundAssign { target, op, value } | Action::BinAssign { target, op, value } => {
                            let current = self.get_variable(target);
                            let val = self.eval_value_expr(value);
                            let new_val = match op {
                                CompoundOp::Add => current + val,
                                CompoundOp::Sub => current - val,
                                CompoundOp::Mul => current * val,
                                CompoundOp::Div => {
                                    if val != 0.0 {
                                        current / val
                                    } else {
                                        current
                                    }
                                }
                            };
                            self.set_variable(target, new_val);
                        }
                        Action::Assign { target, value } => {
                            self.set_variable(target, *value);
                        }
                        Action::Quit => {
                            quit = true;
                        }
                    }
                }
            }
        }
        quit
    }

    fn eval_value_expr(&self, expr: &ValueExpr) -> f64 {
        match expr {
            ValueExpr::Literal(v) => *v,
            ValueExpr::BinOp { left, op, right } => {
                let rhs = self.get_variable(right);
                match op {
                    CompoundOp::Add => left + rhs,
                    CompoundOp::Sub => left - rhs,
                    CompoundOp::Mul => left * rhs,
                    CompoundOp::Div => if rhs != 0.0 { left / rhs } else { *left },
                }
            }
        }
    }

    fn get_variable(&self, name: &str) -> f64 {
        // Check user variables first, then intrinsics
        if let Some(v) = self.variables.get(name) {
            return *v;
        }
        match name {
            "center_x" => self.center_x,
            "center_y" => self.center_y,
            "zoom" => self.zoom,
            "mouse_x" => self.mouse_x,
            "mouse_y" => self.mouse_y,
            "time" => self.time,
            "paused" => if self.paused { 1.0 } else { 0.0 },
            "frame" => self.frame as f64,
            "width" => self.width as f64,
            "height" => self.height as f64,
            _ => 0.0,
        }
    }

    fn set_variable(&mut self, name: &str, value: f64) {
        // Apply range clamping/wrapping
        let clamped = if let Some(range) = self.var_ranges.get(name) {
            if range.wrap {
                let span = range.max - range.min;
                if span > 0.0 {
                    let mut v = value;
                    while v > range.max {
                        v -= span;
                    }
                    while v < range.min {
                        v += span;
                    }
                    v
                } else {
                    value
                }
            } else {
                value.clamp(range.min, range.max)
            }
        } else {
            value
        };

        // Set intrinsic or user variable
        match name {
            "center_x" => {
                self.center_x = clamped;
                self.accum_dirty = true;
            }
            "center_y" => {
                self.center_y = clamped;
                self.accum_dirty = true;
            }
            "zoom" => {
                self.zoom = clamped;
                self.accum_dirty = true;
            }
            "paused" => self.paused = clamped != 0.0,
            "frame" => self.frame = clamped as u64,
            _ => {
                self.variables.insert(name.to_string(), clamped);
            }
        }
    }

    /// Get the pixel buffer to display.
    /// If the last frame used GPU rendering, dispatch the GPU and present.
    /// Returns true if GPU rendered (caller should NOT upload_and_present).
    pub fn render_gpu_frame(&self, display: &Display) -> bool {
        if !self.gpu_rendered_this_frame {
            return false;
        }
        // GPU pixel kernel path
        if let Some(ref gpu) = self.gpu_backend {
            gpu.render(
                display,
                self.center_x,
                self.center_y,
                self.zoom,
                0xFFFFFFFF,
                0,
                self.time,
            );
            return true;
        }
        // GPU sim kernel path — present the pixel buffer from the runner
        if let Some(ref runner) = self.gpu_sim_runner {
            if let Some(ref buf_name) = self.gpu_pixel_buffer_name {
                runner.present_pixels(display, buf_name);
                return true;
            }
        }
        false
    }

    /// For headless GPU rendering (--output, --bench): render to CPU pixel buffer.
    pub fn execute_gpu_headless(&mut self) {
        if !self.has_gpu_kernels {
            return;
        }
        // Find the WGSL source
        let wgsl = self.kernels.values().find_map(|entry| {
            if let CompiledKernelEntry::Gpu { wgsl_source } = entry {
                Some(wgsl_source.clone())
            } else {
                None
            }
        });
        if let Some(source) = wgsl {
            let (gpu, device, queue) = GpuBackend::new_headless(
                self.width, self.height, 256, &source,
            );
            gpu.dispatch_compute(
                &device, &queue,
                self.center_x, self.center_y, self.zoom,
                0xFFFFFFFF, 0, self.time,
            );
            self.pixel_buffer = gpu.readback_pixels(&device, &queue);
        }
    }

    pub fn display_pixels(&self) -> &[u32] {
        if let Some(ref disp) = self.display_buffer {
            if self.accum.is_some() {
                return disp;
            }
        }
        &self.pixel_buffer
    }

    pub fn title(&self) -> String {
        if let Some(ref t) = self.config.title {
            return t.clone();
        }
        if let Some(ref path) = self.config_path {
            let p = std::path::Path::new(path);
            let name = p.file_stem().and_then(|s| s.to_str()).unwrap_or("pixel-doodle");
            return format!("pixel-doodle — {}", name);
        }
        "pixel-doodle".to_string()
    }

    /// Returns true if progressive rendering is active and not yet converged.
    pub fn needs_continuous_redraw(&self) -> bool {
        if self.animated {
            return true;
        }
        if let Some(accum) = &self.accum {
            return !accum.is_converged();
        }
        // Simulations always animate (they have buffers/loops)
        if !self.buffers.is_empty() {
            return true;
        }
        // GPU simulations have buffers in the GPU runner, not self.buffers
        self.gpu_sim_runner.is_some()
    }

    pub fn accumulation_info(&self) -> Option<(u32, u32)> {
        self.accum
            .as_ref()
            .map(|a| (a.sample_count, a.max_samples))
    }
}

// ── Helper functions ──

fn with_pool<F: FnOnce() + Send>(pool: &Option<rayon::ThreadPool>, f: F) {
    match pool {
        Some(pool) => pool.install(f),
        None => f(),
    }
}

fn resolve_path(path: &str, base_dir: &Path) -> std::path::PathBuf {
    if path.starts_with("@root/") {
        // Find project root by looking for Cargo.toml
        let mut dir = base_dir.to_path_buf();
        loop {
            if dir.join("Cargo.toml").exists() {
                return dir.join(&path[6..]);
            }
            if !dir.pop() {
                // Fallback: use base_dir
                return base_dir.join(&path[6..]);
            }
        }
    } else {
        base_dir.join(path)
    }
}

fn find_binding(bindings: &[BufferBinding], slot_name: &str) -> Option<String> {
    bindings
        .iter()
        .find(|b| b.param_name == slot_name)
        .map(|b| b.buffer_name.clone())
}

fn has_accumulate_step(steps: &[PipelineStep]) -> bool {
    steps.iter().any(|s| matches!(s, PipelineStep::Accumulate { .. }))
}

fn find_accumulate_samples(steps: &[PipelineStep]) -> u32 {
    for step in steps {
        if let PipelineStep::Accumulate { samples, .. } = step {
            return *samples;
        }
    }
    256
}

fn compile_pixel_kernel(
    backend_name: &str,
    kernel: &Kernel,
    user_args: &[UserArgSlot],
) -> Result<Box<dyn jit::CompiledKernel>, String> {
    match backend_name {
        #[cfg(feature = "cranelift-backend")]
        "cranelift" => {
            let backend = jit::cranelift::CraneliftBackend;
            Ok(jit::JitBackend::compile(&backend, kernel, user_args))
        }
        #[cfg(feature = "llvm-backend")]
        "llvm" => {
            let backend = jit::llvm::LlvmBackend;
            Ok(jit::JitBackend::compile(&backend, kernel, user_args))
        }
        other => Err(format!("unknown backend '{}'", other)),
    }
}

fn compile_sim_kernel(
    backend_name: &str,
    kernel: &Kernel,
    user_args: &[UserArgSlot],
) -> Result<Box<dyn jit::CompiledSimKernel>, String> {
    match backend_name {
        #[cfg(feature = "cranelift-backend")]
        "cranelift" => {
            let backend = jit::cranelift::CraneliftBackend;
            Ok(jit::JitBackend::compile_sim(&backend, kernel, user_args))
        }
        #[cfg(feature = "llvm-backend")]
        "llvm" => {
            let backend = jit::llvm::LlvmBackend;
            Ok(jit::JitBackend::compile_sim(&backend, kernel, user_args))
        }
        other => Err(format!("unknown backend '{}'", other)),
    }
}

/// Map a winit KeyCode to the string name used in .pdp files.
pub fn key_code_to_name(code: KeyCode) -> Option<&'static str> {
    Some(match code {
        KeyCode::Space => "space",
        KeyCode::Period => "period",
        KeyCode::Comma => "comma",
        KeyCode::ArrowLeft => "left",
        KeyCode::ArrowRight => "right",
        KeyCode::ArrowUp => "up",
        KeyCode::ArrowDown => "down",
        KeyCode::Equal | KeyCode::NumpadAdd => "plus",
        KeyCode::Minus | KeyCode::NumpadSubtract => "minus",
        KeyCode::BracketLeft => "bracket_left",
        KeyCode::BracketRight => "bracket_right",
        KeyCode::Digit0 | KeyCode::Numpad0 => "0",
        KeyCode::Digit1 | KeyCode::Numpad1 => "1",
        KeyCode::Digit2 | KeyCode::Numpad2 => "2",
        KeyCode::Digit3 | KeyCode::Numpad3 => "3",
        KeyCode::KeyR => "r",
        KeyCode::Escape => "escape",
        KeyCode::KeyQ => "q",
        _ => return None,
    })
}
