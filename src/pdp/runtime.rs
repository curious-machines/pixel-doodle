use std::collections::HashMap;
use std::path::Path;

use winit::keyboard::KeyCode;

use crate::display::Display;
use crate::gpu::GpuBackend;
use crate::gpu::sim_runner::GpuSimRunner;
use crate::jit::{self, SimTileKernelFn, TileKernelFn};
use crate::kernel_ir::Kernel;
use crate::progressive::AccumulationBuffer;
use crate::render;

use super::ast::*;

/// Resolved kernel ready for execution.
enum CompiledKernelEntry {
    Pixel {
        func: TileKernelFn,
        _compiled: Box<dyn jit::CompiledKernel>,
    },
    Sim {
        func: SimTileKernelFn,
        _compiled: Box<dyn jit::CompiledSimKernel>,
        /// Read buffer slot names in order.
        read_slots: Vec<String>,
        /// Write buffer slot names in order.
        write_slots: Vec<String>,
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
    /// Parsed (but not compiled) kernel IR, keyed by kernel name.
    kernel_ir: HashMap<String, Kernel>,
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
            kernel_ir: HashMap::new(),
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

    /// Parse and compile all declared kernels (top-level + selected pipeline).
    pub fn compile_kernels(&mut self) -> Result<(), String> {
        // Collect kernel declarations from top-level AND selected pipeline
        let mut kernel_decls: Vec<KernelDecl> = Vec::new();
        // Find the selected pipeline's kernels
        let pipelines = self.config.pipelines.clone();
        let selected_name = if pipelines.len() == 1 {
            pipelines[0].name.clone()
        } else if self.backend_name == "gpu" {
            Some("gpu".to_string())
        } else {
            Some("cpu".to_string())
        };
        for p in &pipelines {
            if pipelines.len() == 1 || p.name == selected_name {
                kernel_decls.extend(p.kernels.iter().cloned());
            }
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
                // Sim kernels go through GpuSimRunner, pixel kernels through GpuBackend
                let entry = if decl.kind == KernelKind::Sim {
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
                    let compiled = compile_pixel_kernel(&backend_name, &ir)?;
                    let func = compiled.function_ptr();
                    self.kernels.insert(
                        decl.name.clone(),
                        CompiledKernelEntry::Pixel {
                            func,
                            _compiled: compiled,
                        },
                    );
                }
                KernelKind::Sim => {
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
                    let compiled = compile_sim_kernel(&backend_name, &ir)?;
                    let func = compiled.function_ptr();
                    self.kernels.insert(
                        decl.name.clone(),
                        CompiledKernelEntry::Sim {
                            func,
                            _compiled: compiled,
                            read_slots,
                            write_slots,
                        },
                    );
                }
                KernelKind::Init => {
                    // Store IR for later use during buffer init
                    self.kernel_ir.insert(decl.name.clone(), ir);
                }
            }
        }
        Ok(())
    }

    /// Initialize all declared buffers (top-level + selected pipeline).
    pub fn init_buffers(&mut self) -> Result<(), String> {
        let mut buf_decls: Vec<BufferDecl> = Vec::new();
        // Add pipeline-scoped buffers
        let pipelines = self.config.pipelines.clone();
        let selected_name = if pipelines.len() == 1 {
            pipelines[0].name.clone()
        } else if self.backend_name == "gpu" {
            Some("gpu".to_string())
        } else {
            Some("cpu".to_string())
        };
        for p in &pipelines {
            if pipelines.len() == 1 || p.name == selected_name {
                buf_decls.extend(p.buffers.iter().cloned());
            }
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
                BufferInit::InitKernel { kernel_name, .. } => {
                    // Compile and run the init kernel on CPU
                    let ir = self
                        .kernel_ir
                        .get(kernel_name)
                        .ok_or_else(|| format!("init kernel '{}' not found", kernel_name))?;

                    // Init kernels always compile with a CPU backend
                    let cpu_backend = if self.backend_name == "gpu" { "cranelift" } else { &self.backend_name };
                    let compiled = compile_sim_kernel(cpu_backend, ir)?;
                    let func = compiled.function_ptr();

                    let mut output_buf = vec![0.0f64; size];
                    let mut pixel_buf = vec![0u32; size];
                    let bufs_in: &[*const f64] = &[];
                    let bufs_out: &[*mut f64] = &[output_buf.as_mut_ptr()];

                    render::render_sim(
                        &mut pixel_buf,
                        self.width as usize,
                        self.height as usize,
                        func,
                        bufs_in,
                        bufs_out,
                        self.tile_height,
                    );
                    output_buf
                }
            };
            self.buffers.insert(decl.name.clone(), data);
        }
        Ok(())
    }

    /// Select which pipeline to use based on backend setting.
    /// Rules:
    /// - One pipeline (named or unnamed): use it
    /// - Multiple pipelines: backend=gpu selects "gpu", otherwise selects "cpu"
    fn selected_pipeline(&self) -> Option<&Pipeline> {
        let pipelines = &self.config.pipelines;
        if pipelines.is_empty() {
            return None;
        }
        if pipelines.len() == 1 {
            return Some(&pipelines[0]);
        }
        // Multiple pipelines — select by backend
        let target = if self.backend_name == "gpu" { "gpu" } else { "cpu" };
        pipelines
            .iter()
            .find(|p| p.name.as_deref() == Some(target))
            .or_else(|| pipelines.first())
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
                    runner.add_buffer(&buf_decl.name, gpu_type.byte_size());
                    match &buf_decl.init {
                        BufferInit::Constant(val) => {
                            runner.init_buffer_constant(&buf_decl.name, *val);
                        }
                        BufferInit::InitKernel { kernel_name, .. } => {
                            // Run init kernel on CPU, then upload to GPU
                            if let Some(ir) = self.kernel_ir.get(kernel_name) {
                                let cpu_backend = if self.backend_name == "gpu" {
                                    "cranelift"
                                } else {
                                    &self.backend_name
                                };
                                if let Ok(compiled) = compile_sim_kernel(cpu_backend, ir) {
                                    let func = compiled.function_ptr();
                                    let size = (self.width * self.height) as usize;
                                    let mut output_buf = vec![0.0f64; size];
                                    let mut pixel_buf = vec![0u32; size];
                                    let bufs_in: &[*const f64] = &[];
                                    let bufs_out: &[*mut f64] = &[output_buf.as_mut_ptr()];
                                    render::render_sim(
                                        &mut pixel_buf,
                                        self.width as usize,
                                        self.height as usize,
                                        func,
                                        bufs_in,
                                        bufs_out,
                                        self.tile_height,
                                    );
                                    runner.upload_f64_data(&buf_decl.name, &output_buf);
                                    eprintln!(
                                        "[gpu] initialized buffer '{}' via init kernel '{}'",
                                        buf_decl.name, kernel_name
                                    );
                                }
                            }
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
                PipelineStep::Display {
                    outputs,
                    kernel_name,
                    input_bindings,
                    args,
                    ..
                } => {
                    self.execute_run_or_display(kernel_name, outputs, input_bindings, args, pool, true);
                }
                PipelineStep::Run {
                    outputs,
                    kernel_name,
                    input_bindings,
                    args,
                    ..
                } => {
                    self.execute_run_or_display(kernel_name, outputs, input_bindings, args, pool, false);
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

    fn execute_run_or_display(
        &mut self,
        kernel_name: &str,
        outputs: &[String],
        input_bindings: &[BufferBinding],
        args: &[NamedArg],
        pool: &Option<rayon::ThreadPool>,
        is_display: bool,
    ) {
        // Handle built-in inject
        if kernel_name == "inject" {
            self.execute_inject(outputs, args);
            return;
        }

        let entry = self.kernels.get(kernel_name);
        match entry {
            Some(CompiledKernelEntry::Pixel { func, .. }) => {
                let kfn = *func;
                let (cx, cy, z, t) = (self.center_x, self.center_y, self.zoom, self.time);
                let w = self.width as usize;
                let h = self.height as usize;
                let th = self.tile_height;
                let buf = &mut self.pixel_buffer;
                let sample_index = 0xFFFFFFFF; // non-progressive by default
                with_pool(pool, || {
                    render::render(buf, w, h, cx, cy, z, kfn, sample_index, t, th);
                });
            }
            Some(CompiledKernelEntry::Sim {
                func,
                read_slots,
                write_slots,
                ..
            }) => {
                let sim_fn = *func;
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

                // For outputs: use the tuple assignment names in order
                let bufs_out: Vec<*mut f64> = if !outputs.is_empty() {
                    outputs
                        .iter()
                        .map(|name| {
                            self.buffers
                                .get_mut(name)
                                .unwrap_or_else(|| panic!("buffer '{}' not found", name))
                                .as_mut_ptr()
                        })
                        .collect()
                } else {
                    write_slots
                        .iter()
                        .map(|slot| {
                            let buf_name = find_binding(input_bindings, slot)
                                .unwrap_or_else(|| slot.clone());
                            self.buffers
                                .get_mut(&buf_name)
                                .unwrap_or_else(|| panic!("buffer '{}' not found", buf_name))
                                .as_mut_ptr()
                        })
                        .collect()
                };

                let pixel_buf = &mut self.pixel_buffer;
                render::render_sim(pixel_buf, w, h, sim_fn, &bufs_in, &bufs_out, th);
            }
            Some(CompiledKernelEntry::Gpu { .. }) => {
                // GPU pixel kernels render directly to the display texture.
                self.gpu_rendered_this_frame = true;
            }
            Some(CompiledKernelEntry::GpuSim { .. }) => {
                // GPU sim kernel — dispatch via GpuSimRunner
                if let Some(runner) = &self.gpu_sim_runner {
                    let mut bindings: Vec<(&str, &str)> = Vec::new();
                    for binding in input_bindings {
                        bindings.push((&binding.param_name, &binding.buffer_name));
                    }
                    // Output buffer names map directly to WGSL var names
                    for out_name in outputs {
                        bindings.push((out_name, out_name));
                    }
                    runner.dispatch(kernel_name, &bindings);
                }
                if is_display {
                    self.gpu_rendered_this_frame = true;
                    if let Some(out) = outputs.first() {
                        self.gpu_pixel_buffer_name = Some(out.clone());
                    }
                }
            }
            None => {
                if is_display {
                    eprintln!("warning: kernel '{}' not found for display", kernel_name);
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
            // Find the display step inside body and run it with sample_index
            for step in body {
                match step {
                    PipelineStep::Display { kernel_name, .. } => {
                        if let Some(CompiledKernelEntry::Pixel { func, .. }) =
                            self.kernels.get(kernel_name)
                        {
                            let kfn = *func;
                            let (cx, cy, z, t) =
                                (self.center_x, self.center_y, self.zoom, self.time);
                            let w = self.width as usize;
                            let h = self.height as usize;
                            let th = self.tile_height;
                            let buf = &mut self.pixel_buffer;
                            with_pool(pool, || {
                                render::render(buf, w, h, cx, cy, z, kfn, sample_index, t, th);
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

    /// Built-in inject: write a value into a buffer around the mouse position.
    fn execute_inject(&mut self, outputs: &[String], args: &[NamedArg]) {
        if outputs.is_empty() {
            return;
        }
        let buf_name = &outputs[0];

        let mut value = 0.0f64;
        let mut radius = 5.0f64;
        let mut falloff = "flat";
        for arg in args {
            match arg.name.as_str() {
                "value" => {
                    if let Literal::Float(v) = &arg.value {
                        value = *v;
                    }
                }
                "radius" => {
                    if let Literal::Float(v) = &arg.value {
                        radius = *v;
                    }
                }
                "falloff" => {
                    // falloff is passed as identifier, stored as string
                }
                _ => {}
            }
        }
        // Check for "falloff" in args — it might be stored differently
        // For now, check the raw string representation
        let use_quadratic = args.iter().any(|a| {
            a.name == "falloff" && matches!(&a.value, Literal::Str(s) if s == "quadratic")
        });
        if use_quadratic {
            falloff = "quadratic";
        }

        let w = self.width as usize;
        let h = self.height as usize;
        let mx = self.mouse_x as isize;
        let my = self.mouse_y as isize;
        let r = radius as isize;

        // Check if this is a GPU buffer
        if let Some(runner) = &self.gpu_sim_runner {
            if !self.buffers.contains_key(buf_name) {
                // GPU buffer — inject via runner
                runner.inject_value(
                    buf_name,
                    mx.max(0) as u32,
                    my.max(0) as u32,
                    r as u32,
                    value,
                );
                return;
            }
        }

        if let Some(buf) = self.buffers.get_mut(buf_name) {
            for dy in -r..=r {
                for dx in -r..=r {
                    let px = mx + dx;
                    let py = my + dy;
                    if px >= 0 && px < w as isize && py >= 0 && py < h as isize {
                        let d2 = (dx * dx + dy * dy) as f64;
                        let r2 = (r * r) as f64;
                        if d2 <= r2 {
                            let idx = py as usize * w + px as usize;
                            if falloff == "quadratic" {
                                let factor = 1.0 - d2 / r2;
                                buf[idx] += value * factor;
                            } else {
                                buf[idx] = value;
                            }
                        }
                    }
                }
            }
        }
    }

    /// Handle a key press event.
    pub fn handle_key_press(&mut self, key_name: &str) {
        let bindings: Vec<KeyBinding> = self.config.key_bindings.clone();
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
                            let new_val = match op {
                                CompoundOp::Add => current + value,
                                CompoundOp::Sub => current - value,
                                CompoundOp::Mul => current * value,
                                CompoundOp::Div => {
                                    if *value != 0.0 {
                                        current / value
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
                    }
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
        // Simulations always animate (they have buffers)
        !self.buffers.is_empty()
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
) -> Result<Box<dyn jit::CompiledKernel>, String> {
    match backend_name {
        #[cfg(feature = "cranelift-backend")]
        "cranelift" => {
            let backend = jit::cranelift::CraneliftBackend;
            Ok(jit::JitBackend::compile(&backend, kernel))
        }
        #[cfg(feature = "llvm-backend")]
        "llvm" => {
            let backend = jit::llvm::LlvmBackend;
            Ok(jit::JitBackend::compile(&backend, kernel))
        }
        other => Err(format!("unknown backend '{}'", other)),
    }
}

fn compile_sim_kernel(
    backend_name: &str,
    kernel: &Kernel,
) -> Result<Box<dyn jit::CompiledSimKernel>, String> {
    match backend_name {
        #[cfg(feature = "cranelift-backend")]
        "cranelift" => {
            let backend = jit::cranelift::CraneliftBackend;
            Ok(jit::JitBackend::compile_sim(&backend, kernel))
        }
        #[cfg(feature = "llvm-backend")]
        "llvm" => {
            let backend = jit::llvm::LlvmBackend;
            Ok(jit::JitBackend::compile_sim(&backend, kernel))
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
        KeyCode::Digit0 | KeyCode::Numpad0 => "digit0",
        KeyCode::Digit1 | KeyCode::Numpad1 => "digit1",
        KeyCode::Digit2 | KeyCode::Numpad2 => "digit2",
        KeyCode::Digit3 | KeyCode::Numpad3 => "digit3",
        KeyCode::KeyR => "r",
        KeyCode::Escape => "escape",
        KeyCode::KeyQ => "q",
        _ => return None,
    })
}
