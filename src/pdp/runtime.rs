use std::collections::HashMap;
use std::path::Path;

use winit::keyboard::KeyCode;

use crate::display::Display;
use crate::gpu::GpuBackend;
use crate::gpu::sim_runner::GpuSimRunner;
use crate::jit::{self, TextureSlot};
#[cfg(any(feature = "cranelift-backend", feature = "llvm-backend"))]
use crate::pdc;
use crate::progressive::AccumulationBuffer;
use crate::texture::TextureData;

use super::ast::*;

/// Resolved kernel ready for execution.
enum CompiledKernelEntry {
    /// GPU pixel kernel — renders directly to display texture.
    Gpu {
        wgsl_source: String,
    },
    /// GPU simulation kernel — dispatched via GpuSimRunner.
    GpuSim {
        wgsl_source: String,
    },
    /// WGSL shader compiled to CPU via naga + Cranelift or LLVM.
    #[cfg(any(feature = "cranelift-backend", feature = "llvm-backend"))]
    GpuCpu {
        compiled: jit::CompiledWgslKernel,
        /// Texture slot names in declaration order.
        tex_slot_names: Vec<String>,
    },
    /// PDC scene kernel — JIT-compiled vector scene description.
    #[cfg(any(feature = "cranelift-backend", feature = "llvm-backend"))]
    Scene {
        compiled: pdc::codegen::CompiledProgram,
    },
}

/// The PDP execution engine. Holds all state for running a config-driven example.
pub struct Runtime {
    config: Config,
    kernels: HashMap<String, CompiledKernelEntry>,
    /// Simulation buffers: name -> f64 array.
    buffers: HashMap<String, Vec<f64>>,
    /// Byte-level simulation buffers for CPU render backend (name -> raw bytes).
    #[cfg(any(feature = "cranelift-backend", feature = "llvm-backend"))]
    gpu_cpu_buffers: HashMap<String, Vec<u8>>,
    /// Auto-allocated storage buffers for pixel kernels (not sim — doesn't trigger continuous redraw).
    #[cfg(any(feature = "cranelift-backend", feature = "llvm-backend"))]
    gpu_cpu_auto_buffers: HashMap<String, Vec<u8>>,
    /// Progressive sample counter for GPU pixel kernels (all GPU backends).
    gpu_sample_index: u32,
    /// Max samples for GPU progressive accumulation (from accumulate block, 0 = single-shot).
    gpu_max_samples: u32,
    /// Loaded textures: name -> RGBA8 data.
    textures: HashMap<String, TextureData>,
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
    mouse_was_down: bool,
    mouse_down_edge: bool,
    mouse_up_edge: bool,

    pub tile_height: usize,
    pub animated: bool,
    /// Where WGSL shaders execute: "gpu" (default) or "cpu".
    render: String,
    /// Which JIT backend for WGSL-on-CPU and PDC: "cranelift" (default) or "llvm".
    codegen: String,
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
    /// Whether the last executed frame used GPU rendering (persists across skipped frames).
    last_frame_was_gpu: bool,
    /// Name of the GPU pixel buffer for sim display steps.
    gpu_pixel_buffer_name: Option<String>,
    /// User args for GPU pixel kernels, resolved at execute_run time.
    gpu_user_args: Vec<f32>,
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

        // Apply builtin defaults
        let mut paused = false;
        for b in &config.builtins {
            if let Some(ref default) = b.default {
                let val = match default {
                    Literal::Float(f) => *f,
                    Literal::Int(i) => *i as f64,
                    Literal::Bool(b) => if *b { 1.0 } else { 0.0 },
                    _ => 0.0,
                };
                match b.name.as_str() {
                    "paused" => paused = val != 0.0,
                    _ => {}
                }
            }
        }

        Runtime {
            config,
            kernels: HashMap::new(),
            buffers: HashMap::new(),
            #[cfg(any(feature = "cranelift-backend", feature = "llvm-backend"))]
            gpu_cpu_buffers: HashMap::new(),
            #[cfg(any(feature = "cranelift-backend", feature = "llvm-backend"))]
            gpu_cpu_auto_buffers: HashMap::new(),
            gpu_sample_index: 0,
            gpu_max_samples: 0,
            textures: HashMap::new(),
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
            paused,
            frame: if paused { 1 } else { 0 },
            frames_executed: 0,
            mouse_down: false,
            mouse_was_down: false,
            mouse_down_edge: false,
            mouse_up_edge: false,
            tile_height: 16,
            animated: false,
            render: "gpu".into(),
            codegen: "cranelift".into(),
            pipeline_name: None,
            base_dir: base_dir.to_path_buf(),
            gpu_backend: None,
            gpu_sim_runner: None,
            has_gpu_kernels: false,
            gpu_rendered_this_frame: false,
            last_frame_was_gpu: false,
            gpu_pixel_buffer_name: None,
            gpu_user_args: Vec::new(),
            config_path: None,
        }
    }

    /// Set the config file path (used in window title fallback).
    pub fn set_config_path(&mut self, path: &str) {
        self.config_path = Some(path.to_string());
    }

    /// Whether WGSL shaders run on CPU (JIT'd) rather than GPU.
    fn is_cpu_render(&self) -> bool {
        self.render == "cpu"
    }

    /// Apply settings from the config's settings block.
    pub fn apply_settings(&mut self) {
        for entry in &self.config.settings.entries {
            match entry.key.as_str() {
                "threads" => { /* handled externally */ }
                "render" => {
                    if let Literal::Str(s) = &entry.value {
                        self.render = s.clone();
                    }
                }
                "codegen" => {
                    if let Literal::Str(s) = &entry.value {
                        self.codegen = s.clone();
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
                "render" => self.render = value.clone(),
                "codegen" => self.codegen = value.clone(),
                "pipeline" => self.pipeline_name = Some(value.clone()),
                "title" => self.config.title = Some(value.clone()),
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

        let base_dir = self.base_dir.clone();
        let is_cpu = self.is_cpu_render();
        let codegen = self.codegen.clone();

        for decl in &kernel_decls {
            let path = resolve_path(&decl.path, &base_dir);

            // GPU kernels (.wgsl) — compile to CPU or defer to GPU
            if path.extension().is_some_and(|e| e == "wgsl") {
                let src = std::fs::read_to_string(&path)
                    .map_err(|e| format!("failed to read kernel '{}': {}", path.display(), e))?;
                if src.contains("params.time") {
                    self.animated = true;
                }

                #[cfg(any(feature = "cranelift-backend", feature = "llvm-backend"))]
                if is_cpu {
                    let compiled = match codegen.as_str() {
                        #[cfg(feature = "cranelift-backend")]
                        "cranelift" => jit::wgsl_cranelift::compile_wgsl(&src)
                            .map_err(|e| format!("WGSL CPU compile '{}': {}", decl.name, e))?,
                        #[cfg(feature = "llvm-backend")]
                        "llvm" => jit::wgsl_llvm::compile_wgsl(&src)
                            .map_err(|e| format!("WGSL CPU compile '{}': {}", decl.name, e))?,
                        _ => return Err(format!("unsupported codegen '{}' (available: cranelift, llvm)", codegen)),
                    };
                    let tex_names: Vec<String> = self
                        .selected_pipeline()
                        .map(|p| p.textures.iter().map(|t| t.name.clone()).collect())
                        .unwrap_or_default();
                    eprintln!("compiled WGSL kernel '{}' to CPU via {} ({} storage buffers)", decl.name, codegen, compiled.num_storage_buffers);
                    self.kernels.insert(decl.name.clone(), CompiledKernelEntry::GpuCpu {
                        compiled,
                        tex_slot_names: tex_names,
                    });
                    continue;
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

            // PDC scene kernels (.pdc) — compile via JIT
            #[cfg(any(feature = "cranelift-backend", feature = "llvm-backend"))]
            if path.extension().is_some_and(|e| e == "pdc") {
                let src = std::fs::read_to_string(&path)
                    .map_err(|e| format!("failed to read PDC kernel '{}': {}", path.display(), e))?;
                let compiled = pdc::compile_for_pipeline_with_codegen(&src, Some(&path), &codegen)
                    .map_err(|e| format!("PDC compile '{}': {}", decl.name, e.format(&src)))?;
                eprintln!("compiled PDC scene kernel '{}' via {}", decl.name, codegen);
                self.kernels.insert(decl.name.clone(), CompiledKernelEntry::Scene {
                    compiled,
                });
                continue;
            }

            return Err(format!(
                "unsupported kernel file extension for '{}' — only .wgsl and .pdc are supported",
                path.display()
            ));
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
            if let Some(gpu_type) = decl.gpu_type {
                // GPU buffers — skip for GPU backend, allocate as bytes for CPU render.
                #[cfg(any(feature = "cranelift-backend", feature = "llvm-backend"))]
                if self.is_cpu_render() {
                    let elem_bytes = gpu_type.byte_size() as usize;
                    let buf_size = size * elem_bytes;
                    let mut data = vec![0u8; buf_size];
                    let BufferInit::Constant(val) = &decl.init;
                    if *val != 0.0 {
                        let component_bytes: Vec<u8> = match gpu_type {
                            GpuElementType::I32 => (*val as i32).to_le_bytes().to_vec(),
                            GpuElementType::U32 => (*val as u32).to_le_bytes().to_vec(),
                            _ => (*val as f32).to_le_bytes().to_vec(),
                        };
                        for chunk in data.chunks_exact_mut(elem_bytes) {
                            for (i, byte) in chunk.iter_mut().enumerate() {
                                *byte = component_bytes[i % component_bytes.len()];
                            }
                        }
                    }
                    self.gpu_cpu_buffers.insert(decl.name.clone(), data);
                }
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

    /// Load all declared textures from the selected pipeline.
    pub fn load_textures(&mut self) -> Result<(), String> {
        let tex_decls: Vec<super::ast::TextureDecl> = match self.selected_pipeline() {
            Some(pipeline) => pipeline.textures.clone(),
            None => return Ok(()),
        };

        for decl in &tex_decls {
            match &decl.init {
                super::ast::TextureInit::File(rel_path) => {
                    let full_path = self.base_dir.join(rel_path);
                    let tex = TextureData::load(&full_path)?;
                    eprintln!(
                        "loaded texture '{}': {}x{} from {}",
                        decl.name, tex.width, tex.height, full_path.display()
                    );
                    self.textures.insert(decl.name.clone(), tex);
                }
            }
        }
        Ok(())
    }

    /// Build a TextureSlot array from loaded textures, ordered by the given names.
    /// Returns an empty vec if no texture names are provided.
    fn build_tex_slots(&self, names: &[String]) -> Vec<TextureSlot> {
        names.iter().map(|name| {
            let tex = self.textures.get(name)
                .unwrap_or_else(|| panic!("texture '{}' not loaded", name));
            TextureSlot {
                data: tex.data.as_ptr(),
                width: tex.width,
                height: tex.height,
            }
        }).collect()
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
            // GPU/GpuCpu kernels handle accumulation internally in the shader —
            // don't create the external AccumulationBuffer for them.
            let uses_internal_accum = self.kernels.values().any(|e| {
                matches!(e, CompiledKernelEntry::Gpu { .. } | CompiledKernelEntry::GpuCpu { .. })
            });
            if uses_internal_accum {
                return;
            }
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
                // Gather textures declared in the selected pipeline
                let tex_names: Vec<String> = self
                    .selected_pipeline()
                    .map(|p| p.textures.iter().map(|t| t.name.clone()).collect())
                    .unwrap_or_default();
                let tex_refs: Vec<&TextureData> = tex_names
                    .iter()
                    .filter_map(|name| self.textures.get(name.as_str()))
                    .collect();
                self.gpu_backend = Some(GpuBackend::new(
                    display, self.width, self.height, 256, &source, &tex_refs,
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

    /// Initialize GPU backends headlessly (no display required).
    /// Used by --output and --bench modes.
    fn init_gpu_sim_headless(&mut self) {
        // GPU sim kernels — create headless GpuSimRunner
        let has_gpu_sim = self.kernels.values().any(|e| matches!(e, CompiledKernelEntry::GpuSim { .. }));
        if has_gpu_sim && self.gpu_sim_runner.is_none() {
            let mut runner = GpuSimRunner::new_headless(self.width, self.height);

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

        // Detect mouse button edges
        self.mouse_down_edge = self.mouse_down && !self.mouse_was_down;
        self.mouse_up_edge = self.mouse_was_down && !self.mouse_down;
        self.mouse_was_down = self.mouse_down;

        // Auto-increment frame when not paused
        if !self.paused {
            self.frame += 1;
        }

        // Only execute pipeline if frame has advanced
        if self.frame <= self.frames_executed && !self.animated && !self.needs_continuous_redraw() {
            return false;
        }

        // Get selected pipeline's steps (clone to avoid borrow issues)
        let steps = self.selected_pipeline().map(|p| p.steps.clone());
        if let Some(steps) = steps {
            self.execute_steps(&steps, pool);
        }

        self.frames_executed = self.frame;
        self.last_frame_was_gpu = self.gpu_rendered_this_frame;
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
                PipelineStep::Swap { a, b, .. } => {
                    let mut swapped = false;
                    // Try CPU f64 buffers first
                    if self.buffers.contains_key(a.as_str()) {
                        let mut buf_a = self.buffers.remove(a).unwrap();
                        let mut buf_b = self.buffers.remove(b).unwrap();
                        std::mem::swap(&mut buf_a, &mut buf_b);
                        self.buffers.insert(a.clone(), buf_a);
                        self.buffers.insert(b.clone(), buf_b);
                        swapped = true;
                    }
                    // Then CPU-rendered byte buffers
                    #[cfg(any(feature = "cranelift-backend", feature = "llvm-backend"))]
                    if !swapped && self.gpu_cpu_buffers.contains_key(a.as_str()) {
                        let mut buf_a = self.gpu_cpu_buffers.remove(a).unwrap();
                        let mut buf_b = self.gpu_cpu_buffers.remove(b).unwrap();
                        std::mem::swap(&mut buf_a, &mut buf_b);
                        self.gpu_cpu_buffers.insert(a.clone(), buf_a);
                        self.gpu_cpu_buffers.insert(b.clone(), buf_b);
                        swapped = true;
                    }
                    // Then GPU
                    if !swapped {
                        if let Some(runner) = &mut self.gpu_sim_runner {
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
                PipelineStep::OnMouse { kind, body, .. } => {
                    let should_run = match kind {
                        MouseEventKind::Mousedown => self.mouse_down,
                        MouseEventKind::Click => self.mouse_down_edge,
                        MouseEventKind::Mouseup => self.mouse_up_edge,
                    };
                    if should_run {
                        self.execute_steps(body, pool);
                    }
                }
            }
        }
    }

    /// Execute a kernel via `run`.
    fn execute_run(
        &mut self,
        kernel_name: &str,
        input_bindings: &[BufferBinding],
        args: &[NamedArg],
        pool: &Option<rayon::ThreadPool>,
    ) {
        // Handle scene kernels separately to avoid borrow conflicts —
        // scene execution needs mutable access to self.gpu_sim_runner and
        // self.variables while self.kernels is borrowed.
        #[cfg(any(feature = "cranelift-backend", feature = "llvm-backend"))]
        if let Some(CompiledKernelEntry::Scene { compiled, .. }) = self.kernels.get(kernel_name) {
            let fn_ptr = compiled.fn_ptr;
            self.execute_scene_kernel(kernel_name, fn_ptr);
            return;
        }

        let entry = self.kernels.get(kernel_name);
        match entry {
            Some(CompiledKernelEntry::Gpu { .. }) => {
                // GPU pixel kernels render directly to the display texture.
                // Resolve user args to f32 values for GPU uniform buffer.
                self.gpu_user_args = args.iter().map(|a| {
                    let val = match &a.value {
                        Literal::Float(f) => *f,
                        Literal::Int(i) => *i as f64,
                        Literal::Bool(b) => if *b { 1.0 } else { 0.0 },
                        Literal::Str(_) => 0.0,
                        Literal::VarRef(name) => self.get_variable(name),
                    };
                    val as f32
                }).collect();
                self.gpu_rendered_this_frame = true;
            }
            Some(CompiledKernelEntry::GpuSim { .. }) => {
                // GPU sim kernel — dispatch via GpuSimRunner
                if let Some(runner) = &self.gpu_sim_runner {
                    let bindings: Vec<(&str, &str)> = input_bindings
                        .iter()
                        .map(|b| (b.param_name.as_str(), b.buffer_name.as_str()))
                        .collect();
                    // Resolve user args and encode with correct types from WGSL Params struct
                    let arg_types = runner.user_arg_types(kernel_name);
                    let mut arg_bytes: Vec<u8> = Vec::new();
                    for (i, a) in args.iter().enumerate() {
                        let val = match &a.value {
                            Literal::Float(f) => *f,
                            Literal::Int(i) => *i as f64,
                            Literal::Bool(b) => if *b { 1.0 } else { 0.0 },
                            Literal::Str(_) => 0.0,
                            Literal::VarRef(name) => self.get_variable(name),
                        };
                        let arg_type = arg_types.get(i).copied()
                            .unwrap_or(crate::gpu::sim_runner::WgslArgType::F32);
                        match arg_type {
                            crate::gpu::sim_runner::WgslArgType::U32 => {
                                arg_bytes.extend_from_slice(&(val as u32).to_le_bytes());
                            }
                            crate::gpu::sim_runner::WgslArgType::I32 => {
                                arg_bytes.extend_from_slice(&(val as i32).to_le_bytes());
                            }
                            crate::gpu::sim_runner::WgslArgType::F32 => {
                                arg_bytes.extend_from_slice(&(val as f32).to_le_bytes());
                            }
                        }
                    }
                    runner.dispatch(kernel_name, &bindings, &arg_bytes);
                }
            }
            #[cfg(any(feature = "cranelift-backend", feature = "llvm-backend"))]
            Some(CompiledKernelEntry::GpuCpu { compiled, tex_slot_names }) => {
                let fn_ptr = compiled.fn_ptr;
                let binding_map = &compiled.binding_map;
                let num_bufs = compiled.num_storage_buffers;
                let params_members = &compiled.params_members;
                let w = self.width;
                let h = self.height;
                let stride = w;

                // Build Params buffer by matching member names to runtime values.
                let aspect = w as f64 / h as f64;
                let view_w = 3.5 / self.zoom;
                let view_h = view_w / aspect;

                let mut params = vec![0u8; 256];
                // Track which members are "user args" (not built-in).
                let mut user_arg_index = 0;
                for member in params_members {
                    let off = member.offset as usize;
                    if off + 4 > params.len() { continue; }
                    match member.name.as_str() {
                        "width" => params[off..off+4].copy_from_slice(&w.to_le_bytes()),
                        "height" => params[off..off+4].copy_from_slice(&h.to_le_bytes()),
                        "stride" => params[off..off+4].copy_from_slice(&stride.to_le_bytes()),
                        "max_iter" => params[off..off+4].copy_from_slice(&256u32.to_le_bytes()),
                        "x_min" => params[off..off+4].copy_from_slice(&((self.center_x - view_w / 2.0) as f32).to_le_bytes()),
                        "y_min" => params[off..off+4].copy_from_slice(&((self.center_y - view_h / 2.0) as f32).to_le_bytes()),
                        "x_step" => params[off..off+4].copy_from_slice(&((view_w / w as f64) as f32).to_le_bytes()),
                        "y_step" => params[off..off+4].copy_from_slice(&((view_h / h as f64) as f32).to_le_bytes()),
                        "sample_index" => {
                            let si = if self.gpu_max_samples > 0 { self.gpu_sample_index } else { 0xFFFFFFFF };
                            params[off..off+4].copy_from_slice(&si.to_le_bytes());
                        }
                        "sample_count" => {
                            let sc = if self.gpu_max_samples > 0 { self.gpu_sample_index + 1 } else { 0 };
                            params[off..off+4].copy_from_slice(&sc.to_le_bytes());
                        }
                        "time" => params[off..off+4].copy_from_slice(&(self.time as f32).to_le_bytes()),
                        name if name.starts_with('_') => {} // padding
                        _ => {
                            // User-defined parameter — match by position in `args`.
                            // Encode according to the WGSL type: f32 for floats, u32 for integers.
                            if let Some(a) = args.get(user_arg_index) {
                                let raw = match &a.value {
                                    Literal::Float(f) => *f,
                                    Literal::Int(i) => *i as f64,
                                    Literal::Bool(b) => if *b { 1.0 } else { 0.0 },
                                    Literal::Str(_) => 0.0,
                                    Literal::VarRef(name) => self.get_variable(name),
                                };
                                if member.is_float {
                                    params[off..off+4].copy_from_slice(&(raw as f32).to_le_bytes());
                                } else {
                                    params[off..off+4].copy_from_slice(&(raw as u32).to_le_bytes());
                                }
                            }
                            user_arg_index += 1;
                        }
                    }
                }

                // Debug: dump first few values of each buffer.
                // Build buffer pointer array by resolving bindings.
                // Each WGSL storage buffer variable maps to a PDP buffer.
                let mut buf_ptrs_base: Vec<usize> = vec![0; num_bufs];

                for binding in input_bindings {
                    let wgsl_name = &binding.param_name;
                    let pdp_name = &binding.buffer_name;
                    if let Some(&buf_idx) = binding_map.get(wgsl_name.as_str()) {
                        // Look in gpu_cpu_buffers first, then pixel_buffer.
                        if let Some(buf) = self.gpu_cpu_buffers.get_mut(pdp_name) {
                            buf_ptrs_base[buf_idx] = buf.as_mut_ptr() as usize;
                        }
                    }
                }

                // If no bindings were specified, auto-allocate buffers for pixel kernels.
                // Binding 0 → pixel_buffer (u32 output), remaining → gpu_cpu_buffers.
                if input_bindings.is_empty() && num_bufs > 0 {
                    buf_ptrs_base[0] = self.pixel_buffer.as_mut_ptr() as usize;
                    let pixel_count = (w * h) as usize;
                    for i in 1..num_bufs {
                        let elem_bytes = compiled.buffer_elem_bytes.get(i).copied().unwrap_or(4) as usize;
                        let auto_name = format!("__auto_buf_{i}");
                        let buf = self.gpu_cpu_auto_buffers.entry(auto_name)
                            .or_insert_with(|| vec![0u8; pixel_count * elem_bytes]);
                        buf_ptrs_base[i] = buf.as_mut_ptr() as usize;
                    }
                }

                // Build texture slots.
                let tex_slots = self.build_tex_slots(tex_slot_names);
                let tex_addr = tex_slots.as_ptr() as usize;

                let th = self.tile_height;
                let params_ref = &params;
                with_pool(pool, || {
                    use rayon::prelude::*;
                    let num_tiles = (h as usize + th - 1) / th;
                    (0..num_tiles).into_par_iter().for_each(|tile| {
                        let row_start = (tile * th) as u32;
                        let row_end = ((tile + 1) * th).min(h as usize) as u32;
                        let buf_ptrs: Vec<*mut u8> = buf_ptrs_base.iter()
                            .map(|&addr| addr as *mut u8)
                            .collect();
                        let tex_ptr = if tex_addr == 0 {
                            std::ptr::null()
                        } else {
                            tex_addr as *const u8
                        };
                        unsafe {
                            fn_ptr(
                                params_ref.as_ptr(),
                                buf_ptrs.as_ptr() as *const *mut u8,
                                tex_ptr,
                                w,
                                h,
                                stride,
                                row_start,
                                row_end,
                            );
                        }
                    });
                });

            }
            #[cfg(any(feature = "cranelift-backend", feature = "llvm-backend"))]
            Some(CompiledKernelEntry::Scene { .. }) => {
                // Handled above via early return
                unreachable!()
            }
            None => {
                eprintln!("warning: kernel '{}' not found", kernel_name);
            }
        }
    }

    /// Execute a PDC scene kernel: run the compiled program, extract the
    /// VectorScene, upload scene data buffers to GpuSimRunner, and set
    /// runtime variables for the rasterizer.
    #[cfg(any(feature = "cranelift-backend", feature = "llvm-backend"))]
    fn execute_scene_kernel(
        &mut self,
        kernel_name: &str,
        fn_ptr: pdc::codegen::PdcSceneFn,
    ) {
        let w = self.width;
        let h = self.height;

        // Execute the compiled PDC main
        let builtins = [w as f64, h as f64];
        let mut scene_builder = pdc::runtime::SceneBuilder::new();
        let mut ctx = pdc::runtime::PdcContext {
            builtins: builtins.as_ptr(),
            scene: &mut scene_builder as *mut _,
        };
        unsafe {
            (fn_ptr)(&mut ctx);
        }

        // Extract VectorScene
        let tile_size = 16u32;
        let tolerance = 0.5f32;
        let scene = pdc::extract_scene(&scene_builder, tolerance, tile_size, w, h);

        eprintln!(
            "[pdc] scene '{}': {} paths, {} segments",
            kernel_name,
            scene.path_colors.len(),
            scene.segments.len(),
        );

        // Upload scene data buffers to GPU or CPU backend
        let tiles_x = (w + tile_size - 1) / tile_size;
        let tiles_y = (h + tile_size - 1) / tile_size;
        let num_paths = scene.path_colors.len() as u32;

        if self.is_cpu_render() {
            self.gpu_cpu_buffers.insert("segments".into(), bytemuck::cast_slice(&scene.segments).to_vec());
            self.gpu_cpu_buffers.insert("seg_path_ids".into(), bytemuck::cast_slice(&scene.seg_path_ids).to_vec());
            self.gpu_cpu_buffers.insert("tile_offsets".into(), bytemuck::cast_slice(&scene.tile_offsets).to_vec());
            self.gpu_cpu_buffers.insert("tile_counts".into(), bytemuck::cast_slice(&scene.tile_counts).to_vec());
            self.gpu_cpu_buffers.insert("tile_indices".into(), bytemuck::cast_slice(&scene.tile_indices).to_vec());
            self.gpu_cpu_buffers.insert("path_colors".into(), bytemuck::cast_slice(&scene.path_colors).to_vec());
            self.gpu_cpu_buffers.insert("path_fill_rules".into(), bytemuck::cast_slice(&scene.path_fill_rules).to_vec());
        } else if let Some(runner) = &mut self.gpu_sim_runner {
            runner.add_buffer_with_data("segments", bytemuck::cast_slice(&scene.segments));
            runner.add_buffer_with_data("seg_path_ids", bytemuck::cast_slice(&scene.seg_path_ids));
            runner.add_buffer_with_data("tile_offsets", bytemuck::cast_slice(&scene.tile_offsets));
            runner.add_buffer_with_data("tile_counts", bytemuck::cast_slice(&scene.tile_counts));
            runner.add_buffer_with_data("tile_indices", bytemuck::cast_slice(&scene.tile_indices));
            runner.add_buffer_with_data("path_colors", bytemuck::cast_slice(&scene.path_colors));
            runner.add_buffer_with_data("path_fill_rules", bytemuck::cast_slice(&scene.path_fill_rules));
        }

        // Set runtime variables for the rasterizer
        self.variables.insert("__scene_tiles_x".to_string(), tiles_x as f64);
        self.variables.insert("__scene_tiles_y".to_string(), tiles_y as f64);
        self.variables.insert("__scene_num_paths".to_string(), num_paths as f64);
    }

    /// Present pixels to screen.
    /// Reset gpu-cpu progressive sampling state: clear accum buffers and sample counter.
    #[cfg(any(feature = "cranelift-backend", feature = "llvm-backend"))]
    fn reset_gpu_cpu_accum(&mut self) {
        self.gpu_sample_index = 0;
        for buf in self.gpu_cpu_buffers.values_mut() {
            buf.fill(0);
        }
        for buf in self.gpu_cpu_auto_buffers.values_mut() {
            buf.fill(0);
        }
    }

    #[cfg(not(any(feature = "cranelift-backend", feature = "llvm-backend")))]
    fn reset_gpu_cpu_accum(&mut self) {}

    fn execute_display(&mut self, buffer_name: Option<&str>) {
        match buffer_name {
            Some(name) => {
                // Check if this is a CPU-rendered buffer — copy to pixel_buffer.
                #[cfg(any(feature = "cranelift-backend", feature = "llvm-backend"))]
                if let Some(buf) = self.gpu_cpu_buffers.get(name) {
                    // Interpret as u32 array and copy to pixel_buffer.
                    let n = self.pixel_buffer.len().min(buf.len() / 4);
                    for i in 0..n {
                        let offset = i * 4;
                        self.pixel_buffer[i] = u32::from_le_bytes([
                            buf[offset], buf[offset + 1], buf[offset + 2], buf[offset + 3],
                        ]);
                    }
                    return;
                }
                // GPU display — present named buffer
                self.gpu_rendered_this_frame = true;
                self.gpu_pixel_buffer_name = Some(name.to_string());
            }
            None => {
                // CPU display — pixel buffer already written by run steps.
                // GPU pixel kernel (real GPU backend) — mark as rendered so
                // main loop uses the GPU present path. GpuCpu kernels write
                // directly to pixel_buffer and use the normal CPU display path.
                if self.has_gpu_kernels && self.gpu_sim_runner.is_none() && !self.is_cpu_render() {
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
        // Detect if the body uses a kernel that does its own internal accumulation
        // (GpuCpu or Gpu kernels with accum buffers managed by the shader).
        let uses_internal_accum = body.iter().any(|step| {
            if let PipelineStep::Run { kernel_name, .. } = step {
                matches!(
                    self.kernels.get(kernel_name),
                    Some(CompiledKernelEntry::Gpu { .. })
                    | Some(CompiledKernelEntry::GpuCpu { .. })
                )
            } else {
                false
            }
        });

        if uses_internal_accum {
            self.execute_accumulate_internal(max_samples, body, pool);
        } else {
            self.execute_accumulate_external(max_samples, body, pool);
        }
    }

    /// Accumulation with runtime-managed AccumulationBuffer.
    fn execute_accumulate_external(
        &mut self,
        max_samples: u32,
        body: &[PipelineStep],
        pool: &Option<rayon::ThreadPool>,
    ) {
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
            return;
        }

        self.execute_steps(body, pool);

        let accum = self.accum.as_mut().unwrap();
        accum.accumulate(&self.pixel_buffer);
        let disp = self.display_buffer.as_mut().unwrap();
        accum.resolve(disp);
    }

    /// Accumulation for GpuCpu/Gpu kernels: shader manages its own accum buffer.
    /// The runtime just passes sample_index/sample_count and tracks convergence.
    fn execute_accumulate_internal(
        &mut self,
        max_samples: u32,
        body: &[PipelineStep],
        pool: &Option<rayon::ThreadPool>,
    ) {
        self.gpu_max_samples = max_samples;

        if self.accum_dirty {
            self.gpu_sample_index = 0;
            self.reset_gpu_cpu_accum();
            self.accum_dirty = false;
        }

        if self.gpu_sample_index >= max_samples {
            return; // converged
        }

        // Execute the body steps — execute_run will pick up gpu_sample_index
        // for GpuCpu kernels; for real GPU kernels render_gpu_frame reads it.
        self.execute_steps(body, pool);

        self.gpu_sample_index += 1;
    }

    /// Handle a keydown event. Returns `true` if the app should quit.
    pub fn handle_keydown(&mut self, key_name: &str) -> bool {
        let bindings: Vec<EventBinding> = self.config.event_bindings.clone();
        let mut quit = false;
        for binding in &bindings {
            if binding.kind == EventKind::Keydown && binding.key_name == key_name {
                quit |= self.execute_actions(&binding.actions);
            }
        }
        quit
    }

    /// Handle a keypress event (fires once on initial press). Returns `true` if the app should quit.
    pub fn handle_keypress(&mut self, key_name: &str) -> bool {
        let bindings: Vec<EventBinding> = self.config.event_bindings.clone();
        let mut quit = false;
        for binding in &bindings {
            if binding.kind == EventKind::Keypress && binding.key_name == key_name {
                quit |= self.execute_actions(&binding.actions);
            }
        }
        quit
    }

    /// Handle a keyup event. Returns `true` if the app should quit.
    pub fn handle_keyup(&mut self, key_name: &str) -> bool {
        let bindings: Vec<EventBinding> = self.config.event_bindings.clone();
        let mut quit = false;
        for binding in &bindings {
            if binding.kind == EventKind::Keyup && binding.key_name == key_name {
                quit |= self.execute_actions(&binding.actions);
            }
        }
        quit
    }

    fn execute_actions(&mut self, actions: &[Action]) -> bool {
        let mut quit = false;
        for action in actions {
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

                self.reset_gpu_cpu_accum();
            }
            "center_y" => {
                self.center_y = clamped;
                self.accum_dirty = true;

                self.reset_gpu_cpu_accum();
            }
            "zoom" => {
                self.zoom = clamped;
                self.accum_dirty = true;

                self.reset_gpu_cpu_accum();
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
    /// Re-present the last GPU frame (for use when pipeline was skipped, e.g. paused).
    pub fn re_present_gpu_frame(&self, display: &Display) -> bool {
        if !self.last_frame_was_gpu {
            return false;
        }
        // GPU pixel kernel path
        if let Some(ref gpu) = self.gpu_backend {
            let (si, sc) = if self.gpu_max_samples > 0 {
                (self.gpu_sample_index, self.gpu_sample_index + 1)
            } else {
                (0xFFFFFFFF, 0)
            };
            gpu.render(
                display,
                self.center_x,
                self.center_y,
                self.zoom,
                si,
                sc,
                self.time,
                &self.gpu_user_args,
            );
            return true;
        }
        // GPU sim kernel path — re-present the pixel buffer
        if let Some(ref runner) = self.gpu_sim_runner {
            if let Some(ref buf_name) = self.gpu_pixel_buffer_name {
                runner.present_pixels(display, buf_name);
                return true;
            }
        }
        false
    }

    pub fn render_gpu_frame(&self, display: &Display) -> bool {
        if !self.gpu_rendered_this_frame {
            return false;
        }
        // GPU pixel kernel path
        if let Some(ref gpu) = self.gpu_backend {
            let (si, sc) = if self.gpu_max_samples > 0 {
                (self.gpu_sample_index, self.gpu_sample_index + 1)
            } else {
                (0xFFFFFFFF, 0)
            };
            gpu.render(
                display,
                self.center_x,
                self.center_y,
                self.zoom,
                si,
                sc,
                self.time,
                &self.gpu_user_args,
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

        let has_gpu_sim = self.kernels.values().any(|e| matches!(e, CompiledKernelEntry::GpuSim { .. }));

        if has_gpu_sim {
            // GPU sim pipeline: init runner headlessly, run init + frame, readback.
            self.init_gpu_sim_headless();
            let pool: Option<rayon::ThreadPool> = None;
            self.execute_init_block(&pool);
            self.execute_frame(0.0, &pool);

            if let Some(ref buf_name) = self.gpu_pixel_buffer_name.clone() {
                if let Some(ref runner) = self.gpu_sim_runner {
                    self.pixel_buffer = runner.readback_buffer(buf_name);
                }
            }
        } else {
            // GPU pixel kernel: create headless context, dispatch once, readback.
            let wgsl = self.kernels.values().find_map(|entry| {
                if let CompiledKernelEntry::Gpu { wgsl_source } = entry {
                    Some(wgsl_source.clone())
                } else {
                    None
                }
            });
            if let Some(source) = wgsl {
                let tex_names: Vec<String> = self
                    .selected_pipeline()
                    .map(|p| p.textures.iter().map(|t| t.name.clone()).collect())
                    .unwrap_or_default();
                let tex_refs: Vec<&TextureData> = tex_names
                    .iter()
                    .filter_map(|name| self.textures.get(name.as_str()))
                    .collect();
                let (gpu, device, queue) = GpuBackend::new_headless(
                    self.width, self.height, 256, &source, &tex_refs,
                );
                gpu.dispatch_compute(
                    &device, &queue,
                    self.center_x, self.center_y, self.zoom,
                    0xFFFFFFFF, 0, self.time,
                    &self.gpu_user_args,
                );
                self.pixel_buffer = gpu.readback_pixels(&device, &queue);
            }
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
        if self.paused {
            return false;
        }
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
        #[cfg(any(feature = "cranelift-backend", feature = "llvm-backend"))]
        if !self.gpu_cpu_buffers.is_empty() {
            return true;
        }
        // GPU progressive accumulation (from accumulate block)
        if self.gpu_max_samples > 0 && self.gpu_sample_index < self.gpu_max_samples {
            return true;
        }
        // GPU simulations have buffers in the GPU runner, not self.buffers
        self.gpu_sim_runner.is_some()
    }

    pub fn accumulation_info(&self) -> Option<(u32, u32)> {
        if let Some(ref a) = self.accum {
            return Some((a.sample_count, a.max_samples));
        }
        if self.gpu_max_samples > 0 {
            return Some((self.gpu_sample_index, self.gpu_max_samples));
        }
        None
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
