//! PDC-driven pipeline runtime.
//!
//! `PdcRuntime` executes a compiled PDC program as the pipeline orchestrator,
//! replacing the declarative PDP config. It implements the same public interface
//! as `pdp::runtime::Runtime` so that `main.rs` can use either.

use std::path::{Path, PathBuf};

use crate::display::Display;
use crate::jit;
use crate::pdc;
use crate::texture::TextureData;
use crate::vector::VectorScene;
use crate::pdc::codegen::{CompiledProgram, StateLayout, PIPELINE_BUILTINS};
use crate::pdc::runtime::{PdcContext, PipelineHost, SceneBuilder};

/// Builtins array indices (must match PIPELINE_BUILTINS order).
#[allow(dead_code)]
mod builtins_idx {
    pub const WIDTH: usize = 0;
    pub const HEIGHT: usize = 1;
    pub const TIME: usize = 2;
    pub const MOUSE_X: usize = 3;
    pub const MOUSE_Y: usize = 4;
    pub const CENTER_X: usize = 5;
    pub const CENTER_Y: usize = 6;
    pub const ZOOM: usize = 7;
    pub const PAUSED: usize = 8;
    pub const FRAME: usize = 9;
    pub const MOUSE_DOWN: usize = 10;
    pub const SAMPLE_INDEX: usize = 11;
    pub const COUNT: usize = 12;
}
use builtins_idx as B;

/// Pipeline host that manages buffers, kernels, and display for PDC scripts.
#[allow(dead_code)]
struct HostState {
    width: u32,
    height: u32,
    /// Pixel output buffer (ARGB u32).
    pixel_buffer: Vec<u32>,
    /// Named WGSL kernels compiled for CPU execution.
    #[cfg(any(feature = "cranelift-backend", feature = "llvm-backend"))]
    cpu_kernels: Vec<CpuKernel>,
    /// Named buffer storage (raw bytes).
    buffers: Vec<NamedBuffer>,
    /// Pending buffer bindings for the next run_kernel call.
    pending_bindings: Vec<BufferBinding>,
    /// Pending kernel args for the next run_kernel call.
    pending_args: Vec<KernelArg>,
    /// Whether display() or display_accumulated() was called this frame.
    display_requested: bool,
    /// Whether the script requested another frame via request_redraw().
    redraw_requested: bool,
    /// Loaded textures, indexed by handle.
    textures: Vec<TextureData>,
    /// Compiled scene programs, indexed by handle.
    scenes: Vec<SceneEntry>,
    /// Accumulation buffer for progressive rendering.
    accum: Option<crate::progressive::AccumulationBuffer>,
    /// Snapshot of builtins for kernel param population.
    /// Updated before each frame by PdcRuntime.
    builtins_snapshot: [f64; B::COUNT],
    /// Base directory for resolving relative paths.
    base_dir: PathBuf,
    /// Render mode: "gpu" or "cpu".
    render: String,
    /// Codegen backend: "cranelift" or "llvm".
    codegen: String,
}

#[allow(dead_code)]
struct NamedBuffer {
    name: String,
    data: Vec<u8>,
    elem_size: usize,
}

#[allow(dead_code)]
struct BufferBinding {
    param_name: String,
    buffer_handle: i32,
    is_output: bool,
}

struct KernelArg {
    name: String,
    value: f64,
}

/// A compiled scene kernel and its last extracted scene data.
struct SceneEntry {
    source: String,
    source_path: PathBuf,
    /// Last extracted scene, with buffer handles for each named buffer.
    scene: Option<VectorScene>,
    /// Buffer handles for scene data: (name, buffer_handle).
    buffer_handles: Vec<(String, i32)>,
    tiles_x: u32,
    tiles_y: u32,
    num_paths: u32,
}

#[cfg(any(feature = "cranelift-backend", feature = "llvm-backend"))]
struct CpuKernel {
    #[allow(dead_code)]
    name: String,
    compiled: jit::CompiledWgslKernel,
}

impl PipelineHost for HostState {
    fn create_buffer(&mut self, type_name: &str, init_value: f64) -> i32 {
        let elem_size = match type_name {
            "gpu_f32" => 4,
            "gpu_i32" | "gpu_u32" => 4,
            "gpu_vec2_f32" => 8,
            "gpu_vec3_f32" => 16, // padded
            "gpu_vec4_f32" => 16,
            _ => 4,
        };
        let count = (self.width as usize) * (self.height as usize);
        let total_bytes = count * elem_size;
        let data = if init_value == 0.0 {
            vec![0u8; total_bytes]
        } else {
            // Initialize with the f32 representation of init_value
            let val = init_value as f32;
            let bytes = val.to_le_bytes();
            let mut data = vec![0u8; total_bytes];
            for chunk in data.chunks_exact_mut(elem_size.min(4)) {
                chunk.copy_from_slice(&bytes[..chunk.len()]);
            }
            data
        };
        let handle = self.buffers.len() as i32;
        self.buffers.push(NamedBuffer {
            name: format!("buffer_{handle}"),
            data,
            elem_size,
        });
        handle
    }

    fn swap_buffers(&mut self, a: i32, b: i32) {
        let (a, b) = (a as usize, b as usize);
        if a < self.buffers.len() && b < self.buffers.len() && a != b {
            // Swap the data vecs
            let ptr_a = &mut self.buffers[a].data as *mut Vec<u8>;
            let ptr_b = &mut self.buffers[b].data as *mut Vec<u8>;
            unsafe { std::ptr::swap(ptr_a, ptr_b); }
        }
    }

    fn load_kernel(&mut self, name: &str, path: &str, _kind: i32) -> i32 {
        let full_path = self.base_dir.join(path);
        let wgsl_src = match std::fs::read_to_string(&full_path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[pdc-runtime] failed to read kernel '{}': {}", full_path.display(), e);
                return -1;
            }
        };

        #[cfg(any(feature = "cranelift-backend", feature = "llvm-backend"))]
        {
            let compiled = match self.codegen.as_str() {
                #[cfg(feature = "cranelift-backend")]
                "cranelift" => jit::wgsl_cranelift::compile_wgsl(&wgsl_src),
                #[cfg(feature = "llvm-backend")]
                "llvm" => jit::wgsl_llvm::compile_wgsl(&wgsl_src),
                _ => Err(format!("unsupported codegen backend '{}'", self.codegen)),
            };
            let compiled = match compiled {
                Ok(k) => k,
                Err(e) => {
                    eprintln!("[pdc-runtime] failed to compile kernel '{}': {}", name, e);
                    return -1;
                }
            };
            let handle = self.cpu_kernels.len() as i32;
            self.cpu_kernels.push(CpuKernel {
                name: name.to_string(),
                compiled,
            });
            handle
        }
        #[cfg(not(any(feature = "cranelift-backend", feature = "llvm-backend")))]
        {
            eprintln!("[pdc-runtime] no JIT backend available for kernel '{}'", name);
            -1
        }
    }

    fn bind_buffer(&mut self, param_name: &str, buffer_handle: i32, is_output: bool) {
        self.pending_bindings.push(BufferBinding {
            param_name: param_name.to_string(),
            buffer_handle,
            is_output,
        });
    }

    fn set_kernel_arg_f64(&mut self, name: &str, value: f64) {
        self.pending_args.push(KernelArg { name: name.to_string(), value });
    }

    fn set_kernel_arg_f32(&mut self, name: &str, value: f32) {
        self.pending_args.push(KernelArg { name: name.to_string(), value: value as f64 });
    }

    fn run_kernel(&mut self, kernel_handle: i32) {
        #[cfg(any(feature = "cranelift-backend", feature = "llvm-backend"))]
        {
            let handle = kernel_handle as usize;
            if handle >= self.cpu_kernels.len() {
                eprintln!("[pdc-runtime] invalid kernel handle {}", kernel_handle);
                self.pending_bindings.clear();
                self.pending_args.clear();
                return;
            }

            // Build buffer pointer array based on bindings
            let kernel = &self.cpu_kernels[handle];
            let num_buffers = kernel.compiled.num_storage_buffers;
            let mut buffer_ptrs: Vec<*mut u8> = vec![std::ptr::null_mut(); num_buffers];

            for binding in &self.pending_bindings {
                let buf_idx = binding.buffer_handle as usize;
                if buf_idx >= self.buffers.len() { continue; }
                if let Some(&slot) = kernel.compiled.binding_map.get(&binding.param_name) {
                    buffer_ptrs[slot] = self.buffers[buf_idx].data.as_mut_ptr();
                }
            }

            // Auto-allocate any unbound buffer slots (e.g., accum buffers for
            // pixel kernels that expect them). Uses buffer_elem_bytes from the
            // compiled kernel to determine per-element size.
            let pixel_count = (self.width as usize) * (self.height as usize);
            for slot in 0..num_buffers {
                if buffer_ptrs[slot].is_null() {
                    let elem_bytes = kernel.compiled.buffer_elem_bytes
                        .get(slot)
                        .copied()
                        .unwrap_or(4) as usize;
                    let auto_buf = self.buffers.len();
                    self.buffers.push(NamedBuffer {
                        name: format!("__auto_{handle}_{slot}"),
                        data: vec![0u8; pixel_count * elem_bytes],
                        elem_size: elem_bytes,
                    });
                    buffer_ptrs[slot] = self.buffers[auto_buf].data.as_mut_ptr();
                }
            }

            // Build params buffer (256 bytes), populating built-in members
            // from the builtins snapshot and user args from pending_args.
            let mut params = [0u8; 256];
            let w = self.width;
            let h = self.height;

            // Compute view transform for pixel kernels
            let center_x = self.builtins_snapshot[B::CENTER_X];
            let center_y = self.builtins_snapshot[B::CENTER_Y];
            let zoom = self.builtins_snapshot[B::ZOOM].max(1e-10);
            let time = self.builtins_snapshot[B::TIME];
            let sample_index = self.builtins_snapshot[B::SAMPLE_INDEX] as u32;
            let aspect = w as f64 / h as f64;
            let view_w = 3.5 / zoom;
            let view_h = view_w / aspect;

            let mut user_arg_index = 0;
            for member in &kernel.compiled.params_members {
                let off = member.offset as usize;
                if off + 4 > params.len() { continue; }
                match member.name.as_str() {
                    "width" => params[off..off + 4].copy_from_slice(&w.to_le_bytes()),
                    "height" => params[off..off + 4].copy_from_slice(&h.to_le_bytes()),
                    "stride" => params[off..off + 4].copy_from_slice(&w.to_le_bytes()),
                    "max_iter" => params[off..off + 4].copy_from_slice(&256u32.to_le_bytes()),
                    "x_min" => params[off..off + 4].copy_from_slice(&((center_x - view_w / 2.0) as f32).to_le_bytes()),
                    "y_min" => params[off..off + 4].copy_from_slice(&((center_y - view_h / 2.0) as f32).to_le_bytes()),
                    "x_step" => params[off..off + 4].copy_from_slice(&((view_w / w as f64) as f32).to_le_bytes()),
                    "y_step" => params[off..off + 4].copy_from_slice(&((view_h / h as f64) as f32).to_le_bytes()),
                    "sample_index" => params[off..off + 4].copy_from_slice(&sample_index.to_le_bytes()),
                    "sample_count" => params[off..off + 4].copy_from_slice(&(sample_index + 1).to_le_bytes()),
                    "time" => params[off..off + 4].copy_from_slice(&(time as f32).to_le_bytes()),
                    name if name.starts_with('_') => {} // padding
                    _ => {
                        // Check pending args first (by name)
                        if let Some(arg) = self.pending_args.iter().find(|a| a.name == member.name) {
                            if member.is_float {
                                params[off..off + 4].copy_from_slice(&(arg.value as f32).to_le_bytes());
                            } else {
                                params[off..off + 4].copy_from_slice(&(arg.value as u32).to_le_bytes());
                            }
                        } else {
                            user_arg_index += 1;
                        }
                    }
                }
            }
            let _ = user_arg_index;

            // Build texture slots from loaded textures
            let tex_slots: Vec<jit::TextureSlot> = self.textures.iter().map(|t| {
                jit::TextureSlot {
                    data: t.data.as_ptr(),
                    width: t.width,
                    height: t.height,
                }
            }).collect();

            // Dispatch across all rows (single-threaded for simplicity)
            let fn_ptr = kernel.compiled.fn_ptr;
            unsafe {
                let params_ptr = params.as_ptr();
                let buffers_ptr = buffer_ptrs.as_ptr() as *const *mut u8;
                let tex_ptr = if tex_slots.is_empty() {
                    std::ptr::null()
                } else {
                    tex_slots.as_ptr() as *const u8
                };
                (fn_ptr)(params_ptr, buffers_ptr, tex_ptr, w, h, w, 0, h);
            }
        }

        self.pending_bindings.clear();
        self.pending_args.clear();
    }

    fn display(&mut self) {
        self.display_requested = true;
    }

    fn display_buffer(&mut self, buffer_handle: i32) {
        let handle = buffer_handle as usize;
        if handle < self.buffers.len() {
            let buf = &self.buffers[handle];
            // Copy buffer data to pixel_buffer (assuming u32 ARGB)
            let pixel_count = self.pixel_buffer.len();
            let src = &buf.data;
            for i in 0..pixel_count.min(src.len() / 4) {
                let offset = i * 4;
                self.pixel_buffer[i] = u32::from_le_bytes([
                    src[offset], src[offset + 1], src[offset + 2], src[offset + 3],
                ]);
            }
        }
        self.display_requested = true;
    }

    fn load_scene(&mut self, _name: &str, path: &str) -> i32 {
        let full_path = self.base_dir.join(path);
        let source = match std::fs::read_to_string(&full_path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[pdc-runtime] failed to read scene '{}': {}", full_path.display(), e);
                return -1;
            }
        };
        let handle = self.scenes.len() as i32;
        self.scenes.push(SceneEntry {
            source,
            source_path: full_path,
            scene: None,
            buffer_handles: Vec::new(),
            tiles_x: 0,
            tiles_y: 0,
            num_paths: 0,
        });
        handle
    }

    fn run_scene(&mut self, handle: i32) {
        let idx = handle as usize;
        if idx >= self.scenes.len() {
            eprintln!("[pdc-runtime] invalid scene handle {}", handle);
            return;
        }

        let w = self.width;
        let h = self.height;
        let tile_size = 16u32;
        let tolerance = 0.5f32;

        let scene_entry = &self.scenes[idx];
        let scene = match pdc::compile_and_run(
            &scene_entry.source, w, h, tolerance, tile_size,
            Some(&scene_entry.source_path),
        ) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[pdc-runtime] scene execution failed: {}", e.format(&self.scenes[idx].source));
                return;
            }
        };

        let tiles_x = (w + tile_size - 1) / tile_size;
        let tiles_y = (h + tile_size - 1) / tile_size;
        let num_paths = scene.path_colors.len() as u32;

        // Convert scene data to buffers
        let buffer_data: Vec<(&str, Vec<u8>)> = vec![
            ("segments", bytemuck_cast_slice(&scene.segments)),
            ("seg_path_ids", bytemuck_cast_u32(&scene.seg_path_ids)),
            ("tile_offsets", bytemuck_cast_u32(&scene.tile_offsets)),
            ("tile_counts", bytemuck_cast_u32(&scene.tile_counts)),
            ("tile_indices", bytemuck_cast_u32(&scene.tile_indices)),
            ("path_colors", bytemuck_cast_u32(&scene.path_colors)),
            ("path_fill_rules", bytemuck_cast_u32(&scene.path_fill_rules)),
        ];

        let mut buffer_handles = Vec::new();
        for (name, data) in buffer_data {
            let buf_handle = self.buffers.len() as i32;
            let elem_size = if name == "segments" { 16 } else { 4 };
            self.buffers.push(NamedBuffer {
                name: format!("__scene_{}_{}", idx, name),
                data,
                elem_size,
            });
            buffer_handles.push((name.to_string(), buf_handle));
        }

        let entry = &mut self.scenes[idx];
        entry.scene = Some(scene);
        entry.buffer_handles = buffer_handles;
        entry.tiles_x = tiles_x;
        entry.tiles_y = tiles_y;
        entry.num_paths = num_paths;
    }

    fn scene_tiles_x(&self, handle: i32) -> f64 {
        self.scenes.get(handle as usize).map_or(0.0, |s| s.tiles_x as f64)
    }

    fn scene_num_paths(&self, handle: i32) -> f64 {
        self.scenes.get(handle as usize).map_or(0.0, |s| s.num_paths as f64)
    }

    fn scene_buffer(&mut self, scene_handle: i32, name: &str) -> i32 {
        let idx = scene_handle as usize;
        if idx >= self.scenes.len() { return -1; }
        self.scenes[idx].buffer_handles.iter()
            .find(|(n, _)| n == name)
            .map(|(_, h)| *h)
            .unwrap_or(-1)
    }

    fn load_texture(&mut self, name: &str, path: &str) -> i32 {
        let full_path = self.base_dir.join(path);
        match TextureData::load(&full_path) {
            Ok(tex) => {
                let handle = self.textures.len() as i32;
                eprintln!("[pdc-runtime] loaded texture '{}' ({}x{}) as handle {}",
                    name, tex.width, tex.height, handle);
                self.textures.push(tex);
                handle
            }
            Err(e) => {
                eprintln!("[pdc-runtime] failed to load texture '{}': {}", full_path.display(), e);
                -1
            }
        }
    }

    fn set_max_samples(&mut self, n: i32) {
        let w = self.width as usize;
        let h = self.height as usize;
        self.accum = Some(crate::progressive::AccumulationBuffer::new(w, h, n as u32));
    }

    fn is_converged(&self) -> bool {
        self.accum.as_ref().map_or(false, |a| a.is_converged())
    }

    fn accumulate_sample(&mut self) {
        if let Some(ref mut accum) = self.accum {
            accum.accumulate(&self.pixel_buffer);
        }
    }

    fn display_accumulated(&mut self) {
        if let Some(ref accum) = self.accum {
            accum.resolve(&mut self.pixel_buffer);
        }
        self.display_requested = true;
    }

    fn reset_accumulation(&mut self) {
        if let Some(ref mut accum) = self.accum {
            accum.reset();
        }
    }

    fn was_display_requested(&self) -> bool { self.display_requested }
    fn clear_display_requested(&mut self) { self.display_requested = false; }
    fn pixel_buffer(&self) -> &[u32] { &self.pixel_buffer }
    fn is_accumulating(&self) -> bool {
        self.accum.as_ref().map_or(false, |a| !a.is_converged())
    }
    fn has_buffers(&self) -> bool {
        !self.buffers.is_empty()
    }
    fn update_dimensions(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.pixel_buffer = vec![0xFF000000u32; (width * height) as usize];
    }
    fn update_builtins(&mut self, builtins: &[f64]) {
        let n = builtins.len().min(B::COUNT);
        self.builtins_snapshot[..n].copy_from_slice(&builtins[..n]);
    }
    fn request_redraw(&mut self) { self.redraw_requested = true; }
    fn was_redraw_requested(&self) -> bool { self.redraw_requested }
    fn clear_redraw_requested(&mut self) { self.redraw_requested = false; }
}

fn bytemuck_cast_u32(data: &[u32]) -> Vec<u8> {
    data.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn bytemuck_cast_slice(data: &[[f32; 4]]) -> Vec<u8> {
    data.iter().flat_map(|seg| {
        seg.iter().flat_map(|f| f.to_le_bytes())
    }).collect()
}

/// PDC-driven pipeline runtime.
///
/// Executes a compiled PDC program as the pipeline orchestrator.
/// Implements the same public interface as `pdp::runtime::Runtime`.
pub struct PdcRuntime {
    compiled: CompiledProgram,
    #[allow(dead_code)]
    state_layout: StateLayout,
    state_block: Box<[u8]>,
    builtins: [f64; B::COUNT],
    scene_builder: SceneBuilder,
    host: Box<dyn PipelineHost>,

    // Public fields matching PDP Runtime interface
    pub width: u32,
    pub height: u32,
    pub mouse_x: f64,
    pub mouse_y: f64,
    pub mouse_down: bool,
    pub has_gpu_kernels: bool,
    pub pixel_buffer: Vec<u32>,
    pub tile_height: usize,

    title: String,
    pub paused: bool,
    frame: u64,
    frames_executed: u64,
    /// Whether this pipeline needs continuous redraws (e.g., uses time or animation).
    animated: bool,
    /// Previous mouse_down state for edge detection.
    mouse_was_down: bool,
}

impl PdcRuntime {
    /// Create a new PDC pipeline runtime from source code.
    pub fn new(
        source: &str,
        source_path: Option<&Path>,
        width: u32,
        height: u32,
        base_dir: &Path,
    ) -> Result<Self, String> {
        let builtins_layout: Vec<(&str, pdc::ast::PdcType)> =
            PIPELINE_BUILTINS.to_vec();

        let (compiled, state_layout) = pdc::compile_only_with_builtins(
            source,
            source_path,
            &builtins_layout,
        ).map_err(|e| e.format(source))?;

        let state_block = state_layout.alloc();
        let pixel_buffer = vec![0xFF000000u32; (width * height) as usize]; // black

        let host: Box<dyn PipelineHost> = Box::new(HostState {
            width,
            height,
            pixel_buffer: vec![0xFF000000u32; (width * height) as usize],
            #[cfg(any(feature = "cranelift-backend", feature = "llvm-backend"))]
            cpu_kernels: Vec::new(),
            buffers: Vec::new(),
            pending_bindings: Vec::new(),
            pending_args: Vec::new(),
            display_requested: false,
            redraw_requested: false,
            textures: Vec::new(),
            scenes: Vec::new(),
            accum: None,
            builtins_snapshot: [0.0; B::COUNT],
            base_dir: base_dir.to_path_buf(),
            render: "cpu".to_string(),
            codegen: if cfg!(feature = "cranelift-backend") {
                "cranelift".to_string()
            } else {
                "llvm".to_string()
            },
        });

        let title = source_path
            .map(|p| p.file_stem().unwrap_or_default().to_string_lossy().to_string())
            .unwrap_or_else(|| "PDC Pipeline".to_string());

        let mut builtins = [0.0f64; B::COUNT];
        builtins[B::WIDTH] = width as f64;
        builtins[B::HEIGHT] = height as f64;
        builtins[B::ZOOM] = 1.0;

        Ok(PdcRuntime {
            compiled,
            state_layout,
            state_block,
            builtins,
            scene_builder: SceneBuilder::new(),
            host,
            width,
            height,
            mouse_x: 0.0,
            mouse_y: 0.0,
            mouse_down: false,
            has_gpu_kernels: false, // PDC runtime is CPU-only for now
            pixel_buffer,
            tile_height: 16,
            title,
            paused: false,
            frame: 0,
            frames_executed: 0,
            animated: false,
            mouse_was_down: false,
        })
    }

    /// Build the PdcContext for calling into JIT'd code.
    fn make_ctx(&mut self) -> PdcContext {
        // Update builtins from runtime state
        self.builtins[B::MOUSE_X] = self.mouse_x;
        self.builtins[B::MOUSE_Y] = self.mouse_y;
        self.builtins[B::PAUSED] = if self.paused { 1.0 } else { 0.0 };
        self.builtins[B::FRAME] = self.frame as f64;
        self.builtins[B::MOUSE_DOWN] = if self.mouse_down { 1.0 } else { 0.0 };

        let host_ptr: *mut Box<dyn PipelineHost> = &mut self.host;

        PdcContext {
            builtins: self.builtins.as_mut_ptr(),
            scene: &mut self.scene_builder as *mut _,
            state: self.state_block.as_mut_ptr(),
            host: host_ptr as *mut u8,
        }
    }

    /// Read back mutable builtins after a PDC call.
    fn read_back_builtins(&mut self) {
        self.paused = self.builtins[B::PAUSED] != 0.0;
        self.frame = self.builtins[B::FRAME] as u64;
    }

    // ── Public interface matching PDP Runtime ──

    pub fn set_config_path(&mut self, _path: &str) {
        // Title already set from source_path
    }

    pub fn apply_settings(&mut self) {
        // Settings are embedded in the PDC source
    }

    pub fn apply_overrides(&mut self, overrides: &[(String, String)]) {
        for (key, value) in overrides {
            match key.as_str() {
                "width" => {
                    if let Ok(v) = value.parse::<u32>() {
                        self.width = v;
                        self.builtins[B::WIDTH] = v as f64;
                        self.pixel_buffer = vec![0xFF000000u32; (self.width * self.height) as usize];
                        self.host.update_dimensions(self.width, self.height);
                    }
                }
                "height" => {
                    if let Ok(v) = value.parse::<u32>() {
                        self.height = v;
                        self.builtins[B::HEIGHT] = v as f64;
                        self.pixel_buffer = vec![0xFF000000u32; (self.width * self.height) as usize];
                        self.host.update_dimensions(self.width, self.height);
                    }
                }
                "render" => {
                    // Accepted but currently only CPU is supported
                }
                "codegen" => {
                    // Accepted for compatibility
                }
                "pipeline" => {
                    // Accepted for compatibility
                }
                _ => {
                    eprintln!("[pdc-runtime] unknown override: {key}={value}");
                }
            }
        }
    }

    pub fn compile_kernels(&mut self) -> Result<(), String> {
        // Kernels are compiled on-demand by the PDC init() function via load_kernel()
        Ok(())
    }

    pub fn init_buffers(&mut self) -> Result<(), String> {
        // Buffers are created on-demand by the PDC init() function via create_buffer()
        Ok(())
    }

    pub fn load_textures(&mut self) -> Result<(), String> {
        Ok(())
    }

    pub fn setup_progressive(&mut self) {
        // Progressive rendering managed by PDC script
    }

    pub fn thread_count(&self) -> Option<usize> {
        None
    }

    /// Initialize: run pdc_main (state init) + call init().
    pub fn execute_init_block(&mut self, _pool: &Option<rayon::ThreadPool>) {
        self.host.update_builtins(&self.builtins);
        let mut ctx = self.make_ctx();

        // Run pdc_main to initialize state block
        unsafe { (self.compiled.fn_ptr)(&mut ctx); }

        // Call init() if defined
        unsafe { self.compiled.call_init(&mut ctx).unwrap(); }

        self.read_back_builtins();

        // Copy host pixel buffer to our pixel buffer
        self.sync_pixels();
    }

    /// Execute one frame. Returns true if pixels were updated.
    pub fn execute_frame(&mut self, time: f64, _pool: &Option<rayon::ThreadPool>) -> bool {
        if !self.paused {
            self.frame += 1;
        }

        // Skip if paused and this frame was already executed
        if self.paused && self.frame <= self.frames_executed {
            return false;
        }

        // Skip if this frame was already executed (static scene, no animation/accumulation)
        if self.frame <= self.frames_executed && !self.animated && !self.host.is_accumulating() {
            return false;
        }

        self.builtins[B::TIME] = time;
        self.host.update_builtins(&self.builtins);
        self.host.clear_display_requested();
        self.host.clear_redraw_requested();

        // Clear accumulated scene data from the previous frame to avoid
        // leaking handles created by per-frame host function calls.
        self.scene_builder = SceneBuilder::new();

        // Mouse edge detection: fire event handlers on transitions
        let mouse_down_edge = self.mouse_down && !self.mouse_was_down;
        let mouse_up_edge = !self.mouse_down && self.mouse_was_down;
        self.mouse_was_down = self.mouse_down;

        let mut ctx = self.make_ctx();

        if mouse_down_edge {
            unsafe { let _ = self.compiled.call_event("on_click", &mut ctx); }
        }
        if self.mouse_down {
            unsafe { let _ = self.compiled.call_event("on_mousedown", &mut ctx); }
        }
        if mouse_up_edge {
            unsafe { let _ = self.compiled.call_event("on_mouseup", &mut ctx); }
        }

        unsafe { self.compiled.call_frame(&mut ctx).unwrap(); }
        self.read_back_builtins();

        if self.host.was_display_requested() {
            self.sync_pixels();
        }

        self.frames_executed = self.frame;
        true
    }

    pub fn init_gpu(&mut self, _display: &Display) {
        // GPU not yet supported in PdcRuntime
    }

    pub fn handle_keypress(&mut self, key_name: &str) -> bool {
        let event_name = format!("on_keypress_{key_name}");
        let mut ctx = self.make_ctx();
        let _handled = unsafe { self.compiled.call_event(&event_name, &mut ctx).unwrap_or(false) };
        self.read_back_builtins();
        false // never quit via keypress for now
    }

    pub fn handle_keydown(&mut self, key_name: &str) -> bool {
        let event_name = format!("on_keydown_{key_name}");
        let mut ctx = self.make_ctx();
        let _handled = unsafe { self.compiled.call_event(&event_name, &mut ctx).unwrap_or(false) };
        self.read_back_builtins();
        false
    }

    pub fn handle_keyup(&mut self, key_name: &str) -> bool {
        let event_name = format!("on_keyup_{key_name}");
        let mut ctx = self.make_ctx();
        let _handled = unsafe { self.compiled.call_event(&event_name, &mut ctx).unwrap_or(false) };
        self.read_back_builtins();
        false
    }

    pub fn render_gpu_frame(&self, _display: &Display) -> bool {
        false // CPU-only for now
    }

    pub fn re_present_gpu_frame(&self, _display: &Display) -> bool {
        false
    }

    pub fn display_pixels(&self) -> &[u32] {
        &self.pixel_buffer
    }

    pub fn needs_continuous_redraw(&self) -> bool {
        if self.paused {
            return false;
        }
        // The PDC script explicitly requests redraws via request_redraw()
        self.host.was_redraw_requested()
    }

    pub fn title(&self) -> String {
        self.title.clone()
    }

    pub fn accumulation_info(&self) -> Option<(u32, u32)> {
        // Access via the concrete type since we need accum field
        // The trait object doesn't expose this directly
        None // TODO: expose via PipelineHost trait if needed for window title
    }

    pub fn execute_gpu_headless(&mut self) {
        // No GPU support yet — fall back to CPU
        self.execute_init_block(&None);
        self.execute_frame(0.0, &None);
    }

    // ── Internal ──

    fn sync_pixels(&mut self) {
        // Copy from host's pixel buffer to our pixel buffer
        let src = self.host.pixel_buffer();
        self.pixel_buffer.copy_from_slice(src);
    }
}
