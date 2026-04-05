//! PDC-driven pipeline runtime.
//!
//! `PdcRuntime` executes a compiled PDC program as the pipeline orchestrator.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::display::Display;
use crate::gpu::{self, GpuBackend, sim_runner::GpuSimRunner};
use crate::jit;
use crate::pdc;
use crate::gpu::GpuElementType;
use crate::texture::TextureData;
use crate::vector::VectorScene;
use crate::pdc::codegen::{CompiledProgram, StateLayout, PIPELINE_BUILTINS};
use crate::pdc::runtime::{PdcContext, PipelineHost, SceneBuilder};

/// Builtins array indices (must match PIPELINE_BUILTINS order).
#[allow(dead_code)]
pub(crate) mod builtins_idx {
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
/// Per-kernel resolved configuration. Separated from HostState
/// so run_kernel can borrow this immutably while mutating buffer data.
///
/// Bindings and args are stored in indexed slots resolved on first use,
/// eliminating per-frame string lookups.
struct KernelConfig {
    /// Per-kernel buffer bindings, indexed by kernel handle. Persist until overwritten.
    bindings: Vec<Vec<BufferBinding>>,
    /// Per-kernel args, indexed by kernel handle. Persist until overwritten.
    args: Vec<Vec<KernelArg>>,
    /// Per-kernel name→index caches for O(1) repeated lookups.
    binding_name_cache: Vec<HashMap<String, usize>>,
    arg_name_cache: Vec<HashMap<String, usize>>,
}

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
    /// Per-kernel bindings and args, in a separate struct for split borrows.
    kernel_config: KernelConfig,
    /// Whether display() was called this frame.
    display_requested: bool,
    /// Loaded textures, indexed by handle.
    textures: Vec<TextureData>,
    /// Compiled scene programs, indexed by handle.
    scenes: Vec<SceneEntry>,
    /// Snapshot of builtins for kernel param population.
    /// Updated before each frame by PdcRuntime.
    builtins_snapshot: [f64; B::COUNT],
    /// Base directory for resolving relative paths.
    base_dir: PathBuf,
    /// Render mode: "gpu" or "cpu".
    render: String,
    /// Codegen backend: "cranelift" or "llvm".
    codegen: String,
    /// GPU sim runner for simulation kernels + buffer management.
    gpu_sim_runner: Option<GpuSimRunner>,
    /// GPU backend for pixel kernels.
    gpu_backend: Option<GpuBackend>,
    /// Whether a GPU render was requested this frame.
    gpu_rendered_this_frame: bool,
    /// WGSL source for deferred GpuBackend creation (pixel kernels).
    gpu_pixel_kernel_source: Option<String>,
    /// Resolved pixel kernel user args for GPU uniform buffer.
    gpu_user_args: Vec<f32>,
    /// Which buffer to present for GPU sim display.
    gpu_pixel_buffer_name: Option<String>,
    /// Whether the last frame used GPU rendering (for re-present).
    last_frame_was_gpu: bool,
    /// Loaded textures for GPU pixel kernel (references by index).
    gpu_textures: Vec<TextureData>,
    /// wgpu device (cloned from display during init_gpu).
    gpu_device: Option<wgpu::Device>,
    /// wgpu queue (cloned from display during init_gpu).
    gpu_queue: Option<wgpu::Queue>,
    /// Kernel handle → name mapping for GPU sim kernels.
    gpu_kernel_names: Vec<String>,
    /// Thread pool for parallel CPU kernel dispatch.
    thread_pool: Option<rayon::ThreadPool>,
    /// Registered keypress handlers: Key tag → JIT'd function pointer.
    keypress_handlers: HashMap<i32, *const u8>,
    /// Registered keydown handlers: Key tag → JIT'd function pointer.
    keydown_handlers: HashMap<i32, *const u8>,
    /// Registered keyup handlers: Key tag → JIT'd function pointer.
    keyup_handlers: HashMap<i32, *const u8>,
    /// Cached buffer handle for auto-allocated PDC pixel kernel output.
    pdc_pixel_buffer: Option<i32>,
    /// Registered mouse down handler.
    mousedown_handler: Option<*const u8>,
    /// Registered mouse up handler.
    mouseup_handler: Option<*const u8>,
    /// Registered click handler.
    click_handler: Option<*const u8>,
}

#[allow(dead_code)]
struct NamedBuffer {
    name: String,
    data: Vec<u8>,
    elem_size: usize,
}

#[derive(Clone)]
#[allow(dead_code)]
struct BufferBinding {
    param_name: String,
    buffer_handle: i32,
    is_output: bool,
    /// Resolved CPU buffer slot (from CompiledWgslKernel::binding_map). -1 = unresolved.
    cpu_slot: i32,
    /// Resolved GPU binding index (from GpuPipeline::binding_map). -1 = unresolved.
    gpu_binding: i32,
}

#[derive(Clone)]
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
        let handle = self.buffers.len() as i32;
        let name = format!("buffer_{handle}");

        // GPU path: create buffer in GpuSimRunner
        if let Some(ref mut runner) = self.gpu_sim_runner {
            let gpu_type = parse_gpu_element_type(type_name);
            runner.add_buffer(&name, gpu_type);
            runner.init_buffer_constant(&name, init_value);
            let elem_size = gpu_type.byte_size() as usize;
            self.buffers.push(NamedBuffer {
                name,
                data: Vec::new(), // GPU-managed, no CPU data
                elem_size,
            });
            return handle;
        }

        // CPU path
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
        self.buffers.push(NamedBuffer {
            name,
            data,
            elem_size,
        });
        handle
    }

    fn buffer_data_ptr(&self, handle: i32) -> *mut u8 {
        let idx = handle as usize;
        if idx < self.buffers.len() && !self.buffers[idx].data.is_empty() {
            self.buffers[idx].data.as_ptr() as *mut u8
        } else {
            std::ptr::null_mut()
        }
    }

    fn buffer_len(&self, handle: i32) -> i32 {
        let idx = handle as usize;
        if idx < self.buffers.len() && self.buffers[idx].elem_size > 0 {
            (self.buffers[idx].data.len() / self.buffers[idx].elem_size) as i32
        } else {
            0
        }
    }

    fn buffer_elem_size(&self, handle: i32) -> i32 {
        let idx = handle as usize;
        if idx < self.buffers.len() {
            self.buffers[idx].elem_size as i32
        } else {
            0
        }
    }

    fn swap_buffers(&mut self, a: i32, b: i32) {
        let (ai, bi) = (a as usize, b as usize);
        if ai < self.buffers.len() && bi < self.buffers.len() && ai != bi {
            // GPU path: swap in GpuSimRunner
            if let Some(ref mut runner) = self.gpu_sim_runner {
                let name_a = self.buffers[ai].name.clone();
                let name_b = self.buffers[bi].name.clone();
                runner.swap_buffers(&name_a, &name_b);
            } else {
                // CPU path: swap the data vecs
                let ptr_a = &mut self.buffers[ai].data as *mut Vec<u8>;
                let ptr_b = &mut self.buffers[bi].data as *mut Vec<u8>;
                unsafe { std::ptr::swap(ptr_a, ptr_b); }
            }
        }
    }

    fn load_kernel(&mut self, name: &str, path: &str, kind: i32) -> i32 {
        let full_path = self.base_dir.join(path);
        let wgsl_src = match std::fs::read_to_string(&full_path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[pdc-runtime] failed to read kernel '{}': {}", full_path.display(), e);
                return -1;
            }
        };

        // GPU path — when GPU resources are available (init_gpu or init_gpu_headless was called)
        let gpu_available = self.gpu_device.is_some() || self.gpu_sim_runner.is_some();
        if self.render == "gpu" && gpu_available {
            if kind == 0 {
                // Pixel kernel — store WGSL source for deferred GpuBackend creation
                self.gpu_pixel_kernel_source = Some(wgsl_src);
                // Return a pseudo-handle; pixel kernel dispatch is handled specially
                return 0;
            } else {
                // Sim kernel (kind=1) — compile into GpuSimRunner pipeline
                if let Some(ref mut runner) = self.gpu_sim_runner {
                    if let Err(e) = runner.add_pipeline(name, &wgsl_src) {
                        eprintln!("[pdc-runtime] failed to compile GPU pipeline '{}': {}", name, e);
                        return -1;
                    }
                    let handle = self.gpu_kernel_names.len() as i32;
                    self.gpu_kernel_names.push(name.to_string());
                    return handle;
                }
            }
        }

        // CPU path
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

    fn bind_buffer(&mut self, kernel_handle: i32, param_name: &str, buffer_handle: i32, is_output: bool) {
        let kh = kernel_handle as usize;
        // Grow per-kernel state if needed
        while self.kernel_config.bindings.len() <= kh {
            self.kernel_config.bindings.push(Vec::new());
        }
        while self.kernel_config.binding_name_cache.len() <= kh {
            self.kernel_config.binding_name_cache.push(HashMap::new());
        }
        // Look up cached index, or insert and cache
        let cache = &mut self.kernel_config.binding_name_cache[kh];
        let bindings = &mut self.kernel_config.bindings[kh];
        if let Some(&idx) = cache.get(param_name) {
            bindings[idx].buffer_handle = buffer_handle;
            bindings[idx].is_output = is_output;
        } else {
            // Resolve CPU slot from compiled kernel binding_map
            #[cfg(any(feature = "cranelift-backend", feature = "llvm-backend"))]
            let cpu_slot = if kh < self.cpu_kernels.len() {
                self.cpu_kernels[kh].compiled.binding_map.get(param_name)
                    .map(|&s| s as i32).unwrap_or(-1)
            } else { -1 };
            #[cfg(not(any(feature = "cranelift-backend", feature = "llvm-backend")))]
            let cpu_slot = -1i32;

            // Resolve GPU binding index from GpuSimRunner pipeline
            let gpu_binding = if let Some(ref runner) = self.gpu_sim_runner {
                let kernel_name = if kh < self.gpu_kernel_names.len() {
                    Some(self.gpu_kernel_names[kh].as_str())
                } else { None };
                kernel_name.and_then(|name| runner.binding_index(name, param_name))
                    .map(|b| b as i32).unwrap_or(-1)
            } else { -1 };

            let idx = bindings.len();
            cache.insert(param_name.to_string(), idx);
            bindings.push(BufferBinding {
                param_name: param_name.to_string(),
                buffer_handle,
                is_output,
                cpu_slot,
                gpu_binding,
            });
        }
    }

    fn set_kernel_arg_f64(&mut self, kernel_handle: i32, name: &str, value: f64) {
        let kh = kernel_handle as usize;
        while self.kernel_config.args.len() <= kh {
            self.kernel_config.args.push(Vec::new());
        }
        while self.kernel_config.arg_name_cache.len() <= kh {
            self.kernel_config.arg_name_cache.push(HashMap::new());
        }
        // Look up cached index, or insert and cache
        let cache = &mut self.kernel_config.arg_name_cache[kh];
        let args = &mut self.kernel_config.args[kh];
        if let Some(&idx) = cache.get(name) {
            args[idx].value = value;
        } else {
            let idx = args.len();
            cache.insert(name.to_string(), idx);
            args.push(KernelArg { name: name.to_string(), value });
        }
    }

    fn set_kernel_arg_f32(&mut self, kernel_handle: i32, name: &str, value: f32) {
        self.set_kernel_arg_f64(kernel_handle, name, value as f64);
    }

    fn run_kernel(&mut self, kernel_handle: i32) {
        let kh = kernel_handle as usize;
        // Borrow kernel config (bindings/args) and mutable state separately.
        // This works because kernel_config is a separate field from buffers, etc.
        let empty_bindings = Vec::new();
        let empty_args = Vec::new();
        let bindings = if kh < self.kernel_config.bindings.len() {
            &self.kernel_config.bindings[kh]
        } else {
            &empty_bindings
        };
        let args = if kh < self.kernel_config.args.len() {
            &self.kernel_config.args[kh]
        } else {
            &empty_args
        };

        // GPU path — when GPU resources are available
        let gpu_available = self.gpu_device.is_some() || self.gpu_sim_runner.is_some();
        if self.render == "gpu" && gpu_available {
            // Check if this is a pixel kernel (handle 0 and gpu_pixel_kernel_source is set)
            if self.gpu_pixel_kernel_source.is_some() && kernel_handle == 0 {
                // Pixel kernel: resolve user args, mark as GPU rendered
                self.gpu_user_args = args.iter()
                    .map(|a| a.value as f32)
                    .collect();
                self.gpu_rendered_this_frame = true;
                return;
            }

            // Sim kernel: dispatch via GpuSimRunner
            if let Some(ref runner) = self.gpu_sim_runner {
                let kernel_name = if kh < self.gpu_kernel_names.len() {
                    self.gpu_kernel_names[kh].clone()
                } else {
                    eprintln!("[pdc-runtime] invalid GPU kernel handle {}", kernel_handle);
                    return;
                };

                // Build bindings: map PDC buffer handles to GpuSimRunner buffer names
                let gpu_bindings: Vec<(String, String)> = bindings.iter()
                    .filter_map(|b| {
                        let buf_idx = b.buffer_handle as usize;
                        if buf_idx < self.buffers.len() {
                            Some((b.param_name.clone(), self.buffers[buf_idx].name.clone()))
                        } else {
                            None
                        }
                    })
                    .collect();
                let binding_refs: Vec<(&str, &str)> = gpu_bindings.iter()
                    .map(|(a, b)| (a.as_str(), b.as_str()))
                    .collect();

                // Resolve user args in WGSL Params struct field order (not PDC insertion order).
                // Each arg is matched by name to the struct field, ensuring correct byte layout.
                let user_args_spec = runner.user_args(&kernel_name);
                let mut arg_bytes: Vec<u8> = Vec::new();
                for spec in user_args_spec {
                    let value = args.iter()
                        .find(|a| a.name == spec.name)
                        .map(|a| a.value)
                        .unwrap_or(0.0);
                    match spec.arg_type {
                        gpu::sim_runner::WgslArgType::U32 => {
                            arg_bytes.extend_from_slice(&(value as u32).to_le_bytes());
                        }
                        gpu::sim_runner::WgslArgType::I32 => {
                            arg_bytes.extend_from_slice(&(value as i32).to_le_bytes());
                        }
                        gpu::sim_runner::WgslArgType::F32 => {
                            arg_bytes.extend_from_slice(&(value as f32).to_le_bytes());
                        }
                    }
                }
                runner.dispatch(&kernel_name, &binding_refs, &arg_bytes);
            }

            return;
        }

        // CPU path
        #[cfg(any(feature = "cranelift-backend", feature = "llvm-backend"))]
        {
            let handle = kernel_handle as usize;
            if handle >= self.cpu_kernels.len() {
                eprintln!("[pdc-runtime] invalid kernel handle {}", kernel_handle);
                return;
            }

            // Build buffer pointer array based on bindings.
            // Uses pre-resolved cpu_slot indices (cached at bind time).
            let kernel = &self.cpu_kernels[handle];
            let num_buffers = kernel.compiled.num_storage_buffers;
            let mut buffer_ptrs: Vec<*mut u8> = vec![std::ptr::null_mut(); num_buffers];

            for binding in bindings {
                let buf_idx = binding.buffer_handle as usize;
                if buf_idx >= self.buffers.len() { continue; }
                let slot = if binding.cpu_slot >= 0 {
                    binding.cpu_slot as usize
                } else if let Some(&s) = kernel.compiled.binding_map.get(&binding.param_name) {
                    s
                } else {
                    continue;
                };
                if slot < num_buffers {
                    buffer_ptrs[slot] = self.buffers[buf_idx].data.as_mut_ptr();
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
                    "sample_count" => params[off..off + 4].copy_from_slice(&sample_index.wrapping_add(1).to_le_bytes()),
                    "time" => params[off..off + 4].copy_from_slice(&(time as f32).to_le_bytes()),
                    name if name.starts_with('_') => {} // padding
                    _ => {
                        // Check kernel args first (by name)
                        if let Some(arg) = args.iter().find(|a| a.name == member.name) {
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

            // Parallel tiled dispatch via rayon.
            let fn_ptr = kernel.compiled.fn_ptr;
            let tile_h = 16usize;
            let params_ref = &params;
            // Convert buffer pointers to usize for Send safety across threads.
            let buf_addrs: Vec<usize> = buffer_ptrs.iter().map(|p| *p as usize).collect();
            let tex_addr = if tex_slots.is_empty() {
                0usize
            } else {
                tex_slots.as_ptr() as usize
            };

            let dispatch = || {
                use rayon::prelude::*;
                let num_tiles = (h as usize + tile_h - 1) / tile_h;
                (0..num_tiles).into_par_iter().for_each(|tile| {
                    let row_start = (tile * tile_h) as u32;
                    let row_end = ((tile + 1) * tile_h).min(h as usize) as u32;
                    let buf_ptrs: Vec<*mut u8> = buf_addrs.iter()
                        .map(|&addr| addr as *mut u8)
                        .collect();
                    let tex_ptr = if tex_addr == 0 {
                        std::ptr::null()
                    } else {
                        tex_addr as *const u8
                    };
                    unsafe {
                        (fn_ptr)(
                            params_ref.as_ptr(),
                            buf_ptrs.as_ptr() as *const *mut u8,
                            tex_ptr,
                            w, h, w,
                            row_start, row_end,
                        );
                    }
                });
            };

            match &self.thread_pool {
                Some(pool) => pool.install(dispatch),
                None => dispatch(),
            }
        }

    }

    fn display(&mut self) {
        let gpu_available = self.gpu_device.is_some() || self.gpu_sim_runner.is_some();
        if self.render == "gpu" && gpu_available && self.gpu_pixel_kernel_source.is_some() {
            // GPU pixel kernel — mark as rendered, actual render in render_gpu_frame
            self.gpu_rendered_this_frame = true;
        }
        self.display_requested = true;
    }

    fn render_pdc_kernel(&mut self, ctx: *mut PdcContext, kernel_fn: *const u8, buffer_handle: i32) -> i32 {
        let w = self.width;
        let h = self.height;
        let pixel_count = (w as usize) * (h as usize);

        // Resolve output buffer: use provided handle or auto-allocate.
        let buf_handle = if buffer_handle >= 0 && (buffer_handle as usize) < self.buffers.len() {
            buffer_handle
        } else {
            // Auto-allocate or reuse cached buffer
            match self.pdc_pixel_buffer {
                Some(h) => h,
                None => {
                    let h = self.create_buffer("gpu_u32", 0.0);
                    self.pdc_pixel_buffer = Some(h);
                    h
                }
            }
        };

        // Ensure buffer is correctly sized
        let buf_idx = buf_handle as usize;
        let needed = pixel_count * 4; // u32 per pixel
        if self.buffers[buf_idx].data.len() != needed {
            self.buffers[buf_idx].data.resize(needed, 0);
        }

        let output_ptr = self.buffers[buf_idx].data.as_mut_ptr() as usize;

        // The JIT'd kernel has signature:
        //   extern "C" fn(*mut PdcContext, i64, i64, i64, i64) -> i64
        // (PDC promotes i32 args to i64 at the ABI level)
        type KernelFn = unsafe extern "C" fn(*mut PdcContext, i64, i64, i64, i64) -> i64;

        // Create a thread-safe PdcContext for kernel execution:
        // - Same builtins, state, and host pointers (read-only during dispatch)
        // - Host pointer is needed for buffer data access (pdc_buffer_data_ptr)
        // - Null scene pointer (not needed for pixel kernels)
        let kernel_ctx = unsafe {
            PdcContext {
                builtins: (*ctx).builtins,
                scene: std::ptr::null_mut(),
                state: (*ctx).state,
                host: (*ctx).host,
            }
        };

        let tile_h = 16usize;
        let fn_addr = kernel_fn as usize;
        // Convert pointers to usize for Send safety across rayon threads.
        let ctx_builtins = kernel_ctx.builtins as usize;
        let ctx_state = kernel_ctx.state as usize;
        let ctx_host = kernel_ctx.host as usize;

        let dispatch = || {
            use rayon::prelude::*;
            let num_tiles = (h as usize + tile_h - 1) / tile_h;
            (0..num_tiles).into_par_iter().for_each(|tile| {
                let row_start = tile * tile_h;
                let row_end = ((tile + 1) * tile_h).min(h as usize);
                let kernel: KernelFn = unsafe { std::mem::transmute(fn_addr) };
                let out = output_ptr as *mut u32;

                // Each thread gets its own PdcContext copy (on stack)
                let mut thread_ctx = PdcContext {
                    builtins: ctx_builtins as *mut f64,
                    scene: std::ptr::null_mut(),
                    state: ctx_state as *mut u8,
                    host: ctx_host as *mut u8,
                };

                for y in row_start..row_end {
                    for x in 0..w as usize {
                        let color = unsafe {
                            kernel(
                                &mut thread_ctx as *mut PdcContext,
                                x as i64,
                                y as i64,
                                w as i64,
                                h as i64,
                            )
                        };
                        unsafe {
                            *out.add(y * w as usize + x) = color as u32;
                        }
                    }
                }
            });
        };

        match &self.thread_pool {
            Some(pool) => pool.install(dispatch),
            None => dispatch(),
        }

        buf_handle
    }

    fn display_buffer(&mut self, buffer_handle: i32) {
        let handle = buffer_handle as usize;
        if handle < self.buffers.len() {
            let has_cpu_data = !self.buffers[handle].data.is_empty();
            if self.gpu_sim_runner.is_some() && !has_cpu_data {
                // GPU sim path — buffer lives on GPU, mark for GPU present
                self.gpu_rendered_this_frame = true;
                self.gpu_pixel_buffer_name = Some(self.buffers[handle].name.clone());
            } else {
                // CPU path — buffer has CPU data (e.g., from PDC pixel kernel),
                // copy to pixel_buffer for display via softbuffer.
                let buf = &self.buffers[handle];
                let pixel_count = self.pixel_buffer.len();
                let src = &buf.data;
                for i in 0..pixel_count.min(src.len() / 4) {
                    let offset = i * 4;
                    self.pixel_buffer[i] = u32::from_le_bytes([
                        src[offset], src[offset + 1], src[offset + 2], src[offset + 3],
                    ]);
                }
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
            let buf_name = format!("__scene_{}_{}", idx, name);
            let elem_size = if name == "segments" { 16 } else { 4 };

            // GPU path: upload scene data to GpuSimRunner
            if let Some(ref mut runner) = self.gpu_sim_runner {
                runner.add_buffer_with_data(&buf_name, &data);
            }

            self.buffers.push(NamedBuffer {
                name: buf_name,
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
                // Store a copy for GPU backend creation if needed
                if self.render == "gpu" {
                    self.gpu_textures.push(tex.clone());
                }
                self.textures.push(tex);
                handle
            }
            Err(e) => {
                eprintln!("[pdc-runtime] failed to load texture '{}': {}", full_path.display(), e);
                -1
            }
        }
    }

    fn was_display_requested(&self) -> bool { self.display_requested }
    fn clear_display_requested(&mut self) { self.display_requested = false; }
    fn pixel_buffer(&self) -> &[u32] { &self.pixel_buffer }
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
    fn set_thread_pool(&mut self, pool: Option<rayon::ThreadPool>) {
        self.thread_pool = pool;
    }

    fn set_render(&mut self, mode: &str) { self.render = mode.to_string(); }
    fn set_codegen(&mut self, backend: &str) { self.codegen = backend.to_string(); }

    fn init_gpu(&mut self, display: &Display) {
        if self.render != "gpu" {
            return;
        }

        // Store device/queue for deferred GpuBackend creation
        self.gpu_device = Some(display.device.clone());
        self.gpu_queue = Some(display.queue.clone());

        // Create GpuSimRunner for sim kernels + buffer management
        if self.gpu_sim_runner.is_none() {
            self.gpu_sim_runner = Some(GpuSimRunner::new(display, self.width, self.height));
        }
    }

    fn gpu_rendered_this_frame(&self) -> bool { self.gpu_rendered_this_frame }

    fn end_frame_gpu(&mut self) {
        self.last_frame_was_gpu = self.gpu_rendered_this_frame;
        self.gpu_rendered_this_frame = false;
    }

    fn last_frame_was_gpu(&self) -> bool { self.last_frame_was_gpu }

    fn render_gpu_frame(&self, display: &Display) -> bool {
        if !self.gpu_rendered_this_frame {
            return false;
        }

        // GPU pixel kernel path
        if let Some(ref gpu) = self.gpu_backend {
            let center_x = self.builtins_snapshot[B::CENTER_X];
            let center_y = self.builtins_snapshot[B::CENTER_Y];
            let zoom = self.builtins_snapshot[B::ZOOM].max(1e-10);
            let time = self.builtins_snapshot[B::TIME];
            let sample_index = self.builtins_snapshot[B::SAMPLE_INDEX] as u32;
            gpu.render(
                display,
                center_x, center_y, zoom,
                sample_index, sample_index.wrapping_add(1),
                time,
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

    fn re_present_gpu_frame(&self, display: &Display) -> bool {
        if !self.last_frame_was_gpu {
            return false;
        }

        // GPU pixel kernel path
        if let Some(ref gpu) = self.gpu_backend {
            let center_x = self.builtins_snapshot[B::CENTER_X];
            let center_y = self.builtins_snapshot[B::CENTER_Y];
            let zoom = self.builtins_snapshot[B::ZOOM].max(1e-10);
            let time = self.builtins_snapshot[B::TIME];
            let sample_index = self.builtins_snapshot[B::SAMPLE_INDEX] as u32;
            gpu.render(
                display,
                center_x, center_y, zoom,
                sample_index, sample_index.wrapping_add(1),
                time,
                &self.gpu_user_args,
            );
            return true;
        }

        // GPU sim kernel path
        if let Some(ref runner) = self.gpu_sim_runner {
            if let Some(ref buf_name) = self.gpu_pixel_buffer_name {
                runner.present_pixels(display, buf_name);
                return true;
            }
        }

        false
    }

    fn is_gpu_render(&self) -> bool { self.render == "gpu" }
    fn render_mode(&self) -> &str { &self.render }
    fn codegen_backend(&self) -> &str { &self.codegen }

    fn init_gpu_headless(&mut self) {
        if self.render != "gpu" {
            return;
        }
        // Create headless GpuSimRunner (owns its own device/queue)
        if self.gpu_sim_runner.is_none() {
            let runner = GpuSimRunner::new_headless(self.width, self.height);
            self.gpu_sim_runner = Some(runner);
        }
        // Mark as having a GPU device so load_kernel/run_kernel use GPU paths.
        // For headless we don't have a wgpu::Device to clone from Display,
        // so we use a sentinel — the actual device lives inside GpuSimRunner.
        // For pixel kernels we'll create a headless GpuBackend in finalize_gpu_pixel_kernel.
        // Set gpu_device to signal "GPU is available" — we use new_headless for the backend.
    }

    fn finalize_gpu_pixel_kernel(&mut self) {
        if self.gpu_backend.is_some() || self.gpu_pixel_kernel_source.is_none() {
            return;
        }
        if let (Some(device), Some(queue)) = (&self.gpu_device, &self.gpu_queue) {
            // Interactive mode: create from stored device/queue
            let source = self.gpu_pixel_kernel_source.as_ref().unwrap();
            let tex_refs: Vec<&TextureData> = self.gpu_textures.iter().collect();
            self.gpu_backend = Some(GpuBackend::build(
                device, queue, self.width, self.height, 256, source, &tex_refs,
            ));
        } else if self.gpu_sim_runner.is_some() {
            // Headless mode: create headless GpuBackend with its own device
            let source = self.gpu_pixel_kernel_source.as_ref().unwrap();
            let tex_refs: Vec<&TextureData> = self.gpu_textures.iter().collect();
            let (backend, device, queue) = GpuBackend::new_headless(
                self.width, self.height, 256, source, &tex_refs,
            );
            self.gpu_backend = Some(backend);
            self.gpu_device = Some(device);
            self.gpu_queue = Some(queue);
        }
    }

    fn readback_gpu_pixels(&mut self) {
        // Pixel kernel path: dispatch headless and read back
        if let Some(ref gpu) = self.gpu_backend {
            if let (Some(device), Some(queue)) = (&self.gpu_device, &self.gpu_queue) {
                let center_x = self.builtins_snapshot[B::CENTER_X];
                let center_y = self.builtins_snapshot[B::CENTER_Y];
                let zoom = self.builtins_snapshot[B::ZOOM].max(1e-10);
                let time = self.builtins_snapshot[B::TIME];
                let sample_index = self.builtins_snapshot[B::SAMPLE_INDEX] as u32;
                gpu.dispatch_compute(
                    device, queue,
                    center_x, center_y, zoom,
                    sample_index, sample_index.wrapping_add(1),
                    time,
                    &self.gpu_user_args,
                );
                self.pixel_buffer = gpu.readback_pixels(device, queue);
                return;
            }
        }

        // Sim kernel path: read back the display buffer from GpuSimRunner
        if let Some(ref runner) = self.gpu_sim_runner {
            if let Some(ref buf_name) = self.gpu_pixel_buffer_name {
                self.pixel_buffer = runner.readback_buffer(buf_name);
            }
        }
    }

    // ── Event handler registration ──

    fn set_keypress_handler(&mut self, key: i32, fn_ptr: *const u8) {
        self.keypress_handlers.insert(key, fn_ptr);
    }
    fn clear_keypress_handler(&mut self, key: i32) {
        self.keypress_handlers.remove(&key);
    }
    fn set_keydown_handler(&mut self, key: i32, fn_ptr: *const u8) {
        self.keydown_handlers.insert(key, fn_ptr);
    }
    fn clear_keydown_handler(&mut self, key: i32) {
        self.keydown_handlers.remove(&key);
    }
    fn set_keyup_handler(&mut self, key: i32, fn_ptr: *const u8) {
        self.keyup_handlers.insert(key, fn_ptr);
    }
    fn clear_keyup_handler(&mut self, key: i32) {
        self.keyup_handlers.remove(&key);
    }
    fn set_mousedown_handler(&mut self, fn_ptr: *const u8) {
        self.mousedown_handler = Some(fn_ptr);
    }
    fn clear_mousedown_handler(&mut self) {
        self.mousedown_handler = None;
    }
    fn set_mouseup_handler(&mut self, fn_ptr: *const u8) {
        self.mouseup_handler = Some(fn_ptr);
    }
    fn clear_mouseup_handler(&mut self) {
        self.mouseup_handler = None;
    }
    fn set_click_handler(&mut self, fn_ptr: *const u8) {
        self.click_handler = Some(fn_ptr);
    }
    fn clear_click_handler(&mut self) {
        self.click_handler = None;
    }
    fn get_keypress_handler(&self, key: i32) -> Option<*const u8> {
        self.keypress_handlers.get(&key).copied()
    }
    fn get_keydown_handler(&self, key: i32) -> Option<*const u8> {
        self.keydown_handlers.get(&key).copied()
    }
    fn get_keyup_handler(&self, key: i32) -> Option<*const u8> {
        self.keyup_handlers.get(&key).copied()
    }
    fn get_mousedown_handler(&self) -> Option<*const u8> {
        self.mousedown_handler
    }
    fn get_mouseup_handler(&self) -> Option<*const u8> {
        self.mouseup_handler
    }
    fn get_click_handler(&self) -> Option<*const u8> {
        self.click_handler
    }
}

/// Map PDC type name strings to GpuElementType.
fn parse_gpu_element_type(type_name: &str) -> GpuElementType {
    match type_name {
        "gpu_f32" => GpuElementType::F32,
        "gpu_i32" => GpuElementType::I32,
        "gpu_u32" => GpuElementType::U32,
        "gpu_vec2_f32" => GpuElementType::Vec2F32,
        "gpu_vec3_f32" => GpuElementType::Vec3F32,
        "gpu_vec4_f32" => GpuElementType::Vec4F32,
        _ => GpuElementType::F32,
    }
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
/// Implements the public interface expected by `main.rs`.
pub struct PdcRuntime {
    compiled: CompiledProgram,
    #[allow(dead_code)]
    state_layout: StateLayout,
    state_block: Box<[u8]>,
    builtins: [f64; B::COUNT],
    scene_builder: SceneBuilder,
    host: Box<dyn PipelineHost>,

    // Public fields used by main.rs
    pub width: u32,
    pub height: u32,
    pub mouse_x: f64,
    pub mouse_y: f64,
    pub mouse_down: bool,
    pub has_gpu_kernels: bool,
    pub pixel_buffer: Vec<u32>,
    pub tile_height: usize,

    title: String,
    paused: bool,
    frame: u64,
    frames_executed: u64,
    /// Whether the last frame() call returned true (requesting another frame).
    frame_requested_redraw: bool,
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
            kernel_config: KernelConfig {
                bindings: Vec::new(),
                args: Vec::new(),
                binding_name_cache: Vec::new(),
                arg_name_cache: Vec::new(),
            },
            display_requested: false,
            textures: Vec::new(),
            scenes: Vec::new(),
            builtins_snapshot: [0.0; B::COUNT],
            base_dir: base_dir.to_path_buf(),
            render: "gpu".to_string(),
            codegen: if cfg!(feature = "cranelift-backend") {
                "cranelift".to_string()
            } else {
                "llvm".to_string()
            },
            gpu_sim_runner: None,
            gpu_backend: None,
            gpu_rendered_this_frame: false,
            gpu_pixel_kernel_source: None,
            gpu_user_args: Vec::new(),
            gpu_pixel_buffer_name: None,
            last_frame_was_gpu: false,
            gpu_textures: Vec::new(),
            gpu_device: None,
            gpu_queue: None,
            gpu_kernel_names: Vec::new(),
            thread_pool: None,
            pdc_pixel_buffer: None,
            keypress_handlers: HashMap::new(),
            keydown_handlers: HashMap::new(),
            keyup_handlers: HashMap::new(),
            mousedown_handler: None,
            mouseup_handler: None,
            click_handler: None,
        });

        let title = source_path
            .map(|p| p.file_stem().unwrap_or_default().to_string_lossy().to_string())
            .unwrap_or_else(|| "PDC Pipeline".to_string());

        let mut builtins = [0.0f64; B::COUNT];
        builtins[B::WIDTH] = width as f64;
        builtins[B::HEIGHT] = height as f64;
        builtins[B::ZOOM] = 1.0;
        // Default to "no accumulation" sentinel — shaders check sample_index
        // != 0xFFFFFFFF before jittering, and sample_count > 0 before
        // accumulating.  Scripts set sample_index = 0 to enable accumulation.
        builtins[B::SAMPLE_INDEX] = 0xFFFFFFFFu32 as f64;

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
            has_gpu_kernels: true, // default render mode is GPU
            pixel_buffer,
            tile_height: 16,
            title,
            paused: false,
            frame: 0,
            frames_executed: 0,
            frame_requested_redraw: false,
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

    // ── Public interface ──

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
                    if value == "gpu" || value == "cpu" {
                        self.host.set_render(value);
                        self.has_gpu_kernels = value == "gpu";
                    }
                }
                "codegen" => {
                    self.host.set_codegen(value);
                }
                "title" => {
                    self.title = value.to_string();
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

    /// Set the thread pool for parallel CPU kernel dispatch.
    /// The pool is moved into the runtime and used for all subsequent kernel calls.
    pub fn set_thread_pool(&mut self, pool: Option<rayon::ThreadPool>) {
        self.host.set_thread_pool(pool);
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

        // Finalize GPU pixel kernel now that init has loaded WGSL source
        self.host.finalize_gpu_pixel_kernel();

        // Update has_gpu_kernels based on render mode after settings have been applied.
        self.has_gpu_kernels = self.host.is_gpu_render();

        // Copy host pixel buffer to our pixel buffer (CPU path only)
        if !self.host.gpu_rendered_this_frame() {
            self.sync_pixels();
        }
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

        // Skip if this frame was already executed and the script didn't
        // request another redraw (static scene, no animation/accumulation).
        if self.frame <= self.frames_executed && !self.frame_requested_redraw {
            return false;
        }

        self.builtins[B::TIME] = time;
        self.host.end_frame_gpu(); // save last_frame_was_gpu, clear gpu_rendered_this_frame
        self.host.update_builtins(&self.builtins);
        self.host.clear_display_requested();

        // Clear accumulated scene data from the previous frame to avoid
        // leaking handles created by per-frame host function calls.
        self.scene_builder = SceneBuilder::new();

        // Mouse edge detection: fire event handlers on transitions
        let mouse_down_edge = self.mouse_down && !self.mouse_was_down;
        let mouse_up_edge = !self.mouse_down && self.mouse_was_down;
        self.mouse_was_down = self.mouse_down;

        let mut ctx = self.make_ctx();

        if mouse_down_edge {
            if let Some(fn_ptr) = self.host.get_click_handler() {
                unsafe { call_handler(fn_ptr, &mut ctx); }
            }
        }
        if self.mouse_down {
            if let Some(fn_ptr) = self.host.get_mousedown_handler() {
                unsafe { call_handler(fn_ptr, &mut ctx); }
            }
        }
        if mouse_up_edge {
            if let Some(fn_ptr) = self.host.get_mouseup_handler() {
                unsafe { call_handler(fn_ptr, &mut ctx); }
            }
        }

        self.frame_requested_redraw = unsafe { self.compiled.call_frame(&mut ctx).unwrap() };
        self.read_back_builtins();

        if self.host.was_display_requested() && !self.host.gpu_rendered_this_frame() {
            self.sync_pixels();
        }

        self.frames_executed = self.frame;
        true
    }

    pub fn init_gpu(&mut self, display: &Display) {
        self.host.init_gpu(display);
        if self.host.is_gpu_render() {
            self.has_gpu_kernels = true;
        }
    }

    /// Initialize GPU resources headlessly (no display/window).
    /// Used for benchmarking and output-only modes.
    pub fn init_gpu_headless(&mut self) {
        self.host.init_gpu_headless();
    }

    /// Dispatch and read back GPU pixel data if GPU rendered this frame.
    /// Used in headless/benchmark modes where there is no display surface.
    pub fn flush_gpu_frame(&mut self) {
        if self.host.gpu_rendered_this_frame() {
            self.host.readback_gpu_pixels();
            self.sync_pixels();
        }
    }

    pub fn handle_keypress(&mut self, key_tag: i32) -> bool {
        if let Some(fn_ptr) = self.host.get_keypress_handler(key_tag) {
            let mut ctx = self.make_ctx();
            unsafe { call_handler(fn_ptr, &mut ctx); }
            self.read_back_builtins();
        }
        false
    }

    pub fn handle_keydown(&mut self, key_tag: i32) -> bool {
        if let Some(fn_ptr) = self.host.get_keydown_handler(key_tag) {
            let mut ctx = self.make_ctx();
            unsafe { call_handler(fn_ptr, &mut ctx); }
            self.read_back_builtins();
        }
        false
    }

    pub fn handle_keyup(&mut self, key_tag: i32) -> bool {
        if let Some(fn_ptr) = self.host.get_keyup_handler(key_tag) {
            let mut ctx = self.make_ctx();
            unsafe { call_handler(fn_ptr, &mut ctx); }
            self.read_back_builtins();
        }
        false
    }

    pub fn render_gpu_frame(&self, display: &Display) -> bool {
        self.host.render_gpu_frame(display)
    }

    pub fn re_present_gpu_frame(&self, display: &Display) -> bool {
        self.host.re_present_gpu_frame(display)
    }

    pub fn display_pixels(&self) -> &[u32] {
        &self.pixel_buffer
    }

    pub fn needs_continuous_redraw(&self) -> bool {
        if self.paused {
            return false;
        }
        // The frame() function returns true to request continuous redraws
        self.frame_requested_redraw
    }

    pub fn title(&self) -> String {
        self.title.clone()
    }

    /// Return the render mode ("gpu" or "cpu").
    pub fn render_mode(&self) -> &str {
        self.host.render_mode()
    }

    /// Return the codegen backend ("cranelift" or "llvm").
    pub fn codegen_backend(&self) -> &str {
        self.host.codegen_backend()
    }

    pub fn execute_gpu_headless(&mut self) {
        // Initialize headless GPU resources (GpuSimRunner with its own device)
        self.host.init_gpu_headless();

        // Run init block — this calls load_kernel/create_buffer which now
        // use the GPU path since gpu_sim_runner is available.
        self.execute_init_block(&None);

        // Run one frame
        self.execute_frame(0.0, &None);

        // Read back GPU pixels to host's pixel_buffer, then sync to ours
        self.host.readback_gpu_pixels();
        self.sync_pixels();
    }

    // ── Internal ──

    fn sync_pixels(&mut self) {
        // Copy from host's pixel buffer to our pixel buffer
        let src = self.host.pixel_buffer();
        self.pixel_buffer.copy_from_slice(src);
    }
}

/// Call a registered event handler function pointer with the given PdcContext.
///
/// # Safety
/// The `fn_ptr` must be a valid JIT'd function with signature `extern "C" fn(*mut PdcContext)`.
unsafe fn call_handler(fn_ptr: *const u8, ctx: &mut PdcContext) {
    unsafe {
        let f: extern "C" fn(*mut PdcContext) = std::mem::transmute(fn_ptr);
        f(ctx);
    }
}

/// Map winit KeyCode to Key enum tag (must match variant order in type_check.rs).
pub fn key_code_to_tag(code: winit::keyboard::KeyCode) -> Option<i32> {
    use winit::keyboard::KeyCode;
    Some(match code {
        KeyCode::Space => 0,           // Key.Space
        KeyCode::ArrowLeft => 1,       // Key.Left
        KeyCode::ArrowRight => 2,      // Key.Right
        KeyCode::ArrowUp => 3,         // Key.Up
        KeyCode::ArrowDown => 4,       // Key.Down
        KeyCode::Equal | KeyCode::NumpadAdd => 5,      // Key.Plus
        KeyCode::Minus | KeyCode::NumpadSubtract => 6,  // Key.Minus
        KeyCode::BracketLeft => 7,     // Key.BracketLeft
        KeyCode::BracketRight => 8,    // Key.BracketRight
        KeyCode::Digit0 | KeyCode::Numpad0 => 9,   // Key.Digit0
        KeyCode::Digit1 | KeyCode::Numpad1 => 10,  // Key.Digit1
        KeyCode::Digit2 | KeyCode::Numpad2 => 11,  // Key.Digit2
        KeyCode::Digit3 | KeyCode::Numpad3 => 12,  // Key.Digit3
        KeyCode::KeyR => 13,           // Key.R
        KeyCode::KeyQ => 14,           // Key.Q
        KeyCode::Escape => 15,         // Key.Escape
        KeyCode::Period => 16,         // Key.Period
        KeyCode::Comma => 17,          // Key.Comma
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    /// Helper to create a PdcRuntime from source, run init + one frame, return pixel buffer.
    fn run_pdc_frame(src: &str, width: u32, height: u32) -> Vec<u32> {
        let base_dir = Path::new(".");
        let mut rt = PdcRuntime::new(src, None, width, height, base_dir)
            .expect("PdcRuntime::new failed");
        rt.host.set_render("cpu");
        rt.execute_init_block(&None);
        rt.execute_frame(0.0, &None);
        rt.pixel_buffer.clone()
    }

    #[test]
    fn render_solid_color() {
        let src = r#"
fn pixel(x: i32, y: i32, w: i32, h: i32) -> i32 {
    return i32(0xFF00FF00)
}

fn frame() -> bool {
    var buf = render(pixel)
    display_buffer(buf)
    return false
}
"#;
        let pixels = run_pdc_frame(src, 4, 4);
        assert_eq!(pixels.len(), 16);
        for &p in &pixels {
            assert_eq!(p, 0xFF00FF00, "expected solid green, got 0x{:08X}", p);
        }
    }

    #[test]
    fn render_gradient_varies_by_x() {
        let src = r#"
fn pixel(x: i32, y: i32, w: i32, h: i32) -> i32 {
    var r = x * 255 / (w - 1)
    return i32(0xFF000000) | (r << 16)
}

fn frame() -> bool {
    var buf = render(pixel)
    display_buffer(buf)
    return false
}
"#;
        let pixels = run_pdc_frame(src, 8, 2);
        // First column should be black (r=0), last column should have r=255
        let first = pixels[0];
        let last = pixels[7];
        assert_eq!(first & 0x00FF0000, 0x00000000, "first pixel red channel should be 0");
        assert_eq!(last & 0x00FF0000, 0x00FF0000, "last pixel red channel should be 255");
        // Pixels should vary across x
        assert_ne!(pixels[0], pixels[7]);
    }

    #[test]
    fn render_with_explicit_buffer() {
        let src = r#"
fn pixel(x: i32, y: i32, w: i32, h: i32) -> i32 {
    return i32(0xFFFF0000)
}

fn frame() -> bool {
    var buf = Buffer.U32()
    render(pixel, buf)
    display_buffer(buf)
    return false
}
"#;
        let pixels = run_pdc_frame(src, 4, 4);
        assert_eq!(pixels.len(), 16);
        for &p in &pixels {
            assert_eq!(p, 0xFFFF0000, "expected solid red, got 0x{:08X}", p);
        }
    }

    #[test]
    fn render_reads_module_scope() {
        let src = r#"
var color_value = i32(0xFF0000FF)

fn pixel(x: i32, y: i32, w: i32, h: i32) -> i32 {
    return color_value
}

fn frame() -> bool {
    var buf = render(pixel)
    display_buffer(buf)
    return false
}
"#;
        let pixels = run_pdc_frame(src, 2, 2);
        for &p in &pixels {
            assert_eq!(p, 0xFF0000FF, "expected blue from module var, got 0x{:08X}", p);
        }
    }

    #[test]
    fn render_multipass_ping_pong() {
        // First pass: write red. Second pass: read from first pass buffer (module scope).
        let src = r#"
var buf_a = Buffer.U32()
var buf_b = Buffer.U32()

fn pass1(x: i32, y: i32, w: i32, h: i32) -> i32 {
    return i32(0xFFFF0000)
}

fn pass2(x: i32, y: i32, w: i32, h: i32) -> i32 {
    return i32(0xFF00FF00)
}

fn frame() -> bool {
    render(pass1, buf_a)
    render(pass2, buf_b)
    display_buffer(buf_b)
    return false
}
"#;
        let pixels = run_pdc_frame(src, 2, 2);
        for &p in &pixels {
            assert_eq!(p, 0xFF00FF00, "expected green from pass2, got 0x{:08X}", p);
        }
    }
}
