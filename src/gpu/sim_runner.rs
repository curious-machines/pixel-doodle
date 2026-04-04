use std::collections::HashMap;

use crate::display::Display;
use super::GpuElementType;

use super::aligned_bytes_per_row;

/// Standard uniform layout for GPU sim kernels.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SimParams {
    width: u32,
    height: u32,
    stride: u32,
    _pad: u32,
}

/// A named GPU storage buffer.
struct GpuBuffer {
    buffer: wgpu::Buffer,
    element_size: u32,
    element_type: GpuElementType,
}

/// Type of a user arg field in the WGSL Params struct.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WgslArgType {
    F32,
    U32,
    I32,
}

/// A compiled GPU compute pipeline with binding metadata.
#[allow(dead_code)]
struct GpuPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    /// Map from WGSL variable name to binding index.
    binding_map: HashMap<String, u32>,
    /// Total number of bindings (including uniform at 0).
    num_bindings: u32,
    /// Types of user arg fields in the Params struct (after SimParams).
    user_arg_types: Vec<WgslArgType>,
    /// User arg specs (names + types) in Params struct field order.
    user_args: Vec<WgslUserArg>,
}

/// Generic GPU simulation runner for named storage buffers and compute pipelines.
/// Dispatches compute shaders with buffer bindings resolved by name.
pub struct GpuSimRunner {
    device: wgpu::Device,
    queue: wgpu::Queue,
    buffers: HashMap<String, GpuBuffer>,
    pipelines: HashMap<String, GpuPipeline>,
    uniform_buffer: wgpu::Buffer,
    width: u32,
    height: u32,
    stride: u32,
}

impl GpuSimRunner {
    /// Create a new GPU sim runner using the display's device.
    pub fn new(display: &Display, width: u32, height: u32) -> Self {
        let stride = aligned_bytes_per_row(width) / 4;

        // 256 bytes: 16 for SimParams + 240 for user args (up to 60 f32 values)
        let uniform_buffer = display.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sim params"),
            size: 256,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Write initial uniform values
        let params = SimParams {
            width,
            height,
            stride,
            _pad: 0,
        };
        display
            .queue
            .write_buffer(&uniform_buffer, 0, bytemuck::bytes_of(&params));

        Self {
            device: display.device.clone(),
            queue: display.queue.clone(),
            buffers: HashMap::new(),
            pipelines: HashMap::new(),
            uniform_buffer,
            width,
            height,
            stride,
        }
    }

    /// Create a new GPU sim runner with its own headless device (no display needed).
    pub fn new_headless(width: u32, height: u32) -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("failed to find a suitable GPU adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("sim-runner headless"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            },
        ))
        .expect("failed to create device");

        let stride = aligned_bytes_per_row(width) / 4;
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sim params"),
            size: 256,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params = SimParams {
            width,
            height,
            stride,
            _pad: 0,
        };
        queue.write_buffer(&uniform_buffer, 0, bytemuck::bytes_of(&params));

        Self {
            device,
            queue,
            buffers: HashMap::new(),
            pipelines: HashMap::new(),
            uniform_buffer,
            width,
            height,
            stride,
        }
    }

    /// Read a named u32 pixel buffer back to CPU memory.
    pub fn readback_buffer(&self, buffer_name: &str) -> Vec<u32> {
        let gpu_buf = match self.buffers.get(buffer_name) {
            Some(b) => b,
            None => {
                eprintln!("warning: GPU buffer '{}' not found for readback", buffer_name);
                return vec![0u32; (self.width * self.height) as usize];
            }
        };

        let buf_size = gpu_buf.buffer.size();
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback staging"),
            size: buf_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback"),
        });
        encoder.copy_buffer_to_buffer(&gpu_buf.buffer, 0, &staging, 0, buf_size);
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::PollType::Wait).expect("GPU poll failed");

        let mapped = slice.get_mapped_range();
        let stride_pixels = self.stride as usize;
        let w = self.width as usize;
        let h = self.height as usize;

        // The GPU buffer has stride-aligned rows; compact to width×height.
        let gpu_data: &[u32] = bytemuck::cast_slice(&mapped);
        let mut out = vec![0u32; w * h];
        for row in 0..h {
            let src_start = row * stride_pixels;
            let dst_start = row * w;
            out[dst_start..dst_start + w].copy_from_slice(&gpu_data[src_start..src_start + w]);
        }
        out
    }

    /// Read back a named buffer as raw bytes.
    pub fn readback_buffer_bytes(&self, buffer_name: &str) -> Vec<u8> {
        let gpu_buf = match self.buffers.get(buffer_name) {
            Some(b) => b,
            None => return Vec::new(),
        };
        let buf_size = gpu_buf.buffer.size();
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback raw"),
            size: buf_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback raw"),
        });
        encoder.copy_buffer_to_buffer(&gpu_buf.buffer, 0, &staging, 0, buf_size);
        self.queue.submit(std::iter::once(encoder.finish()));
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::PollType::Wait).expect("GPU poll failed");
        slice.get_mapped_range().to_vec()
    }

    /// Allocate a named GPU storage buffer.
    pub fn add_buffer(
        &mut self,
        name: &str,
        element_type: GpuElementType,
    ) {
        let element_size = element_type.byte_size();
        let buf_size = (self.stride * self.height) as u64 * element_size as u64;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(name),
            size: buf_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.buffers.insert(
            name.to_string(),
            GpuBuffer {
                buffer,
                element_size,
                element_type,
            },
        );
    }

    /// Create or replace a named GPU buffer with the given data.
    ///
    /// Unlike `add_buffer`, this accepts arbitrary-sized data (not limited
    /// to width×height). Used for scene data buffers (segments, tile indices,
    /// etc.) whose size depends on the scene content.
    pub fn add_buffer_with_data(&mut self, name: &str, data: &[u8]) {
        let buf_size = data.len() as u64;
        // Ensure at least 16 bytes (wgpu minimum buffer size for bindings)
        let buf_size = buf_size.max(16);
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(name),
            size: buf_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !data.is_empty() {
            self.queue.write_buffer(&buffer, 0, data);
        }
        self.buffers.insert(
            name.to_string(),
            GpuBuffer {
                buffer,
                element_size: 1,
                element_type: GpuElementType::U32, // placeholder — WGSL knows the real type
            },
        );
    }

    /// Compile a WGSL shader and register it as a named pipeline.
    pub fn add_pipeline(
        &mut self,
        name: &str,
        wgsl_source: &str,
    ) -> Result<(), String> {
        let binding_map = parse_wgsl_bindings(wgsl_source);
        let user_args_parsed = parse_wgsl_user_args(wgsl_source);
        let user_arg_types: Vec<WgslArgType> = user_args_parsed.iter().map(|a| a.arg_type).collect();
        let num_bindings = binding_map
            .values()
            .copied()
            .max()
            .map(|m| m + 1)
            .unwrap_or(1); // at least binding 0 (uniform)

        // Build bind group layout entries
        let mut entries: Vec<wgpu::BindGroupLayoutEntry> = Vec::new();
        for i in 0..num_bindings {
            if i == 0 {
                // Binding 0 is always the uniform buffer
                entries.push(wgpu::BindGroupLayoutEntry {
                    binding: i,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                });
            } else {
                // Determine if this binding is read-only or read-write
                let is_read_only = is_binding_read_only(wgsl_source, i);
                entries.push(wgpu::BindGroupLayoutEntry {
                    binding: i,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: is_read_only,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                });
            }
        }

        let bind_group_layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(name),
            entries: &entries,
        });

        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(name),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(wgsl_source.into()),
        });

        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(name),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        self.pipelines.insert(
            name.to_string(),
            GpuPipeline {
                pipeline,
                bind_group_layout,
                binding_map,
                num_bindings,
                user_arg_types,
                user_args: user_args_parsed,
            },
        );
        Ok(())
    }

    /// Dispatch a compute shader with named buffer bindings and optional user args.
    ///
    /// `bindings` maps WGSL variable names to buffer names.
    /// `user_arg_bytes` are raw bytes written to the uniform buffer after SimParams
    /// (offset 16). The caller is responsible for encoding each arg with the correct
    /// type (f32 or u32) to match the WGSL kernel's Params struct layout.
    /// The uniform buffer (binding 0) is always bound automatically.
    pub fn dispatch(
        &self,
        pipeline_name: &str,
        bindings: &[(&str, &str)],
        user_arg_bytes: &[u8],
    ) {
        // Write user args to uniform buffer after SimParams (16 bytes)
        if !user_arg_bytes.is_empty() {
            self.queue.write_buffer(&self.uniform_buffer, 16, user_arg_bytes);
        }

        let gpu_pipeline = match self.pipelines.get(pipeline_name) {
            Some(p) => p,
            None => {
                eprintln!("warning: GPU pipeline '{}' not found", pipeline_name);
                return;
            }
        };

        // Build bind group entries
        let mut bg_entries: Vec<wgpu::BindGroupEntry> = Vec::new();

        // Binding 0: uniform buffer (always)
        bg_entries.push(wgpu::BindGroupEntry {
            binding: 0,
            resource: self.uniform_buffer.as_entire_binding(),
        });

        // Map named bindings to binding indices
        for (wgsl_name, buffer_name) in bindings {
            if let Some(&binding_idx) = gpu_pipeline.binding_map.get(*wgsl_name) {
                if let Some(gpu_buf) = self.buffers.get(*buffer_name) {
                    bg_entries.push(wgpu::BindGroupEntry {
                        binding: binding_idx,
                        resource: gpu_buf.buffer.as_entire_binding(),
                    });
                } else {
                    eprintln!(
                        "warning: GPU buffer '{}' not found for binding '{}'",
                        buffer_name, wgsl_name
                    );
                }
            }
            // If wgsl_name not in binding_map, skip (it might be an output assignment)
        }

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(pipeline_name),
            layout: &gpu_pipeline.bind_group_layout,
            entries: &bg_entries,
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(pipeline_name),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(pipeline_name),
                timestamp_writes: None,
            });
            pass.set_pipeline(&gpu_pipeline.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let wg_x = (self.width + 15) / 16;
            let wg_y = (self.height + 15) / 16;
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Return the WGSL user arg types for a named pipeline.
    pub fn user_arg_types(&self, pipeline_name: &str) -> &[WgslArgType] {
        self.pipelines.get(pipeline_name)
            .map(|p| p.user_arg_types.as_slice())
            .unwrap_or(&[])
    }

    /// Return the WGSL user arg specs (names + types) for a named pipeline.
    pub fn user_args(&self, pipeline_name: &str) -> &[WgslUserArg] {
        self.pipelines.get(pipeline_name)
            .map(|p| p.user_args.as_slice())
            .unwrap_or(&[])
    }

    /// Return the wgpu binding index for a named buffer parameter in a pipeline.
    pub fn binding_index(&self, pipeline_name: &str, param_name: &str) -> Option<u32> {
        self.pipelines.get(pipeline_name)
            .and_then(|p| p.binding_map.get(param_name).copied())
    }

    /// Swap two named buffer entries (O(1) pointer swap).
    pub fn swap_buffers(&mut self, a: &str, b: &str) {
        if let (Some(buf_a), Some(buf_b)) = (self.buffers.remove(a), self.buffers.remove(b)) {
            self.buffers.insert(a.to_string(), buf_b);
            self.buffers.insert(b.to_string(), buf_a);
        }
    }

    /// Copy a u32 pixel buffer to the display texture and present.
    pub fn present_pixels(&self, display: &Display, buffer_name: &str) {
        let gpu_buf = match self.buffers.get(buffer_name) {
            Some(b) => b,
            None => {
                eprintln!("warning: GPU pixel buffer '{}' not found", buffer_name);
                return;
            }
        };

        let aligned_bpr = aligned_bytes_per_row(self.width);

        let mut encoder = display
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("present pixels"),
            });

        encoder.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo {
                buffer: &gpu_buf.buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(aligned_bpr),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::TexelCopyTextureInfo {
                texture: display.texture(),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );

        display.present_with_commands(encoder.finish());
    }

    /// Initialize a buffer with constant data (zero-fill or value-fill).
    pub fn init_buffer_constant(
        &self,
        buffer_name: &str,
        value: f64,
    ) {
        let gpu_buf = match self.buffers.get(buffer_name) {
            Some(b) => b,
            None => return,
        };
        let total_elements = self.stride * self.height;
        let elem_size = gpu_buf.element_size as usize;
        let mut data = vec![0u8; total_elements as usize * elem_size];

        if value != 0.0 {
            let component_bytes = match gpu_buf.element_type {
                GpuElementType::I32 => (value as i32).to_le_bytes(),
                GpuElementType::U32 => (value as u32).to_le_bytes(),
                _ => (value as f32).to_le_bytes(),
            };
            let components_per_element = elem_size / 4;
            for i in 0..total_elements as usize {
                for j in 0..components_per_element {
                    let offset = i * elem_size + j * 4;
                    data[offset..offset + 4].copy_from_slice(&component_bytes);
                }
            }
        }

        self.queue.write_buffer(&gpu_buf.buffer, 0, &data);
    }

    /// Get the stride (pixels per row including alignment padding).
    #[allow(dead_code)]
    pub fn stride(&self) -> u32 {
        self.stride
    }
}

// ── WGSL binding parser ──

/// Parse WGSL source to extract variable name → binding index mapping.
///
/// Scans for patterns like:
///   `@binding(N) var<storage, read> name: array<...>;`
///   `@binding(N) var<uniform> name: Params;`
///
/// Returns only storage buffer bindings (not the uniform at binding 0).
fn parse_wgsl_bindings(source: &str) -> HashMap<String, u32> {
    let mut map = HashMap::new();

    for line in source.lines() {
        let line = line.trim();

        // Look for @binding(N)
        if let Some(binding_start) = line.find("@binding(") {
            let after_binding = &line[binding_start + 9..];
            if let Some(paren_end) = after_binding.find(')') {
                let binding_str = &after_binding[..paren_end];
                if let Ok(binding_idx) = binding_str.trim().parse::<u32>() {
                    // Skip binding 0 (uniform) — we handle that automatically
                    if binding_idx == 0 {
                        continue;
                    }
                    // Find the variable name: `var<...> name:`
                    if let Some(var_pos) = line.find("var<") {
                        let after_var = &line[var_pos..];
                        if let Some(gt_pos) = after_var.find('>') {
                            let after_gt = after_var[gt_pos + 1..].trim();
                            // Name is everything up to the next ':'
                            if let Some(colon_pos) = after_gt.find(':') {
                                let name = after_gt[..colon_pos].trim();
                                if !name.is_empty() {
                                    map.insert(name.to_string(), binding_idx);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    map
}

/// Check if a binding at a given index is read-only in the WGSL source.
fn is_binding_read_only(source: &str, binding_idx: u32) -> bool {
    let pattern = format!("@binding({})", binding_idx);
    for line in source.lines() {
        if line.contains(&pattern) {
            // Check if it's `var<storage, read>` (read-only) vs `var<storage, read_write>`
            if line.contains("read_write") {
                return false;
            }
            if line.contains("var<storage") {
                return true; // `var<storage, read>` or just `var<storage>`
            }
        }
    }
    false // default to read-write if unclear
}

/// Parse the WGSL `Params` struct to extract user arg types.
///
/// The Params struct has a fixed SimParams header (width, height, stride, _pad)
/// followed by user-defined fields. This function returns the types of those
/// user-defined fields in declaration order, skipping padding fields (names
/// starting with `_`).
/// A user arg field parsed from the WGSL Params struct: name and type.
#[derive(Debug, Clone)]
pub struct WgslUserArg {
    pub name: String,
    pub arg_type: WgslArgType,
}

fn parse_wgsl_user_args(source: &str) -> Vec<WgslUserArg> {
    let sim_params = ["width", "height", "stride"];
    let mut args = Vec::new();
    let mut in_params = false;

    for line in source.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("struct Params") {
            in_params = true;
            continue;
        }
        if in_params {
            if trimmed == "}" || trimmed == "};" {
                break;
            }
            // Parse "name: type," lines
            if let Some(colon_pos) = trimmed.find(':') {
                let name = trimmed[..colon_pos].trim();
                let type_str = trimmed[colon_pos + 1..].trim().trim_end_matches(',');
                // Skip SimParams fields and padding
                if sim_params.contains(&name) || name.starts_with('_') {
                    continue;
                }
                let arg_type = match type_str {
                    "u32" => WgslArgType::U32,
                    "i32" => WgslArgType::I32,
                    _ => WgslArgType::F32,
                };
                args.push(WgslUserArg { name: name.to_string(), arg_type });
            }
        }
    }
    args
}

#[cfg(test)]
fn parse_wgsl_user_arg_types(source: &str) -> Vec<WgslArgType> {
    parse_wgsl_user_args(source).into_iter().map(|a| a.arg_type).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compare CPU JIT and GPU execution of a WGSL sim kernel on identical data.
    ///
    /// Creates a field with known velocity+density, runs one advect step on both
    /// GPU and CPU, and verifies the results match within float precision.
    #[cfg(feature = "cranelift-backend")]
    #[test]
    fn advect_cpu_gpu_match() {
        use crate::jit;

        let w: u32 = 400;
        let h: u32 = 300;
        let advect_src = std::fs::read_to_string("examples/sim/smoke/smoke_advect.wgsl")
            .expect("failed to read advect kernel");
        let inject_src = std::fs::read_to_string("examples/sim/shared/inject_vec.wgsl")
            .expect("failed to read inject kernel");

        // --- Create seed via inject kernel (matches smoke init) ---
        // First, run inject on both GPU and CPU to produce the field data
        let zero_field = vec![0u8; (w * h) as usize * 16]; // vec4<f32> = 16 bytes

        // Inject params: x, y, radius, value, falloff_quadratic, component
        let inject_x = w as f32 / 2.0;
        let inject_y = h as f32 * 0.8;
        let inject_radius = 30.0f32;
        let inject_value = -5.0f32;
        let inject_falloff = 1.0f32;
        let inject_component = 1.0f32; // velocity.y

        let mut inject_args = Vec::new();
        inject_args.extend_from_slice(&inject_x.to_le_bytes());
        inject_args.extend_from_slice(&inject_y.to_le_bytes());
        inject_args.extend_from_slice(&inject_radius.to_le_bytes());
        inject_args.extend_from_slice(&inject_value.to_le_bytes());
        inject_args.extend_from_slice(&inject_falloff.to_le_bytes());
        inject_args.extend_from_slice(&inject_component.to_le_bytes());
        // Two padding floats to match Params struct
        inject_args.extend_from_slice(&0.0f32.to_le_bytes());
        inject_args.extend_from_slice(&0.0f32.to_le_bytes());

        // --- GPU inject (use add_buffer for stride-padded allocation, matching production) ---
        let mut gpu_runner = GpuSimRunner::new_headless(w, h);
        gpu_runner.add_buffer("buf_in", GpuElementType::Vec4F32);
        gpu_runner.init_buffer_constant("buf_in", 0.0);
        gpu_runner.add_buffer("buf_out", GpuElementType::Vec4F32);
        gpu_runner.init_buffer_constant("buf_out", 0.0);
        gpu_runner.add_pipeline("inject", &inject_src).expect("GPU inject compile failed");
        gpu_runner.dispatch("inject", &[("buf_in", "buf_in"), ("buf_out", "buf_out")], &inject_args);
        // Swap: buf_out has velocity data, make it buf_in for advect
        gpu_runner.swap_buffers("buf_in", "buf_out");

        // Now inject density (component 2)
        let mut inject_args2 = Vec::new();
        inject_args2.extend_from_slice(&inject_x.to_le_bytes());
        inject_args2.extend_from_slice(&inject_y.to_le_bytes());
        inject_args2.extend_from_slice(&inject_radius.to_le_bytes());
        inject_args2.extend_from_slice(&1.0f32.to_le_bytes()); // density value
        inject_args2.extend_from_slice(&inject_falloff.to_le_bytes());
        inject_args2.extend_from_slice(&2.0f32.to_le_bytes()); // component = z (density)
        inject_args2.extend_from_slice(&0.0f32.to_le_bytes());
        inject_args2.extend_from_slice(&0.0f32.to_le_bytes());
        gpu_runner.dispatch("inject", &[("buf_in", "buf_in"), ("buf_out", "buf_out")], &inject_args2);
        gpu_runner.swap_buffers("buf_in", "buf_out");

        // Read back GPU field after inject (raw, width-packed)
        let gpu_field_raw = {
            let gpu_buf = gpu_runner.buffers.get("buf_in").unwrap();
            let buf_size = gpu_buf.buffer.size();
            let staging = gpu_runner.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("test readback inject"),
                size: buf_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut encoder = gpu_runner.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor { label: Some("readback") },
            );
            encoder.copy_buffer_to_buffer(&gpu_buf.buffer, 0, &staging, 0, buf_size);
            gpu_runner.queue.submit(std::iter::once(encoder.finish()));
            let slice = staging.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            gpu_runner.device.poll(wgpu::PollType::Wait).unwrap();
            slice.get_mapped_range().to_vec()
        };

        // --- CPU inject ---
        let cpu_inject = jit::wgsl_cranelift::compile_wgsl(&inject_src)
            .expect("CPU inject compile failed");
        let mut cpu_buf_in = zero_field.clone();
        let mut cpu_buf_out = zero_field.clone();

        let mut inject_params = [0u8; 256];
        inject_params[0..4].copy_from_slice(&w.to_le_bytes());
        inject_params[4..8].copy_from_slice(&h.to_le_bytes());
        inject_params[8..12].copy_from_slice(&w.to_le_bytes()); // stride = width
        // _pad = 0
        // User args at offset 16
        inject_params[16..16 + inject_args.len()].copy_from_slice(&inject_args);

        let mut inject_buf_ptrs: Vec<*mut u8> = vec![std::ptr::null_mut(); cpu_inject.num_storage_buffers];
        if let Some(&slot) = cpu_inject.binding_map.get("buf_in") {
            inject_buf_ptrs[slot] = cpu_buf_in.as_mut_ptr();
        }
        if let Some(&slot) = cpu_inject.binding_map.get("buf_out") {
            inject_buf_ptrs[slot] = cpu_buf_out.as_mut_ptr();
        }
        unsafe {
            (cpu_inject.fn_ptr)(
                inject_params.as_ptr(),
                inject_buf_ptrs.as_ptr() as *const *mut u8,
                std::ptr::null(), w, h, w, 0, h,
            );
        }
        // Swap: buf_out → buf_in
        std::mem::swap(&mut cpu_buf_in, &mut cpu_buf_out);

        // Second inject (density)
        inject_params[16..16 + inject_args2.len()].copy_from_slice(&inject_args2);
        if let Some(&slot) = cpu_inject.binding_map.get("buf_in") {
            inject_buf_ptrs[slot] = cpu_buf_in.as_mut_ptr();
        }
        if let Some(&slot) = cpu_inject.binding_map.get("buf_out") {
            inject_buf_ptrs[slot] = cpu_buf_out.as_mut_ptr();
        }
        unsafe {
            (cpu_inject.fn_ptr)(
                inject_params.as_ptr(),
                inject_buf_ptrs.as_ptr() as *const *mut u8,
                std::ptr::null(), w, h, w, 0, h,
            );
        }
        std::mem::swap(&mut cpu_buf_in, &mut cpu_buf_out);

        // --- Compare inject results ---
        let gpu_inject_floats: &[f32] = bytemuck::cast_slice(&gpu_field_raw);
        let cpu_inject_floats: &[f32] = bytemuck::cast_slice(&cpu_buf_in);
        let mut inject_max_diff: f32 = 0.0;
        let mut inject_diff_count = 0;
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) as usize;
                for c in 0..4 {
                    let gi = idx * 4 + c;
                    if gi < gpu_inject_floats.len() {
                        let diff = (gpu_inject_floats[gi] - cpu_inject_floats[gi]).abs();
                        if diff > inject_max_diff { inject_max_diff = diff; }
                        if diff > 1e-4 { inject_diff_count += 1; }
                    }
                }
            }
        }
        eprintln!("Inject comparison: max_diff={:.6}, mismatches={}", inject_max_diff, inject_diff_count);
        assert!(
            inject_max_diff < 1e-4,
            "CPU/GPU inject mismatch: max_diff={:.6}, {} differing components",
            inject_max_diff, inject_diff_count,
        );

        // --- Now run advect on both ---
        // Use the GPU-injected field as the canonical field_in for both paths
        // (to isolate advect from inject differences)
        let field_bytes: Vec<u8> = cpu_buf_in.clone(); // use CPU inject result (verified identical to GPU)

        // --- GPU advect (use add_buffer for stride-padded allocation, matching production) ---
        let mut gpu_runner = GpuSimRunner::new_headless(w, h);
        gpu_runner.add_buffer("field_in", GpuElementType::Vec4F32);
        gpu_runner.queue.write_buffer(
            &gpu_runner.buffers.get("field_in").unwrap().buffer,
            0,
            &field_bytes,
        );
        gpu_runner.add_buffer("field_out", GpuElementType::Vec4F32);
        gpu_runner
            .add_pipeline("advect", &advect_src)
            .expect("GPU pipeline compile failed");
        gpu_runner.dispatch(
            "advect",
            &[("field_in", "field_in"), ("field_out", "field_out")],
            &[],
        );

        // Readback GPU result — raw bytes, width-packed (kernels index with y*width+x)
        let gpu_raw = {
            let gpu_buf = gpu_runner.buffers.get("field_out").unwrap();
            let buf_size = gpu_buf.buffer.size();
            let staging = gpu_runner.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("test readback"),
                size: buf_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut encoder = gpu_runner.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor { label: Some("test readback") },
            );
            encoder.copy_buffer_to_buffer(&gpu_buf.buffer, 0, &staging, 0, buf_size);
            gpu_runner.queue.submit(std::iter::once(encoder.finish()));
            let slice = staging.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            gpu_runner.device.poll(wgpu::PollType::Wait).unwrap();
            let mapped = slice.get_mapped_range();
            mapped.to_vec()
        };
        // GPU buffer is stride-padded but kernel writes at y*width+x, so extract width-packed
        let gpu_floats: &[f32] = bytemuck::cast_slice(&gpu_raw);
        let mut gpu_field = vec![[0.0f32; 4]; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) as usize;
                let base = idx * 4;
                if base + 3 < gpu_floats.len() {
                    gpu_field[idx] = [
                        gpu_floats[base],
                        gpu_floats[base + 1],
                        gpu_floats[base + 2],
                        gpu_floats[base + 3],
                    ];
                }
            }
        }

        // --- CPU path ---
        let compiled = jit::wgsl_cranelift::compile_wgsl(&advect_src)
            .expect("CPU compile failed");
        let mut cpu_field_in = field_bytes.clone();
        let mut cpu_field_out = vec![0u8; field_bytes.len()];

        // Build params: width, height, stride, _pad (16 bytes)
        let mut params = [0u8; 256];
        params[0..4].copy_from_slice(&w.to_le_bytes());
        params[4..8].copy_from_slice(&h.to_le_bytes());
        params[8..12].copy_from_slice(&w.to_le_bytes()); // stride = width on CPU
        // _pad = 0 (already zero)

        // Build buffer pointers — must match binding_map order
        let mut buf_ptrs: Vec<*mut u8> = vec![std::ptr::null_mut(); compiled.num_storage_buffers];
        if let Some(&slot) = compiled.binding_map.get("field_in") {
            buf_ptrs[slot] = cpu_field_in.as_mut_ptr();
        }
        if let Some(&slot) = compiled.binding_map.get("field_out") {
            buf_ptrs[slot] = cpu_field_out.as_mut_ptr();
        }

        unsafe {
            (compiled.fn_ptr)(
                params.as_ptr(),
                buf_ptrs.as_ptr() as *const *mut u8,
                std::ptr::null(),
                w,
                h,
                w, // stride = width
                0,
                h, // process all rows
            );
        }

        // Parse CPU result
        let cpu_floats: &[f32] = bytemuck::cast_slice(&cpu_field_out);
        let mut cpu_field = vec![[0.0f32; 4]; (w * h) as usize];
        for i in 0..(w * h) as usize {
            let base = i * 4;
            cpu_field[i] = [
                cpu_floats[base],
                cpu_floats[base + 1],
                cpu_floats[base + 2],
                cpu_floats[base + 3],
            ];
        }

        // --- Compare ---
        let mut max_diff: f32 = 0.0;
        let mut diff_count = 0;
        let mut first_mismatch = None;
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) as usize;
                for c in 0..4 {
                    let diff = (gpu_field[idx][c] - cpu_field[idx][c]).abs();
                    if diff > max_diff {
                        max_diff = diff;
                    }
                    if diff > 1e-4 {
                        diff_count += 1;
                        if first_mismatch.is_none() {
                            first_mismatch = Some((x, y, c, gpu_field[idx][c], cpu_field[idx][c]));
                        }
                    }
                }
            }
        }

        if let Some((x, y, c, gpu_v, cpu_v)) = first_mismatch {
            eprintln!(
                "First mismatch at ({}, {}) component {}: GPU={:.6}, CPU={:.6}, diff={:.6}",
                x, y, c, gpu_v, cpu_v, (gpu_v - cpu_v).abs()
            );
            eprintln!(
                "Total mismatches: {} / {} (max_diff={:.6})",
                diff_count,
                w * h * 4,
                max_diff
            );
            // Print a few rows around the mismatch to help debug
            eprintln!("GPU field_out at y={}: {:?}", y, &gpu_field[(y * w) as usize..(y * w + w) as usize].iter().map(|v| v[2]).collect::<Vec<_>>()[..10]);
            eprintln!("CPU field_out at y={}: {:?}", y, &cpu_field[(y * w) as usize..(y * w + w) as usize].iter().map(|v| v[2]).collect::<Vec<_>>()[..10]);
        }

        assert!(
            max_diff < 1e-4,
            "CPU/GPU advect mismatch: max_diff={:.6}, {} differing components",
            max_diff,
            diff_count,
        );
    }

    #[test]
    fn parse_bindings_gray_scott() {
        let wgsl = r#"
            @group(0) @binding(0) var<uniform> params: Params;
            @group(0) @binding(1) var<storage, read> field_in: array<vec2<f32>>;
            @group(0) @binding(2) var<storage, read_write> field_out: array<vec2<f32>>;
        "#;
        let map = parse_wgsl_bindings(wgsl);
        assert_eq!(map.get("field_in"), Some(&1));
        assert_eq!(map.get("field_out"), Some(&2));
        assert!(!map.contains_key("params")); // binding 0 excluded
    }

    #[test]
    fn parse_bindings_jacobi() {
        let wgsl = r#"
            @group(0) @binding(0) var<uniform> params: Params;
            @group(0) @binding(1) var<storage, read> div_in: array<f32>;
            @group(0) @binding(2) var<storage, read> press_in: array<f32>;
            @group(0) @binding(3) var<storage, read_write> press_out: array<f32>;
        "#;
        let map = parse_wgsl_bindings(wgsl);
        assert_eq!(map.get("div_in"), Some(&1));
        assert_eq!(map.get("press_in"), Some(&2));
        assert_eq!(map.get("press_out"), Some(&3));
    }

    #[test]
    fn is_read_only_check() {
        let wgsl = r#"
            @group(0) @binding(1) var<storage, read> field_in: array<vec2<f32>>;
            @group(0) @binding(2) var<storage, read_write> field_out: array<vec2<f32>>;
        "#;
        assert!(is_binding_read_only(wgsl, 1));
        assert!(!is_binding_read_only(wgsl, 2));
    }

    #[test]
    fn parse_user_arg_types_no_user_args() {
        let wgsl = r#"
struct Params {
    width: u32,
    height: u32,
    stride: u32,
    _pad: u32,
}
        "#;
        let types = parse_wgsl_user_arg_types(wgsl);
        assert!(types.is_empty());
    }

    #[test]
    fn parse_user_arg_types_all_u32() {
        let wgsl = r#"
struct Params {
    width: u32,
    height: u32,
    stride: u32,
    _pad: u32,
    tile_size: u32,
    tiles_x: u32,
    num_paths: u32,
    _pad2: u32,
}
        "#;
        let types = parse_wgsl_user_arg_types(wgsl);
        assert_eq!(types, vec![WgslArgType::U32, WgslArgType::U32, WgslArgType::U32]);
    }

    #[test]
    fn parse_user_arg_types_mixed() {
        let wgsl = r#"
struct Params {
    width: u32,
    height: u32,
    stride: u32,
    _pad: u32,
    speed: f32,
    count: u32,
    threshold: f32,
}
        "#;
        let types = parse_wgsl_user_arg_types(wgsl);
        assert_eq!(types, vec![WgslArgType::F32, WgslArgType::U32, WgslArgType::F32]);
    }

    #[test]
    fn parse_user_arg_types_with_i32() {
        let wgsl = r#"
struct Params {
    width: u32,
    height: u32,
    stride: u32,
    _pad: u32,
    offset: i32,
}
        "#;
        let types = parse_wgsl_user_arg_types(wgsl);
        assert_eq!(types, vec![WgslArgType::I32]);
    }

    #[test]
    fn parse_user_arg_types_semicolon_style() {
        // WGSL allows both `,` and `;` terminated struct fields
        let wgsl = r#"
struct Params {
    width: u32,
    height: u32,
    stride: u32,
    _pad: u32,
    tile_size: u32,
};
        "#;
        let types = parse_wgsl_user_arg_types(wgsl);
        assert_eq!(types, vec![WgslArgType::U32]);
    }

    #[test]
    fn parse_bindings_tile_raster() {
        let wgsl = r#"
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> segments: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> seg_path_ids: array<u32>;
@group(0) @binding(3) var<storage, read> tile_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> tile_counts: array<u32>;
@group(0) @binding(5) var<storage, read> tile_indices: array<u32>;
@group(0) @binding(6) var<storage, read> path_colors: array<u32>;
@group(0) @binding(7) var<storage, read_write> pixels: array<u32>;
@group(0) @binding(8) var<storage, read> path_fill_rules: array<u32>;
        "#;
        let map = parse_wgsl_bindings(wgsl);
        assert_eq!(map.len(), 8); // 8 storage bindings (binding 0 excluded)
        assert_eq!(map.get("segments"), Some(&1));
        assert_eq!(map.get("seg_path_ids"), Some(&2));
        assert_eq!(map.get("tile_offsets"), Some(&3));
        assert_eq!(map.get("tile_counts"), Some(&4));
        assert_eq!(map.get("tile_indices"), Some(&5));
        assert_eq!(map.get("path_colors"), Some(&6));
        assert_eq!(map.get("pixels"), Some(&7));
        assert_eq!(map.get("path_fill_rules"), Some(&8));

        // Check read-only detection
        assert!(is_binding_read_only(wgsl, 1));  // segments: read
        assert!(!is_binding_read_only(wgsl, 7)); // pixels: read_write
        assert!(is_binding_read_only(wgsl, 8));  // path_fill_rules: read
    }
}
