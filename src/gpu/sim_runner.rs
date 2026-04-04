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
        let user_arg_types = parse_wgsl_user_arg_types(wgsl_source);
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
fn parse_wgsl_user_arg_types(source: &str) -> Vec<WgslArgType> {
    let sim_params = ["width", "height", "stride"];
    let mut types = Vec::new();
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
                types.push(arg_type);
            }
        }
    }
    types
}

#[cfg(test)]
mod tests {
    use super::*;

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
