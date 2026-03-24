use std::collections::HashMap;

use crate::display::Display;

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
}

/// Generic GPU simulation runner driven by PDC pipeline steps.
///
/// Holds named GPU storage buffers and compiled compute pipelines.
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

        let uniform_buffer = display.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sim params"),
            size: std::mem::size_of::<SimParams>() as u64,
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

    /// Allocate a named GPU storage buffer.
    pub fn add_buffer(
        &mut self,
        name: &str,
        element_size: u32,
    ) {
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
            },
        );
        Ok(())
    }

    /// Dispatch a compute shader with named buffer bindings.
    ///
    /// `bindings` maps WGSL variable names to PDC buffer names.
    /// The uniform buffer (binding 0) is always bound automatically.
    pub fn dispatch(
        &self,
        pipeline_name: &str,
        bindings: &[(&str, &str)],
    ) {
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

    /// Write injection data to a GPU buffer around a position.
    /// `value` is the raw bytes for one element.
    pub fn inject_raw(
        &self,
        buffer_name: &str,
        px: u32,
        py: u32,
        radius: u32,
        value: &[u8],
    ) {
        let gpu_buf = match self.buffers.get(buffer_name) {
            Some(b) => b,
            None => return,
        };
        let elem_size = gpu_buf.element_size as usize;
        let w = self.width as i32;
        let h = self.height as i32;
        let r = radius as i32;

        for dy in -r..=r {
            for dx in -r..=r {
                let x = px as i32 + dx;
                let y = py as i32 + dy;
                if x >= 0 && x < w && y >= 0 && y < h {
                    let d2 = (dx * dx + dy * dy) as f32;
                    let r2 = (r * r) as f32;
                    if d2 <= r2 {
                        let offset =
                            (y as u64 * self.stride as u64 + x as u64) * elem_size as u64;
                        self.queue.write_buffer(&gpu_buf.buffer, offset, value);
                    }
                }
            }
        }
    }

    /// Inject a f64 value into a GPU buffer, converting to the buffer's element type.
    /// For vec types, the value is written to the first component; others are zeroed.
    pub fn inject_value(
        &self,
        buffer_name: &str,
        px: u32,
        py: u32,
        radius: u32,
        value: f64,
    ) {
        let gpu_buf = match self.buffers.get(buffer_name) {
            Some(b) => b,
            None => return,
        };
        let elem_size = gpu_buf.element_size as usize;
        let mut bytes = vec![0u8; elem_size];
        let val_f32 = value as f32;

        match elem_size {
            4 => {
                // f32, i32, or u32
                bytes.copy_from_slice(&val_f32.to_le_bytes());
            }
            8 => {
                // vec2f — write value to both components for reaction-diffusion (u,v)
                // For gray-scott: inject v=value → vec2(0, value)
                // Default: write to second component (v channel)
                let zero_bytes = 0.0f32.to_le_bytes();
                let val_bytes = val_f32.to_le_bytes();
                bytes[0..4].copy_from_slice(&zero_bytes);
                bytes[4..8].copy_from_slice(&val_bytes);
            }
            16 => {
                // vec4f — write value to density (4th component) for smoke
                let val_bytes = val_f32.to_le_bytes();
                bytes[12..16].copy_from_slice(&val_bytes);
            }
            _ => {
                bytes[0..4].copy_from_slice(&val_f32.to_le_bytes());
            }
        }

        self.inject_raw(buffer_name, px, py, radius, &bytes);
    }

    /// Upload CPU-side f64 data to a GPU buffer, converting to the buffer's element type.
    pub fn upload_f64_data(
        &self,
        buffer_name: &str,
        data: &[f64],
    ) {
        let gpu_buf = match self.buffers.get(buffer_name) {
            Some(b) => b,
            None => return,
        };
        let elem_size = gpu_buf.element_size as usize;
        let stride = self.stride as usize;
        let w = self.width as usize;
        let h = self.height as usize;

        // Allocate stride-aligned GPU buffer
        let mut gpu_data = vec![0u8; stride * h * elem_size];

        for row in 0..h {
            for col in 0..w {
                let src_idx = row * w + col;
                let dst_idx = row * stride + col;
                let dst_offset = dst_idx * elem_size;
                let val = data.get(src_idx).copied().unwrap_or(0.0) as f32;
                let val_bytes = val.to_le_bytes();

                match elem_size {
                    4 => {
                        gpu_data[dst_offset..dst_offset + 4].copy_from_slice(&val_bytes);
                    }
                    8 => {
                        // vec2f: put value in first component
                        gpu_data[dst_offset..dst_offset + 4].copy_from_slice(&val_bytes);
                    }
                    16 => {
                        // vec4f: put value in first component
                        gpu_data[dst_offset..dst_offset + 4].copy_from_slice(&val_bytes);
                    }
                    _ => {
                        gpu_data[dst_offset..dst_offset + 4].copy_from_slice(&val_bytes);
                    }
                }
            }
        }

        self.queue.write_buffer(&gpu_buf.buffer, 0, &gpu_data);
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
            let val_f32 = value as f32;
            let bytes = val_f32.to_le_bytes();
            // Fill each f32 component with the value
            let floats_per_element = elem_size / 4;
            for i in 0..total_elements as usize {
                for j in 0..floats_per_element {
                    let offset = i * elem_size + j * 4;
                    data[offset..offset + 4].copy_from_slice(&bytes);
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
}
