use crate::display::Display;

const COPY_BYTES_PER_ROW_ALIGNMENT: u32 = 256;

/// Round up bytes_per_row to the required wgpu alignment.
fn aligned_bytes_per_row(width: u32) -> u32 {
    let unaligned = width * 4;
    let align = COPY_BYTES_PER_ROW_ALIGNMENT;
    (unaligned + align - 1) / align * align
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    width: u32,
    height: u32,
    max_iter: u32,
    stride: u32, // pixels per row in storage buffer (may be > width due to alignment)
    x_min: f32,
    y_min: f32,
    x_step: f32,
    y_step: f32,
}

pub struct GpuBackend {
    compute_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
    storage_buffer: wgpu::Buffer,
    width: u32,
    height: u32,
    max_iter: u32,
    stride: u32,
}

impl GpuBackend {
    pub fn new(display: &Display, width: u32, height: u32, max_iter: u32) -> Self {
        let device = &display.device;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mandelbrot compute"),
            source: wgpu::ShaderSource::Wgsl(include_str!("mandelbrot.wgsl").into()),
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gpu compute"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gpu compute"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mandelbrot compute"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let aligned_bpr = aligned_bytes_per_row(width);
        let stride = aligned_bpr / 4; // pixels per row in storage

        let storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pixel output"),
            size: (aligned_bpr * height) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        Self {
            compute_pipeline,
            bind_group_layout,
            uniform_buffer,
            storage_buffer,
            width,
            height,
            max_iter,
            stride,
        }
    }

    /// Dispatch the compute shader and copy results to the display texture.
    pub fn render(
        &self,
        display: &Display,
        center_x: f64,
        center_y: f64,
        zoom: f64,
    ) {
        let aspect = self.width as f64 / self.height as f64;
        let view_w = 3.5 / zoom;
        let view_h = view_w / aspect;
        let x_min = center_x - view_w / 2.0;
        let y_min = center_y - view_h / 2.0;
        let x_step = view_w / self.width as f64;
        let y_step = view_h / self.height as f64;

        let params = Params {
            width: self.width,
            height: self.height,
            max_iter: self.max_iter,
            stride: self.stride,
            x_min: x_min as f32,
            y_min: y_min as f32,
            x_step: x_step as f32,
            y_step: y_step as f32,
        };

        display
            .queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&params));

        let bind_group = display
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gpu compute"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.storage_buffer.as_entire_binding(),
                    },
                ],
            });

        let mut encoder =
            display
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("gpu compute"),
                });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mandelbrot"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.compute_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let wg_x = (self.width + 15) / 16;
            let wg_y = (self.height + 15) / 16;
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        let aligned_bpr = aligned_bytes_per_row(self.width);

        // Copy storage buffer → display texture
        encoder.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo {
                buffer: &self.storage_buffer,
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
}
