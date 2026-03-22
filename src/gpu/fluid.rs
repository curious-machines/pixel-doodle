use crate::display::Display;
use crate::simulation::GrayScottParams;

use super::aligned_bytes_per_row;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuParams {
    width: u32,
    height: u32,
    stride: u32,
    _pad0: u32,
    feed: f32,
    kill: f32,
    du: f32,
    dv: f32,
    dt: f32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

pub struct GpuFluidBackend {
    step_pipeline: wgpu::ComputePipeline,
    vis_pipeline: wgpu::ComputePipeline,
    step_layout: wgpu::BindGroupLayout,
    vis_layout: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
    /// Ping-pong field buffers: vec2<f32> per cell (u, v)
    field_a: wgpu::Buffer,
    field_b: wgpu::Buffer,
    /// Pixel output buffer for visualization (stride-aligned rows)
    pixel_buffer: wgpu::Buffer,
    width: u32,
    height: u32,
    stride: u32,
    /// Which buffer is currently "input" (false = A is input, true = B is input)
    ping: bool,
}

impl GpuFluidBackend {
    pub fn new(display: &Display, width: u32, height: u32, params: &GrayScottParams) -> Self {
        Self::build(&display.device, &display.queue, width, height, params)
    }

    fn build(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        params: &GrayScottParams,
    ) -> Self {
        let source = include_str!("gray_scott.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gray-scott compute"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });

        // Step bind group layout: uniform, field_in (read), field_out (read_write)
        let step_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gray-scott step"),
            entries: &[
                bgl_uniform(0),
                bgl_storage_ro(1),
                bgl_storage_rw(2),
            ],
        });

        // Vis bind group layout: uniform, field (read), pixels (read_write)
        let vis_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gray-scott vis"),
            entries: &[
                bgl_uniform(0),
                bgl_storage_ro(1),
                bgl_storage_rw(2),
            ],
        });

        let step_pipe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gray-scott step"),
            bind_group_layouts: &[&step_layout],
            push_constant_ranges: &[],
        });

        let vis_pipe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gray-scott vis"),
            bind_group_layouts: &[&vis_layout],
            push_constant_ranges: &[],
        });

        let step_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gray-scott step"),
            layout: Some(&step_pipe_layout),
            module: &shader,
            entry_point: Some("step"),
            compilation_options: Default::default(),
            cache: None,
        });

        let vis_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gray-scott vis"),
            layout: Some(&vis_pipe_layout),
            module: &shader,
            entry_point: Some("visualize"),
            compilation_options: Default::default(),
            cache: None,
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gray-scott params"),
            size: std::mem::size_of::<GpuParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let n = (width * height) as u64;
        let field_size = n * 8; // vec2<f32> = 8 bytes per cell
        let aligned_bpr = aligned_bytes_per_row(width);
        let stride = aligned_bpr / 4; // pixels per row in storage

        let field_a = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("field A"),
            size: field_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let field_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("field B"),
            size: field_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pixel_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pixel output"),
            size: (aligned_bpr * height) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Upload initial state
        let mut init_data = vec![[1.0f32, 0.0f32]; n as usize];
        let mut rng = 12345u64;
        let mut next = || -> u64 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            rng
        };
        let radius: usize = 6;
        for _ in 0..20 {
            let sx = (next() as usize) % (width as usize - 2 * radius) + radius;
            let sy = (next() as usize) % (height as usize - 2 * radius) + radius;
            for dy in -(radius as isize)..=(radius as isize) {
                for dx in -(radius as isize)..=(radius as isize) {
                    if dx * dx + dy * dy > (radius * radius) as isize {
                        continue;
                    }
                    let x = (sx as isize + dx) as usize;
                    let y = (sy as isize + dy) as usize;
                    init_data[y * width as usize + x] = [0.5, 0.25];
                }
            }
        }
        queue.write_buffer(&field_a, 0, bytemuck::cast_slice(&init_data));

        // Upload params
        let gpu_params = GpuParams {
            width,
            height,
            stride,
            _pad0: 0,
            feed: params.feed,
            kill: params.kill,
            du: params.du,
            dv: params.dv,
            dt: params.dt,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };
        queue.write_buffer(&uniform_buffer, 0, bytemuck::bytes_of(&gpu_params));

        Self {
            step_pipeline,
            vis_pipeline,
            step_layout,
            vis_layout,
            uniform_buffer,
            field_a,
            field_b,
            pixel_buffer,
            width,
            height,
            stride,
            ping: false,
        }
    }

    /// Run N substeps and render to the display.
    pub fn step_and_render(&mut self, display: &Display, substeps: u32) {
        let device = &display.device;

        let wg_x = (self.width + 15) / 16;
        let wg_y = (self.height + 15) / 16;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gray-scott"),
        });

        // Run substeps
        for _ in 0..substeps {
            let (input, output) = if self.ping {
                (&self.field_b, &self.field_a)
            } else {
                (&self.field_a, &self.field_b)
            };

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gray-scott step"),
                layout: &self.step_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: input.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output.as_entire_binding(),
                    },
                ],
            });

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("gray-scott step"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.step_pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(wg_x, wg_y, 1);
            }

            self.ping = !self.ping;
        }

        // Visualization pass: read from current input (which is the result of last step)
        let current_field = if self.ping { &self.field_b } else { &self.field_a };

        let vis_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gray-scott vis"),
            layout: &self.vis_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: current_field.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.pixel_buffer.as_entire_binding(),
                },
            ],
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gray-scott vis"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.vis_pipeline);
            pass.set_bind_group(0, &vis_bind_group, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // Copy pixel buffer to display texture
        let aligned_bpr = self.stride * 4;
        encoder.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo {
                buffer: &self.pixel_buffer,
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

    /// Inject chemical V at pixel position (for mouse interaction).
    pub fn inject(&self, queue: &wgpu::Queue, px: u32, py: u32, radius: u32) {
        // Write directly to the current input field
        let field = if self.ping { &self.field_b } else { &self.field_a };
        let w = self.width;
        let h = self.height;
        let r = radius as i32;

        for dy in -r..=r {
            for dx in -r..=r {
                if dx * dx + dy * dy > r * r {
                    continue;
                }
                let x = ((px as i32 + dx).rem_euclid(w as i32)) as u32;
                let y = ((py as i32 + dy).rem_euclid(h as i32)) as u32;
                let offset = ((y * w + x) as u64) * 8;
                let data: [f32; 2] = [0.0, 1.0];
                queue.write_buffer(field, offset, bytemuck::cast_slice(&data));
            }
        }
    }
}

fn bgl_uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_storage_ro(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_storage_rw(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
