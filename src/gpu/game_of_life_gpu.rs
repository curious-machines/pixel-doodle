use crate::display::Display;

use super::aligned_bytes_per_row;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuParams {
    width: u32,
    height: u32,
    stride: u32,
    _pad0: u32,
    center_x: f32,
    center_y: f32,
    zoom: f32,
    _pad1: u32,
}

pub struct GpuGameOfLifeBackend {
    step_pipeline: wgpu::ComputePipeline,
    vis_pipeline: wgpu::ComputePipeline,
    step_layout: wgpu::BindGroupLayout,
    vis_layout: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
    /// Ping-pong grid buffers: i32 per cell (age encoding)
    grid_a: wgpu::Buffer,
    grid_b: wgpu::Buffer,
    /// Pixel output buffer for visualization (stride-aligned rows)
    pixel_buffer: wgpu::Buffer,
    width: u32,
    height: u32,
    stride: u32,
    /// Which buffer is currently "input" (false = A is input, true = B is input)
    ping: bool,
}

impl GpuGameOfLifeBackend {
    pub fn new(display: &Display, width: u32, height: u32, density: f64) -> Self {
        Self::build(&display.device, &display.queue, width, height, density)
    }

    fn build(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        density: f64,
    ) -> Self {
        let source = include_str!("game_of_life.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("game-of-life compute"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });

        let step_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gol step"),
            entries: &[
                bgl_uniform(0),
                bgl_storage_ro(1),
                bgl_storage_rw(2),
            ],
        });

        let vis_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gol vis"),
            entries: &[
                bgl_uniform(0),
                bgl_storage_ro(1),
                bgl_storage_rw(2),
            ],
        });

        let step_pipe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gol step"),
            bind_group_layouts: &[&step_layout],
            push_constant_ranges: &[],
        });

        let vis_pipe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gol vis"),
            bind_group_layouts: &[&vis_layout],
            push_constant_ranges: &[],
        });

        let step_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gol step"),
            layout: Some(&step_pipe_layout),
            module: &shader,
            entry_point: Some("step"),
            compilation_options: Default::default(),
            cache: None,
        });

        let vis_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gol vis"),
            layout: Some(&vis_pipe_layout),
            module: &shader,
            entry_point: Some("visualize"),
            compilation_options: Default::default(),
            cache: None,
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gol params"),
            size: std::mem::size_of::<GpuParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let n = (width * height) as u64;
        let grid_size = n * 4; // i32 = 4 bytes per cell
        let aligned_bpr = aligned_bytes_per_row(width);
        let stride = aligned_bpr / 4;

        let grid_a = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grid A"),
            size: grid_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let grid_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grid B"),
            size: grid_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pixel_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pixel output"),
            size: (aligned_bpr * height) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Initialize grid with random density
        let mut init_data = vec![0i32; n as usize];
        let mut rng = 98765u64;
        let mut next_rng = || -> f64 {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (rng >> 33) as f64 / (1u64 << 31) as f64
        };
        for cell in init_data.iter_mut() {
            if next_rng() < density {
                *cell = 1; // alive, age=1
            }
        }
        queue.write_buffer(&grid_a, 0, bytemuck::cast_slice(&init_data));

        // Upload params
        let gpu_params = GpuParams {
            width,
            height,
            stride,
            _pad0: 0,
            center_x: 0.0,
            center_y: 0.0,
            zoom: 1.0,
            _pad1: 0,
        };
        queue.write_buffer(&uniform_buffer, 0, bytemuck::bytes_of(&gpu_params));

        Self {
            step_pipeline,
            vis_pipeline,
            step_layout,
            vis_layout,
            uniform_buffer,
            grid_a,
            grid_b,
            pixel_buffer,
            width,
            height,
            stride,
            ping: false,
        }
    }

    /// Update viewport parameters for visualization.
    pub fn update_viewport(&self, queue: &wgpu::Queue, center_x: f32, center_y: f32, zoom: f32) {
        let gpu_params = GpuParams {
            width: self.width,
            height: self.height,
            stride: self.stride,
            _pad0: 0,
            center_x,
            center_y,
            zoom,
            _pad1: 0,
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&gpu_params));
    }

    /// Map screen pixel coordinates to grid cell coordinates through the viewport.
    pub fn screen_to_grid(&self, screen_x: f32, screen_y: f32, center_x: f32, center_y: f32, zoom: f32) -> (u32, u32) {
        let fw = self.width as f32;
        let fh = self.height as f32;
        let gx = (screen_x - fw * 0.5) / zoom + center_x * fw;
        let gy = (screen_y - fh * 0.5) / zoom + center_y * fh;
        let gx = ((gx.floor() as i32).rem_euclid(self.width as i32)) as u32;
        let gy = ((gy.floor() as i32).rem_euclid(self.height as i32)) as u32;
        (gx, gy)
    }

    /// Run one generation step and render to the display.
    pub fn step_and_render(&mut self, display: &Display) {
        let device = &display.device;

        let wg_x = (self.width + 15) / 16;
        let wg_y = (self.height + 15) / 16;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("game-of-life"),
        });

        // Step: ping-pong
        let (input, output) = if self.ping {
            (&self.grid_b, &self.grid_a)
        } else {
            (&self.grid_a, &self.grid_b)
        };

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gol step"),
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
                label: Some("gol step"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.step_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        self.ping = !self.ping;

        // Visualization pass: read from result of step
        let current_grid = if self.ping { &self.grid_b } else { &self.grid_a };

        let vis_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gol vis"),
            layout: &self.vis_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: current_grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.pixel_buffer.as_entire_binding(),
                },
            ],
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gol vis"),
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

    /// Render current state without stepping (for paused display).
    pub fn render_current(&self, display: &Display) {
        let device = &display.device;
        let wg_x = (self.width + 15) / 16;
        let wg_y = (self.height + 15) / 16;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gol vis-only"),
        });

        let current_grid = if self.ping { &self.grid_b } else { &self.grid_a };

        let vis_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gol vis"),
            layout: &self.vis_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: current_grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.pixel_buffer.as_entire_binding(),
                },
            ],
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gol vis"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.vis_pipeline);
            pass.set_bind_group(0, &vis_bind_group, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

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

    /// Inject live cells at pixel position (for mouse interaction).
    pub fn inject(&self, queue: &wgpu::Queue, px: u32, py: u32, radius: u32) {
        let grid = if self.ping { &self.grid_b } else { &self.grid_a };
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
                let offset = ((y * w + x) as u64) * 4;
                let data: [i32; 1] = [1]; // alive, age=1
                queue.write_buffer(grid, offset, bytemuck::cast_slice(&data));
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
