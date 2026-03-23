use crate::display::Display;

use super::aligned_bytes_per_row;

const JACOBI_ITERATIONS: u32 = 40;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuSmokeParams {
    width: u32,
    height: u32,
    stride: u32,
    _pad0: u32,
    dt: f32,
    dissipation: f32,
    buoyancy: f32,
    _pad1: u32,
}

pub struct SmokeParams {
    pub dt: f32,
    pub dissipation: f32,
    pub buoyancy: f32,
}

impl Default for SmokeParams {
    fn default() -> Self {
        Self {
            dt: 4.0,
            dissipation: 0.998,
            buoyancy: 0.08,
        }
    }
}

pub struct GpuSmokeBackend {
    advect_pipeline: wgpu::ComputePipeline,
    div_pipeline: wgpu::ComputePipeline,
    jacobi_pipeline: wgpu::ComputePipeline,
    project_pipeline: wgpu::ComputePipeline,
    vis_pipeline: wgpu::ComputePipeline,

    advect_layout: wgpu::BindGroupLayout,
    div_layout: wgpu::BindGroupLayout,
    jacobi_layout: wgpu::BindGroupLayout,
    project_layout: wgpu::BindGroupLayout,
    vis_layout: wgpu::BindGroupLayout,

    uniform_buffer: wgpu::Buffer,
    /// Ping-pong field buffers: vec4<f32>(vx, vy, density, 0) per cell
    field_a: wgpu::Buffer,
    field_b: wgpu::Buffer,
    /// Ping-pong pressure buffers: f32 per cell
    pressure_a: wgpu::Buffer,
    pressure_b: wgpu::Buffer,
    /// Divergence buffer: f32 per cell
    div_buf: wgpu::Buffer,
    /// Pixel output buffer (stride-aligned)
    pixel_buffer: wgpu::Buffer,

    width: u32,
    height: u32,
    stride: u32,
    /// false = field_a is current input, true = field_b is current input
    ping: bool,
}

impl GpuSmokeBackend {
    pub fn new(display: &Display, width: u32, height: u32, params: &SmokeParams) -> Self {
        Self::build(&display.device, &display.queue, width, height, params)
    }

    fn build(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        params: &SmokeParams,
    ) -> Self {
        let source = include_str!("smoke.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("smoke compute"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });

        // --- Bind group layouts ---
        // advect: uniform, field_in(vec4 RO), field_out(vec4 RW)
        let advect_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("smoke advect"),
            entries: &[bgl_uniform(0), bgl_storage_ro(1), bgl_storage_rw(2)],
        });

        // divergence: uniform, field_in(vec4 RO), div_out(f32 RW)
        let div_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("smoke div"),
            entries: &[bgl_uniform(0), bgl_storage_ro(1), bgl_storage_rw(2)],
        });

        // jacobi: uniform, div_in(f32 RO), pressure_in(f32 RO), pressure_out(f32 RW)
        let jacobi_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("smoke jacobi"),
            entries: &[
                bgl_uniform(0),
                bgl_storage_ro(1),
                bgl_storage_ro(2),
                bgl_storage_rw(3),
            ],
        });

        // project: uniform, pressure_in(f32 RO), field_in(vec4 RO), field_out(vec4 RW)
        let project_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("smoke project"),
            entries: &[
                bgl_uniform(0),
                bgl_storage_ro(1),
                bgl_storage_ro(2),
                bgl_storage_rw(3),
            ],
        });

        // visualize: uniform, field_in(vec4 RO), pixels_out(u32 RW)
        let vis_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("smoke vis"),
            entries: &[bgl_uniform(0), bgl_storage_ro(1), bgl_storage_rw(2)],
        });

        // --- Pipelines ---
        let make_pipeline =
            |label: &str, layout: &wgpu::BindGroupLayout, entry: &str| -> wgpu::ComputePipeline {
                let pipe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(label),
                    bind_group_layouts: &[layout],
                    push_constant_ranges: &[],
                });
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout: Some(&pipe_layout),
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
            };

        let advect_pipeline = make_pipeline("smoke advect", &advect_layout, "advect");
        let div_pipeline = make_pipeline("smoke div", &div_layout, "divergence");
        let jacobi_pipeline = make_pipeline("smoke jacobi", &jacobi_layout, "jacobi");
        let project_pipeline = make_pipeline("smoke project", &project_layout, "project");
        let vis_pipeline = make_pipeline("smoke vis", &vis_layout, "visualize");

        // --- Buffers ---
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("smoke params"),
            size: std::mem::size_of::<GpuSmokeParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let n = (width * height) as u64;
        let field_size = n * 16; // vec4<f32> = 16 bytes
        let scalar_size = n * 4; // f32 = 4 bytes
        let aligned_bpr = aligned_bytes_per_row(width);
        let stride = aligned_bpr / 4;

        let make_field_buf = |label: &str| -> wgpu::Buffer {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: field_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        let make_scalar_buf = |label: &str| -> wgpu::Buffer {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: scalar_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        let field_a = make_field_buf("smoke field A");
        let field_b = make_field_buf("smoke field B");
        let pressure_a = make_scalar_buf("smoke pressure A");
        let pressure_b = make_scalar_buf("smoke pressure B");
        let div_buf = make_scalar_buf("smoke divergence");

        let pixel_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("smoke pixels"),
            size: (aligned_bpr * height) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Initialize fields to zero (no smoke, no velocity)
        let zero_field = vec![[0.0f32; 4]; n as usize];
        queue.write_buffer(&field_a, 0, bytemuck::cast_slice(&zero_field));

        let zero_scalar = vec![0.0f32; n as usize];
        queue.write_buffer(&pressure_a, 0, bytemuck::cast_slice(&zero_scalar));

        // Upload params
        let gpu_params = GpuSmokeParams {
            width,
            height,
            stride,
            _pad0: 0,
            dt: params.dt,
            dissipation: params.dissipation,
            buoyancy: params.buoyancy,
            _pad1: 0,
        };
        queue.write_buffer(&uniform_buffer, 0, bytemuck::bytes_of(&gpu_params));

        Self {
            advect_pipeline,
            div_pipeline,
            jacobi_pipeline,
            project_pipeline,
            vis_pipeline,
            advect_layout,
            div_layout,
            jacobi_layout,
            project_layout,
            vis_layout,
            uniform_buffer,
            field_a,
            field_b,
            pressure_a,
            pressure_b,
            div_buf,
            pixel_buffer,
            width,
            height,
            stride,
            ping: false,
        }
    }

    /// Run one simulation step and render to display.
    pub fn step_and_render(&mut self, display: &Display) {
        let device = &display.device;
        let wg_x = (self.width + 15) / 16;
        let wg_y = (self.height + 15) / 16;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("smoke"),
        });

        // --- 1. Advect: field_current -> field_other ---
        {
            let (input, output) = if self.ping {
                (&self.field_b, &self.field_a)
            } else {
                (&self.field_a, &self.field_b)
            };

            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("smoke advect"),
                layout: &self.advect_layout,
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

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("smoke advect"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.advect_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        self.ping = !self.ping;

        // --- 2. Divergence: field_current -> div_buf ---
        {
            let current = if self.ping { &self.field_b } else { &self.field_a };

            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("smoke div"),
                layout: &self.div_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: current.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.div_buf.as_entire_binding(),
                    },
                ],
            });

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("smoke div"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.div_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // --- 3. Jacobi pressure solve (N iterations) ---
        {
            let mut pressure_ping = false; // false = pressure_a is input
            for _ in 0..JACOBI_ITERATIONS {
                let (p_in, p_out) = if pressure_ping {
                    (&self.pressure_b, &self.pressure_a)
                } else {
                    (&self.pressure_a, &self.pressure_b)
                };

                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("smoke jacobi"),
                    layout: &self.jacobi_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.uniform_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: self.div_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: p_in.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: p_out.as_entire_binding(),
                        },
                    ],
                });

                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("smoke jacobi"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.jacobi_pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(wg_x, wg_y, 1);

                pressure_ping = !pressure_ping;
            }

            // After even iterations, result is in pressure_a; after odd, in pressure_b.
            // JACOBI_ITERATIONS=40 is even, so result is in pressure_a. We use that below.
        }

        // --- 4. Project: field_current + pressure -> field_other ---
        {
            let (f_in, f_out) = if self.ping {
                (&self.field_b, &self.field_a)
            } else {
                (&self.field_a, &self.field_b)
            };
            // After 40 (even) Jacobi iterations starting from pressure_a,
            // result is back in pressure_a.
            let pressure = &self.pressure_a;

            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("smoke project"),
                layout: &self.project_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: pressure.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: f_in.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: f_out.as_entire_binding(),
                    },
                ],
            });

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("smoke project"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.project_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        self.ping = !self.ping;

        // --- 5. Visualize: field_current -> pixels ---
        {
            let current = if self.ping { &self.field_b } else { &self.field_a };

            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("smoke vis"),
                layout: &self.vis_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: current.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.pixel_buffer.as_entire_binding(),
                    },
                ],
            });

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("smoke vis"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.vis_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // --- 6. Copy pixel buffer to display texture ---
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

    /// Inject smoke density and upward velocity at pixel position.
    pub fn inject(&self, queue: &wgpu::Queue, px: u32, py: u32, radius: u32) {
        let field = if self.ping { &self.field_b } else { &self.field_a };
        let w = self.width;
        let h = self.height;
        let r = radius as i32;

        for dy in -r..=r {
            for dx in -r..=r {
                let d2 = dx * dx + dy * dy;
                let r2 = r * r;
                if d2 > r2 {
                    continue;
                }

                let x = (px as i32 + dx).clamp(0, w as i32 - 1) as u32;
                let y = (py as i32 + dy).clamp(0, h as i32 - 1) as u32;
                let offset = ((y * w + x) as u64) * 16; // vec4<f32> = 16 bytes

                let t = 1.0 - (d2 as f32 / r2 as f32);
                // vx=0, vy=upward impulse, density=smoke, pad=0
                let data: [f32; 4] = [0.0, -3.0 * t, 0.5 * t, 0.0];
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
