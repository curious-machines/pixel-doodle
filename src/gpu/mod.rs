pub mod sim_runner;

use crate::display::Display;
use crate::texture::TextureData;

const COPY_BYTES_PER_ROW_ALIGNMENT: u32 = 256;

/// Round up bytes_per_row to the required wgpu alignment.
pub fn aligned_bytes_per_row(width: u32) -> u32 {
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
    stride: u32,
    x_min: f32,
    y_min: f32,
    x_step: f32,
    y_step: f32,
    sample_index: u32,
    sample_count: u32,
    time: f32,
    _pad: [u32; 1],
}

/// A GPU-side texture resource (texture view + uploaded data).
struct GpuTexture {
    view: wgpu::TextureView,
}

pub struct GpuBackend {
    compute_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
    storage_buffer: wgpu::Buffer,
    accum_buffer: wgpu::Buffer,
    /// Sampler shared by all textures (created only when textures are present).
    sampler: Option<wgpu::Sampler>,
    /// GPU texture resources, one per declared texture.
    gpu_textures: Vec<GpuTexture>,
    width: u32,
    height: u32,
    max_iter: u32,
    stride: u32,
}

impl GpuBackend {
    /// Build pipeline and buffers on the given device.
    fn build(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        max_iter: u32,
        wgsl_source: &str,
        textures: &[&TextureData],
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("compute kernel"),
            source: wgpu::ShaderSource::Wgsl(wgsl_source.into()),
        });

        // Base layout entries: uniform (0), storage output (1), accum (2)
        let mut layout_entries = vec![
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
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ];

        // If textures are present: binding 3 = sampler, binding 4+ = texture_2d
        if !textures.is_empty() {
            layout_entries.push(wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            });
            for i in 0..textures.len() {
                layout_entries.push(wgpu::BindGroupLayoutEntry {
                    binding: 4 + i as u32,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                });
            }
        }

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gpu compute"),
                entries: &layout_entries,
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gpu compute"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gpu compute"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // 256 bytes: 48 for Params + 208 for user args (up to 52 f32 values)
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: 256,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let aligned_bpr = aligned_bytes_per_row(width);
        let stride = aligned_bpr / 4;

        let storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pixel output"),
            size: (aligned_bpr * height) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let accum_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("accumulation"),
            size: (stride * height) as u64 * 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Upload textures to GPU
        let mut gpu_textures = Vec::new();
        let sampler = if !textures.is_empty() {
            for (i, tex) in textures.iter().enumerate() {
                let size = wgpu::Extent3d {
                    width: tex.width,
                    height: tex.height,
                    depth_or_array_layers: 1,
                };
                let gpu_tex = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(&format!("texture_{i}")),
                    size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &gpu_tex,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    &tex.data,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(tex.width * 4),
                        rows_per_image: Some(tex.height),
                    },
                    size,
                );
                let view = gpu_tex.create_view(&wgpu::TextureViewDescriptor::default());
                gpu_textures.push(GpuTexture { view });
            }
            Some(device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("tex_sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }))
        } else {
            None
        };

        Self {
            compute_pipeline,
            bind_group_layout,
            uniform_buffer,
            storage_buffer,
            accum_buffer,
            sampler,
            gpu_textures,
            width,
            height,
            max_iter,
            stride,
        }
    }

    /// Build bind group entries including textures if present.
    fn bind_group_entries(&self) -> Vec<wgpu::BindGroupEntry<'_>> {
        let mut entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: self.uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: self.storage_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: self.accum_buffer.as_entire_binding(),
            },
        ];
        if let Some(ref sampler) = self.sampler {
            entries.push(wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(sampler),
            });
            for (i, gpu_tex) in self.gpu_textures.iter().enumerate() {
                entries.push(wgpu::BindGroupEntry {
                    binding: 4 + i as u32,
                    resource: wgpu::BindingResource::TextureView(&gpu_tex.view),
                });
            }
        }
        entries
    }

    pub fn new(
        display: &Display,
        width: u32,
        height: u32,
        max_iter: u32,
        wgsl_source: &str,
        textures: &[&TextureData],
    ) -> Self {
        Self::build(&display.device, &display.queue, width, height, max_iter, wgsl_source, textures)
    }

    /// Zero the accumulation buffer (call on pan/zoom reset).
    #[allow(dead_code)]
    pub fn reset_accumulation(&self, queue: &wgpu::Queue) {
        let size = (self.stride * self.height) as u64 * 16;
        let zeroes = vec![0u8; size as usize];
        queue.write_buffer(&self.accum_buffer, 0, &zeroes);
    }

    fn view_params(
        &self,
        center_x: f64,
        center_y: f64,
        zoom: f64,
        sample_index: u32,
        sample_count: u32,
        time: f64,
    ) -> Params {
        let aspect = self.width as f64 / self.height as f64;
        let view_w = 3.5 / zoom;
        let view_h = view_w / aspect;
        let x_min = center_x - view_w / 2.0;
        let y_min = center_y - view_h / 2.0;
        let x_step = view_w / self.width as f64;
        let y_step = view_h / self.height as f64;

        Params {
            width: self.width,
            height: self.height,
            max_iter: self.max_iter,
            stride: self.stride,
            x_min: x_min as f32,
            y_min: y_min as f32,
            x_step: x_step as f32,
            y_step: y_step as f32,
            sample_index,
            sample_count,
            time: time as f32,
            _pad: [0; 1],
        }
    }

    /// Write params + user args to the uniform buffer.
    fn write_uniforms(&self, queue: &wgpu::Queue, params: &Params, user_args: &[f32]) {
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(params));
        if !user_args.is_empty() {
            let arg_bytes: Vec<u8> = user_args.iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            queue.write_buffer(
                &self.uniform_buffer,
                std::mem::size_of::<Params>() as u64,
                &arg_bytes,
            );
        }
    }

    /// Dispatch the compute shader and copy results to the display texture.
    pub fn render(
        &self,
        display: &Display,
        center_x: f64,
        center_y: f64,
        zoom: f64,
        sample_index: u32,
        sample_count: u32,
        time: f64,
        user_args: &[f32],
    ) {
        let params = self.view_params(center_x, center_y, zoom, sample_index, sample_count, time);
        self.write_uniforms(&display.queue, &params, user_args);

        let entries = self.bind_group_entries();
        let bind_group = display
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gpu compute"),
                layout: &self.bind_group_layout,
                entries: &entries,
            });

        let mut encoder =
            display
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("gpu compute"),
                });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compute"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.compute_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let wg_x = (self.width + 15) / 16;
            let wg_y = (self.height + 15) / 16;
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        let aligned_bpr = aligned_bytes_per_row(self.width);

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

    /// Create a headless GPU context (no window/surface) for benchmarking/output.
    pub fn new_headless(
        width: u32,
        height: u32,
        max_iter: u32,
        wgsl_source: &str,
        textures: &[&TextureData],
    ) -> (Self, wgpu::Device, wgpu::Queue) {
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
                label: Some("pixel-doodle headless"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            },
        ))
        .expect("failed to create device");
        let backend = Self::build(&device, &queue, width, height, max_iter, wgsl_source, textures);
        (backend, device, queue)
    }

    /// Dispatch the compute shader without presentation (headless).
    pub fn dispatch_compute(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        center_x: f64,
        center_y: f64,
        zoom: f64,
        sample_index: u32,
        sample_count: u32,
        time: f64,
        user_args: &[f32],
    ) {
        let params = self.view_params(center_x, center_y, zoom, sample_index, sample_count, time);
        self.write_uniforms(queue, &params, user_args);

        let entries = self.bind_group_entries();
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu compute"),
            layout: &self.bind_group_layout,
            entries: &entries,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gpu compute"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compute"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.compute_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let wg_x = (self.width + 15) / 16;
            let wg_y = (self.height + 15) / 16;
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        device.poll(wgpu::PollType::Wait).expect("GPU poll failed");
    }

    /// Read back the storage buffer contents as a Vec<u32> pixel buffer.
    pub fn readback_pixels(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<u32> {
        let aligned_bpr = aligned_bytes_per_row(self.width);
        let buf_size = (aligned_bpr * self.height) as u64;

        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback staging"),
            size: buf_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback"),
        });
        encoder.copy_buffer_to_buffer(&self.storage_buffer, 0, &staging, 0, buf_size);
        queue.submit(std::iter::once(encoder.finish()));
        device.poll(wgpu::PollType::Wait).expect("GPU poll failed");

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        device.poll(wgpu::PollType::Wait).expect("GPU poll failed");
        rx.recv().unwrap().expect("buffer map failed");

        let data = slice.get_mapped_range();
        let stride = self.stride as usize;
        let w = self.width as usize;
        let h = self.height as usize;
        let all_pixels: &[u32] = bytemuck::cast_slice(&data);

        let mut out = Vec::with_capacity(w * h);
        for row in 0..h {
            out.extend_from_slice(&all_pixels[row * stride..row * stride + w]);
        }
        drop(data);
        staging.unmap();
        out
    }
}
