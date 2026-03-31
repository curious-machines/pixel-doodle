use crate::display::Display;
use crate::gpu::aligned_bytes_per_row;

use super::VectorScene;

/// GPU tile-based vector rasterizer.
///
/// Manages its own compute pipeline and buffers. Shares the Display's
/// wgpu device and queue. Dispatches a tile rasterization kernel that
/// computes per-pixel winding numbers from line segments.
pub struct VectorTileRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    uniform_buf: wgpu::Buffer,
    pixel_buf: wgpu::Buffer,
    // Scene data buffers — created on upload, None initially
    segment_buf: Option<wgpu::Buffer>,
    seg_path_id_buf: Option<wgpu::Buffer>,
    tile_offset_buf: Option<wgpu::Buffer>,
    tile_count_buf: Option<wgpu::Buffer>,
    tile_index_buf: Option<wgpu::Buffer>,
    path_color_buf: Option<wgpu::Buffer>,
    path_fill_rule_buf: Option<wgpu::Buffer>,
    // Dimensions
    width: u32,
    height: u32,
    stride: u32,
    tile_size: u32,
    tiles_x: u32,
    tiles_y: u32,
    num_paths: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    width: u32,
    height: u32,
    stride: u32,
    tile_size: u32,
    tiles_x: u32,
    num_paths: u32,
    _pad0: u32,
    _pad1: u32,
}

impl VectorTileRenderer {
    /// Create a new tile renderer.
    ///
    /// `wgsl_source` is the tile rasterization shader source.
    /// The `TILE_SIZE` constant in the shader must match `tile_size`.
    pub fn new(display: &Display, tile_size: u32, wgsl_source: &str) -> Self {
        let device = display.device.clone();
        let queue = display.queue.clone();

        let width = display.width();
        let height = display.height();
        let stride = aligned_bytes_per_row(width) / 4;
        let tiles_x = (width + tile_size - 1) / tile_size;
        let tiles_y = (height + tile_size - 1) / tile_size;

        // Compile shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tile rasterizer"),
            source: wgpu::ShaderSource::Wgsl(wgsl_source.into()),
        });

        // Bind group layout: uniform + 7 storage buffers
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tile rasterizer"),
            entries: &[
                // binding 0: uniform params
                bgl_entry(0, wgpu::BufferBindingType::Uniform),
                // binding 1: segments (read)
                bgl_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                // binding 2: seg_path_ids (read)
                bgl_entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                // binding 3: tile_offsets (read)
                bgl_entry(3, wgpu::BufferBindingType::Storage { read_only: true }),
                // binding 4: tile_counts (read)
                bgl_entry(4, wgpu::BufferBindingType::Storage { read_only: true }),
                // binding 5: tile_indices (read)
                bgl_entry(5, wgpu::BufferBindingType::Storage { read_only: true }),
                // binding 6: path_colors (read)
                bgl_entry(6, wgpu::BufferBindingType::Storage { read_only: true }),
                // binding 7: pixels (read_write)
                bgl_entry(7, wgpu::BufferBindingType::Storage { read_only: false }),
                // binding 8: path_fill_rules (read)
                bgl_entry(8, wgpu::BufferBindingType::Storage { read_only: true }),
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tile rasterizer"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tile rasterizer"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Uniform buffer
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tile params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Pixel output buffer (stride-aligned for copy_buffer_to_texture)
        let pixel_buf_size = (stride * height) as u64 * 4;
        let pixel_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tile pixels"),
            size: pixel_buf_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            uniform_buf,
            pixel_buf,
            segment_buf: None,
            seg_path_id_buf: None,
            tile_offset_buf: None,
            tile_count_buf: None,
            tile_index_buf: None,
            path_color_buf: None,
            path_fill_rule_buf: None,
            width,
            height,
            stride,
            tile_size,
            tiles_x,
            tiles_y,
            num_paths: 0,
        }
    }

    /// Upload scene data to GPU buffers.
    ///
    /// Recreates scene buffers sized to the actual data.
    pub fn upload_scene(&mut self, scene: &VectorScene) {
        self.num_paths = scene.path_colors.len() as u32;

        // Segments: vec4<f32> each (16 bytes)
        self.segment_buf = Some(create_and_write_buffer(
            &self.device,
            &self.queue,
            "segments",
            bytemuck::cast_slice(&scene.segments),
        ));

        // Segment path IDs: u32 each
        self.seg_path_id_buf = Some(create_and_write_buffer(
            &self.device,
            &self.queue,
            "seg_path_ids",
            bytemuck::cast_slice(&scene.seg_path_ids),
        ));

        // Tile offsets: u32 each
        self.tile_offset_buf = Some(create_and_write_buffer(
            &self.device,
            &self.queue,
            "tile_offsets",
            bytemuck::cast_slice(&scene.tile_offsets),
        ));

        // Tile counts: u32 each
        self.tile_count_buf = Some(create_and_write_buffer(
            &self.device,
            &self.queue,
            "tile_counts",
            bytemuck::cast_slice(&scene.tile_counts),
        ));

        // Tile indices: u32 each
        self.tile_index_buf = Some(create_and_write_buffer(
            &self.device,
            &self.queue,
            "tile_indices",
            bytemuck::cast_slice(&scene.tile_indices),
        ));

        // Path colors: u32 each
        self.path_color_buf = Some(create_and_write_buffer(
            &self.device,
            &self.queue,
            "path_colors",
            bytemuck::cast_slice(&scene.path_colors),
        ));

        // Path fill rules: u32 each
        self.path_fill_rule_buf = Some(create_and_write_buffer(
            &self.device,
            &self.queue,
            "path_fill_rules",
            bytemuck::cast_slice(&scene.path_fill_rules),
        ));
    }

    /// Dispatch the tile rasterization kernel and present to screen.
    pub fn render(&self, display: &Display) {
        // Ensure scene has been uploaded
        let segment_buf = self.segment_buf.as_ref().expect("scene not uploaded");
        let seg_path_id_buf = self.seg_path_id_buf.as_ref().expect("scene not uploaded");
        let tile_offset_buf = self.tile_offset_buf.as_ref().expect("scene not uploaded");
        let tile_count_buf = self.tile_count_buf.as_ref().expect("scene not uploaded");
        let tile_index_buf = self.tile_index_buf.as_ref().expect("scene not uploaded");
        let path_color_buf = self.path_color_buf.as_ref().expect("scene not uploaded");
        let path_fill_rule_buf = self.path_fill_rule_buf.as_ref().expect("scene not uploaded");

        // Write uniform params
        let params = Params {
            width: self.width,
            height: self.height,
            stride: self.stride,
            tile_size: self.tile_size,
            tiles_x: self.tiles_x,
            num_paths: self.num_paths,
            _pad0: 0,
            _pad1: 0,
        };
        self.queue
            .write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&params));

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tile rasterizer"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: segment_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: seg_path_id_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: tile_offset_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: tile_count_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: tile_index_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: path_color_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.pixel_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: path_fill_rule_buf.as_entire_binding(),
                },
            ],
        });

        // Dispatch compute + copy to display texture
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tile rasterizer"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tile rasterizer"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(self.tiles_x, self.tiles_y, 1);
        }

        // Copy pixel buffer to display texture
        let aligned_bpr = aligned_bytes_per_row(self.width);
        encoder.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo {
                buffer: &self.pixel_buf,
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

/// Helper to create a bind group layout entry for a buffer binding.
fn bgl_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Create a GPU storage buffer and write data to it.
fn create_and_write_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    data: &[u8],
) -> wgpu::Buffer {
    // wgpu requires storage buffers to be at least 1 element; use 4 bytes minimum
    let size = (data.len() as u64).max(4);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    if !data.is_empty() {
        queue.write_buffer(&buffer, 0, data);
    }
    buffer
}
