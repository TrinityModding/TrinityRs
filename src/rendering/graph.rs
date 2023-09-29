use std::collections::HashMap;
use std::mem::size_of;
use std::num::NonZeroU64;
use std::rc::Rc;
use std::sync::{Arc, mpsc, RwLock};
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::mpsc::Receiver;
use std::thread;
use ultraviolet::{Mat4, Vec4};
use vulkano::buffer::{Buffer, BufferContents, BufferCreateFlags, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract, RenderingAttachmentInfo, RenderingInfo};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::{Device, Queue};
use vulkano::{DeviceSize, NonExhaustive};
use vulkano::command_buffer::ResourceInCommand::IndexBuffer;
use vulkano::memory::allocator::{AllocationCreateInfo, AllocationType, DeviceLayout, FreeListAllocator, MemoryAllocatePreference, MemoryTypeFilter, StandardMemoryAllocator, Suballocation, Suballocator};
use vulkano::memory::allocator::suballocator::Region;
use vulkano::memory::DeviceAlignment;
use vulkano::pipeline::graphics::viewport::Viewport;
use vulkano::pipeline::{GraphicsPipeline, PipelineLayout};
use vulkano::render_pass::{AttachmentLoadOp, AttachmentStoreOp};
use vulkano::swapchain::Swapchain;
use crate::io::model::MeshBufferInfo;
use crate::rendering::renderer::{Recorder, Renderer};
use crate::rendering::texture_manager::TextureManager;
use crate::WindowFrameBuffer;

const POS_ATTRIB_VERTEX_BUFFER_INITIAL_SIZE: u64 = 6_000000;
const COLOR_ATTRIBS_VERTEX_BUFFER_INITIAL_SIZE: u64 = 14_000000;
const INDEX_BUFFER_INITIAL_SIZE: u64 = 6_000000;
const INSTANCE_INFO_BUFFER_INITIAL_SIZE: u64 = 32_000000;

pub trait RenderingInstance {
    fn get_models(&mut self) -> Vec<Arc<MeshLocation>>;

    fn get_instance_index(&mut self) -> u64;
}

pub struct SceneGraph {
    pub pipeline_layout: Arc<PipelineLayout>,
    pub fbo: Rc<RwLock<WindowFrameBuffer>>,
    pub allocator: Arc<StandardMemoryAllocator>,
    pub texture_manager: TextureManager,
    pub buffers: HashMap<Arc<GraphicsPipeline>, RwLock<BufferStorage>>,
    pub shader_map: HashMap<String, Arc<GraphicsPipeline>>,
}

#[derive(BufferContents)]
#[repr(C)]
pub struct PushConstants {
    pub instance_offset: u64,
}

impl Recorder for SceneGraph {
    fn record(&self, builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>, device: Arc<Device>, swapchain: Arc<Swapchain>, viewport: Viewport, image_index: usize) {
        let framebuffer = self.fbo.read().unwrap();
        let proj_matrix = SceneGraph::perspective(90f32, viewport.extent[0] / viewport.extent[1], 0.1, 100.0);

        builder
            .begin_rendering(RenderingInfo {
                color_attachments: vec![Some(RenderingAttachmentInfo {
                    load_op: AttachmentLoadOp::Clear,
                    store_op: AttachmentStoreOp::Store,
                    clear_value: Some([0.0, 0.0, 1.0, 1.0].into()),
                    ..RenderingAttachmentInfo::image_view(framebuffer.swapchain_image_views[image_index].clone())
                })],
                depth_attachment: Some(RenderingAttachmentInfo {
                    load_op: AttachmentLoadOp::Clear,
                    store_op: AttachmentStoreOp::DontCare,
                    clear_value: Some(1f32.into()),
                    ..RenderingAttachmentInfo::image_view(framebuffer.depth_image_view.clone())
                }),
                ..Default::default()
            }).unwrap()
            .set_viewport(0, [viewport.clone()].into_iter().collect()).unwrap();

        for entry in &self.buffers {
            let buffer_guard = entry.1.read().unwrap();
            let raw_idx_buffer: Subbuffer<[u8]> = buffer_guard.index_buffer.buffer.into();
            let idx_buffer: &Subbuffer<[u32]> = raw_idx_buffer.reinterpret_ref();

            builder
                .bind_pipeline_graphics(entry.0.clone()).unwrap()
                .bind_vertex_buffers(0, [buffer_guard.pos_vertex_buffer.buffer.clone().into(), buffer_guard.color_vertex_buffer.buffer.clone().into()]).unwrap()
                .bind_index_buffer(idx_buffer.clone()).unwrap();
            // .bind_descriptor_sets(PipelineBindPoint::Graphics, self.pipeline_layout.clone(), 0, self.set.clone()).unwrap();

            for instance in &buffer_guard.instances {
                let mut instance_guard = instance.write().unwrap();
                builder.push_constants(self.pipeline_layout.clone(), 0, PushConstants { instance_offset: 0 }).unwrap();

                for sub_mesh in &instance_guard.get_models() {
                    // TODO: use indirect calls at some point
                    builder.draw_indexed(sub_mesh.idx_count as u32, 1, sub_mesh.idx_offset as u32, 0, 0).unwrap();
                }
            }
        }

        builder.end_rendering().unwrap();
    }
}

impl SceneGraph {
    pub fn new(window_framebuffer: Rc<RwLock<WindowFrameBuffer>>, layout: Arc<PipelineLayout>, renderer: &Renderer) -> SceneGraph {
        // The index of the currently most up-to-date texture. The worker thread swaps the index after
        // every finished write, which is always done to the, at that point in time, unused texture.
        let current_texture_index = Arc::new(AtomicBool::new(false));
        // Current generation, used to notify the worker thread of when a texture is no longer read.
        let current_generation = Arc::new(AtomicU64::new(0));
        let (channel, receiver) = mpsc::channel();
        run_worker(
            receiver,
            renderer.transfer_queue.clone(),
            current_texture_index.clone(),
            current_generation.clone(),
            renderer.swapchain.image_count(),
            renderer.allocator.clone(),
            renderer.command_buffer_allocator.clone(),
        );

        let d_layout = layout.set_layouts().get(0).unwrap();
        let set = PersistentDescriptorSet::new_variable(
            &renderer.descriptor_set_allocator,
            d_layout.clone(),
            5,
            [WriteDescriptorSet::image_view_sampler_array(
                0,
                0,
                [],
            )],
            [],
        ).unwrap();

        SceneGraph {
            pipeline_layout: layout,
            fbo: window_framebuffer,
            allocator: renderer.allocator.clone(),
            shader_map: HashMap::new(),
            texture_manager: TextureManager::new(),
            buffers: HashMap::new(),
        }
    }

    fn perspective(vertical_fov: f32, aspect_ratio: f32, z_near: f32, z_far: f32) -> Mat4 {
        let t = (vertical_fov / 2.0).tan();
        let sy = 1.0 / t;
        let sx = sy / aspect_ratio;
        let r = z_far / (z_far - z_near);

        Mat4::new(
            Vec4::new(sx, 0.0, 0.0, 0.0),
            Vec4::new(0.0, -sy, 0.0, 0.0),
            Vec4::new(0.0, 0.0, r, 1.0),
            Vec4::new(0.0, 0.0, -z_near * r, 0.0),
        )
    }

    pub fn add_shader(&mut self, name: &str, pipeline: Arc<GraphicsPipeline>, allocator: Arc<StandardMemoryAllocator>, device: Arc<Device>) {
        self.shader_map.entry(name.to_string())
            .or_insert(pipeline.clone());
        self.buffers.entry(pipeline)
            .or_insert(RwLock::new(BufferStorage::new(allocator, device.clone())));
    }

    pub fn upload(&self, shader_name: &String, mesh_info: MeshBufferInfo) -> Arc<MeshLocation> {
        let shader = self.shader_map.get(shader_name).expect(format!("Model requested shader that was not implemented: '{}'", shader_name).as_str()).clone();
        let mut buffers = self.buffers.get(&shader).unwrap().write().unwrap();

        // Upload Vertex Data
        match shader_name.as_str() {
            "Standard" | "FresnelBlend" | "EyeClearCoat" | "SSS" => {
                let element_count = mesh_info.positions.len();
                {
                    #[derive(BufferContents)]
                    #[repr(C)]
                    pub struct PosElement {
                        pub position: [f32; 3],
                    }

                    let pos_buffer_element_size = size_of::<PosElement>();
                    let pos_sub_buffer: Subbuffer<[PosElement]> = buffers.pos_vertex_buffer.get_slice(pos_buffer_element_size, element_count).reinterpret();
                    let mut write_guard = pos_sub_buffer.write().unwrap();
                    for (o, i) in write_guard.iter_mut().zip(mesh_info.positions) {
                        *o = PosElement { position: [i.x, i.y, i.z] };
                    }
                }

                {
                    #[derive(BufferContents)]
                    #[repr(C)]
                    pub struct ColElement {
                        pub uv: [f32; 2],
                    }

                    let col_buffer_element_size = size_of::<ColElement>();
                    let col_sub_buffer: Subbuffer<[ColElement]> = buffers.color_vertex_buffer.get_slice(col_buffer_element_size, element_count).reinterpret();
                    let mut write_guard = col_sub_buffer.write().unwrap();
                    for (o, i) in write_guard.iter_mut().zip(mesh_info.uvs) {
                        *o = ColElement { uv: [i.x, i.y] };
                    }
                }
            }
            _ => panic!("No method to convert mesh data for shader")
        }

        // Write index buffer
        let index_start_pos = buffers.index_buffer.position;
        let index_count = mesh_info.indices.len();
        let index_sub_buffer: Subbuffer<[u32]> = buffers.index_buffer.get_slice(size_of::<u32>(), mesh_info.indices.len()).reinterpret();
        let mut write_guard = index_sub_buffer.write().unwrap();
        for (o, i) in write_guard.iter_mut().zip(mesh_info.indices) {
            *o = i;
        }

        Arc::new(MeshLocation {
            shader,
            idx_offset: index_start_pos,
            idx_count: index_count as u64,
        })
    }
}

// The job of this worker is to move data from transfer buffers into buffers on the Device
#[allow(clippy::too_many_arguments)]
fn run_worker(
    channel: mpsc::Receiver<()>,
    transfer_queue: Arc<Queue>,
    current_texture_index: Arc<AtomicBool>,
    current_generation: Arc<AtomicU64>,
    swapchain_image_count: u32,
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
) {
    thread::spawn(move || {
        let mut builder = AutoCommandBufferBuilder::primary(
            command_buffer_allocator.as_ref(),
            transfer_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        ).unwrap();
    });
}

/// Sub-allocatable buffer that uses the transfer queue in order to upload data to Device Buffers
struct ManagedBuffer {
    pub buffer: Arc<Buffer>,
    pub allocator: Arc<dyn Suballocator>,
    pub sub_alloc_alignment: DeviceAlignment,
}

impl ManagedBuffer {
    pub fn new(allocator: Arc<StandardMemoryAllocator>, buffer_usage: BufferUsage, size: DeviceSize, device: Arc<Device>) -> ManagedBuffer {
        let alignment = device.physical_device().properties().non_coherent_atom_size;

        let buffer = Buffer::new(
            allocator,
            BufferCreateInfo {
                usage: buffer_usage,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            DeviceLayout::from_size_alignment(size, DeviceAlignment::MIN.as_devicesize()).unwrap(),
        ).unwrap();

        ManagedBuffer {
            buffer,
            allocator: Arc::new(FreeListAllocator::new(Region::new(0, size).unwrap())),
            sub_alloc_alignment: alignment,
        }
    }

    pub fn transfer<T>(&mut self, size: DeviceSize, transfer_buffer: Subbuffer<[u8]>, renderer: &Renderer) -> Suballocation {
        let device_allocation = self.allocator.allocate(
            DeviceLayout::from_size_alignment(size, self.sub_alloc_alignment.as_devicesize()).unwrap(),
            AllocationType::Linear,
            DeviceAlignment::MIN,
        ).unwrap();

        let mut builder = AutoCommandBufferBuilder::primary(
            &renderer.command_buffer_allocator,
            renderer.transfer_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        ).unwrap();

        let copy_info = CopyBufferInfo::buffers(transfer_buffer, self.buffer.into());
        copy_info.regions[0].size = device_allocation.size;
        copy_info.regions[0].dst_offset = device_allocation.offset;
        let cmd_buffer = builder
            .copy_buffer(copy_info).unwrap()
            .build().unwrap();

        //TODO: make this happen on the transfer thread and somehow let the model references know that their model is ready for rendering
        let _ = cmd_buffer.execute(renderer.graphics_queue.clone()).unwrap();

        return device_allocation;
    }
}

pub struct BufferStorage {
    /// Just the position attribute
    pub pos_vertex_buffer: ManagedBuffer,
    /// Attributes used for colour
    pub color_vertex_buffer: ManagedBuffer,
    pub index_buffer: ManagedBuffer,
    pub instance_buffer: ManagedBuffer,
    /// All instances that should be rendered with the buffers
    pub instances: Vec<Arc<RwLock<Box<dyn RenderingInstance>>>>,
}

impl BufferStorage {
    pub fn new(allocator: Arc<StandardMemoryAllocator>, device: Arc<Device>) -> BufferStorage {
        BufferStorage {
            pos_vertex_buffer: ManagedBuffer::new(
                allocator.clone(),
                BufferUsage::VERTEX_BUFFER,
                POS_ATTRIB_VERTEX_BUFFER_INITIAL_SIZE,
                device.clone()
            ),
            color_vertex_buffer: ManagedBuffer::new(
                allocator.clone(),
                BufferUsage::VERTEX_BUFFER,
                COLOR_ATTRIBS_VERTEX_BUFFER_INITIAL_SIZE,
                device.clone()
            ),
            index_buffer: ManagedBuffer::new(
                allocator.clone(),
                BufferUsage::INDEX_BUFFER,
                INDEX_BUFFER_INITIAL_SIZE,
                device.clone()
            ),
            instance_buffer: ManagedBuffer::new(
                allocator.clone(),
                BufferUsage::STORAGE_BUFFER,
                INSTANCE_INFO_BUFFER_INITIAL_SIZE,
                device.clone()
            ),
            instances: Vec::new(),
        }
    }
}

pub struct MeshLocation {
    pub shader: Arc<GraphicsPipeline>,
    pub idx_offset: u64,
    pub idx_count: u64,
}
