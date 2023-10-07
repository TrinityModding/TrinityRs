use crate::io::flatbuffers::trmsh_generated::titan::model::MaterialInfo;
use crate::io::model::MeshBufferInfo;
use crate::rendering::renderer::{Recorder, Renderer};
use crate::rendering::texture_manager::TextureManager;
use crate::WindowFrameBuffer;
use std::collections::HashMap;
use std::mem::size_of;
use std::ops::Add;
use std::rc::Rc;
use std::sync::{Arc, Mutex, RwLock};
use ultraviolet::{Mat4, Rotor3, Similarity3, Vec3, Vec4};
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, PrimaryAutoCommandBuffer,
    PrimaryCommandBufferAbstract, RenderingAttachmentInfo, RenderingInfo,
};
use vulkano::descriptor_set::PersistentDescriptorSet;
use vulkano::device::Device;
use vulkano::memory::allocator::suballocator::Region;
use vulkano::memory::allocator::{
    AllocationCreateInfo, AllocationType, DeviceLayout, FreeListAllocator, MemoryTypeFilter,
    StandardMemoryAllocator, Suballocation, Suballocator,
};
use vulkano::memory::DeviceAlignment;
use vulkano::pipeline::graphics::viewport::Viewport;
use vulkano::pipeline::{GraphicsPipeline, PipelineBindPoint, PipelineLayout};
use vulkano::render_pass::{AttachmentLoadOp, AttachmentStoreOp};
use vulkano::swapchain::Swapchain;
use vulkano::DeviceSize;
use crate::rendering::shaders::ShaderCollection;

const POS_ATTRIB_VERTEX_BUFFER_INITIAL_SIZE: u64 = 6_000000;
const COLOR_ATTRIBS_VERTEX_BUFFER_INITIAL_SIZE: u64 = 14_000000;
const INDEX_BUFFER_INITIAL_SIZE: u64 = 6_000000;
const INSTANCE_INFO_BUFFER_INITIAL_SIZE: u64 = 32_000000;

#[derive(BufferContents)]
#[repr(C)]
pub struct PushConstants {
    pub instance_address: u64,
}

#[derive(Clone, Debug)]
pub struct Camera {
    translation: Vec3,
    rotation: Rotor3,
    cached_transform: Mat4,
}

impl Camera {
    pub fn new() -> Camera {
        Camera {
            translation: Vec3::new(0.0, 0.0, 0.0),
            rotation: Rotor3::from_euler_angles(0.0, 0.0, 0.0),
            cached_transform: Mat4::identity(),
        }
    }

    pub fn update(&mut self) {
        let mut transform = Similarity3::identity();
        transform.append_translation(self.translation);
        transform.append_rotation(self.rotation);
        self.cached_transform = transform.into_homogeneous_matrix();
    }

    pub fn translate(&mut self, x: f32, y: f32, z: f32) {
        self.translation = self.translation.add(Vec3::new(x, y, z));
        self.update();
    }

    pub fn get_matrix(&self) -> Mat4 {
        self.cached_transform.clone()
    }
}

pub struct SceneGraph {
    pub descriptor_set: Option<Arc<PersistentDescriptorSet>>,
    pub layout: Arc<PipelineLayout>,
    pub fbo: Rc<RwLock<WindowFrameBuffer>>,
    pub allocator: Arc<StandardMemoryAllocator>,
    pub texture_manager: TextureManager,
    pub buffers: HashMap<Arc<GraphicsPipeline>, RwLock<BufferStorage>>,
    pub shader_map: HashMap<String, Arc<GraphicsPipeline>>,
}

impl Recorder for SceneGraph {
    #[allow(unused_variables)]
    fn record(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<
            PrimaryAutoCommandBuffer<Arc<StandardCommandBufferAllocator>>,
            Arc<StandardCommandBufferAllocator>,
        >,
        device: Arc<Device>,
        swapchain: Arc<Swapchain>,
        viewport: Viewport,
        image_index: usize,
    ) {
        let framebuffer = self.fbo.read().unwrap();
        let proj_matrix =
            SceneGraph::perspective(90f32, viewport.extent[0] / viewport.extent[1], 0.1, 100.0);

        builder
            .begin_rendering(RenderingInfo {
                color_attachments: vec![Some(RenderingAttachmentInfo {
                    load_op: AttachmentLoadOp::Clear,
                    store_op: AttachmentStoreOp::Store,
                    clear_value: Some([0.0, 0.0, 1.0, 1.0].into()),
                    ..RenderingAttachmentInfo::image_view(
                        framebuffer.swapchain_image_views[image_index].clone(),
                    )
                })],
                depth_attachment: Some(RenderingAttachmentInfo {
                    load_op: AttachmentLoadOp::Clear,
                    store_op: AttachmentStoreOp::DontCare,
                    clear_value: Some(1f32.into()),
                    ..RenderingAttachmentInfo::image_view(framebuffer.depth_image_view.clone())
                }),
                ..Default::default()
            })
            .unwrap()
            .set_viewport(0, [viewport.clone()].into_iter().collect())
            .unwrap();

        drop(framebuffer);

        builder
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.layout.clone(),
                0,
                self.get_texture_descriptor(),
            )
            .unwrap();

        for entry in &self.buffers {
            let buffer_guard = entry.1.read().unwrap();
            let raw_idx_buffer: Subbuffer<[u8]> = buffer_guard.index_buffer.buffer.clone().into();
            let idx_buffer: &Subbuffer<[u32]> = raw_idx_buffer.reinterpret_ref();

            builder
                .bind_pipeline_graphics(entry.0.clone())
                .unwrap()
                .bind_vertex_buffers(
                    0,
                    [
                        buffer_guard.pos_vertex_buffer.buffer.clone().into(),
                        buffer_guard.color_vertex_buffer.buffer.clone().into(),
                    ],
                )
                .unwrap()
                .bind_index_buffer(idx_buffer.clone())
                // .bind_descriptor_sets(PipelineBindPoint::Graphics, self.layout.clone(), 0, self.set.clone()).unwrap()
                .unwrap();

            for instance in &buffer_guard.instances {
                builder
                    .push_constants(
                        self.layout.clone(),
                        0,
                        PushConstants {
                            instance_address: 0,
                        },
                    )
                    .unwrap();

                builder
                    .draw_indexed(
                        instance.0.idx_count,
                        1,
                        instance.0.idx_offset,
                        instance.0.vertex_offset,
                        0,
                    )
                    .unwrap();
            }
        }

        builder.end_rendering().unwrap();
    }
}

impl SceneGraph {
    pub fn new(
        window_framebuffer: Rc<RwLock<WindowFrameBuffer>>,
        shaders: ShaderCollection,
        renderer: &Renderer,
    ) -> Arc<Mutex<SceneGraph>> {
        let scene_graph = Arc::new(Mutex::new(SceneGraph {
            descriptor_set: None,
            layout: shaders.layout.clone(),
            fbo: window_framebuffer,
            allocator: renderer.allocator.clone(),
            shader_map: HashMap::new(),
            texture_manager: TextureManager::new(
                shaders.layout.clone(),
                renderer.descriptor_set_allocator.clone(),
                renderer.device.clone(),
            ),
            buffers: HashMap::new(),
        }));

        let mut scene_guard = scene_graph.lock().unwrap();
        for shader in shaders.shaders {
            scene_guard.shader_map
                .entry(shader.0.to_string())
                .or_insert(shader.1.clone());
            scene_guard.buffers
                .entry(shader.1)
                .or_insert(RwLock::new(BufferStorage::new(renderer.allocator.clone())));
        }

        drop(scene_guard);
        scene_graph
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

    pub fn add_model(&mut self, meshes: Vec<Arc<MeshLocation>>) {
        for x in meshes {
            let buffer = self.buffers.get(&x.shader).unwrap();
            let mut write_guard = buffer.write().unwrap();
            write_guard.instances.push((x.clone(), 0));
        }
    }

    fn new_transfer_buffer<T>(&self, size: DeviceSize) -> Subbuffer<[T]>
        where
            T: BufferContents,
    {
        Buffer::new_slice(
            self.allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            size,
        )
            .unwrap()
    }

    pub fn write_mesh_to_renderer(
        &mut self,
        shader_to_mesh_map: HashMap<String, (Arc<MeshBufferInfo>, Vec<MaterialInfo>)>,
        renderer: &mut Renderer,
    ) -> Vec<Arc<MeshLocation>> {
        let mut written_models = Vec::new();

        for entry in shader_to_mesh_map {
            let shader_name = entry.0;
            let mesh_info = entry.1.0;

            let shader = self
                .shader_map
                .get(shader_name.as_str())
                .unwrap_or_else(|| panic!("Unimplemented shader requested: '{}'", shader_name))
                .clone();
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

                        let transfer_buffer: Subbuffer<[PosElement]> =
                            self.new_transfer_buffer(element_count as DeviceSize);
                        let mut write_guard = transfer_buffer.write().unwrap();
                        for (o, i) in write_guard.iter_mut().zip(mesh_info.positions.clone()) {
                            *o = PosElement {
                                position: [i.x, i.y, i.z],
                            };
                        }

                        drop(write_guard);
                        buffers.pos_vertex_buffer.transfer(
                            (size_of::<PosElement>() * element_count) as DeviceSize,
                            transfer_buffer.clone().reinterpret(),
                            renderer,
                        );
                    }

                    {
                        #[derive(BufferContents)]
                        #[repr(C)]
                        pub struct ColElement {
                            pub uv: [f32; 2],
                        }

                        let transfer_buffer: Subbuffer<[ColElement]> =
                            self.new_transfer_buffer(element_count as DeviceSize);
                        let mut write_guard = transfer_buffer.write().unwrap();
                        for (o, i) in write_guard.iter_mut().zip(mesh_info.uvs.clone()) {
                            *o = ColElement { uv: [i.x, i.y] };
                        }

                        drop(write_guard);
                        buffers.color_vertex_buffer.transfer(
                            (size_of::<ColElement>() * element_count) as DeviceSize,
                            transfer_buffer.clone().reinterpret(),
                            renderer,
                        );
                    }
                }
                _ => panic!("No method to convert mesh data for shader"),
            }

            // Write index buffer
            let idx_count = mesh_info.indices.len();
            let transfer_buffer: Subbuffer<[u32]> =
                self.new_transfer_buffer(idx_count as DeviceSize);
            let mut write_guard = transfer_buffer.write().unwrap();
            for (o, i) in write_guard.iter_mut().zip(mesh_info.indices.clone()) {
                *o = i;
            }

            drop(write_guard);
            buffers.index_buffer.transfer(
                (mesh_info.indices.len() * size_of::<u32>()) as DeviceSize,
                transfer_buffer.clone().reinterpret(),
                renderer,
            );

            for sub_mesh in entry.1.1 {
                written_models.push(Arc::new(MeshLocation {
                    shader: shader.clone(),
                    idx_offset: sub_mesh.poly_offset(),
                    idx_count: sub_mesh.poly_count(),
                    vertex_offset: buffers.vertex_offset,
                }));
            }

            // Make sure the next meshes are offset correctly
            buffers.vertex_offset += mesh_info.positions.len() as i32;
        }

        written_models
    }

    pub fn get_texture_descriptor(&mut self) -> Arc<PersistentDescriptorSet> {
        if self.descriptor_set.is_none() {
            self.descriptor_set = Some(self.texture_manager.generate_descriptor_set());
        }

        self.descriptor_set.clone().unwrap()
    }
}

pub struct ManagedBuffer {
    pub buffer: Arc<Buffer>,
    pub allocator: Arc<dyn Suballocator>,
    pub sub_alloc_alignment: DeviceAlignment,
}

impl ManagedBuffer {
    pub fn new(
        allocator: Arc<StandardMemoryAllocator>,
        buffer_usage: BufferUsage,
        size: DeviceSize,
    ) -> ManagedBuffer {
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
        )
            .unwrap();

        ManagedBuffer {
            buffer,
            allocator: Arc::new(FreeListAllocator::new(Region::new(0, size).unwrap())),
            sub_alloc_alignment: DeviceAlignment::MIN,
        }
    }

    pub fn transfer(
        &mut self,
        size: DeviceSize,
        transfer_buffer: Subbuffer<[u8]>,
        renderer: &mut Renderer,
    ) -> Suballocation {
        let device_allocation = self
            .allocator
            .allocate(
                DeviceLayout::from_size_alignment(size, DeviceAlignment::MIN.as_devicesize())
                    .unwrap(),
                AllocationType::Linear,
                DeviceAlignment::MIN,
            )
            .unwrap();

        let mut copy_info = CopyBufferInfo::buffers(transfer_buffer, self.buffer.clone().into());
        copy_info.regions[0].size = device_allocation.size;
        copy_info.regions[0].dst_offset = device_allocation.offset;

        let mut builder = AutoCommandBufferBuilder::primary(
            &renderer.command_buffer_allocator,
            renderer.graphics_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
            .unwrap();

        builder.copy_buffer(copy_info).unwrap();
        let cmd_buffer = builder.build().unwrap();

        //TODO: make this happen on the transfer thread and somehow let the model references know that their model is ready for rendering
        let _ = cmd_buffer.execute(renderer.graphics_queue.clone()).unwrap();
        device_allocation
    }
}

pub struct BufferStorage {
    pub vertex_offset: i32,
    /// Just the position attribute
    pub pos_vertex_buffer: ManagedBuffer,
    /// Attributes used for colour
    pub color_vertex_buffer: ManagedBuffer,
    pub index_buffer: ManagedBuffer,
    pub instance_buffer: ManagedBuffer,
    /// All instances that should be rendered with the buffers
    pub instances: Vec<(Arc<MeshLocation>, u64)>,
}

impl BufferStorage {
    pub fn new(allocator: Arc<StandardMemoryAllocator>) -> BufferStorage {
        BufferStorage {
            vertex_offset: 0,
            pos_vertex_buffer: ManagedBuffer::new(
                allocator.clone(),
                BufferUsage::VERTEX_BUFFER | BufferUsage::TRANSFER_DST,
                POS_ATTRIB_VERTEX_BUFFER_INITIAL_SIZE,
            ),
            color_vertex_buffer: ManagedBuffer::new(
                allocator.clone(),
                BufferUsage::VERTEX_BUFFER | BufferUsage::TRANSFER_DST,
                COLOR_ATTRIBS_VERTEX_BUFFER_INITIAL_SIZE,
            ),
            index_buffer: ManagedBuffer::new(
                allocator.clone(),
                BufferUsage::INDEX_BUFFER | BufferUsage::TRANSFER_DST,
                INDEX_BUFFER_INITIAL_SIZE,
            ),
            instance_buffer: ManagedBuffer::new(
                allocator.clone(),
                BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                INSTANCE_INFO_BUFFER_INITIAL_SIZE,
            ),
            instances: Vec::new(),
        }
    }
}

pub struct MeshLocation {
    pub shader: Arc<GraphicsPipeline>,
    pub idx_offset: u32,
    pub idx_count: u32,
    ///a constant value that should be added to each index in the index buffer to produce the final vertex number to be used.
    pub vertex_offset: i32,
}
