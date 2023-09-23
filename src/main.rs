use std::io::Cursor;
use std::ops::Add;
use std::sync::{Arc, Mutex};
use ultraviolet::{Mat4, Rotor3, Similarity3, Vec3, Vec4};

use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, IndexBuffer, Subbuffer};
use vulkano::buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferToImageInfo, DrawIndexedIndirectCommand, PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract, RenderingAttachmentInfo, RenderingInfo};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::descriptor_set::layout::DescriptorBindingFlags;
use vulkano::device::Device;
use vulkano::DeviceSize;
use vulkano::format::Format;
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::image::sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo};
use vulkano::image::view::ImageView;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter};
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::pipeline::graphics::color_blend::ColorBlendState;
use vulkano::pipeline::graphics::depth_stencil::DepthStencilState;
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::{FrontFace, RasterizationState};
use vulkano::pipeline::graphics::subpass::PipelineRenderingCreateInfo;
use vulkano::pipeline::graphics::vertex_input::{VertexInputAttributeDescription, VertexInputBindingDescription, VertexInputRate, VertexInputState};
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::render_pass::{AttachmentLoadOp, AttachmentStoreOp};
use vulkano::swapchain::{Swapchain, SwapchainCreateInfo};
use vulkano::sync::GpuFuture;
use winit::{
    event_loop::EventLoop,
    window::WindowBuilder,
};
use winit::event::{Event, VirtualKeyCode, WindowEvent};
use winit::event::ElementState::{Pressed, Released};
use winit::event_loop::ControlFlow;

use crate::io::model::{Attribute, AttributeSize, IndexLayout, RenderModel, RootModel};
use crate::rendering::renderer::{Recorder, Renderer};

mod rendering;
mod io;

mod vs {
    vulkano_shaders::shader! {
            ty: "vertex",
            path: "shaders/vertex.glsl"
            }
}

mod fs {
    vulkano_shaders::shader! {
            ty: "fragment",
            path: "shaders/fragment.glsl"
            }
}

#[derive(Clone, Debug)]
struct Camera {
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

struct WindowFrameBuffer {
    pub swapchain_image_views: Vec<Arc<ImageView>>,
    pub depth_image_view: Arc<ImageView>,
}

impl WindowFrameBuffer {
    pub fn new(viewport: &mut Viewport, memory_allocator: Arc<dyn MemoryAllocator>, images: &[Arc<Image>]) -> WindowFrameBuffer {
        let extent = images[0].extent();
        viewport.extent = [extent[0] as f32, extent[1] as f32];

        let depth_image_view = ImageView::new_default(
            Image::new(
                memory_allocator,
                ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format: Format::D16_UNORM,
                    extent: images[0].extent(),
                    usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT | ImageUsage::TRANSIENT_ATTACHMENT,
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            ).unwrap(),
        ).unwrap();

        let swapchain_image_views = images
            .iter()
            .map(|image| ImageView::new_default(image.clone()).unwrap())
            .collect::<Vec<_>>();

        WindowFrameBuffer {
            swapchain_image_views,
            depth_image_view,
        }
    }

    pub fn update(&mut self, viewport: &mut Viewport, memory_allocator: Arc<dyn MemoryAllocator>, images: &[Arc<Image>]) {
        let extent = images[0].extent();
        viewport.extent = [extent[0] as f32, extent[1] as f32];

        self.depth_image_view = ImageView::new_default(
            Image::new(
                memory_allocator,
                ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format: Format::D16_UNORM,
                    extent: images[0].extent(),
                    usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT | ImageUsage::TRANSIENT_ATTACHMENT,
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            ).unwrap(),
        ).unwrap();

        self.swapchain_image_views = images
            .iter()
            .map(|image| ImageView::new_default(image.clone()).unwrap())
            .collect::<Vec<_>>()
    }
}

fn main() {
    let event_loop = EventLoop::new();
    let window = Arc::new(WindowBuilder::new()
        .with_title("trinity-rs")
        .build(&event_loop).unwrap());

    // Renderer Setup
    let info = Renderer::new(window.clone(), &event_loop);
    let mut renderer = info.0;
    let images = info.1;
    let fbo = Arc::new(Mutex::new(WindowFrameBuffer::new(&mut renderer.viewport, renderer.allocator.clone(), images.as_slice())));

    // Graph/Scene Setup
    let model = RenderModel::from_trmdl(String::from("A:/PokemonScarlet/pokemon/data/pm0855/pm0855_00_00/pm0855_00_00.trmdl"), renderer.allocator.clone());
    let model_transform = Arc::new(Mutex::new(Similarity3::identity()));
    model_transform.lock().unwrap().append_translation(Vec3::new(0.0, -0.3, 0.0));

    // TODO: clean this up
    let mut uploads = AutoCommandBufferBuilder::primary(
        &renderer.command_buffer_allocator,
        renderer.queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    ).unwrap();

    let mascot_texture = {
        let png_bytes = include_bytes!("A:/PokemonScarlet/pokemon/data/pm0855/pm0855_00_00/pm0855_00_00_body_alb.png").to_vec();
        let cursor = Cursor::new(png_bytes);
        let decoder = png::Decoder::new(cursor);
        let mut reader = decoder.read_info().unwrap();
        let info = reader.info();
        let extent = [info.width, info.height, 1];

        let upload_buffer = Buffer::new_slice(
            renderer.allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            (info.width * info.height * 4) as DeviceSize,
        ).unwrap();

        reader.next_frame(&mut upload_buffer.write().unwrap()).unwrap();

        let image = Image::new(
            renderer.allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::R8G8B8A8_SRGB,
                extent,
                usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        ).unwrap();

        uploads
            .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
                upload_buffer,
                image.clone(),
            )).unwrap();

        ImageView::new_default(image).unwrap()
    };

    let sampler = Sampler::new(
        renderer.device.clone(),
        SamplerCreateInfo {
            mag_filter: Filter::Nearest,
            min_filter: Filter::Nearest,
            address_mode: [SamplerAddressMode::Repeat; 3],
            ..Default::default()
        },
    ).unwrap();

    renderer.previous_frame_end = Some(
        uploads
            .build().unwrap()
            .execute(renderer.queue.clone()).unwrap()
            .boxed(),
    );

    let camera = Arc::new(Mutex::new(Camera::new()));
    let triangle_renderer = ModelRenderGraph::new(&mut renderer, model, 0, camera.clone(), fbo.clone(), model_transform.clone(), mascot_texture, sampler); // load the non-lod mesh
    renderer.add_recorder(triangle_renderer);

    // Logic Setup
    let mut move_forward = false;
    let mut move_backward = false;
    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => {
                renderer.recreate_swapchain = true;
            }
            Event::WindowEvent {
                event: WindowEvent::KeyboardInput {
                    device_id: _device_id, input, is_synthetic: _is_synthetic
                },
                ..
            } => {
                if input.state == Pressed {
                    match input.virtual_keycode {
                        Some(VirtualKeyCode::W) => {
                            move_forward = true;
                            move_backward = false;
                        }
                        Some(VirtualKeyCode::S) => {
                            move_forward = false;
                            move_backward = true;
                        }
                        _ => {}
                    }
                }

                if input.state == Released {
                    match input.virtual_keycode {
                        Some(VirtualKeyCode::W) => {
                            move_forward = false;
                        }
                        Some(VirtualKeyCode::S) => {
                            move_backward = false;
                        }
                        _ => {}
                    }
                }
            }
            Event::RedrawEventsCleared => {
                let image_extent: [u32; 2] = renderer.window.inner_size().into();
                if image_extent.contains(&0) {
                    return;
                }
                renderer.previous_frame_end.as_mut().unwrap().cleanup_finished();
                if renderer.recreate_swapchain {
                    let (new_swapchain, new_images) = renderer.swapchain
                        .recreate(SwapchainCreateInfo {
                            image_extent,
                            ..renderer.swapchain.create_info()
                        })
                        .expect("failed to recreate swapchain");

                    renderer.swapchain = new_swapchain;
                    fbo.lock().unwrap().update(&mut renderer.viewport, renderer.allocator.clone(), new_images.as_slice());
                    renderer.recreate_swapchain = false;
                }
                renderer.render();

                model_transform.lock().unwrap().append_rotation(Rotor3::from_euler_angles(0.0, 0.0, 0.05));

                let mut cam = camera.lock().unwrap();
                if move_forward {
                    cam.translate(0.0, 0.0, 0.01);
                }
                if move_backward {
                    cam.translate(0.0, 0.0, -0.01);
                }
            }
            _ => (),
        }
    });
}

struct ModelRenderGraph {
    fbo: Arc<Mutex<WindowFrameBuffer>>,
    camera: Arc<Mutex<Camera>>,
    pipeline: Arc<GraphicsPipeline>,
    indirect_buffer: Subbuffer<[DrawIndexedIndirectCommand]>,
    target_mesh: RenderModel,
    model_transform: Arc<Mutex<Similarity3>>,
    set: Arc<PersistentDescriptorSet>,
}

impl ModelRenderGraph {
    pub fn new(renderer: &mut Renderer, model: RootModel, mesh_idx: usize, camera: Arc<Mutex<Camera>>, fbo: Arc<Mutex<WindowFrameBuffer>>, model_transform: Arc<Mutex<Similarity3>>, tex: Arc<ImageView>, sampler: Arc<Sampler>) -> ModelRenderGraph {
        let pipeline = {
            let vs = vs::load(renderer.device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();
            let fs = fs::load(renderer.device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();

            let stages = [
                PipelineShaderStageCreateInfo::new(vs),
                PipelineShaderStageCreateInfo::new(fs),
            ];

            let mut layout_create_info = PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages);

            // Adjust the info for set 0, binding 0 to make it variable with 2 descriptors.
            let binding = layout_create_info.set_layouts[0]
                .bindings
                .get_mut(&0).unwrap();
            binding.binding_flags |= DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT;
            binding.descriptor_count = 1;

            let layout = PipelineLayout::new(
                renderer.device.clone(),
                layout_create_info
                    .into_pipeline_layout_create_info(renderer.device.clone())
                    .unwrap(),
            ).unwrap();

            let subpass = PipelineRenderingCreateInfo {
                color_attachment_formats: vec![Some(renderer.swapchain.image_format())],
                depth_attachment_format: Some(Format::D16_UNORM),
                ..Default::default()
            };

            let mut vertex_attributes = Vec::new();
            let mut vertex_bindings = Vec::new();
            let total_size = calculate_element_size(&model.attributes);
            let mut offset = 0;
            for attr_idx in 0..model.attributes.len() {
                let attribute = model.attributes.get(attr_idx).unwrap();
                let vk_type = match attribute.size {
                    AttributeSize::None(_, _) => panic!("impossible"),
                    AttributeSize::Rgba8UNorm(_, _) => Format::R8G8B8A8_UNORM,
                    AttributeSize::Rgba8Unsigned(_, _) => Format::R8G8B8A8_UINT,
                    AttributeSize::R32UInt(_, _) => Format::R32_UINT,
                    AttributeSize::R32Int(_, _) => Format::R32_SINT,
                    AttributeSize::Rgba16UNorm(_, _) => Format::R16G16B16A16_UNORM,
                    AttributeSize::Rgba16Float(_, _) => Format::R16G16B16A16_SFLOAT,
                    AttributeSize::Rg32Float(_, _) => Format::R32G32_SFLOAT,
                    AttributeSize::Rgb32Float(_, _) => Format::R32G32B32_SFLOAT,
                    AttributeSize::Rgba32Float(_, _) => Format::R32G32B32A32_SFLOAT,
                };
                let size = attribute.get_size();

                vertex_attributes.push((attr_idx as u32, VertexInputAttributeDescription {
                    binding: 0,
                    format: vk_type,
                    offset,
                }));

                vertex_bindings.push((0, VertexInputBindingDescription {
                    stride: total_size,
                    input_rate: VertexInputRate::Vertex,
                }));

                offset += size;
            }

            let vertex_layout = VertexInputState::new()
                .attributes(vertex_attributes)
                .bindings(vertex_bindings);

            GraphicsPipeline::new(
                renderer.device.clone(),
                None,
                GraphicsPipelineCreateInfo {
                    stages: stages.into_iter().collect(),
                    vertex_input_state: Some(vertex_layout),
                    input_assembly_state: Some(InputAssemblyState::default()),
                    viewport_state: Some(ViewportState::viewport_dynamic_scissor_irrelevant()),
                    rasterization_state: Some(RasterizationState::default().front_face(FrontFace::Clockwise)),
                    depth_stencil_state: Some(DepthStencilState::simple_depth_test()),
                    multisample_state: Some(MultisampleState::default()),
                    color_blend_state: Some(ColorBlendState::new(subpass.color_attachment_formats.len() as u32, )),
                    // depth_stencil_state: Some(DepthStencilState::simple_depth_test()),
                    subpass: Some(subpass.into()),
                    ..GraphicsPipelineCreateInfo::layout(layout)
                },
            ).unwrap()
        };


        let layout = pipeline.layout().set_layouts().get(0).unwrap();
        let set = PersistentDescriptorSet::new_variable(
            &renderer.descriptor_set_allocator,
            layout.clone(),
            1,
            [WriteDescriptorSet::image_view_sampler_array(
                0,
                0,
                [
                    (tex as _, sampler.clone())
                ],
            )],
            [],
        ).unwrap();

        let indirect_args_pool = SubbufferAllocator::new(
            renderer.allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::INDIRECT_BUFFER | BufferUsage::STORAGE_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );

        let target_mesh = model.meshes.get(mesh_idx).unwrap().to_owned();

        let indirect_buffer = indirect_args_pool
            .allocate_slice(target_mesh.draw_calls.len() as _).unwrap();
        indirect_buffer
            .write().unwrap()
            .copy_from_slice(target_mesh.draw_calls.as_slice());

        ModelRenderGraph {
            fbo,
            camera,
            target_mesh,
            pipeline,
            indirect_buffer,
            model_transform,
            set
        }
    }
}

fn calculate_element_size(attribs: &Vec<Attribute>) -> u32 {
    let mut total_size = 0;
    for attrib in attribs {
        total_size += attrib.get_size();
    }
    total_size
}

impl Recorder for ModelRenderGraph {
    fn record(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        _device: Arc<Device>,
        _swapchain: Arc<Swapchain>,
        viewport: Viewport,
        image_index: usize,
    ) {
        let framebuffer = self.fbo.lock().unwrap();
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

        pub fn perspective(vertical_fov: f32, aspect_ratio: f32, z_near: f32, z_far: f32) -> Mat4 {
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

        let proj_matrix = perspective(90f32, viewport.extent[0] / viewport.extent[1], 0.1, 100.0);
        let model_matrix: Mat4 = self.model_transform.lock().unwrap().into_homogeneous_matrix();

        let push_constants = vs::PushConstantData {
            projMat: proj_matrix.into(),
            viewMat: self.camera.lock().unwrap().get_matrix().into(),
            modelTransform: model_matrix.into(),
        };

        builder
            .push_constants(self.pipeline.layout().clone(), 0, push_constants).unwrap()
            .bind_pipeline_graphics(self.pipeline.clone()).unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                self.set.clone()
            ).unwrap()
            .bind_vertex_buffers(0, self.target_mesh.vertex_buffer.clone()).unwrap();

        let src = self.target_mesh.index_buffer.clone();
        match self.target_mesh.idx_layout {
            IndexLayout::UInt8(_) => builder.bind_index_buffer(src).unwrap(),
            IndexLayout::UInt16(_) => builder.bind_index_buffer(IndexBuffer::U16(Subbuffer::reinterpret::<[u16]>(src))).unwrap(),
            IndexLayout::UInt32(_) => builder.bind_index_buffer(IndexBuffer::U32(Subbuffer::reinterpret::<[u32]>(src))).unwrap(),
            IndexLayout::UInt64(_) => panic!("64 bit index buffers not supported"),
        };

        builder.draw_indexed_indirect(self.indirect_buffer.to_owned()).unwrap();

        builder.end_rendering().unwrap();
    }
}
