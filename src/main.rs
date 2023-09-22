use std::ops::Add;
use std::sync::{Arc, Mutex};
use ultraviolet::{Mat4, Vec3};

use vulkano::buffer::{BufferUsage, IndexBuffer, Subbuffer};
use vulkano::buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DrawIndexedIndirectCommand, PrimaryAutoCommandBuffer, RenderingAttachmentInfo, RenderingInfo};
use vulkano::device::Device;
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::memory::allocator::MemoryTypeFilter;
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::pipeline::graphics::color_blend::ColorBlendState;
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::{FrontFace, RasterizationState};
use vulkano::pipeline::graphics::subpass::PipelineRenderingCreateInfo;
use vulkano::pipeline::graphics::vertex_input::{VertexInputAttributeDescription, VertexInputBindingDescription, VertexInputRate, VertexInputState};
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::render_pass::{AttachmentLoadOp, AttachmentStoreOp};
use vulkano::swapchain::Swapchain;
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
    rotation: Vec3,
    cached_transform: Mat4,
}

impl Camera {
    pub fn new() -> Camera {
        Camera {
            translation: Vec3::new(0.0, 0.0, 0.0),
            rotation: Vec3::new(0.0, 0.0, 0.0),
            cached_transform: Mat4::identity().inversed()
        }
    }

    pub fn update(&mut self) {
        self.cached_transform = Mat4::identity().translated(&self.translation);
    }

    pub fn translate(&mut self, x: f32, y: f32, z: f32) {
        self.translation = self.translation.add(Vec3::new(x, y, z));
        self.update();
    }

    pub fn get_matrix(&self) -> Mat4 {
        self.cached_transform.clone()
    }
}

fn main() {
    let event_loop = EventLoop::new();
    let window = Arc::new(WindowBuilder::new()
        .with_title("trinity-rs")
        .build(&event_loop).unwrap());

    let mut renderer = Renderer::new(window.clone(), &event_loop);
    let model = RenderModel::from_trmdl(String::from("A:/PokemonScarlet/pokemon/data/pm0855/pm0855_00_00/pm0855_00_00.trmdl"), renderer.allocator.clone());

    let mut camera = Arc::new(Mutex::new(Camera::new()));
    let triangle_renderer = ModelRenderGraph::new(&mut renderer, model, 0, camera.clone()); // load the non-lod mesh
    renderer.record(triangle_renderer);

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
                        },
                        Some(VirtualKeyCode::S) => {
                            move_forward = false;
                            move_backward = true;
                        },
                        _ => {},
                    }
                }

                if input.state == Released {
                    match input.virtual_keycode {
                        Some(VirtualKeyCode::W) => {
                            move_forward = false;
                        },
                        Some(VirtualKeyCode::S) => {
                            move_backward = false;
                        },
                        _ => {},
                    }
                }
            }
            Event::RedrawEventsCleared => {
                renderer.render();
                let mut cam = camera.lock().unwrap();

                if move_forward {
                    cam.translate(0.0, 0.0, 0.1);
                    println!("{}", cam.translation.z);
                }
                if move_backward {
                    cam.translate(0.0, 0.0, -0.1);
                    println!("{}",cam.translation.z);
                }
            },
            _ => (),
        }
    });
}

struct ModelRenderGraph {
    camera: Arc<Mutex<Camera>>,
    pipeline: Arc<GraphicsPipeline>,
    indirect_buffer: Subbuffer<[DrawIndexedIndirectCommand]>,
    target_mesh: RenderModel,
    model_transform: Mat4
}

impl ModelRenderGraph {
    pub fn new(renderer: &mut Renderer, model: RootModel, mesh_idx: usize, camera: Arc<Mutex<Camera>>) -> ModelRenderGraph {
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

            let layout = PipelineLayout::new(
                renderer.device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                    .into_pipeline_layout_create_info(renderer.device.clone())
                    .unwrap(),
            ).unwrap();

            let subpass = PipelineRenderingCreateInfo {
                color_attachment_formats: vec![Some(renderer.swapchain.image_format())],
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

            println!("Vertex Layout for model: {:?}", model.attributes);

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
                    rasterization_state: Some(RasterizationState::default()
                        .front_face(FrontFace::Clockwise)),
                    multisample_state: Some(MultisampleState::default()),
                    color_blend_state: Some(ColorBlendState::new(
                        subpass.color_attachment_formats.len() as u32,
                    )),
                    // depth_stencil_state: Some(DepthStencilState::simple_depth_test()),
                    subpass: Some(subpass.into()),
                    ..GraphicsPipelineCreateInfo::layout(layout)
                },
            ).unwrap()
        };

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
            camera,
            target_mesh,
            pipeline,
            indirect_buffer,
            model_transform: Mat4::identity(),
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
        window_image_views: &Vec<Arc<ImageView>>,
        viewport: Viewport,
        image_index: usize,
    ) {
        builder
            .begin_rendering(RenderingInfo {
                color_attachments: vec![Some(RenderingAttachmentInfo {
                    load_op: AttachmentLoadOp::Clear,
                    store_op: AttachmentStoreOp::Store,
                    clear_value: Some([0.0, 0.0, 1.0, 1.0].into()),
                    ..RenderingAttachmentInfo::image_view(
                        window_image_views[image_index].clone(),
                    )
                })],
                // depth_attachment: Some(RenderingAttachmentInfo {
                //     load_op: AttachmentLoadOp::Clear,
                //     store_op: AttachmentStoreOp::Store,
                //     clear_value: Some([0.0, 0.0, 0.0, 1.0].into()),
                //     ..RenderingAttachmentInfo::image_view(
                //         window_image_views[image_index].clone(),
                //     )
                // }),
                ..Default::default()
            }).unwrap()
            .set_viewport(0, [viewport.clone()].into_iter().collect()).unwrap();

        let proj_matrix = ultraviolet::projection::rh_ydown::perspective_vk(90f32, viewport.extent[0] / viewport.extent[1], 0.1, 1000.0);

        let push_constants = vs::PushConstantData {
            projMat: proj_matrix.into(),
            viewMat: self.camera.lock().unwrap().get_matrix().into(),
            modelTransform: self.model_transform.into(),
        };

        builder
            .push_constants(self.pipeline.layout().clone(), 0, push_constants).unwrap()
            .bind_pipeline_graphics(self.pipeline.clone()).unwrap()
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
