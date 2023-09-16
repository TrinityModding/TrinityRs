use std::mem::transmute;
use std::ptr::null;
use std::sync::Arc;

use bytemuck::cast;
use vulkano::buffer::{BufferContents, BufferUsage, IndexBuffer, Subbuffer};
use vulkano::buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DrawIndexedIndirectCommand, PrimaryAutoCommandBuffer, RenderingAttachmentInfo, RenderingInfo};
use vulkano::device::Device;
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::memory::allocator::MemoryTypeFilter;
use vulkano::pipeline::{GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::pipeline::graphics::color_blend::ColorBlendState;
use vulkano::pipeline::graphics::depth_stencil::DepthStencilState;
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::subpass::PipelineRenderingCreateInfo;
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition, VertexInputAttributeDescription, VertexInputBindingDescription, VertexInputRate, VertexInputState};
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::render_pass::{AttachmentLoadOp, AttachmentStoreOp};
use vulkano::swapchain::Swapchain;
use winit::{
    event_loop::EventLoop,
    window::WindowBuilder,
};
use winit::event::{Event, WindowEvent};
use winit::event_loop::ControlFlow;

use crate::io::model::{AttributeSize, IndexLayout, RenderModel, RootModel};
use crate::rendering::renderer::{Recorder, Renderer};

mod rendering;
mod io;

fn main() {
    let event_loop = EventLoop::new();
    let window = Arc::new(WindowBuilder::new()
        .with_title("trinity-rs")
        .build(&event_loop).unwrap());

    let mut renderer = Renderer::new(window.clone(), &event_loop);
    let model = RenderModel::from_trmdl(String::from("F:/PokemonModels/SV/pokemon/data/pm0197/pm0197_00_00/pm0197_00_00.trmdl"), renderer.allocator.clone());

    let triangle_renderer = ModelRenderGraph::new(&mut renderer, model, 0); // load the non-lod mesh
    renderer.record(triangle_renderer);

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
            Event::RedrawEventsCleared => {
                renderer.render();
            }
            _ => (),
        }
    });
}

#[derive(Clone)]
struct ModelRenderGraph {
    pipeline: Arc<GraphicsPipeline>,
    indirect_buffer: Subbuffer<[DrawIndexedIndirectCommand]>,
    target_mesh: RenderModel,
}

impl ModelRenderGraph {
    pub fn new(renderer: &mut Renderer, model: RootModel, mesh_idx: usize) -> ModelRenderGraph {
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
            let mut size = 0;
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

                vertex_attributes.push((attr_idx as u32, VertexInputAttributeDescription {
                    binding: 0,
                    format: vk_type,
                    offset: size,
                }));

                vertex_bindings.push((0, VertexInputBindingDescription {
                    stride: size,
                    input_rate: VertexInputRate::Vertex,
                }));

                size += match attribute.size {
                    AttributeSize::None(_, s) => s as u32,
                    AttributeSize::Rgba8UNorm(_, s) => s as u32,
                    AttributeSize::Rgba8Unsigned(_, s) => s as u32,
                    AttributeSize::R32UInt(_, s) => s as u32,
                    AttributeSize::R32Int(_, s) => s as u32,
                    AttributeSize::Rgba16UNorm(_, s) => s as u32,
                    AttributeSize::Rgba16Float(_, s) => s as u32,
                    AttributeSize::Rg32Float(_, s) => s as u32,
                    AttributeSize::Rgb32Float(_, s) => s as u32,
                    AttributeSize::Rgba32Float(_, s) => s as u32,
                };
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
                    rasterization_state: Some(RasterizationState::default()),
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
            target_mesh,
            pipeline,
            indirect_buffer,
        }
    }
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
            .set_viewport(0, [viewport].into_iter().collect()).unwrap();

        unsafe {
            let src = self.target_mesh.index_buffer.clone();
            let index_buffer = match self.target_mesh.idx_layout {
                IndexLayout::UInt8(_) => IndexBuffer::U8(transmute(src)),
                IndexLayout::UInt16(_) => IndexBuffer::U16(transmute(src)),
                IndexLayout::UInt32(_) => IndexBuffer::U32(transmute(src)),
                IndexLayout::UInt64(_) => panic!("64 bit index buffers not supported"),
            };

            builder
                .bind_pipeline_graphics(self.pipeline.clone()).unwrap()
                .bind_vertex_buffers(0, self.target_mesh.vertex_buffer.clone()).unwrap()
                .bind_index_buffer(index_buffer).unwrap()
                .draw_indexed_indirect(self.indirect_buffer.to_owned()).unwrap();
        }

        builder.end_rendering().unwrap();
    }
}
