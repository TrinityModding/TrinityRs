use std::sync::Arc;

use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DrawIndirectCommand, PrimaryAutoCommandBuffer, RenderingAttachmentInfo, RenderingInfo};
use vulkano::device::Device;
use vulkano::image::view::ImageView;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::{GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::pipeline::graphics::color_blend::ColorBlendState;
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::subpass::PipelineRenderingCreateInfo;
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition};
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
use crate::io::model::RenderModel;

use crate::rendering::renderer::{Recorder, Renderer};

mod rendering;
mod io;

fn main() {
    let model = RenderModel::from_trmdl(String::from("F:/PokemonModels/SV/pokemon/data/pm0197/pm0197_00_00/pm0197_00_00.trmdl"));

    let event_loop = EventLoop::new();
    let window = Arc::new(WindowBuilder::new()
        .with_title("trinity-rs")
        .build(&event_loop).unwrap());

    let mut renderer = Renderer::new(window.clone(), &event_loop);
    let triangle_renderer = TriangleRenderer::new(&mut renderer);
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

#[derive(BufferContents, Vertex)]
#[repr(C)]
pub struct Vertex2D {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}

#[derive(Clone)]
struct TriangleRenderer {
    vertex_buffer: Subbuffer<[Vertex2D]>,
    pipeline: Arc<GraphicsPipeline>,
    indirect_buffer: Subbuffer<[DrawIndirectCommand]>,
}

impl TriangleRenderer {
    pub fn new(renderer: &mut Renderer) -> TriangleRenderer {
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(renderer.device.clone()));

        let vertices = [
            Vertex2D {
                position: [0.0, -1.0],
            },
            Vertex2D {
                position: [-1.0, 1.0],
            },
            Vertex2D {
                position: [1.0, 1.0],
            },
        ];
        let vertex_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vertices,
        ).unwrap();

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

            let vertex_input_state = Vertex2D::per_vertex()
                .definition(&vs.info().input_interface)
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

            GraphicsPipeline::new(
                renderer.device.clone(),
                None,
                GraphicsPipelineCreateInfo {
                    stages: stages.into_iter().collect(),
                    vertex_input_state: Some(vertex_input_state),
                    input_assembly_state: Some(InputAssemblyState::default()),
                    viewport_state: Some(ViewportState::viewport_dynamic_scissor_irrelevant()),
                    rasterization_state: Some(RasterizationState::default()),
                    multisample_state: Some(MultisampleState::default()),
                    color_blend_state: Some(ColorBlendState::new(
                        subpass.color_attachment_formats.len() as u32,
                    )),
                    subpass: Some(subpass.into()),
                    ..GraphicsPipelineCreateInfo::layout(layout)
                },
            ).unwrap()
        };

        let indirect_args_pool = SubbufferAllocator::new(
            memory_allocator,
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::INDIRECT_BUFFER | BufferUsage::STORAGE_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );

        let indirect_commands = [DrawIndirectCommand {
            vertex_count: vertex_buffer.len() as u32,
            instance_count: 1,
            first_vertex: 0,
            first_instance: 0,
        }];

        let indirect_buffer = indirect_args_pool
            .allocate_slice(indirect_commands.len() as _)
            .unwrap();
        indirect_buffer
            .write()
            .unwrap()
            .copy_from_slice(&indirect_commands);

        let indirect_buffer = indirect_args_pool
            .allocate_slice(indirect_commands.len() as _)
            .unwrap();
        indirect_buffer
            .write()
            .unwrap()
            .copy_from_slice(&indirect_commands);

        TriangleRenderer {
            indirect_buffer,
            vertex_buffer,
            pipeline,
        }
    }
}

impl Recorder for TriangleRenderer {
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
                ..Default::default()
            }).unwrap()
            .set_viewport(0, [viewport].into_iter().collect()).unwrap()
            .bind_pipeline_graphics(self.pipeline.clone()).unwrap()
            .bind_vertex_buffers(0, self.vertex_buffer.clone()).unwrap()
            .draw_indirect(self.indirect_buffer.to_owned()).unwrap()
            .end_rendering().unwrap();
    }
}
