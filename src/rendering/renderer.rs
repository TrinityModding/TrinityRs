use std::sync::Arc;

use vulkano::{buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage}, command_buffer::allocator::StandardCommandBufferAllocator, device::{
    Device, DeviceCreateInfo, DeviceExtensions, Features, physical::PhysicalDevice, physical::PhysicalDeviceType,
    QueueCreateInfo, QueueFlags,
}, image::{Image, ImageUsage, view::ImageView}, instance::{Instance, InstanceCreateFlags, InstanceCreateInfo}, memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator}, pipeline::{
    graphics::{
        color_blend::ColorBlendState,
        GraphicsPipelineCreateInfo,
        input_assembly::InputAssemblyState,
        multisample::MultisampleState,
        rasterization::RasterizationState,
        subpass::PipelineRenderingCreateInfo,
        vertex_input::{Vertex, VertexDefinition},
        viewport::{Viewport, ViewportState},
    },
    GraphicsPipeline,
    layout::PipelineDescriptorSetLayoutCreateInfo, PipelineLayout, PipelineShaderStageCreateInfo,
}, swapchain::{
    Surface, Swapchain, SwapchainCreateInfo,
}, sync::{self, GpuFuture}, Validated, Version, VulkanError, VulkanLibrary};
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, RenderingAttachmentInfo, RenderingInfo};
use vulkano::device::Queue;
use vulkano::render_pass::{AttachmentLoadOp, AttachmentStoreOp};
use vulkano::swapchain::{acquire_next_image, SwapchainPresentInfo};
use winit::event_loop::EventLoop;
use winit::window::Window;

#[derive(BufferContents, Vertex)]
#[repr(C)]
pub struct OurVertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}

pub struct Renderer {
    pub(crate) recreate_swapchain: bool,
    pub attachment_image_views: Vec<Arc<ImageView>>,
    pub previous_frame_end: Option<Box<dyn GpuFuture>>,
    pub command_buffer_allocator: StandardCommandBufferAllocator,
    pub swapchain: Arc<Swapchain>,
    pub window: Arc<Window>,
    pub queue: Arc<Queue>,
    pub pipeline: Arc<GraphicsPipeline>,
    pub vertex_buffer: Subbuffer<[OurVertex]>,
    pub device: Arc<Device>,
    pub viewport: Viewport,
}

impl Renderer {
    pub fn new(window: Arc<Window>, event_loop: &EventLoop<()>) -> Renderer {
        let library = VulkanLibrary::new().unwrap();
        let required_extensions = Surface::required_extensions(&event_loop);

        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY, // Include MoltenVK in search
                enabled_extensions: required_extensions,
                ..Default::default()
            },
        ).unwrap();

        let surface = Surface::from_window(instance.clone(), window.clone()).unwrap();

        let mut device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };

        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .unwrap()
            .filter(|p| {
                supports_dynamic_rendering(p.as_ref())
            })
            .filter(|p| {
                p.supported_extensions().contains(&device_extensions)
            })
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        // We select a queue family that supports graphics operations. When drawing to
                        // a window surface, as we do in this example, we also need to check that
                        // queues in this queue family are capable of presenting images to the surface.
                        q.queue_flags.intersects(QueueFlags::GRAPHICS)
                            && p.surface_support(i as u32, &surface).unwrap_or(false)
                    })

                    .map(|i| (p, i as u32))
            })

            .min_by_key(|(p, _)| {
                match p.properties().device_type {
                    PhysicalDeviceType::DiscreteGpu => 0,
                    PhysicalDeviceType::IntegratedGpu => 1,
                    PhysicalDeviceType::VirtualGpu => 2,
                    PhysicalDeviceType::Cpu => 3,
                    PhysicalDeviceType::Other => 4,
                    _ => 5,
                }
            })
            .expect("no suitable physical device found");

        println!(
            "Using device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
        );

        if physical_device.api_version() < Version::V1_3 {
            device_extensions.khr_dynamic_rendering = true;
        }

        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],

                enabled_extensions: device_extensions,

                enabled_features: Features {
                    dynamic_rendering: true,
                    ..Features::empty()
                },

                ..Default::default()
            },
        )
            .unwrap();

        let queue = queues.next().unwrap();

        let (mut swapchain, images) = {
            let surface_capabilities = device
                .physical_device()
                .surface_capabilities(&surface, Default::default())
                .unwrap();

            // Choosing the internal format that the images will have.
            let image_format = device
                .physical_device()
                .surface_formats(&surface, Default::default())
                .unwrap()[0]
                .0;

            Swapchain::new(
                device.clone(),
                surface,
                SwapchainCreateInfo {
                    min_image_count: surface_capabilities.min_image_count.max(2),
                    image_format,
                    image_extent: window.inner_size().into(),
                    image_usage: ImageUsage::COLOR_ATTACHMENT,
                    composite_alpha: surface_capabilities
                        .supported_composite_alpha
                        .into_iter()
                        .next()
                        .unwrap(),

                    ..Default::default()
                },
            )
                .unwrap()
        };

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        let vertices = [
            OurVertex {
                position: [-0.5, -0.25],
            },
            OurVertex {
                position: [0.0, 0.5],
            },
            OurVertex {
                position: [0.25, -0.1],
            },
        ];
        let vertex_buffer = Buffer::from_iter(
            memory_allocator,
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
        )
            .unwrap();

        mod vs {
            vulkano_shaders::shader! {
            ty: "vertex",
            src: r"
                #version 450

                layout(location = 0) in vec2 position;

                void main() {
                    gl_Position = vec4(position, 0.0, 1.0);
                }
            ",
        }
        }

        mod fs {
            vulkano_shaders::shader! {
            ty: "fragment",
            src: r"
                #version 450

                layout(location = 0) out vec4 f_color;

                void main() {
                    f_color = vec4(1.0, 0.0, 0.0, 1.0);
                }
            ",
        }
        }

        let pipeline = {
            let vs = vs::load(device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();
            let fs = fs::load(device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();

            let vertex_input_state = OurVertex::per_vertex()
                .definition(&vs.info().input_interface)
                .unwrap();

            let stages = [
                PipelineShaderStageCreateInfo::new(vs),
                PipelineShaderStageCreateInfo::new(fs),
            ];

            let layout = PipelineLayout::new(
                device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap(),
            ).unwrap();

            let subpass = PipelineRenderingCreateInfo {
                color_attachment_formats: vec![Some(swapchain.image_format())],
                ..Default::default()
            };

            GraphicsPipeline::new(
                device.clone(),
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

        let mut viewport = Viewport {
            offset: [0.0, 0.0],
            extent: [0.0, 0.0],
            depth_range: 0.0..=1.0,
        };

        Renderer {
            swapchain,
            window,
            queue,
            // TODO: remove these part specific ones
            pipeline,
            vertex_buffer,
            recreate_swapchain: false,
            attachment_image_views: update_viewport(&images, &mut viewport),
            previous_frame_end: Some(sync::now(device.clone()).boxed()),
            command_buffer_allocator: StandardCommandBufferAllocator::new(device.clone(), Default::default()),
            device,
            viewport
        }
    }

    pub fn render(&mut self) {
        let image_extent: [u32; 2] = self.window.inner_size().into();

        if image_extent.contains(&0) {
            return;
        }

        self.previous_frame_end.as_mut().unwrap().cleanup_finished();

        if self.recreate_swapchain {
            let (new_swapchain, new_images) = self.swapchain
                .recreate(SwapchainCreateInfo {
                    image_extent,
                    ..self.swapchain.create_info()
                })
                .expect("failed to recreate swapchain");

            self.swapchain = new_swapchain;
            self.attachment_image_views = update_viewport(&new_images, &mut self.viewport);
            self.recreate_swapchain = false;
        }

        let (image_index, suboptimal, acquire_future) =
            match acquire_next_image(self.swapchain.clone(), None).map_err(Validated::unwrap) {
                Ok(r) => r,
                Err(VulkanError::OutOfDate) => {
                    self.recreate_swapchain = true;
                    return;
                }
                Err(e) => panic!("failed to acquire next image: {e}"),
            };

        if suboptimal {
            self.recreate_swapchain = true;
        }

        let mut builder = AutoCommandBufferBuilder::primary(
            &self.command_buffer_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
            .unwrap();

        builder
            .begin_rendering(RenderingInfo {
                color_attachments: vec![Some(RenderingAttachmentInfo {
                    load_op: AttachmentLoadOp::Clear,
                    store_op: AttachmentStoreOp::Store,
                    clear_value: Some([0.0, 0.0, 1.0, 1.0].into()),
                    ..RenderingAttachmentInfo::image_view(
                        self.attachment_image_views[image_index as usize].clone(),
                    )
                })],
                ..Default::default()
            }).unwrap()
            .set_viewport(0, [self.viewport.clone()].into_iter().collect()).unwrap()
            .bind_pipeline_graphics(self.pipeline.clone()).unwrap()
            .bind_vertex_buffers(0, self.vertex_buffer.clone()).unwrap()
            .draw(self.vertex_buffer.len() as u32, 1, 0, 0).unwrap()
            .end_rendering().unwrap();

        // Finish building the command buffer by calling `build`.
        let command_buffer = builder.build().unwrap();

        let render_future = self.previous_frame_end
            .take()
            .unwrap()
            .join(acquire_future)
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(
                self.queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(self.swapchain.clone(), image_index),
            )
            .then_signal_fence_and_flush();

        match render_future.map_err(Validated::unwrap) {
            Ok(future) => {
                self.previous_frame_end = Some(future.boxed());
            }
            Err(VulkanError::OutOfDate) => {
                self.recreate_swapchain = true;
                self.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
            }
            Err(e) => {
                println!("failed to flush render_future: {e}");
                self.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
            }
        }
    }
}

fn update_viewport(
    images: &[Arc<Image>],
    viewport: &mut Viewport,
) -> Vec<Arc<ImageView>> {
    let extent = images[0].extent();
    viewport.extent = [extent[0] as f32, extent[1] as f32];

    images
        .iter()
        .map(|image| ImageView::new_default(image.clone()).unwrap())
        .collect::<Vec<_>>()
}

fn supports_dynamic_rendering(p: &PhysicalDevice) -> bool {
    p.api_version() >= Version::V1_3 || p.supported_extensions().khr_dynamic_rendering
}
