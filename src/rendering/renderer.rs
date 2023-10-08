use std::sync::{Arc, Mutex};

use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::device::Queue;
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageType};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryAllocator, StandardMemoryAllocator};
use vulkano::swapchain::{acquire_next_image, SwapchainPresentInfo};
use vulkano::{
    command_buffer::allocator::StandardCommandBufferAllocator,
    device::{
        physical::PhysicalDevice, physical::PhysicalDeviceType, Device, DeviceCreateInfo,
        DeviceExtensions, Features, QueueCreateInfo, QueueFlags,
    },
    image::ImageUsage,
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    pipeline::graphics::viewport::Viewport,
    swapchain::{Surface, Swapchain, SwapchainCreateInfo},
    sync::{self, GpuFuture},
    Validated, Version, VulkanError, VulkanLibrary,
};
use winit::event_loop::EventLoop;
use winit::window::Window;

pub trait Recorder {
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
    );
}

pub struct Renderer {
    pub command_buffer_recorders: Vec<Arc<Mutex<dyn Recorder>>>,
    pub recreate_swapchain: bool,
    pub previous_frame_end: Option<Box<dyn GpuFuture>>,
    pub command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    pub descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    pub swapchain: Arc<Swapchain>,
    pub window: Arc<Window>,
    pub graphics_queue: Arc<Queue>,
    pub transfer_queue: Arc<Queue>,
    pub device: Arc<Device>,
    pub viewport: Viewport,
    pub allocator: Arc<StandardMemoryAllocator>,
}

impl Renderer {
    pub fn add_recorder(&mut self, predicate: Arc<Mutex<dyn Recorder>>) {
        self.command_buffer_recorders.push(predicate);
    }

    pub fn new(window: Arc<Window>, event_loop: &EventLoop<()>) -> (Renderer, Vec<Arc<Image>>) {
        let library = VulkanLibrary::new().unwrap();
        let required_extensions = Surface::required_extensions(&event_loop);
        let enabled_layers = vec![String::from("VK_LAYER_KHRONOS_validation")];

        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY, // Include MoltenVK in search
                enabled_extensions: required_extensions,
                enabled_layers,
                ..Default::default()
            },
        )
        .unwrap();

        let surface = Surface::from_window(instance.clone(), window.clone()).unwrap();

        let mut device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };

        let (physical_device, graphics_family_index) = instance
            .enumerate_physical_devices()
            .unwrap()
            .filter(|p| supports_dynamic_rendering(p.as_ref()))
            .filter(|p| p.supported_extensions().contains(&device_extensions))
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        q.queue_flags.intersects(QueueFlags::GRAPHICS)
                            && p.surface_support(i as u32, &surface).unwrap_or(false)
                    })
                    .map(|i| (p, i as u32))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 5,
                PhysicalDeviceType::IntegratedGpu => 1, // TODO: remember to undo this! Testing done on I-GPU to make sure it can run well
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            })
            .expect("no suitable physical device found");

        println!("max_descriptor_set_samplers {}", physical_device.properties().max_descriptor_set_samplers);
        println!("max_per_stage_descriptor_sampled_images {}", physical_device.properties().max_per_stage_descriptor_sampled_images);

        println!(
            "Using device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
        );

        if physical_device.api_version() < Version::V1_3 {
            device_extensions.khr_dynamic_rendering = true;
        }

        let transfer_family_index = physical_device
            .queue_family_properties()
            .iter()
            .enumerate()
            .filter(|(_, q)| q.queue_flags.intersects(QueueFlags::TRANSFER))
            .min_by_key(|(_, q)| q.queue_flags.count())
            .unwrap()
            .0 as u32;

        let (device, mut queues) = {
            let mut queue_create_infos = vec![QueueCreateInfo {
                queue_family_index: graphics_family_index,
                ..Default::default()
            }];

            if transfer_family_index != graphics_family_index {
                queue_create_infos.push(QueueCreateInfo {
                    queue_family_index: transfer_family_index,
                    ..Default::default()
                });
            }

            Device::new(
                physical_device,
                DeviceCreateInfo {
                    queue_create_infos,
                    enabled_extensions: device_extensions,

                    enabled_features: Features {
                        dynamic_rendering: true,
                        multi_draw_indirect: true,
                        descriptor_binding_variable_descriptor_count: true,
                        runtime_descriptor_array: true,
                        shader_int64: true,
                        buffer_device_address: true,
                        ..Features::empty()
                    },

                    ..Default::default()
                },
            )
            .unwrap()
        };

        let graphics_queue = queues.next().unwrap();
        // If we didn't get a dedicated transfer queue, fall back to the graphics queue for transfers.
        let transfer_queue = queues.next().unwrap_or_else(|| graphics_queue.clone());

        let (swapchain, images) = {
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

        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: [0.0, 0.0],
            depth_range: 0.0..=1.0,
        };

        (
            Renderer {
                command_buffer_recorders: vec![],
                swapchain,
                window,
                graphics_queue,
                transfer_queue,
                recreate_swapchain: false,
                previous_frame_end: Some(sync::now(device.clone()).boxed()),
                command_buffer_allocator: Arc::new(StandardCommandBufferAllocator::new(
                    device.clone(),
                    Default::default(),
                )),
                descriptor_set_allocator: Arc::new(StandardDescriptorSetAllocator::new(
                    device.clone(),
                )),
                device: device.clone(),
                viewport,
                allocator: Arc::new(StandardMemoryAllocator::new_default(device.clone())),
            },
            images,
        )
    }

    pub fn render(&mut self) {
        let (image_index, suboptimal, acquire_future) =
            match acquire_next_image(self.swapchain.clone(), None).map_err(Validated::unwrap) {
                Ok(r) => r,
                Err(VulkanError::OutOfDate) => {
                    self.recreate_swapchain = true;
                    return;
                }
                Err(_e) => panic!("failed to acquire next image: {_e}"),
            };

        if suboptimal {
            self.recreate_swapchain = true;
        }

        let mut builder = AutoCommandBufferBuilder::primary(
            &self.command_buffer_allocator,
            self.graphics_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        for x in &self.command_buffer_recorders {
            x.lock().unwrap().record(
                &mut builder,
                self.device.clone(),
                self.swapchain.clone(),
                self.viewport.clone(),
                image_index as usize,
            );
        }

        // Finish building the command buffer by calling `build`.
        let command_buffer = builder.build().unwrap();

        let render_future = self
            .previous_frame_end
            .take()
            .unwrap()
            .join(acquire_future)
            .then_execute(self.graphics_queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(
                self.graphics_queue.clone(),
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
            Err(_e) => {
                println!("failed to flush render_future: {_e}");
                self.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
            }
        }
    }
}

fn supports_dynamic_rendering(p: &PhysicalDevice) -> bool {
    p.api_version() >= Version::V1_3 || p.supported_extensions().khr_dynamic_rendering
}

pub struct WindowFrameBuffer {
    pub swapchain_image_views: Vec<Arc<ImageView>>,
    pub depth_image_view: Arc<ImageView>,
}

impl WindowFrameBuffer {
    pub fn new(
        viewport: &mut Viewport,
        memory_allocator: Arc<dyn MemoryAllocator>,
        images: &[Arc<Image>],
    ) -> WindowFrameBuffer {
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
            )
            .unwrap(),
        )
        .unwrap();

        let swapchain_image_views = images
            .iter()
            .map(|image| ImageView::new_default(image.clone()).unwrap())
            .collect::<Vec<_>>();

        WindowFrameBuffer {
            swapchain_image_views,
            depth_image_view,
        }
    }

    pub fn update(
        &mut self,
        viewport: &mut Viewport,
        memory_allocator: Arc<dyn MemoryAllocator>,
        images: &[Arc<Image>],
    ) {
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
            )
            .unwrap(),
        )
        .unwrap();

        self.swapchain_image_views = images
            .iter()
            .map(|image| ImageView::new_default(image.clone()).unwrap())
            .collect::<Vec<_>>()
    }
}
