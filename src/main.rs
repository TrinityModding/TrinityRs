use crate::io::model::from_trmdl;
use crate::rendering::graph::{Camera, SceneGraph};
use std::rc::Rc;
use std::sync::{Arc, Mutex, RwLock};
use ultraviolet::{Rotor3, Similarity3, Vec3};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryCommandBufferAbstract,
};
use vulkano::image::sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo};
use vulkano::swapchain::SwapchainCreateInfo;
use vulkano::sync::GpuFuture;
use winit::event::ElementState::{Pressed, Released};
use winit::event::{Event, VirtualKeyCode, WindowEvent};
use winit::event_loop::ControlFlow;
use winit::{event_loop::EventLoop, window::WindowBuilder};

use crate::rendering::renderer::{Renderer, WindowFrameBuffer};
use crate::rendering::shaders::load_shaders;

mod io;
mod rendering;

fn main() {
    let event_loop = EventLoop::new();
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("trinity-rs")
            .build(&event_loop)
            .unwrap(),
    );

    // Renderer Setup
    let info = Renderer::new(window.clone(), &event_loop);
    let mut renderer = info.0;

    let shaders = load_shaders(&renderer);

    // Graph setup
    let fbo = Rc::new(RwLock::new(WindowFrameBuffer::new(
        &mut renderer.viewport,
        renderer.allocator.clone(),
        info.1.as_slice(),
    )));
    let graph = SceneGraph::new(fbo.clone(), shaders, &renderer, Camera::new());
    let mut graph_lock = graph.lock().unwrap();

    // Load model into graph
    let _lod_mesh = from_trmdl(
        String::from("pikachu/pm0025_00_00.trmdl"),
        &mut graph_lock,
        &mut renderer,
    );

    graph_lock.add_model(_lod_mesh.get_best());

    let model_transform = Arc::new(Mutex::new(Similarity3::identity()));
    model_transform
        .lock()
        .unwrap()
        .append_translation(Vec3::new(0.0, -0.8, 0.0));

    let mut uploads = AutoCommandBufferBuilder::primary(
        &renderer.command_buffer_allocator,
        renderer.graphics_queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();
    
    graph_lock
        .texture_manager
        .upload_all(&renderer, &mut uploads);

    let _sampler = Sampler::new(
        renderer.device.clone(),
        SamplerCreateInfo {
            mag_filter: Filter::Nearest,
            min_filter: Filter::Nearest,
            address_mode: [SamplerAddressMode::Repeat; 3],
            ..Default::default()
        },
    )
    .unwrap();

    renderer.previous_frame_end = Some(
        uploads
            .build()
            .unwrap()
            .execute(renderer.graphics_queue.clone())
            .unwrap()
            .boxed(),
    );

    drop(graph_lock);

    renderer.add_recorder(graph);

    let camera = Arc::new(Mutex::new(Camera::new()));

    // Logic Setup
    let mut move_forward = false;
    let mut move_backward = false;
    event_loop.run(move |event, _, control_flow| match event {
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
            event:
                WindowEvent::KeyboardInput {
                    device_id: _device_id,
                    input,
                    is_synthetic: _is_synthetic,
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
            renderer
                .previous_frame_end
                .as_mut()
                .unwrap()
                .cleanup_finished();
            if renderer.recreate_swapchain {
                let (new_swapchain, new_images) = renderer
                    .swapchain
                    .recreate(SwapchainCreateInfo {
                        image_extent,
                        ..renderer.swapchain.create_info()
                    })
                    .expect("failed to recreate swapchain");

                renderer.swapchain = new_swapchain;
                fbo.write().unwrap().update(
                    &mut renderer.viewport,
                    renderer.allocator.clone(),
                    new_images.as_slice(),
                );
                renderer.recreate_swapchain = false;
            }
            renderer.render();

            model_transform
                .lock()
                .unwrap()
                .append_rotation(Rotor3::from_euler_angles(0.0, 0.0, 0.05));

            let mut cam = camera.lock().unwrap();
            if move_forward {
                cam.translate(0.0, 0.0, 0.01);
            }
            if move_backward {
                cam.translate(0.0, 0.0, -0.01);
            }
        }
        _ => (),
    });
}
