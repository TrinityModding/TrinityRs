mod rendering;

use std::sync::Arc;

use winit::{
    event_loop::EventLoop,
    window::WindowBuilder,
};
use winit::event::{Event, WindowEvent};
use winit::event_loop::ControlFlow;
use crate::rendering::renderer::Renderer;

fn main() {
    let event_loop = EventLoop::new();
    let window = Arc::new(WindowBuilder::new().build(&event_loop).unwrap());

    let mut renderer = Renderer::new(window.clone(), &event_loop);

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
