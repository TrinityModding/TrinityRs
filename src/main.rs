use crate::io::model::from_trmdl;
use crate::rendering::graph::SceneGraph;
use crate::rendering::pipeline::{PipelineCreationInfo, VertexAttributeInfo};
use std::fmt::Debug;
use std::fs;
use std::fs::File;
use std::io::Read;
use std::ops::Add;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::{Arc, Mutex, RwLock};
use ultraviolet::{Mat4, Rotor3, Similarity3, Vec3};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryCommandBufferAbstract,
};
use vulkano::descriptor_set::layout::DescriptorBindingFlags;
use vulkano::format::Format;
use vulkano::image::sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo};
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::shader::{EntryPoint, ShaderModule, ShaderModuleCreateInfo};
use vulkano::swapchain::SwapchainCreateInfo;
use vulkano::sync::GpuFuture;
use winit::event::ElementState::{Pressed, Released};
use winit::event::{Event, VirtualKeyCode, WindowEvent};
use winit::event_loop::ControlFlow;
use winit::{event_loop::EventLoop, window::WindowBuilder};

use crate::rendering::renderer::{Renderer, WindowFrameBuffer};
use crate::rendering::texture_manager::PngTextureUploader;

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

    // Load shaders
    let standard_stages = [
        PipelineShaderStageCreateInfo::new(read_entrypoint("standard.vs", &renderer)),
        PipelineShaderStageCreateInfo::new(read_entrypoint("standard.fs", &renderer)),
    ];

    let mut layout_create_info = PipelineDescriptorSetLayoutCreateInfo::from_stages(&standard_stages);

    let binding = layout_create_info.set_layouts[0]
        .bindings
        .get_mut(&0).unwrap();
    binding.binding_flags |= DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT;
    binding.descriptor_count = 1;

    // This layout is shared with every shader so it doesn't matter what shader gets this
    let layout = PipelineLayout::new(
        renderer.device.clone(),
        layout_create_info
            .into_pipeline_layout_create_info(renderer.device.clone())
            .unwrap(),
    )
    .unwrap();

    let _standard = PipelineCreationInfo {
        attributes: vec![
            VertexAttributeInfo {
                binding: 0,
                format: Format::R32G32B32_SFLOAT,
            }, // Pos
            VertexAttributeInfo {
                binding: 1,
                format: Format::R32G32_SFLOAT,
            }, // TexCoords
        ],
    }
    .create(&renderer, standard_stages, layout.clone());

    // Graph setup
    let fbo = Rc::new(RwLock::new(WindowFrameBuffer::new(
        &mut renderer.viewport,
        renderer.allocator.clone(),
        info.1.as_slice(),
    )));
    let graph = SceneGraph::new(fbo.clone(), layout, &renderer);
    let mut graph_lock = graph.lock().unwrap();

    graph_lock.add_shader(
        "Standard",
        _standard.clone(),
        renderer.allocator.clone(),
        renderer.device.clone(),
    );
    graph_lock.add_shader(
        "SSS",
        _standard.clone(),
        renderer.allocator.clone(),
        renderer.device.clone(),
    );
    graph_lock.add_shader(
        "EyeClearCoat",
        _standard.clone(),
        renderer.allocator.clone(),
        renderer.device.clone(),
    );
    graph_lock.add_shader(
        "FresnelBlend",
        _standard.clone(),
        renderer.allocator.clone(),
        renderer.device.clone(),
    );

    // Load model into graph
    let pokemon = "pm1018";
    let _lod_mesh = from_trmdl(
        format!(
            "A:/PokemonScarlet/pokemon/data/{}/{}_00_00/{}_00_00.trmdl",
            pokemon, pokemon, pokemon
        ),
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
        .queue(Box::new(PngTextureUploader::new(
            fs::read(PathBuf::from("C:/Users/Hayden/Desktop/fallback.png")).unwrap(),
        )));
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

fn read_entrypoint(shader_name: &str, renderer: &Renderer) -> EntryPoint {
    let code = read_spirv_words_from_file(format!("shaders/{}.spv", shader_name));
    let module = unsafe {
        ShaderModule::new(renderer.device.clone(), ShaderModuleCreateInfo::new(&code)).unwrap()
    };
    module.entry_point("main").unwrap()
}

fn read_spirv_words_from_file(path: impl AsRef<Path>) -> Vec<u32> {
    let path = path.as_ref();
    let mut bytes = Vec::new();
    let mut file = File::open(path)
        .unwrap_or_else(|err| panic!("can't open file `{}`: {}.", path.display(), err));

    file.read_to_end(&mut bytes).unwrap();
    vulkano::shader::spirv::bytes_to_words(&bytes)
        .unwrap_or_else(|err| panic!("file `{}`: {}", path.display(), err))
        .into_owned()
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
