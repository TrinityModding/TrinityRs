use std::collections::HashMap;
use std::sync::Arc;
use vulkano::descriptor_set::layout::DescriptorBindingFlags;
use vulkano::format::Format;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::shader::{EntryPoint, ShaderModule};
use crate::rendering::pipeline::{PipelineCreationInfo, VertexAttributeInfo};
use crate::rendering::renderer::Renderer;

vulkano_shaders::shader!(
        shaders: {
            standard_vs: {
                ty: "vertex",
                path: "shaders/standard.vs.glsl"
            },
            standard_fs: {
                ty: "fragment",
                path: "shaders/standard.fs.glsl"
            }
        }
);

pub fn load_shaders(renderer: &Renderer) -> ShaderCollection {
    let device = renderer.device.clone();

    let standard_stages = [
        PipelineShaderStageCreateInfo::new(read_entrypoint(
            load_standard_vs(device.clone()).unwrap(),
        )),
        PipelineShaderStageCreateInfo::new(read_entrypoint(
            load_standard_fs(device.clone()).unwrap(),
        )),
    ];

    let mut descriptor_layout_creation_info =
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&standard_stages);

    // Textures
    let texture_binding = descriptor_layout_creation_info.set_layouts[0]
        .bindings
        .get_mut(&1)
        .unwrap();
    texture_binding.binding_flags |= DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT;
    texture_binding.descriptor_count = renderer.device.physical_device().properties().max_per_stage_resources - 10; // Take 10 off just to be safe

    let layout = PipelineLayout::new(
        device.clone(),
        descriptor_layout_creation_info
            .into_pipeline_layout_create_info(device.clone())
            .unwrap(),
    ).unwrap();

    let standard_pipeline = PipelineCreationInfo {
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
    }.create(renderer, standard_stages, layout.clone());

    let mut shaders = HashMap::new();
    shaders.insert("Standard", standard_pipeline.clone());
    shaders.insert("SSS", standard_pipeline.clone()); // TODO: implement this shader properly
    shaders.insert("Unlit", standard_pipeline.clone()); // TODO: implement this shader properly
    shaders.insert("EyeClearCoat", standard_pipeline.clone()); // TODO: implement this shader properly
    shaders.insert("FresnelBlend", standard_pipeline.clone()); // TODO: implement this shader properly

    ShaderCollection {
        layout,
        shaders,
    }
}

fn read_entrypoint(module: Arc<ShaderModule>) -> EntryPoint {
    module.entry_point("main").unwrap()
}

pub struct ShaderCollection {
    pub layout: Arc<PipelineLayout>,
    pub shaders: HashMap<&'static str, Arc<GraphicsPipeline>>,
}
