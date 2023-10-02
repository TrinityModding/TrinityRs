use std::collections::HashMap;
use crate::rendering::renderer::Renderer;
use std::mem::size_of;
use std::sync::Arc;
use itertools::Itertools;
use vulkano::format::Format;
use vulkano::pipeline::graphics::color_blend::ColorBlendState;
use vulkano::pipeline::graphics::depth_stencil::DepthStencilState;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::{FrontFace, RasterizationState};
use vulkano::pipeline::graphics::subpass::PipelineRenderingCreateInfo;
use vulkano::pipeline::graphics::vertex_input::{
    VertexInputAttributeDescription, VertexInputBindingDescription, VertexInputRate,
    VertexInputState,
};
use vulkano::pipeline::graphics::viewport::ViewportState;
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::{GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo};

pub struct VertexAttributeInfo {
    pub binding: u32,
    pub format: Format,
}

impl VertexAttributeInfo {
    pub fn get_attribute_size(&self) -> u32 {
        match self.format {
            Format::R32G32B32_SFLOAT => size_of::<f32>() as u32 * 3,
            Format::R32G32_SFLOAT => size_of::<f32>() as u32 * 2,
            _ => panic!("Unimplemented vertex attribute format {:?}", self.format),
        }
    }
}

pub struct PipelineCreationInfo {
    pub attributes: Vec<VertexAttributeInfo>,
}

impl PipelineCreationInfo {
    pub fn create(
        &self,
        renderer: &Renderer,
        stages: [PipelineShaderStageCreateInfo; 2],
        layout: Arc<PipelineLayout>,
    ) -> Arc<GraphicsPipeline> {
        let element_sizes: HashMap<u32, u32> =
            self.attributes.iter().group_by(|x| x.binding)
                .into_iter()
                .map(|(binding, group)| {
                    let size_sum: u32 = group.map(|x| x.get_attribute_size()).sum();
                    (binding, size_sum)
                })
                .collect();

        let mut vertex_attributes = Vec::new();
        let mut vertex_bindings = Vec::new();
        let mut offsets: HashMap<u32, u32> = HashMap::new();
        for i in 0..self.attributes.len() {
            let attribute = self.attributes.get(i).unwrap();
            let size = attribute.get_attribute_size();

            vertex_attributes.push((
                i as u32,
                VertexInputAttributeDescription {
                    binding: attribute.binding,
                    format: attribute.format,
                    offset: *offsets.entry(attribute.binding).or_insert(0),
                },
            ));
            vertex_bindings.push((
                attribute.binding,
                VertexInputBindingDescription {
                    stride: *element_sizes.get(&attribute.binding).unwrap(),
                    input_rate: VertexInputRate::Vertex,
                },
            ));

            if let Some(value) = offsets.get_mut(&attribute.binding) {
                *value += size;
            }
        }

        let vertex_attribs = VertexInputState::new()
            .attributes(vertex_attributes)
            .bindings(vertex_bindings);

        let subpass = PipelineRenderingCreateInfo {
            color_attachment_formats: vec![Some(renderer.swapchain.image_format())],
            depth_attachment_format: Some(Format::D16_UNORM),
            ..Default::default()
        };

        GraphicsPipeline::new(
            renderer.device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(vertex_attribs),
                input_assembly_state: Some(InputAssemblyState::default()),
                viewport_state: Some(ViewportState::viewport_dynamic_scissor_irrelevant()),
                rasterization_state: Some(
                    RasterizationState::default().front_face(FrontFace::Clockwise),
                ),
                depth_stencil_state: Some(DepthStencilState::simple_depth_test()),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState::new(
                    subpass.color_attachment_formats.len() as u32,
                )),
                subpass: Some(subpass.into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )
            .unwrap()
    }
}
