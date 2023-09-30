use crate::io::model::{Attribute, AttributeFormat};
use vulkano::format::Format;
use vulkano::pipeline::graphics::vertex_input::{
    VertexInputAttributeDescription, VertexInputBindingDescription, VertexInputRate,
    VertexInputState,
};

pub fn create_vertex_layout(attribs: &Vec<Attribute>) -> VertexInputState {
    let mut vertex_attributes = Vec::new();
    let mut vertex_bindings = Vec::new();
    let total_size = calculate_element_size(attribs);
    let mut offset = 0;
    for attr_idx in 0..attribs.len() {
        let attribute = attribs.get(attr_idx).unwrap();
        let vk_type = match attribute.format {
            AttributeFormat::None(_, _) => panic!("None attribute exists"),
            AttributeFormat::Rgba8UNorm(_, _) => Format::R8G8B8A8_UNORM,
            AttributeFormat::Rgba8Unsigned(_, _) => Format::R8G8B8A8_UINT,
            AttributeFormat::R32UInt(_, _) => Format::R32_UINT,
            AttributeFormat::R32Int(_, _) => Format::R32_SINT,
            AttributeFormat::Rgba16UNorm(_, _) => Format::R16G16B16A16_UNORM,
            AttributeFormat::Rgba16Float(_, _) => Format::R16G16B16A16_SFLOAT,
            AttributeFormat::Rg32Float(_, _) => Format::R32G32_SFLOAT,
            AttributeFormat::Rgb32Float(_, _) => Format::R32G32B32_SFLOAT,
            AttributeFormat::Rgba32Float(_, _) => Format::R32G32B32A32_SFLOAT,
        };
        let size = attribute.get_size();

        vertex_attributes.push((
            attr_idx as u32,
            VertexInputAttributeDescription {
                binding: 0,
                format: vk_type,
                offset,
            },
        ));
        vertex_bindings.push((
            0,
            VertexInputBindingDescription {
                stride: total_size,
                input_rate: VertexInputRate::Vertex,
            },
        ));
        offset += size;
    }

    VertexInputState::new()
        .attributes(vertex_attributes)
        .bindings(vertex_bindings)
}

fn calculate_element_size(attribs: &Vec<Attribute>) -> u32 {
    let mut total_size = 0;
    for attrib in attribs {
        total_size += attrib.get_size();
    }
    total_size
}
