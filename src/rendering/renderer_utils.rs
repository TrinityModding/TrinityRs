use std::io::Cursor;
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CopyBufferToImageInfo, PrimaryAutoCommandBuffer};
use vulkano::DeviceSize;
use vulkano::format::Format;
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::image::view::ImageView;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::pipeline::graphics::vertex_input::{VertexInputAttributeDescription, VertexInputBindingDescription, VertexInputRate, VertexInputState};
use crate::io::model::{Attribute, AttributeSize};
use crate::rendering::renderer::Renderer;

pub fn load_png(png_bytes: Vec<u8>, renderer: &Renderer, uploads: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>) -> Arc<ImageView> {
    let cursor = Cursor::new(png_bytes);
    let decoder = png::Decoder::new(cursor);
    let mut reader = decoder.read_info().unwrap();
    let info = reader.info();
    let extent = [info.width, info.height, 1];

    let upload_buffer = Buffer::new_slice(
        renderer.allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        (info.width * info.height * 4) as DeviceSize,
    ).unwrap();

    reader.next_frame(&mut upload_buffer.write().unwrap()).unwrap();

    let image = Image::new(
        renderer.allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: Format::R8G8B8A8_SRGB,
            extent,
            usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
            ..Default::default()
        },
        AllocationCreateInfo::default(),
    ).unwrap();

    uploads
        .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
            upload_buffer,
            image.clone(),
        )).unwrap();

    ImageView::new_default(image).unwrap()
}

pub fn create_vertex_layout(attribs: &Vec<Attribute>) -> VertexInputState {
    let mut vertex_attributes = Vec::new();
    let mut vertex_bindings = Vec::new();
    let total_size = calculate_element_size(attribs);
    let mut offset = 0;
    for attr_idx in 0..attribs.len() {
        let attribute = attribs.get(attr_idx).unwrap();
        let vk_type = match attribute.size {
            AttributeSize::None(_, _) => panic!("None attribute exists"),
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
        let size = attribute.get_size();

        vertex_attributes.push((attr_idx as u32, VertexInputAttributeDescription {
            binding: 0,
            format: vk_type,
            offset,
        }));
        vertex_bindings.push((0, VertexInputBindingDescription {
            stride: total_size,
            input_rate: VertexInputRate::Vertex,
        }));
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