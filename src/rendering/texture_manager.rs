use crate::rendering::renderer::Renderer;
use png::Reader;
use std::cell::RefCell;
use std::io::Cursor;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CopyBufferToImageInfo, PrimaryAutoCommandBuffer,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::Device;
use vulkano::format::Format;
use vulkano::image::sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo};
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::pipeline::PipelineLayout;
use vulkano::DeviceSize;

pub trait TextureUploader {
    fn get_width(&self) -> u32;
    fn get_height(&self) -> u32;
    fn upload(&mut self, buffer: &Subbuffer<[u8]>);
}

pub struct PngTextureUploader {
    reader: Rc<RefCell<Reader<Cursor<Vec<u8>>>>>,
    width: u32,
    height: u32,
}

impl PngTextureUploader {
    pub fn new(png_bytes: Vec<u8>) -> PngTextureUploader {
        let cursor = Cursor::new(png_bytes);
        let decoder = png::Decoder::new(cursor);
        let reader = Rc::new(RefCell::new(decoder.read_info().unwrap()));
        let info_reader = reader.clone();
        let borrow = info_reader.borrow();
        let info = borrow.info();

        PngTextureUploader {
            reader,
            width: info.width,
            height: info.height,
        }
    }
}

impl TextureUploader for PngTextureUploader {
    fn get_width(&self) -> u32 {
        self.width
    }

    fn get_height(&self) -> u32 {
        self.height
    }

    fn upload(&mut self, buffer: &Subbuffer<[u8]>) {
        self.reader
            .borrow_mut()
            .next_frame(&mut buffer.write().unwrap())
            .unwrap();
    }
}

pub struct TextureManager {
    sampler: Arc<Sampler>,
    layout: Arc<PipelineLayout>,
    allocator: Arc<StandardDescriptorSetAllocator>,
    next_texture_id: u32,
    textures_to_upload: Vec<Rc<Mutex<Box<dyn TextureUploader>>>>,
    textures: Vec<Arc<ImageView>>,
}

impl TextureManager {
    pub fn new(
        layout: Arc<PipelineLayout>,
        allocator: Arc<StandardDescriptorSetAllocator>,
        device: Arc<Device>,
    ) -> TextureManager {
        let sampler = Sampler::new(
            device,
            SamplerCreateInfo {
                mag_filter: Filter::Nearest,
                min_filter: Filter::Nearest,
                address_mode: [SamplerAddressMode::Repeat; 3],
                ..Default::default()
            },
        )
        .unwrap();

        TextureManager {
            sampler,
            layout,
            allocator,
            next_texture_id: 0,
            textures_to_upload: Vec::new(),
            textures: Vec::new(),
        }
    }

    pub fn generate_descriptor_set(&self) -> Arc<PersistentDescriptorSet> {
        let textures: Vec<(Arc<ImageView>, Arc<Sampler>)> = self
            .textures
            .iter()
            .map(|item| (Arc::clone(item), self.sampler.clone()))
            .collect();

        let d_layout = self.layout.set_layouts().get(0).unwrap();
        PersistentDescriptorSet::new_variable(
            &self.allocator,
            d_layout.clone(),
            self.textures.len() as u32,
            [WriteDescriptorSet::image_view_sampler_array(0, 0, textures)],
            [],
        )
        .unwrap()
    }

    pub fn queue(&mut self, texture_uploader: Box<dyn TextureUploader>) -> u32 {
        self.textures_to_upload
            .push(Rc::new(Mutex::new(texture_uploader)));
        self.next_texture_id += 1;
        return self.next_texture_id - 1;
    }

    pub fn upload_all(
        &mut self,
        renderer: &Renderer,
        uploads: &mut AutoCommandBufferBuilder<
            PrimaryAutoCommandBuffer<Arc<StandardCommandBufferAllocator>>,
            Arc<StandardCommandBufferAllocator>,
        >,
    ) {
        for x in &self.textures_to_upload {
            let mut guard = x.lock().unwrap();

            let extent = [guard.get_width(), guard.get_height(), 1];

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
                (guard.get_width() * guard.get_height() * 4) as DeviceSize,
            )
            .unwrap();

            guard.upload(&upload_buffer);

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
            )
            .unwrap();

            uploads
                .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
                    upload_buffer,
                    image.clone(),
                ))
                .unwrap();

            self.textures.push(ImageView::new_default(image).unwrap());
        }
    }
}
