use std::collections::HashMap;
use std::fs;
use std::mem::size_of;
use std::path::PathBuf;
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::{DrawIndexedIndirectCommand};
use vulkano::half::f16;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use crate::io::flatbuffers::trmbf_generated::titan::model::root_as_trmbf;
use crate::io::flatbuffers::trmdl_generated::titan::model::root_as_trmdl;
use crate::io::flatbuffers::trmsh_generated::titan::model::{PolygonType, root_as_trmsh, Type, VertexAttribute};
use crate::io::flatbuffers::trmtr_generated::titan::model::root_as_trmtr_unchecked;
use crate::rendering::texture_manager::{PngTextureUploader, TextureManager};

pub const RGBA8_UNORM: AttributeSize = AttributeSize::Rgba8UNorm(20, size_of::<u8>() * 4);
pub const RGBA8_UNSIGNED: AttributeSize = AttributeSize::Rgba8Unsigned(22, size_of::<u8>() * 4);
pub const R32_UINT: AttributeSize = AttributeSize::R32UInt(36, size_of::<u32>());
pub const R32_INT: AttributeSize = AttributeSize::R32Int(37, size_of::<i32>());
pub const RGBA16_UNORM: AttributeSize = AttributeSize::Rgba16UNorm(39, size_of::<u16>() * 4);
pub const RGBA16_FLOAT: AttributeSize = AttributeSize::Rgba16Float(43, size_of::<f16>() * 4);
pub const RG32_FLOAT: AttributeSize = AttributeSize::Rg32Float(48, size_of::<f32>() * 2);
pub const RGB32_FLOAT: AttributeSize = AttributeSize::Rgb32Float(51, size_of::<f32>() * 3);
pub const RGBA32_FLOAT: AttributeSize = AttributeSize::Rgba32Float(54, size_of::<f32>() * 4);

/// Holds the sub-models inside of a model
#[derive(Clone, Debug)]
pub struct MeshGroup {
    pub sub_meshes: Vec<SubMesh>,
    pub vertex_buffer: Subbuffer<[u8]>,
    pub index_buffer: Subbuffer<[u8]>,
    pub idx_layout: IndexLayout,
    pub attributes: Vec<Attribute>,
}

#[derive(Clone, Debug)]
pub struct SubMesh {
    pub draw_calls: Vec<MaterialDrawCall>,
}

#[derive(Clone, Debug)]
pub struct MaterialDrawCall {
    pub cmd: DrawIndexedIndirectCommand,
    pub texture_idx: u32,
}

impl SubMesh {
    pub fn from_trmdl(file_path: String, allocator: Arc<StandardMemoryAllocator>, texture_manager: &mut TextureManager) -> HashMap<String, Vec<Arc<MeshGroup>>> {
        let mut path = PathBuf::new();
        path.push(file_path);
        let file_bytes = fs::read(path.to_str().unwrap()).unwrap();
        let trmdl = root_as_trmdl(file_bytes.as_slice()).unwrap();
        path.pop();

        let trmtr_path = trmdl.materials().unwrap().get(0);
        let trmtr_bytes = fs::read(path.join(trmtr_path).to_str().unwrap()).unwrap();
        let material = unsafe { root_as_trmtr_unchecked(trmtr_bytes.as_slice()) };

        // Map from material id to renderer's texture ID
        let mut render_texture_id_map = HashMap::new();
        for i in 0..material.materials().unwrap().len() {
            let material = material.materials().unwrap().get(i);

            for tex in material.textures().unwrap() {
                if tex.texture_name().unwrap().eq("BaseColorMap") {
                    println!("{}", path.join(String::from(tex.texture_file().unwrap()).replace(".bntx", ".png")).to_str().unwrap());
                    let tex_bytes = fs::read(path.join(String::from(tex.texture_file().unwrap()).replace(".bntx", ".png")).to_str().unwrap()).unwrap();

                    let renderer_id = texture_manager.queue(Box::new(PngTextureUploader::new(tex_bytes)));
                    render_texture_id_map.entry(String::from(material.name().unwrap())).or_insert(renderer_id);
                    break;
                }
            }
        }

        let mut result_map: HashMap<String, Vec<Arc<MeshGroup>>> = HashMap::new();
        trmdl.meshes().unwrap().iter().for_each(|x| {
            let trmsh_path = x.filename().unwrap();
            let trmbf_path = String::from(trmsh_path).replace(".trmsh", ".trmbf");

            let trmsh_bytes = fs::read(path.join(trmsh_path).to_str().unwrap()).unwrap();
            let trmbf_bytes = fs::read(path.join(trmbf_path).to_str().unwrap()).unwrap();

            let trmsh = root_as_trmsh(trmsh_bytes.as_slice()).unwrap();
            let trmbf = root_as_trmbf(trmbf_bytes.as_slice()).unwrap();
            let mut models = Vec::new();

            for mesh_idx in 0..trmsh.meshes().unwrap().len() {
                let info = trmsh.meshes().unwrap().get(mesh_idx);
                let data = trmbf.buffers().unwrap().get(mesh_idx);
                let vertex_buffer = data.vertex_buffer().unwrap().get(0);
                let idx_buffer = data.index_buffer().unwrap().get(0);
                let idx_layout = IndexLayout::get(info.polygon_type()).unwrap();
                let raw_attributes = info.attributes().unwrap().get(0);
                let mut meshes = Vec::new();

                let mut attributes = Vec::new();
                for attr in raw_attributes.attrs().unwrap() {
                    attributes.push(Attribute {
                        type_: AttributeType::get(attr.attribute()).unwrap(),
                        size: AttributeSize::get(attr.type_()).unwrap(),
                    });
                }

                // draw calls for different materials
                let mut draw_calls = Vec::new();
                for material in info.materials().unwrap() {
                    draw_calls.push(MaterialDrawCall {
                        cmd: DrawIndexedIndirectCommand {
                            index_count: material.poly_count(),
                            instance_count: 1, // TODO: get info somehow on how many instances there are to control this instead of letting this control it
                            first_index: material.poly_offset(),
                            vertex_offset: 0,
                            first_instance: 0,
                        },
                        texture_idx: *render_texture_id_map.get(material.material_name().unwrap()).unwrap(),
                    });
                }

                meshes.push(SubMesh {
                    draw_calls
                });

                let vertex_data = vertex_buffer.buffer().unwrap().bytes();
                let vertex_buffer = Buffer::new_slice(
                    allocator.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::VERTEX_BUFFER,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                    vertex_data.len() as u64,
                ).unwrap();
                vertex_buffer.write().unwrap().copy_from_slice(vertex_data);

                let index_data = idx_buffer.buffer().unwrap().bytes();
                let index_buffer = Buffer::new_slice(
                    allocator.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::INDEX_BUFFER,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                    index_data.len() as u64,
                ).unwrap();
                index_buffer.write().unwrap().copy_from_slice(index_data);

                models.push(Arc::new(MeshGroup {
                    sub_meshes: meshes,
                    vertex_buffer,
                    index_buffer,
                    idx_layout,
                    attributes,
                }));
            }

            result_map
                .entry(String::from(x.filename().unwrap()))
                .or_insert(models);
        });

        result_map
    }
}

#[derive(Clone, Debug)]
pub enum IndexLayout {
    UInt8(u32),
    UInt16(u32),
    UInt32(u32),
    UInt64(u32),
}

impl IndexLayout {
    fn get(i: PolygonType) -> Option<IndexLayout> {
        match i.0 {
            0 => Some(IndexLayout::UInt8(size_of::<u8>() as u32)),
            1 => Some(IndexLayout::UInt16(size_of::<u16>() as u32)),
            2 => Some(IndexLayout::UInt32(size_of::<u32>() as u32)),
            3 => Some(IndexLayout::UInt64(size_of::<u64>() as u32)),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum AttributeType {
    None,
    Position,
    Normal,
    Tangent,
    BiNormal,
    Color,
    TextureCoords,
    BlendIndices,
    BlendWeights,
}

impl AttributeType {
    fn get(id: VertexAttribute) -> Option<AttributeType> {
        match id.0 {
            0 => Some(AttributeType::None),
            1 => Some(AttributeType::Position),
            2 => Some(AttributeType::Normal),
            3 => Some(AttributeType::Tangent),
            4 => Some(AttributeType::BiNormal),
            5 => Some(AttributeType::Color),
            6 => Some(AttributeType::TextureCoords),
            7 => Some(AttributeType::BlendIndices),
            8 => Some(AttributeType::BlendWeights),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum AttributeSize {
    None(i64, usize),
    Rgba8UNorm(i64, usize),
    Rgba8Unsigned(i64, usize),
    R32UInt(i64, usize),
    R32Int(i64, usize),
    Rgba16UNorm(i64, usize),
    Rgba16Float(i64, usize),
    Rg32Float(i64, usize),
    Rgb32Float(i64, usize),
    Rgba32Float(i64, usize),
}

impl AttributeSize {
    fn get(id: Type) -> Option<AttributeSize> {
        match id.0 {
            0 => Some(AttributeSize::None(0, 0)),
            20 => Some(RGBA8_UNORM),
            22 => Some(RGBA8_UNSIGNED),
            36 => Some(R32_UINT),
            37 => Some(R32_INT),
            39 => Some(RGBA16_UNORM),
            43 => Some(RGBA16_FLOAT),
            48 => Some(RG32_FLOAT),
            51 => Some(RGB32_FLOAT),
            54 => Some(RGBA32_FLOAT),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Attribute {
    pub type_: AttributeType,
    pub size: AttributeSize,
}

impl Attribute {
    pub fn new(type_: AttributeType, size: AttributeSize) -> Attribute {
        Attribute {
            type_,
            size,
        }
    }

    pub fn get_size(&self) -> u32 {
        match self.size {
            AttributeSize::None(_, s) => s as u32,
            AttributeSize::Rgba8UNorm(_, s) => s as u32,
            AttributeSize::Rgba8Unsigned(_, s) => s as u32,
            AttributeSize::R32UInt(_, s) => s as u32,
            AttributeSize::R32Int(_, s) => s as u32,
            AttributeSize::Rgba16UNorm(_, s) => s as u32,
            AttributeSize::Rgba16Float(_, s) => s as u32,
            AttributeSize::Rg32Float(_, s) => s as u32,
            AttributeSize::Rgb32Float(_, s) => s as u32,
            AttributeSize::Rgba32Float(_, s) => s as u32,
        }
    }
}
