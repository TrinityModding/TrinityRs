use crate::io::flatbuffers::trmbf_generated::titan::model::root_as_trmbf;
use crate::io::flatbuffers::trmdl_generated::titan::model::{root_as_trmdl, trmeshes};
use crate::io::flatbuffers::trmsh_generated::titan::model::{
    root_as_trmsh, MaterialInfo, PolygonType, Type, VertexAttribute,
};
use crate::io::flatbuffers::trmtr_generated::titan::model::root_as_trmtr_unchecked;
use crate::rendering::graph::{MeshLocation, SceneGraph};
use crate::rendering::renderer::Renderer;
use bytemuck::cast_slice;
use byteorder::{LittleEndian, ReadBytesExt};
use std::collections::HashMap;
use std::fs;
use std::io::{Cursor, Read};
use std::mem::size_of;
use std::ops::Div;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use ultraviolet::{Vec2, Vec3, Vec4};
use vulkano::half::f16;

pub const RGBA8_UNORM: AttributeFormat = AttributeFormat::Rgba8UNorm(20, size_of::<u8>() * 4);
pub const RGBA8_UNSIGNED: AttributeFormat = AttributeFormat::Rgba8Unsigned(22, size_of::<u8>() * 4);
pub const R32_UINT: AttributeFormat = AttributeFormat::R32UInt(36, size_of::<u32>());
pub const R32_INT: AttributeFormat = AttributeFormat::R32Int(37, size_of::<i32>());
pub const RGBA16_UNORM: AttributeFormat = AttributeFormat::Rgba16UNorm(39, size_of::<u16>() * 4);
pub const RGBA16_FLOAT: AttributeFormat = AttributeFormat::Rgba16Float(43, size_of::<f16>() * 4);
pub const RG32_FLOAT: AttributeFormat = AttributeFormat::Rg32Float(48, size_of::<f32>() * 2);
pub const RGB32_FLOAT: AttributeFormat = AttributeFormat::Rgb32Float(51, size_of::<f32>() * 3);
pub const RGBA32_FLOAT: AttributeFormat = AttributeFormat::Rgba32Float(54, size_of::<f32>() * 4);

pub struct LodMesh {
    meshes: Vec<Vec<Arc<MeshLocation>>>,
}

impl LodMesh {
    pub fn get_best(&self) -> Vec<Arc<MeshLocation>> {
        self.meshes.get(0).unwrap().clone()
    }
}

pub fn from_trmdl(
    file_path: String,
    render_graph: &mut SceneGraph,
    renderer: &mut Renderer,
) -> LodMesh {
    let mut path = PathBuf::from(file_path);
    let file_bytes = fs::read(path.to_str().unwrap()).unwrap();
    let trmdl = root_as_trmdl(file_bytes.as_slice()).unwrap();
    path.pop();

    let trmtr_path = trmdl.materials().unwrap().get(0);
    let trmtr_bytes = fs::read(path.join(trmtr_path).to_str().unwrap()).unwrap();
    let material = unsafe { root_as_trmtr_unchecked(trmtr_bytes.as_slice()) };

    let mut material_to_shader_map = HashMap::new();
    for material in &material.materials().unwrap() {
        material_to_shader_map
            .entry(material.name().unwrap())
            .or_insert(material.shaders().unwrap().get(0).shader_name().unwrap());
    }

    let lod_info = trmdl.lods().unwrap().get(0);
    println!("Lod Set Type: {}", lod_info.lod_type().unwrap());
    LodMesh {
        meshes: lod_info
            .index()
            .unwrap()
            .iter()
            .map(|x| trmdl.meshes().unwrap().get(x.unk0() as usize))
            .map(|x1| read_mesh(x1, &path, &material_to_shader_map, render_graph, renderer))
            .collect(),
    }
}

pub struct MeshBufferInfo {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub tangents: Vec<Vec3>,
    pub bi_normals: Vec<Vec3>,
    pub colors: Vec<Vec<u8>>,
    pub uvs: Vec<Vec2>,
    pub bone_ids: Vec<Vec<u8>>,
    pub bone_weights: Vec<Vec4>,
    pub indices: Vec<u32>,
}

fn read_mesh(
    reference: trmeshes,
    path: &Path,
    material_to_shader_map: &HashMap<&str, &str>,
    render_graph: &mut SceneGraph,
    renderer: &mut Renderer,
) -> Vec<Arc<MeshLocation>> {
    let trmsh_path = reference.filename().unwrap();
    let trmsh_bytes = fs::read(path.join(trmsh_path).to_str().unwrap()).unwrap();
    let trmsh = root_as_trmsh(trmsh_bytes.as_slice()).unwrap();

    let trmbf_path = String::from(trmsh_path).replace(".trmsh", ".trmbf");
    let trmbf_bytes = fs::read(path.join(trmbf_path).to_str().unwrap()).unwrap();
    let trmbf = root_as_trmbf(trmbf_bytes.as_slice()).unwrap();

    let mut shader_to_mesh_map = HashMap::new();
    for mesh_idx in 0..trmsh.meshes().unwrap().len() {
        let info = trmsh.meshes().unwrap().get(mesh_idx);
        let data = trmbf.buffers().unwrap().get(mesh_idx);

        // Attributes
        let raw_attributes = info.attributes().unwrap().get(0);
        let mut attributes = Vec::new();
        for attr in raw_attributes.attrs().unwrap() {
            attributes.push(Attribute {
                type_: AttributeType::get(attr.attribute()).unwrap(),
                format: AttributeFormat::get(attr.type_()).unwrap(),
            });
        }

        // Vertices
        let vertex_buffer = data
            .vertex_buffer()
            .unwrap()
            .get(0)
            .buffer()
            .unwrap()
            .bytes();
        let vertex_count = vertex_buffer.len()
            / (attributes.iter().map(|attr| attr.get_size()).sum::<u32>() as usize);
        let mut vertex_reader = Cursor::new(vertex_buffer);
        let mut mesh_buffer_info = MeshBufferInfo {
            positions: Vec::new(),
            normals: Vec::new(),
            tangents: Vec::new(),
            bi_normals: Vec::new(),
            colors: Vec::new(),
            uvs: Vec::new(),
            bone_ids: Vec::new(),
            bone_weights: Vec::new(),
            indices: Vec::new(),
        };

        for _i in 0..vertex_count {
            for attribute in &attributes {
                match attribute.type_ {
                    AttributeType::Position => mesh_buffer_info
                        .positions
                        .push(read_vec3(&attribute.format, &mut vertex_reader)),
                    AttributeType::Normal => mesh_buffer_info
                        .normals
                        .push(read_vec3(&attribute.format, &mut vertex_reader)),
                    AttributeType::Tangent => mesh_buffer_info
                        .tangents
                        .push(read_vec3(&attribute.format, &mut vertex_reader)),
                    AttributeType::BiNormal => mesh_buffer_info
                        .bi_normals
                        .push(read_vec3(&attribute.format, &mut vertex_reader)),
                    AttributeType::Color => mesh_buffer_info
                        .colors
                        .push(read_bytes4(&attribute.format, &mut vertex_reader)),
                    AttributeType::TextureCoords => mesh_buffer_info
                        .uvs
                        .push(read_vec2(&attribute.format, &mut vertex_reader)),
                    AttributeType::BlendIndices => mesh_buffer_info
                        .bone_ids
                        .push(read_bytes4(&attribute.format, &mut vertex_reader)),
                    AttributeType::BlendWeights => {
                        let w = (vertex_reader.read_u16::<LittleEndian>().unwrap() as f32) / 65535.0;
                        let x = (vertex_reader.read_u16::<LittleEndian>().unwrap() as f32) / 65535.0;
                        let y = (vertex_reader.read_u16::<LittleEndian>().unwrap() as f32) / 65535.0;
                        let z = (vertex_reader.read_u16::<LittleEndian>().unwrap() as f32) / 65535.0;
                        let div = x + y + z + w;
                        mesh_buffer_info
                            .bone_weights
                            .push(Vec4::new(x, y, z, w).div(Vec4::new(div, div, div, div)));
                    }
                    _ => panic!("Unhandled attribute type {:?}", attribute.type_),
                }
            }
        }

        // Indices
        let idx_buffer = data.index_buffer().unwrap().get(0).buffer().unwrap();
        let idx_layout = IndexLayout::get(info.polygon_type()).unwrap();
        match idx_layout {
            IndexLayout::UInt8(_) => {
                for i in idx_buffer {
                    mesh_buffer_info.indices.push(i as u32);
                }
            }
            IndexLayout::UInt16(_) => {
                let u64_indices: &[u16] = cast_slice(idx_buffer.bytes());
                for i in u64_indices {
                    mesh_buffer_info.indices.push(*i as u32);
                }
            }
            IndexLayout::UInt32(_) => {
                let u64_indices: &[u32] = cast_slice(idx_buffer.bytes());
                for i in u64_indices {
                    mesh_buffer_info.indices.push(*i);
                }
            }
            IndexLayout::UInt64(_) => {
                let u64_indices: &[u64] = cast_slice(idx_buffer.bytes());
                for i in u64_indices {
                    mesh_buffer_info.indices.push(*i as u32);
                }
            }
        }

        let info_ref = Arc::new(mesh_buffer_info);

        // Reorganise data to make it more readable for us
        for sub_mesh in info.materials().unwrap() {
            let material_name = sub_mesh.material_name().unwrap();
            let shader = material_to_shader_map.get(material_name).unwrap();

            shader_to_mesh_map
                .entry(String::from(*shader))
                .or_insert(Vec::new())
                .push((sub_mesh, info_ref.clone()));
        }
    }

    write_mesh_to_renderer(shader_to_mesh_map, render_graph, renderer)
}

fn write_mesh_to_renderer(
    shader_to_mesh_map: HashMap<String, Vec<(MaterialInfo, Arc<MeshBufferInfo>)>>,
    render_graph: &mut SceneGraph,
    renderer: &mut Renderer,
) -> Vec<Arc<MeshLocation>> {
    let mut written_models = Vec::new();

    for entry in shader_to_mesh_map {
        let shader_name = entry.0;

        for sub_mesh in entry.1 {
            let idx_offset = sub_mesh.0.poly_offset() as usize;
            let idx_count = sub_mesh.0.poly_count() as usize;

            let mesh_indices = &sub_mesh.1.indices.as_slice()[idx_offset..(idx_offset + idx_count)];
            let smallest_idx = mesh_indices.iter().cloned().min().unwrap() as usize;
            let biggest_idx = mesh_indices.iter().cloned().max().unwrap() as usize;

            let sub_mesh_buffer_info = MeshBufferInfo {
                positions: if sub_mesh.1.positions.is_empty() {
                    Vec::new()
                } else {
                    sub_mesh.1.positions.as_slice()[smallest_idx..biggest_idx].to_vec()
                },
                normals: if sub_mesh.1.normals.is_empty() {
                    Vec::new()
                } else {
                    sub_mesh.1.normals.as_slice()[smallest_idx..biggest_idx].to_vec()
                },
                tangents: if sub_mesh.1.tangents.is_empty() {
                    Vec::new()
                } else {
                    sub_mesh.1.tangents.as_slice()[smallest_idx..biggest_idx].to_vec()
                },
                bi_normals: if sub_mesh.1.bi_normals.is_empty() {
                    Vec::new()
                } else {
                    sub_mesh.1.bi_normals.as_slice()[smallest_idx..biggest_idx].to_vec()
                },
                colors: if sub_mesh.1.colors.is_empty() {
                    Vec::new()
                } else {
                    sub_mesh.1.colors.as_slice()[smallest_idx..biggest_idx].to_vec()
                },
                uvs: if sub_mesh.1.uvs.is_empty() {
                    Vec::new()
                } else {
                    sub_mesh.1.uvs.as_slice()[smallest_idx..biggest_idx].to_vec()
                },
                bone_ids: if sub_mesh.1.bone_ids.is_empty() {
                    Vec::new()
                } else {
                    sub_mesh.1.bone_ids.as_slice()[smallest_idx..biggest_idx].to_vec()
                },
                bone_weights: if sub_mesh.1.bone_weights.is_empty() {
                    Vec::new()
                } else {
                    sub_mesh.1.bone_weights.as_slice()[smallest_idx..biggest_idx].to_vec()
                },
                indices: if sub_mesh.1.indices.is_empty() {
                    Vec::new()
                } else {
                    mesh_indices
                        .iter()
                        .cloned()
                        .map(|x| x - (smallest_idx as u32))
                        .collect()
                },
            };

            written_models.push(render_graph.upload(&shader_name, sub_mesh_buffer_info, renderer));
        }
    }

    written_models
}

fn read_bytes4(format: &AttributeFormat, reader: &mut Cursor<&[u8]>) -> Vec<u8> {
    match format {
        AttributeFormat::Rgba8UNorm(_, _) => vec![
            reader.read_u8().unwrap(),
            reader.read_u8().unwrap(),
            reader.read_u8().unwrap(),
            reader.read_u8().unwrap(),
        ],
        AttributeFormat::Rgba8Unsigned(_, _) => {
            let w = reader.read_u8().unwrap();
            let x = reader.read_u8().unwrap();
            let y = reader.read_u8().unwrap();
            let z = reader.read_u8().unwrap();
            vec![w, x, y, z]
        },
        _ => panic!("Unhandled vertex attribute format {:?}", format),
    }
}

fn read_vec3(format: &AttributeFormat, reader: &mut Cursor<&[u8]>) -> Vec3 {
    match format {
        AttributeFormat::Rgb32Float(_, _) => Vec3::new(
            reader.read_f32::<LittleEndian>().unwrap(),
            reader.read_f32::<LittleEndian>().unwrap(),
            reader.read_f32::<LittleEndian>().unwrap(),
        ),
        AttributeFormat::Rgba16Float(_, _) => {
            let x = f32::from(read_f16(reader)); // Ignored. Maybe padding?
            let y = f32::from(read_f16(reader));
            let z = f32::from(read_f16(reader));
            let _w = f32::from(read_f16(reader));

            Vec3::new(x, y, z)
        },
        _ => panic!("Unhandled vertex attribute format {:?}", format),
    }
}

fn read_vec2(format: &AttributeFormat, reader: &mut Cursor<&[u8]>) -> Vec2 {
    match format {
        AttributeFormat::Rg32Float(_, _) => Vec2::new(
            reader.read_f32::<LittleEndian>().unwrap(),
            1.0 - reader.read_f32::<LittleEndian>().unwrap(),
        ),
        _ => panic!("Unhandled vertex attribute format {:?}", format),
    }
}

fn read_f16(cursor: &mut Cursor<&[u8]>) -> f16 {
    let mut buf = [0; 2];
    cursor.read_exact(&mut buf).unwrap();
    f16::from_le_bytes(buf)
}

#[derive(Clone, Debug)]
pub enum IndexLayout {
    UInt8(usize),
    UInt16(usize),
    UInt32(usize),
    UInt64(usize),
}

impl IndexLayout {
    fn get(i: PolygonType) -> Option<IndexLayout> {
        match i.0 {
            0 => Some(IndexLayout::UInt8(size_of::<u8>())),
            1 => Some(IndexLayout::UInt16(size_of::<u16>())),
            2 => Some(IndexLayout::UInt32(size_of::<u32>())),
            3 => Some(IndexLayout::UInt64(size_of::<u64>())),
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
pub enum AttributeFormat {
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

impl AttributeFormat {
    fn get(id: Type) -> Option<AttributeFormat> {
        match id.0 {
            0 => Some(AttributeFormat::None(0, 0)),
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
    pub format: AttributeFormat,
}

impl Attribute {
    pub fn get_size(&self) -> u32 {
        match self.format {
            AttributeFormat::None(_, s) => s as u32,
            AttributeFormat::Rgba8UNorm(_, s) => s as u32,
            AttributeFormat::Rgba8Unsigned(_, s) => s as u32,
            AttributeFormat::R32UInt(_, s) => s as u32,
            AttributeFormat::R32Int(_, s) => s as u32,
            AttributeFormat::Rgba16UNorm(_, s) => s as u32,
            AttributeFormat::Rgba16Float(_, s) => s as u32,
            AttributeFormat::Rg32Float(_, s) => s as u32,
            AttributeFormat::Rgb32Float(_, s) => s as u32,
            AttributeFormat::Rgba32Float(_, s) => s as u32,
        }
    }
}
