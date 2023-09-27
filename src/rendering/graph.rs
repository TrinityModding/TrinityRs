use std::collections::HashMap;
use std::sync::Arc;
use vulkano::buffer::Subbuffer;
use vulkano::pipeline::GraphicsPipeline;

const VERTEX_BUFFER_INITIAL_SIZE: u64 = 16_000000; // 16mb
const INDEX_BUFFER_INITIAL_SIZE: u64 = 4_000000; // 4mb
const STORAGE_BUFFER_INITIAL_SIZE: u64 = 32_000000; // 32mb

pub struct RenderGraph {
    buffers: HashMap<ShaderType, BufferStorage>,
}

pub struct ShaderType {
    pipeline: GraphicsPipeline
}

pub struct BufferStorage {
    pos_vertex_buffer: Subbuffer<[u8]>, // Just the position attribute
    color_vertex_buffer: Subbuffer<[u8]>, // Attributes used for colour
    index_buffer: Subbuffer<[u8]>,
    instance_buffer: Subbuffer<[u8]>,
}

pub struct ModelLocation {
    pub shader: Arc<ShaderType>,
    pub idx_offsets: u64,
    pub idx_counts: u64
}
