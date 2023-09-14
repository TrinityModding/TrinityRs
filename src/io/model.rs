use std::fs;
use std::path::{PathBuf};

use crate::io::flatbuffers::trmdl_generated::titan::model::root_as_trmdl;
use crate::io::flatbuffers::trmtr_generated::titan::model::root_as_trmtr;

pub struct RenderModel {}

impl RenderModel {
    pub fn from_trmdl(file_path: String) -> RenderModel {
        let mut path = PathBuf::new();
        path.push(file_path);
        let file_bytes = fs::read(path.to_str().unwrap()).unwrap();
        let trmdl = root_as_trmdl(file_bytes.as_slice()).unwrap();
        path.pop();

        path.push(trmdl.materials().unwrap().get(0));
        let trmtr_bytes = fs::read(path.to_str().unwrap()).unwrap();
        let material = root_as_trmtr(trmtr_bytes.as_slice());
        path.pop();

        RenderModel {}
    }
}