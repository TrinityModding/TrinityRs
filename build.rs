use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=shaders/standard.vs.glsl");
    println!("cargo:rerun-if-changed=shaders/standard.fs.glsl");

    compile_vertex_shader("standard");
    compile_fragment_shader("standard");
}

fn compile_vertex_shader(shader_name: &str) {
    Command::new("glslc")
        .arg("--target-env=vulkan1.3")
        .arg("-fshader-stage=vertex")
        .arg("-O")
        .arg("-o")
        .arg(format!("shaders/{}.vs.spv", shader_name))
        .arg(format!("shaders/{}.vs.glsl", shader_name))
        .output()
        .expect("failed to compile vertex shader");
}

fn compile_fragment_shader(shader_name: &str) {
    Command::new("glslc")
        .arg("--target-env=vulkan1.3")
        .arg("-fshader-stage=fragment")
        .arg("-O")
        .arg("-o")
        .arg(format!("shaders/{}.fs.spv", shader_name))
        .arg(format!("shaders/{}.fs.glsl", shader_name))
        .output()
        .expect("failed to compile vertex shader");
}
