#version 450
#extension GL_ARB_gpu_shader_int64 : enable

layout(location = 0) in vec3 position;
layout(location = 1) in vec2 uv;

layout(location = 0) out vec2 outUv;
layout(location = 1) out int outTexture;

layout(push_constant) uniform PushConstantData {
    uint64_t instanceAddress;
} constants;

void main() {
//    mat4 worldSpace = constants.projMat * constants.viewMat;
//    gl_Position = worldSpace * constants.modelTransform * vec4(position, 1.0);

    gl_Position = vec4(position, 1.0);
    outUv = uv;
    outTexture = 0;
}