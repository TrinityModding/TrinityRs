#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require

layout(location = 0) in vec3 position;
layout(location = 1) in vec2 uv;

layout(location = 0) out vec2 outUv;
layout(location = 1) out int outTexture;

layout(buffer_reference, std430, buffer_reference_align = 16) readonly buffer Instance {
    mat4 transform;
};

layout(push_constant) uniform PushConstantData {
    mat4 projMat;
    mat4 viewMat;
    Instance instanceAddress;
} constants;

void main() {
    mat4 worldSpace = constants.projMat * constants.viewMat;
    Instance instance = constants.instanceAddress;
    gl_Position = worldSpace * vec4(position, 1.0);
    gl_Position = vec4(position, 1.0);
    outUv = uv;
    outTexture = 0;
}