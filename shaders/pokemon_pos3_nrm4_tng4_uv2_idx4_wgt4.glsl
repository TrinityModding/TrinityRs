#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec4 normal;
layout(location = 2) in vec4 tangent;
layout(location = 3) in vec2 uv;
layout(location = 4) in uint[] blendIndices;
layout(location = 5) in vec4 blendWeights;

layout(location = 0) out vec2 outUv;
layout(location = 1) out int textureId;

layout(push_constant) uniform PushConstantData {
    mat4 projMat;
    mat4 viewMat;
    mat4 modelTransform;
    int textureId;
} constants;

void main() {
    mat4 worldSpace = constants.projMat * constants.viewMat;
    gl_Position = worldSpace * constants.modelTransform * vec4(position, 1.0);
    outUv = uv;
    textureId = constants.textureId;
}