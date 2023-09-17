#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec4 tangent;
layout(location = 3) in vec2 uv;
layout(location = 4) in uint[] blendIndices;
layout(location = 5) in vec4 blendWeights;

layout(location = 0) out vec3 outColor;

layout(push_constant) uniform PushConstantData {
    mat4 projMat;
    mat4 viewMat;
} constants;

void main() {
    const vec3 colors[3] = vec3[3](
        vec3(1.0f, 0.0f, 0.0f), //red
        vec3(0.0f, 1.0f, 0.0f), //green
        vec3(00.f, 0.0f, 1.0f)  //blue
    );

    gl_Position = constants.projMat * constants.viewMat * vec4(position, 1.0);
    outColor = colors[gl_VertexIndex % 3];
}