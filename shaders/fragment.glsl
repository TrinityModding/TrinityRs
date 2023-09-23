#version 450
#extension GL_EXT_nonuniform_qualifier : enable

layout(location = 0) in vec2 coords;
layout(location = 0) out vec4 fragColor;

layout(set = 0, binding = 0) uniform sampler2D textures[];

void main() {
    fragColor = texture(nonuniformEXT(textures[0]), vec2(coords.x, 1.0 - coords.y));
}