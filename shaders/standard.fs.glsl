#version 450
#extension GL_EXT_nonuniform_qualifier : enable

layout(location = 0) in vec2 coords;
layout(location = 1) flat in int textureId;
layout(location = 0) out vec4 fragColor;

layout(set = 0, binding = 0) uniform sampler s;
layout(set = 0, binding = 1) uniform texture2D textures[];

void main() {
    vec4 diffuseColor = texture(sampler2D(nonuniformEXT(textures[textureId]), s), vec2(coords.x, coords.y));

    fragColor = diffuseColor;
}