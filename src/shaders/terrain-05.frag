#version 330 core

in float diffuse;
smooth in vec2 tcoord;

uniform sampler2D texImage;

out vec4 color;

void main()
{
  vec3 tcolor = diffuse*(texture(texImage, tcoord).rgb);
  color = vec4(tcolor, 1.0);
}
