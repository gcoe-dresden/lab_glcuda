#version 330 core

layout(location=0) in vec4 in_position;
layout(location=1) in vec4 in_normal;
layout(location=2) in vec2 in_texcoord;

uniform mat4 mvp;

out float diffuse;
smooth out vec2 tcoord;

void main()
{
  diffuse = 0.1+clamp(dot(in_normal.xyz, vec3(0.15,0.15,0.5)),0.0,1.0);
  tcoord = in_texcoord;
  gl_Position = mvp * in_position;
}
