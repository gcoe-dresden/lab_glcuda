#version 330 core

layout(location=0) in vec4 in_position;
layout(location=1) in vec4 in_normal;

uniform mat4 mvp;

out vec3 fcolor;

void main()
{
  float diffuse = 0.1+clamp(dot(in_normal.xyz, vec3(0.15,0.15,0.5)),0.0,1.0);
  //  float diffuse = 0.0+dot(in_normal.xyz, vec3(0.0,0.0,1.0));
  fcolor = vec3(diffuse);
  gl_Position = mvp * in_position;
}
