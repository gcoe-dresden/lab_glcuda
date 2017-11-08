#version 330 core

layout(location=0) in vec4 in_position;
layout(location=1) in vec4 in_color;

uniform mat4 modelview;

out vec4 vColor;

void main()
{
  vColor = in_color;
  gl_Position = modelview * in_position;
}
