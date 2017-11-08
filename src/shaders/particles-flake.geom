#version 330 core

layout(points) in;
layout(triangle_strip,max_vertices=4) out;

in vec4 vColor[];
out vec4 fColor;

uniform mat4 mvp;
       
void main()
{
  fColor = vColor[0];
  // points are placed in [-0.5*frame_width .. 0.5*frame_width, -0.5*frame_height .. 0.5*frame_height]
  //  so size should be measured like in number of pixels
  float size = 10.f;
  vec4 center = gl_in[0].gl_Position;
  gl_Position = center + vec4(-size, -size, 0.0f, 0.0f);    // 1:bottom-left
  gl_Position = mvp*gl_Position;
  EmitVertex();
  gl_Position = center + vec4( size, -size, 0.0f, 0.0f);    // 2:bottom-right
  gl_Position = mvp*gl_Position;
  EmitVertex();
  gl_Position = center + vec4(-size,  size, 0.0f, 0.0f);    // 3:top-left
  gl_Position = mvp*gl_Position;
  EmitVertex();
  gl_Position = center + vec4( size,  size, 0.0f, 0.0f);    // 4:top-right
  gl_Position = mvp*gl_Position;
  EmitVertex();
  EndPrimitive();
}
