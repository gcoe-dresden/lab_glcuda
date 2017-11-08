#version 330 core

layout(points) in;
layout(points,max_vertices=1) out;

in vec4 vColor[];
out vec4 fColor;

uniform mat4 projection;

void main()
{
  vec4 center = gl_in[0].gl_Position;//!< Die Positionen sind jetzt im View-Space (Kamerazentrum == Ursprung)
  fColor = vColor[0];
  gl_Position = projection * center;
  gl_PointSize = 2.0;
  EmitVertex();
  EndPrimitive();
}
