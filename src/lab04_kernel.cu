
#include <cuda.h>
#include <cuda_runtime.h>

#define GLM_FORCE_CUDA
#include <glm/glm.hpp> // vec3, vec4, ivec4, mat4
/*
 *
 */
__global__ void d_advance( float4* verts,
                           int mesh_width, int mesh_height,
                           float time,
                           float delta)
{
  int i,j;
  for (i = blockIdx.y * blockDim.y + threadIdx.y;
       i < mesh_height;
       i += blockDim.y * gridDim.y)
  {
    for (j = blockIdx.x * blockDim.x + threadIdx.x;
         j < mesh_width;
         j += blockDim.x * gridDim.x)
    {
      float x = -1.0f + 2.0f*j/mesh_width;
      float y = -1.0f + 2.0f*i/mesh_height;
      float distance = sqrt(x*x+y*y);
      float z = exp(-2.0f*distance)*sin(24*distance-0.01f*time);
      verts[j + i*mesh_width].z = 0.1f*z;
    }
  }
}


/*
 * calculate normals
 */
__global__ void d_normals( float4* verts,
                           int anchor_w, int anchor_h,
                           int mesh_width, int mesh_height)
{
  int i,j;
  float4* normals = verts + mesh_width*mesh_height;
  for (i = blockIdx.y * blockDim.y + threadIdx.y; // 0..3
       i < anchor_h;
       i += blockDim.y * gridDim.y) // +4
  {
    for (j = blockIdx.x * blockDim.x + threadIdx.x; // 0..31, 32..63, ...
         j < anchor_w;
         j += blockDim.x * gridDim.x) // +32*(xxx*numSMs)

    {
      /**      v1
       *       |
       * v4 -- v0 -- v2
       *       |
       *       v3
       */
      int vertexIndex = ( i * mesh_width ) + j + mesh_width + 1;
      glm::vec3 v0 (verts[ vertexIndex ].x,
                    verts[ vertexIndex ].y,
                    verts[ vertexIndex ].z);
      glm::vec3 v1 (verts[ vertexIndex-mesh_width ].x,
                    verts[ vertexIndex-mesh_width ].y,
                    verts[ vertexIndex-mesh_width ].z);
      glm::vec3 v2 (verts[ vertexIndex+1 ].x,
                    verts[ vertexIndex+1 ].y,
                    verts[ vertexIndex+1 ].z);
      glm::vec3 v3 (verts[ vertexIndex+mesh_width ].x,
                    verts[ vertexIndex+mesh_width ].y,
                    verts[ vertexIndex+mesh_width ].z);
      glm::vec3 v4 (verts[ vertexIndex-1 ].x,
                    verts[ vertexIndex-1 ].y,
                    verts[ vertexIndex-1 ].z);

      glm::vec3 normal_v12 = glm::normalize( glm::cross( v1-v0, v2-v0 ) );
      glm::vec3 normal_v23 = glm::normalize( glm::cross( v2-v0, v3-v0 ) );
      glm::vec3 normal_v34 = glm::normalize( glm::cross( v3-v0, v4-v0 ) );
      glm::vec3 normal_v41 = glm::normalize( glm::cross( v4-v0, v1-v0 ) );
      glm::vec3 normal = glm::normalize( normal_v12 + normal_v23 + normal_v34 + normal_v41 );
      normals[vertexIndex].x = normal.x;
      normals[vertexIndex].y = normal.y;
      normals[vertexIndex].z = normal.z;
    }
  }
}

void kernel_advance(float4* verts,
                    int mesh_width, int mesh_height,
                    double time,
                    int numSMs,
                    double delta)
{
  dim3 threads(32, 4);
  dim3 blocks( 16*numSMs );
    
  d_advance<<<blocks, threads>>>( verts,
                                  mesh_width, mesh_height,
                                  static_cast<float>(time),
                                  static_cast<float>(delta));


  d_normals<<<blocks, threads>>>( verts,
                                  mesh_width-2, mesh_height-2,
                                  mesh_width, mesh_height);
}
