/**
- reduced number of particles
- only using modelviewprojection (mvp) matrix for shader
 + vertex and geometry shader changed to produce rectangles
- C++11 random number generator
- random alpha values
*/

#include <glad/glad.h>

#include "cuda_globals.hpp"

#include <glm/glm.hpp> // vec3, vec4, ivec4, mat4
#include <glm/gtc/type_ptr.hpp> // value_ptr
#include <glm/gtc/matrix_transform.hpp> // ortho, rotate, scale, perspective

#include <GLFW/glfw3.h>

#include <iostream>
#include <fstream>
#include <string>
#include <thread>
#include <iomanip>
#include <random>

#define CHECK_GLERROR() checkGLError(__FILE__, __LINE__)

/* global variables */
cudaGraphicsResource* cuda_vbo_resource = NULL;
GLuint  vao             = 0;
GLuint  vbo             = 0;
void*   d_state_buffer  = NULL;
float   speed           = 1.0f;
int reset = 1; // initial particle distribution on first run_cuda()
int frame_width;
int frame_height;
int numSMs;
double fps;
double delta;

static constexpr int MESH_WIDTH = 16;
static constexpr int MESH_HEIGHT = 16;
static constexpr int MESH_COUNT = MESH_WIDTH * MESH_HEIGHT;

GLFWwindow* window;

void kernel_reset(float4* verts, float4* states,
                  int ww, int wh,
                  int mesh_width, int mesh_height,
                  int numSMs);

void kernel_advance(float4* verts, float4* states,
                    float mx, float my,
                    int mesh_count,
                    float speed,
                    int numSMs,
                    double delta);


/* function declarations */
void create_buffers();
void run_cuda(double delta);
void cleanup();
void computeFPS();

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

/**
 * Basic OpenGL Error Handling.
 */
inline int checkGLError(const char* _file, const int _line)
{
  GLuint err = glGetError();
  if (err != GL_NO_ERROR){
    std::string errstr="";
    switch(err) {
    case GL_INVALID_ENUM: errstr = "GL_INVALID_ENUM"; break;
    case GL_INVALID_VALUE: errstr = "GL_INVALID_VALUE"; break;
    case GL_INVALID_OPERATION: errstr = "GL_INVALID_OPERATION"; break;
    case GL_STACK_OVERFLOW: errstr = "GL_STACK_OVERFLOW"; break;
    case GL_STACK_UNDERFLOW: errstr = "GL_STACK_UNDERFLOW"; break;
    case GL_OUT_OF_MEMORY: errstr = "GL_OUT_OF_MEMORY"; break;
    case GL_INVALID_FRAMEBUFFER_OPERATION: errstr = "GL_INVALID_FRAMEBUFFER_OPERATION"; break;
    }
    std::cerr << _file << ":" << _line << ": OpenGL Error: " << err <<" = "<< errstr <<std::endl;
  }
  return err;
}

/**
 * Check if shader compiler has returned errors.
 */
void checkShaderCompileStatus(GLuint shader) {
  GLint shader_ok;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &shader_ok);
  if (shader_ok != GL_TRUE)
  {
    GLsizei log_length;
    char info_log[8192];
    glGetShaderInfoLog(shader, 8192, &log_length,info_log);
    std::cerr << "ERROR: Failed to compile shader" << std::endl;
    std::cerr << " message: " << info_log << std::endl;
    shader = 0;
  }else
    std::cerr << "Shader " << shader << " compiled." << std::endl;
}

/**
 * Check if linker has returned errors.
 */
void checkProgramLinkStatus(GLuint program) {
  GLint program_ok;

  glGetProgramiv(program, GL_LINK_STATUS, &program_ok);
  if (program_ok != GL_TRUE)
  {
    GLsizei log_length;
    char info_log[8192];
    glGetProgramInfoLog(program, 8192, &log_length, info_log);
    std::cerr << "ERROR: Failed to link program" << std::endl;
    std::cerr << " message: " << info_log << std::endl;
    program = 0;
  }
}

/**
 * Get shader source as string.
 */
std::string readFile(const char *_filepath) {
  std::string content;
  std::ifstream filestream(_filepath, std::ios::in);
  if(!filestream.is_open()) {
    std::cerr << "Could not read file " << _filepath << std::endl;
    return "";
  }
  std::string line = "";
  while(!filestream.eof()) {
    std::getline(filestream, line);
    content.append(line + "\n");
  }
  filestream.close();
  return content;
}

/// Error callback for glfw.
static void error_callback(int error, const char* description)
{
  std::cerr << "Error: " << description << std::endl;
}

/// Key event handler for glfw.
static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    glfwSetWindowShouldClose(window, GLFW_TRUE);
  if (key == GLFW_KEY_V && action == GLFW_PRESS) {
    static int vsync = 1;
    // enable v-sync, doc: number of screen updates to wait from the time glfwSwapBuffers was called before swapping the buffers and returning.
    glfwSwapInterval( (vsync=1-vsync) );
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
  glViewport(0, 0, width, height);
  reset = 1;
  frame_width = width;
  frame_height = height;
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
  speed += 0.1*yoffset;
  std::cout << speed << std::endl;
}

//-----------------------------------------------------------------------------

/**
 *
 */
int main(int argc, char** argv)
{

  glfwSetErrorCallback(error_callback);

  if (!glfwInit())
    exit(EXIT_FAILURE);

  // we want OpenGL 3.3
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // we do not want to use deprecated stuff


  // set screen size
  window = glfwCreateWindow(640, 480, "Snowflake", NULL, NULL);
  if (!window)
  {
    std::cerr << "Failed to open GLFW window. Probably a GPU and OpenGL driver issue." << std::endl;
    glfwTerminate();
    exit(EXIT_FAILURE);
  }
  // doc: On some machines screen coordinates and pixels are the same, but on others they will not be.  // So we get the real frame size in pixels here
  glfwGetFramebufferSize(window, &frame_width, &frame_height);

  glfwSetKeyCallback(window, key_callback);
  glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
  glfwSetScrollCallback(window, scroll_callback);

  glfwMakeContextCurrent(window);
  /// we need this to get OpenGL functions / extensions
  gladLoadGLLoader((GLADloadproc) glfwGetProcAddress);
  // enable v-sync, doc: number of screen updates to wait from the time glfwSwapBuffers was called before swapping the buffers and returning.
  glfwSwapInterval(1);

  // --- FPS ---
  std::thread([&]() {
      while (1)
      {
        std::ostringstream ss;
        ss << std::fixed << delta << std::setprecision(1) << "    [" << fps << " FPS]";
        glfwSetWindowTitle(window, ss.str().c_str());
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
      }
    }).detach();

  // --- CUDA ---
  unsigned int gl_device_count = 0;
  int gl_device_id = 0;
  /* 'cudaGLGetDevices' is more for multi-GPU context, since an error message here can be misleading,
     if no CUDA capable device within OpenGL context can be found.
     If you have a notebook with 2 GPUs (e.g. Intel, Nvidia), then the application probably runs OpenGL on Intel GPU.
     In this case, the CUDA GPU cannot be used for OpenGL interops and calls like 'cudaGraphicsMapResources' will fail.
     On linux, you can use 'optirun ./app' to use Nvidia GPU to run OpenGL.
  */
  //CHECK_CUDA(cudaGLGetDevices(&gl_device_count, &gl_device_id, 1, cudaGLDeviceListAll));
  //std::cerr << gl_device_count << " " << gl_device_id << std::endl;

  std::cout << listCudaDevices().str();
  CHECK_CUDA( cudaSetDevice(0) );
  cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

  // --- OpenGL calls ---

  // --- create OpenGL vertex buffer and CUDA buffer
  create_buffers();

  // --- load and compile shader ---
  GLuint vertex_shader, geom_shader, fragment_shader, program;

  std::string vertex_shader_text   = readFile("shaders/particles-flake.vert");
  std::string geom_shader_text     = readFile("shaders/particles-flake.geom");
  std::string fragment_shader_text = readFile("shaders/particles-flake.frag");
  const char* c_vertex_shader_text   = vertex_shader_text.c_str();
  const char* c_geom_shader_text     = geom_shader_text.c_str();
  const char* c_fragment_shader_text = fragment_shader_text.c_str();

  vertex_shader = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vertex_shader, // shader id
                 1, // count
                 reinterpret_cast<const GLchar**>(&c_vertex_shader_text), // array of strings
                 NULL // array of lengths
    );
  glCompileShader(vertex_shader);
  checkShaderCompileStatus(vertex_shader);

  geom_shader = glCreateShader(GL_GEOMETRY_SHADER);
  glShaderSource(geom_shader, // shader id
                 1, // count
                 reinterpret_cast<const GLchar**>(&c_geom_shader_text), // array of strings
                 NULL // array of lengths
    );
  glCompileShader(geom_shader);
  checkShaderCompileStatus(geom_shader);

  fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragment_shader, // shader id
                 1, // count
                 reinterpret_cast<const GLchar**>(&c_fragment_shader_text), // array of strings
                 NULL // array of lengths
    );
  glCompileShader(fragment_shader);
  checkShaderCompileStatus(fragment_shader);
  CHECK_GLERROR();

  // --- create shader program, link compiled shader objects

  program = glCreateProgram();
  glAttachShader(program, vertex_shader);
  glAttachShader(program, geom_shader);
  glAttachShader(program, fragment_shader);
  glLinkProgram(program);
  checkProgramLinkStatus(program);
  glUseProgram(program);
  CHECK_GLERROR();

  // additive blending
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE);
  glEnable(GL_POINT_SMOOTH);
  glEnable(GL_PROGRAM_POINT_SIZE);
  glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);

  double lastTime = glfwGetTime();
  while (!glfwWindowShouldClose(window))
  {
    double currentTime = glfwGetTime();
    delta = 100.0*(currentTime-lastTime);
    lastTime = currentTime;
    // off screen computation
    run_cuda(delta);

    glm::mat4 model(1.0f);
    // ortho ( left, right, bottom, top, zNear, zFar )
    glm::mat4 projection = glm::ortho( -0.5f*frame_width, 0.5f*frame_width,
                                       -0.5f*frame_height, 0.5f*frame_height,
                                       1.f, -1.f );
    glm::mat4 view(1.0f);
    glm::mat4 mvp = projection*view*model;

    int loc_mvp = glGetUniformLocation(program, "mvp");
    // glUniformMatrix4fv( location, count, transpose, pointer)
    glUniformMatrix4fv(loc_mvp, 1, GL_FALSE, glm::value_ptr(mvp));

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // render from the vbo

    glBindVertexArray(vao);
    glDrawArrays(GL_POINTS, 0, MESH_COUNT);
    glBindVertexArray(0);

    glfwSwapBuffers(window);
    glfwPollEvents();
    computeFPS();
  }
  cleanup();

  glfwDestroyWindow(window);


  glfwTerminate();

  cudaDeviceReset();
  return EXIT_SUCCESS;
}


/* CUDA kernel preparation & execution
 * Take care, that OpenGL buffers are not bound, so CUDA can use them
 */
void run_cuda(double delta)
{
  float4 *dptr;
  size_t num_bytes;
  // map OpenGL buffer object to CUDA
  // Note: OpenGL binding and CUDA mapping on a buffer are mutually exclusive
  //  you will not be able to map a buffer if it is bound by OpenGL
  //
  // DEPRECATED: cudaGLMapBufferObject((void**)&dptr, vbo);
  CHECK_CUDA(cudaGraphicsMapResources(1, &cuda_vbo_resource, 0));
  CHECK_CUDA(cudaGraphicsResourceGetMappedPointer((void **)&dptr,
                                                  &num_bytes,
                                                  cuda_vbo_resource));

  if(reset) {
    kernel_reset( dptr, static_cast<float4*>(d_state_buffer),
                  frame_width, frame_height,
                  MESH_WIDTH, MESH_HEIGHT,
                  numSMs );
    reset = 0;
  }else{
    double xpos, ypos;
    // this returns cursor pos in screen coordinates (maybe you have to convert to frame size coordinates)
    glfwGetCursorPos(window, &xpos, &ypos);
    float msx = (xpos - (frame_width>>1));
    float msy = ((frame_height>>1) - ypos);

    kernel_advance( dptr, static_cast<float4*>(d_state_buffer),
                    msx, msy,
                    MESH_COUNT,
                    speed, numSMs, delta );
  }

  CHECK_LAST("Kernel launch failed.");
  // unmap buffer object
  // DEPRECATED: checkCudaErrors(cudaGLUnmapBufferObject(vbo));
  CHECK_CUDA(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));
}


/**
 * Create VBO and device buffer
 * We'll need a few buffers for out particle system.
 * 1) A buffer which holds the renderable vertices
 * 2) A state buffer which holds particle speed and mass
 */
void create_buffers()
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> runif(0,1);
  /* OpenGL: 
    * Create an OpenGL buffer for the vertices + color information.
    * Init Data and upload it to GPU via OpenGL glBufferData();
    */

  GLfloat *h_data = new GLfloat[MESH_COUNT*8];

  // initializing vertex values
  for (unsigned int i = 0; i < MESH_COUNT*4; i = i+4)
  {
      h_data[i] = 0.0f;   // vertex.x
      h_data[i+1] = 0.0f; // vertex.y
      h_data[i+2] = 0.0f; // vertex.z
      h_data[i+3] = 1.0f; // vertex.w
  }
  // initializing color values
  for (unsigned int i = MESH_COUNT*4; i < MESH_COUNT*8; i = i+4)
  {
      // color is set in the kernel code but for debug purposes we'll set it
      // to white here
      h_data[i] = 1.0f;     // Red
      h_data[i+1] = 1.0f;   // Green
      h_data[i+2] = 1.0f;   // Blue
      h_data[i+3] = 0.05f+0.5*runif(gen);  // Alpha
  }
  // create buffer object
  glGenBuffers(1, &vbo);
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);

  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  // initialize buffer object by uploading host data
  glBufferData( GL_ARRAY_BUFFER,
                MESH_COUNT * 8 * sizeof(GLfloat),
                h_data,
                GL_DYNAMIC_DRAW );

  // vertex attribute
  glVertexAttribPointer(0,             // layout index for position in shader
                        4,             // number of components
                        GL_FLOAT,      // type
                        GL_FALSE,      // do not normalize values
                        0,             // stride in bytes
                        0              // offset in array for first component
    );
  glEnableVertexAttribArray(0); // enable vertex attribute with layout index 0

  // color attribute
  glVertexAttribPointer(1,             // layout index for position in shader
                        4,             // number of components
                        GL_FLOAT,      // type
                        GL_FALSE,      // do not normalize values
                        0,             // stride in bytes
                        (GLvoid*) (4*MESH_COUNT*sizeof(GLfloat)) // offset in array for first component
    );
  glEnableVertexAttribArray(1); // enable vertex attribute with layout index 1

  // release buffer
  glBindVertexArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  // if something went wrong in OpenGL
  CHECK_GLERROR();
  // ------------------------------------------------------------------------
  /* CUDA: register OpenGL buffer with CUDA*/
  CHECK_CUDA( cudaGraphicsGLRegisterBuffer( &cuda_vbo_resource,
                                            vbo,
                                            cudaGraphicsMapFlagsNone ) );

  /* each particle has its own state
      with a force(vec3) and particles mass value */
  GLfloat* h_state = new float[MESH_COUNT*4];

  for(unsigned int i=0; i<MESH_COUNT*4; i+=4)
  {
    h_state[i]   = 0.0f;  // force.x
    h_state[i+1] = 0.0f;  // force.y
    h_state[i+2] = 0.0f;  // force.z
    // random mass of particle
    h_state[i+3] = 75.0f + runif(gen) * 25.0f;
  }
  // malloc for cuda: cudaMalloc(devicePtr,size)
  CHECK_CUDA( cudaMalloc( &d_state_buffer,
                          MESH_COUNT*sizeof(float)*4 ) );
  // transfer host data to gpu: cudaMemcpy(devicePtr, hostPtr, size, kind)
  CHECK_CUDA( cudaMemcpy( d_state_buffer,
                          (const void*)h_state,
                          MESH_COUNT*sizeof(float)*4,
                          cudaMemcpyHostToDevice ) );
  /* We won't need this stuff again on the host system */
  delete[] h_data;
  delete[] h_state;
}

/**
 *
 */
void cleanup()
{
  // unregister this buffer object with CUDA
  cudaGraphicsUnregisterResource(cuda_vbo_resource);

  glDeleteVertexArrays(1, &vao);
  glDeleteBuffers(1, &vbo);
  CHECK_CUDA( cudaFree(d_state_buffer) );
  d_state_buffer=nullptr;
}

/**
 *
 */
void computeFPS()
{
  static double lastTime = glfwGetTime();
  static size_t nbFrames = 0;
  nbFrames++;
  double currentTime = glfwGetTime();
  double delta = currentTime - lastTime;
  if ( delta >= 1.0 ){
    fps = double(nbFrames) / delta;
    nbFrames = 0;
    lastTime = currentTime;
  }
}
