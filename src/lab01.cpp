// this is an update to the old example from https://github.com/wjakob/glfw/blob/master/examples/simple.c

#include <glm/glm.hpp> // vec3, vec4, ivec4, mat4
#include <glm/gtc/type_ptr.hpp> // value_ptr
#include <glm/gtc/matrix_transform.hpp> // ortho, rotate, scale, perspective

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <fstream>

#define CHECK_GLERROR() checkGLError(__FILE__, __LINE__)

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

// -----------------------------------------------------------------------------

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
  }
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

// -----------------------------------------------------------------------------

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
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

/**
 * Vertex data (x,y,z) for our triangle.
 */
static glm::vec3 vertices[3] =
{
  { -0.6f, -0.4f, 0.f },
  {  0.6f, -0.4f, 0.f },
  {   0.f,  0.6f, 0.f }
};

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

int main(void)
{
  GLFWwindow* window;

  glfwSetErrorCallback(error_callback);

  if (!glfwInit())
    exit(EXIT_FAILURE);

  // we want OpenGL 3.3
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // we do not want to use deprecated stuff

  window = glfwCreateWindow(640, 480, "Simple example", NULL, NULL);
  if (!window)
  {
    std::cerr << "Failed to open GLFW window. Probably a GPU and OpenGL driver issue." << std::endl;
    glfwTerminate();
    exit(EXIT_FAILURE);
  }

  glfwSetKeyCallback(window, key_callback);

  glfwMakeContextCurrent(window);
  /// we need this to get OpenGL functions / extensions
  gladLoadGLLoader((GLADloadproc) glfwGetProcAddress);
  // enable v-sync, doc: number of screen updates to wait from the time glfwSwapBuffers was called before swapping the buffers and returning.
  glfwSwapInterval(1);

  // ----------------------------------------------------------------------

  // Shader

  // --- load and compile shader ---
  GLuint vertex_shader, fragment_shader, program;

  std::string vertex_shader_text = readFile("shaders/simple.vert");
  std::string fragment_shader_text = readFile("shaders/simple.frag");
  const char* c_vertex_shader_text = vertex_shader_text.c_str();
  const char* c_fragment_shader_text = fragment_shader_text.c_str();

  vertex_shader = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vertex_shader, // shader id
                 1, // count
                 reinterpret_cast<const GLchar**>(&c_vertex_shader_text), // array of strings
                 NULL // array of lengths
    );
  glCompileShader(vertex_shader);
  checkShaderCompileStatus(vertex_shader);

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
  glAttachShader(program, fragment_shader);
  glLinkProgram(program);
  checkProgramLinkStatus(program);
  glUseProgram(program);
  CHECK_GLERROR();

  // ----------------------------------------------------------------------

  // --- create triangle as vertex array object ---
  /*
Comment from https://www.cs.utexas.edu/~fussell/courses/cs384g-fall2013/lectures/lectureXX-OpenGL.pdf
    Vertex data must be stored in a VBO, and associated with a VAO
    - The code-flow is similar to configuring a VAO
    - generate VBO names by calling glGenBuffers()

    - bind a specific VBO for initialization by calling
    glBindBuffer(GL_ARRAY_BUFFER,...)

    - load data into VBO using
    glBufferData(GL_ARRAY_BUFFER,...)

    - bind VAO for use in rendering
    glBindVertexArray()
  */

  // vertex array object needed in OpenGL core context
  GLuint vao;
  GLuint vertex_buffer;

  glGenBuffers(1, &vertex_buffer);
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);

  glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
  CHECK_GLERROR();

  // location also can be obtained by glGetVertexAttribLocation()
  glVertexAttribPointer(0,             // layout index for position in shader
                        3,             // number of components
                        GL_FLOAT,      // type
                        GL_FALSE,      // do not normalize values
                        0,             // stride in bytes
                        0              // offset in array for first component
    );
  glEnableVertexAttribArray(0); // enable vertex attribute with layout index 0

  // unbind
  glBindVertexArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  CHECK_GLERROR();

  while (!glfwWindowShouldClose(window))
  {
    float ratio;
    int width, height;

    glfwGetFramebufferSize(window, &width, &height);
    glViewport(0, 0, width, height);
    ratio = width / (float) height;

    glm::mat4 model(1.0f);
    // ortho ( left, right, bottom, top, zNear, zFar )
    glm::mat4 projection = glm::ortho( -ratio, ratio, -1.f, 1.f, 1.f, -1.f );
    glm::mat4 view(1.0f);
    // rotate( mat4, angle, axis )
    view = glm::rotate(view,
                       static_cast<float>( glfwGetTime() ),
                       glm::vec3(0,0,1.f));
    glm::mat4 mvp = projection*view*model; // model-view-projection


    glClear(GL_COLOR_BUFFER_BIT);
    glUniformMatrix4fv(0, 1, GL_FALSE, glm::value_ptr(mvp));

    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    // unbind
    glBindVertexArray(0);

    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  glfwDestroyWindow(window);

  glfwTerminate();
  exit(EXIT_SUCCESS);
}
