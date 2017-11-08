// http://www.glfw.org/docs/latest/quick.html

#include <glad/glad.h>

/* from glfw:
    Do not include the OpenGL header yourself, as GLFW does this for you in a platform-independent way
    Do not include windows.h or other platform-specific headers unless you plan on using those APIs yourself
    If you do need to include such headers, include them before the GLFW header and it will detect this
*/
#include <GLFW/glfw3.h>
#include <math.h>

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
  CHECK_GLERROR();

  while (!glfwWindowShouldClose(window))
  {
    // the background color
    glClearColor(0.,sin(glfwGetTime()),0.,1.);
    glClear(GL_COLOR_BUFFER_BIT);



    CHECK_GLERROR();
    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  glfwDestroyWindow(window);

  glfwTerminate();
  exit(EXIT_SUCCESS);
}
