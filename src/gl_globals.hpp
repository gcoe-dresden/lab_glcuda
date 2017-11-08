/*****************************************************************************/
/**
 * @file
 * @brief Some globals.
 * @author Matthias Werner
 * @sa http://11235813tdd.blogspot.de/
 *****************************************************************************/

#ifndef GL_GLOBALS_HPP_
#define GL_GLOBALS_HPP_

//-----------------------------------------------------------------------------

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <Windows.h>
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/freeglut.h>
#include <GL/wglext.h>
#include <time.h>
typedef unsigned int uint;
#define M_WINDOWS 1
#endif

#if defined(LINUX) || defined(__linux)
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/freeglut.h>
#include <GL/glx.h>
#include <string.h>
#include <sys/time.h>
#endif

#include <iostream>
#include <stdexcept>

#define CHECK_GLERROR() checkGLError(__FILE__, __LINE__)

inline int checkGLError(const char* f, const int line)
{
  GLuint err = glGetError();
  if (err != GL_NO_ERROR){
    std::cerr << f << ":" << line << ": OpenGL Error: '" << gluErrorString(err) <<"'"<<std::endl;
  }
  return err;
}

inline float calculate_fps()
{
  static int frame_count = 0;
  static int current_time = 0, previous_time = 0;
  static float fps = 0;
  int time_interval = 0;
  //  Increase frame count
  frame_count++;

  //  Get the number of milliseconds since glutInit called
  //  (or first call to glutGet(GLUT ELAPSED TIME)).
  current_time = glutGet(GLUT_ELAPSED_TIME);

  //  Calculate time passed
  //int time_interval = current_time - previous_time;
  time_interval = current_time - previous_time;

  if (time_interval > 1000)
  {
    //  calculate the number of frames per second
    fps = frame_count / (time_interval / 1000.0f);

    //  Set time
    previous_time = current_time;

    //  Reset frame count
    frame_count = 0;
  }
  return fps;
}


#endif /* GL_GLOBALS_HPP_ */
