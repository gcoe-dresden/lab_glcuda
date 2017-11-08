CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
set(CMAKE_DISABLE_SOURCE_CHANGES  ON)

set(CMAKE_COLOR_MAKEFILE ON)
# And optionally
#set(CMAKE_VERBOSE_MAKEFILE ON)


if ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
  message(SEND_ERROR "In-source builds are not allowed.")
endif ()
if (WIN32)
  set(CMAKE_SHARED_LIBRARY_PREFIX "")
endif ()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})
# set project build type
IF(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
   SET(CMAKE_BUILD_TYPE Release)
ENDIF(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
MESSAGE(">> Setting up ${CMAKE_BUILD_TYPE} build")


include(CheckCCompilerFlag)
include(CheckCXXCompilerFlag)


if (CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|Intel)$")
  check_cxx_compiler_flag("-std=c++14" HAS_CPP14_FLAG)
  check_cxx_compiler_flag("-std=c++11" HAS_CPP11_FLAG)

  if (HAS_CPP14_FLAG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
  elseif (HAS_CPP11_FLAG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  else()
    message(FATAL_ERROR "Unsupported compiler -- C++11 support required!")
  endif()
endif()

# --- CUDA ---

find_package(CUDA REQUIRED)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  if(CUDA_VERSION_STRING VERSION_EQUAL 8.0 OR CUDA_VERSION_STRING VERSION_GREATER 8.0)
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 6.0
        OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 6.0)
      message(FATAL_ERROR "CUDA 8.0 does not support gcc6+. Please choose an older gcc version like:\n
cmake -DCMAKE_C_COMPILER=`which gcc-5` -DCMAKE_CXX_COMPILER=`which g++-5` ..")
    endif()
  elseif(CUDA_VERSION_STRING VERSION_EQUAL 7.5 OR CUDA_VERSION_STRING VERSION_GREATER 7.5)
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0
        OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 5.0)
      message(WARNING "CUDA 7.5 does not support gcc5+. Please choose an older gcc version like:\n
cmake -DCMAKE_C_COMPILER=`which gcc-4` -DCMAKE_CXX_COMPILER=`which g++-4` ..")
    endif()
  endif()
endif()

include(FindCUDA)
include_directories(${CUDA_INCLUDE_DIRS})

list(APPEND CUDA_NVCC_FLAGS "-std=c++11 -gencode arch=compute_30,code=compute_30")
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  list(APPEND CUDA_NVCC_FLAGS "-g -lineinfo")
endif()

# --- GLFW ---
# re-used cmake settings from nanogui cmake files

if(NOT IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/ext/glfw/src")
  message(FATAL_ERROR "The dependency repositories (GLFW, etc.) are missing! "
    "You probably did not clone the project with --recursive. It is possible to recover "
    "by calling \"git submodule update --init --recursive\"")
endif()

set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL " " FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL " " FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL " " FORCE)
set(GLFW_BUILD_INSTALL OFF CACHE BOOL " " FORCE)
set(GLFW_INSTALL OFF CACHE BOOL " " FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL " " FORCE)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # Quench annoying deprecation warnings when compiling GLFW on OSX
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-deprecated-declarations")
endif()

# Compile GLAD and GLFW
list(APPEND EXTRA_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/ext/glfw/deps/glad.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/ext/glfw/deps/glad/glad.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/ext/glfw/deps/KHR/khrplatform.h")
if (MSVC)
  set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/ext/glad.c"
    PROPERTIES COMPILE_FLAGS "/wd4055 ")
endif()

add_subdirectory(ext/glfw ext_build/glfw)
include_directories(ext/glfw/include ext/glfw/deps/ ext/glfw/deps/glad)


# --- OpenGL ---

# find_package(OpenGL REQUIRED)
# include_directories( ${OPENGL_INCLUDE_DIRS})

# --- glm (header only library for OpenGL math) ---
set(GLM_TEST_ENABLE_FAST_MATH OFF CACHE BOOL " " FORCE)
include_directories(ext/glm)

# ---

add_subdirectory(src)
