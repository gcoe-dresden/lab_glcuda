cmake_minimum_required (VERSION 3.1)

include(cmake/safeguards.cmake)

project(glcuda
  VERSION 0.1.0
  LANGUAGES C CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.0)
    message(FATAL_ERROR "gcc5.0 or newer required.")
  endif()
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# --- Common ---

add_library(Common INTERFACE)
add_library(glcuda::common ALIAS Common)

target_include_directories(Common INTERFACE
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/inc>
  )

# --- CUDA ---
find_package(CUDA REQUIRED)

list(APPEND CUDA_NVCC_FLAGS "-arch=sm_30")
list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
list(APPEND CUDA_NVCC_FLAGS "-lineinfo")

add_library(Cuda INTERFACE)
add_library(glcuda::cuda ALIAS Cuda)

# workaround for FindCUDA that uses plain link_libraries
set(CUDA_LIBRARIES PRIVATE ${CUDA_LIBRARIES})

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

# Compile GLAD and GLFW
if (MSVC)
  set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/ext/glad.c"
    PROPERTIES COMPILE_FLAGS "/wd4055 ")
endif()

#...(dir binary-dir)
add_subdirectory(ext/glfw ext_build/glfw)

target_include_directories(Common INTERFACE
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ext/glfw/include>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ext/glfw/deps>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ext/glfw/deps/glad>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ext/glfw/deps/KHR>
  )

# --- glm (header only library for OpenGL math) ---
set(GLM_TEST_ENABLE_FAST_MATH OFF CACHE BOOL " " FORCE)

target_include_directories(Common INTERFACE
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ext/glm>
)

# ---

add_library(glcuda_glad STATIC
  "${CMAKE_SOURCE_DIR}/ext/glfw/deps/glad.c")
target_link_libraries(glcuda_glad PRIVATE Common)

target_link_libraries(Common INTERFACE
  glcuda_glad
  glfw
  ${OPENGL_LIBRARIES}
  ${GLEW_LIBRARY}
  )

add_subdirectory(src)
