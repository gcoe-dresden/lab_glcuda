cmake_minimum_required(VERSION 2.8)

# ---

add_custom_target(
  copy_shader_files
  ${CMAKE_COMMAND}
  -D SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
  -D DESTINATION_DIR=${CMAKE_BINARY_DIR}
  -P ${CMAKE_CURRENT_SOURCE_DIR}/shaders/copy_shaders.cmake
  COMMENT "copy shader source files." VERBATIM
  )


if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lab00.cpp")
add_executable(lab00 ${EXTRA_SOURCES} lab00.cpp)
target_link_libraries(lab00 glfw ${OPENGL_LIBRARIES} ${GLEW_LIBRARY})
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lab01.cpp")
add_executable(lab01 ${EXTRA_SOURCES} lab01.cpp)
target_link_libraries(lab01 glfw ${OPENGL_LIBRARIES} ${GLEW_LIBRARY})
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lab02.cpp")
add_executable(lab02 ${EXTRA_SOURCES} lab02.cpp)
target_link_libraries(lab02 glfw ${OPENGL_LIBRARIES} ${GLEW_LIBRARY})
endif()


if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lab02-solution.cpp")
  add_executable(lab02-solution ${EXTRA_SOURCES} lab02-solution.cpp)
  target_link_libraries(lab02-solution glfw ${OPENGL_LIBRARIES} ${GLEW_LIBRARY})
  add_dependencies (lab02-solution copy_shader_files)
endif()

if(CUDA_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lab03.cpp")
  cuda_add_executable(lab03 ${EXTRA_SOURCES} lab03.cpp lab03_kernel.cu)
  target_link_libraries(lab03 glfw ${CUDA_LIBRARIES} ${OPENGL_LIBRARIES} ${GLEW_LIBRARY})
endif()

if(CUDA_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lab03-flake.cpp")
  cuda_add_executable(lab03-flake ${EXTRA_SOURCES} lab03-flake.cpp lab03-flake_kernel.cu)
  target_link_libraries(lab03-flake glfw ${CUDA_LIBRARIES} ${OPENGL_LIBRARIES} ${GLEW_LIBRARY})
endif()

if(CUDA_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lab04.cpp")
  cuda_add_executable(lab04 ${EXTRA_SOURCES} lab04.cpp lab04_kernel.cu)
  target_link_libraries(lab04 glfw ${CUDA_LIBRARIES} ${OPENGL_LIBRARIES} ${GLEW_LIBRARY})
endif()

if(CUDA_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lab05.cpp")
  cuda_add_executable(lab05 ${EXTRA_SOURCES} lab05.cpp lab05_kernel.cu)
  target_link_libraries(lab05 glfw ${CUDA_LIBRARIES} ${OPENGL_LIBRARIES} ${GLEW_LIBRARY})
endif()

if(CUDA_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lab05-surface.cpp")
  cuda_add_executable(lab05-surface ${EXTRA_SOURCES} lab05-surface.cpp lab05-surface_kernel.cu)
  target_link_libraries(lab05-surface glfw ${CUDA_LIBRARIES} ${OPENGL_LIBRARIES} ${GLEW_LIBRARY})
endif()
