
add_custom_target(
  copy_shader_files
  ${CMAKE_COMMAND}
  -D SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
  -D DESTINATION_DIR=${CMAKE_BINARY_DIR}
  -P ${CMAKE_CURRENT_SOURCE_DIR}/shaders/copy_shaders.cmake
  COMMENT "copy shader source files." VERBATIM
  )


if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lab00.cpp")
add_executable(lab00 lab00.cpp)
target_link_libraries(lab00 PRIVATE Common)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lab01.cpp")
add_executable(lab01 lab01.cpp)
target_link_libraries(lab01 PRIVATE Common)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lab02.cpp")
add_executable(lab02 lab02.cpp)
target_link_libraries(lab02 PRIVATE Common)
endif()


if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lab02-solution.cpp")
  add_executable(lab02-solution lab02-solution.cpp)
  target_link_libraries(lab02-solution PRIVATE Common)
  add_dependencies (lab02-solution copy_shader_files)
endif()

if(CUDA_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lab03.cpp")
  cuda_add_executable(lab03 lab03.cpp lab03_kernel.cu)
  target_link_libraries(lab03 PRIVATE Common Cuda)
endif()

if(CUDA_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lab03-flake.cpp")
  cuda_add_executable(lab03-flake lab03-flake.cpp lab03-flake_kernel.cu)
  target_link_libraries(lab03-flake PRIVATE Common Cuda)
endif()

if(CUDA_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lab04.cpp")
  cuda_add_executable(lab04 lab04.cpp lab04_kernel.cu)
  target_link_libraries(lab04 PRIVATE Common Cuda)
endif()

if(CUDA_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lab05.cpp")
  cuda_add_executable(lab05 lab05.cpp lab05_kernel.cu)
  target_link_libraries(lab05 PRIVATE Common Cuda)
endif()

if(CUDA_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lab05-surface.cpp")
  cuda_add_executable(lab05-surface lab05-surface.cpp lab05-surface_kernel.cu)
  target_link_libraries(lab05-surface PRIVATE Common Cuda)
endif()
