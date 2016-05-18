cmake_minimum_required(VERSION 2.8.11)

set(THRUST_MODULE_SOURCE
  ${CMAKE_CURRENT_LIST_DIR}/Image.cuh
  ${CMAKE_CURRENT_LIST_DIR}/thrust_operators.cuh
  ${CMAKE_CURRENT_LIST_DIR}/Image.cu
  )

list(APPEND CUDA_CPP_FILES
  ${THRUST_MODULE_SOURCE}
  )

cuda_add_executable(image_tests
  ${CMAKE_CURRENT_LIST_DIR}/image_tests_console.cpp
  ${THRUST_MODULE_SOURCE}
)

include_directories(${CMAKE_CURRENT_LIST_DIR})
