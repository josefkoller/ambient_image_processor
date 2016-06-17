cmake_minimum_required(VERSION 2.8.11)

  
list(APPEND CUDA_CPP_FILES
  ${CMAKE_CURRENT_LIST_DIR}/cuda_helper.in
  ${CMAKE_CURRENT_LIST_DIR}/cuda_host_helper.in
  )

configure_file(${CMAKE_CURRENT_LIST_DIR}/cuda_helper.in ${CMAKE_BINARY_DIR}/generated/cuda_helper.cuh)
configure_file(${CMAKE_CURRENT_LIST_DIR}/cuda_host_helper.in ${CMAKE_BINARY_DIR}/generated/cuda_host_helper.cuh)
include_directories(${CMAKE_BINARY_DIR}/generated/)

include_directories(${CMAKE_CURRENT_LIST_DIR})
