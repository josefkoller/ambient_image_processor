cmake_minimum_required(VERSION 2.8.11)

  
list(APPEND CUDA_CPP_FILES
  ${CMAKE_CURRENT_LIST_DIR}/cuda_helper.cuh
  )

include_directories(${CMAKE_CURRENT_LIST_DIR})
