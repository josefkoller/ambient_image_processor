cmake_minimum_required(VERSION 2.8.11)


list(APPEND SOURCE_H_FILES
  ${CMAKE_CURRENT_LIST_DIR}/CudaImageOperationsProcessor.h
  )


list(APPEND SOURCE_CPP_FILES
  ${CMAKE_CURRENT_LIST_DIR}/CudaImageOperationsProcessor.cpp
  )


list(APPEND CUDA_CPP_FILES
  ${CMAKE_CURRENT_LIST_DIR}/binary_operation.cu
  ${CMAKE_CURRENT_LIST_DIR}/multiply.cu
  ${CMAKE_CURRENT_LIST_DIR}/divide.cu
  ${CMAKE_CURRENT_LIST_DIR}/add.cu
  ${CMAKE_CURRENT_LIST_DIR}/subtract.cu
  ${CMAKE_CURRENT_LIST_DIR}/convolution3x3.cu
  ${CMAKE_CURRENT_LIST_DIR}/convolution3x3x3.cu
  ${CMAKE_CURRENT_LIST_DIR}/add_constant.cu
  ${CMAKE_CURRENT_LIST_DIR}/multiply_constant.cu
  ${CMAKE_CURRENT_LIST_DIR}/cosine_transform.cu
  ${CMAKE_CURRENT_LIST_DIR}/inverse_cosine_transform.cu
  ${CMAKE_CURRENT_LIST_DIR}/divergence.cu
  ${CMAKE_CURRENT_LIST_DIR}/solve_poisson_in_cosine_domain.cu
  )

include_directories(${CMAKE_CURRENT_LIST_DIR})
