cmake_minimum_required(VERSION 2.8.11)

list(APPEND SOURCE_H_FILES
  ${CMAKE_CURRENT_LIST_DIR}/ConjugateGradientAlgorithm.h
  ${CMAKE_CURRENT_LIST_DIR}/ConjugateGradientWidget.h
  )

list(APPEND SOURCE_UI_FILES
  ${CMAKE_CURRENT_LIST_DIR}/ConjugateGradientWidget.ui
  )

list(APPEND SOURCE_CPP_FILES
  ${CMAKE_CURRENT_LIST_DIR}/ConjugateGradientAlgorithm.cpp
  ${CMAKE_CURRENT_LIST_DIR}/ConjugateGradientWidget.cpp
  )

list(APPEND CUDA_CPP_FILES
  ${CMAKE_CURRENT_LIST_DIR}/ImageMatrix.h
  ${CMAKE_CURRENT_LIST_DIR}/ImageVectorOperations.h

  ${CMAKE_CURRENT_LIST_DIR}/image_matrix_kernels.cu
  ${CMAKE_CURRENT_LIST_DIR}/image_vector_operations_kernels.cu

  ${CMAKE_CURRENT_LIST_DIR}/ImageMatrix.cpp
  ${CMAKE_CURRENT_LIST_DIR}/ImageVectorOperations.cpp

  ${CMAKE_CURRENT_LIST_DIR}/ImageMatrixGradientFactory.h
  ${CMAKE_CURRENT_LIST_DIR}/ImageMatrixGradientFactory.cpp
  )
include_directories(${CMAKE_CURRENT_LIST_DIR})
