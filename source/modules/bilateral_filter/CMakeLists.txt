cmake_minimum_required(VERSION 2.8.11)

list(APPEND SOURCE_H_FILES
  ${CMAKE_CURRENT_LIST_DIR}/BilateralFilterWidget.h
  ${CMAKE_CURRENT_LIST_DIR}/BilateralFilterProcessor.h
  )

list(APPEND SOURCE_UI_FILES
  ${CMAKE_CURRENT_LIST_DIR}/BilateralFilterWidget.ui
  )

list(APPEND SOURCE_CPP_FILES
  ${CMAKE_CURRENT_LIST_DIR}/BilateralFilterWidget.cpp
  ${CMAKE_CURRENT_LIST_DIR}/BilateralFilterProcessor.cpp
  )

list(APPEND CUDA_CPP_FILES
  ${CMAKE_CURRENT_LIST_DIR}/bilateral_filter_kernel.cu
  )

include_directories(${CMAKE_CURRENT_LIST_DIR})
