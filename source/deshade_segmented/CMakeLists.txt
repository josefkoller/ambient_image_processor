cmake_minimum_required(VERSION 2.8.11)

list(APPEND SOURCE_H_FILES
  ${CMAKE_CURRENT_LIST_DIR}/DeshadeSegmentedWidget.h
  ${CMAKE_CURRENT_LIST_DIR}/DeshadeSegmentedProcessor.h
  )

list(APPEND SOURCE_UI_FILES
  ${CMAKE_CURRENT_LIST_DIR}/DeshadeSegmentedWidget.ui
  )

list(APPEND SOURCE_CPP_FILES
  ${CMAKE_CURRENT_LIST_DIR}/DeshadeSegmentedWidget.cpp
  ${CMAKE_CURRENT_LIST_DIR}/DeshadeSegmentedProcessor.cpp
  )

#list(APPEND CUDA_CPP_FILES
#  ${CMAKE_CURRENT_LIST_DIR}/non_local_gradient_kernel.cu
#  )

