cmake_minimum_required(VERSION 2.8.11)

list(APPEND SOURCE_H_FILES
  ${CMAKE_CURRENT_LIST_DIR}/NonLocalGradientWidget.h
  ${CMAKE_CURRENT_LIST_DIR}/NonLocalGradientKernelWidget.h
  ${CMAKE_CURRENT_LIST_DIR}/NonLocalGradientProcessor.h
  )

list(APPEND SOURCE_UI_FILES
  ${CMAKE_CURRENT_LIST_DIR}/NonLocalGradientWidget.ui
  ${CMAKE_CURRENT_LIST_DIR}/NonLocalGradientKernelWidget.ui
  )

list(APPEND SOURCE_CPP_FILES
  ${CMAKE_CURRENT_LIST_DIR}/NonLocalGradientWidget.cpp
  ${CMAKE_CURRENT_LIST_DIR}/NonLocalGradientKernelWidget.cpp
  ${CMAKE_CURRENT_LIST_DIR}/NonLocalGradientProcessor.cpp
  )


list(APPEND CUDA_CPP_FILES
  ${CMAKE_CURRENT_LIST_DIR}/non_local_gradient_kernel.cu
  )

include_directories(${CMAKE_CURRENT_LIST_DIR})
