cmake_minimum_required(VERSION 2.8.11)

list(APPEND SOURCE_H_FILES
  ${CMAKE_CURRENT_LIST_DIR}/HistogramWidget.h
  ${CMAKE_CURRENT_LIST_DIR}/HistogramProcessor.h
  ${CMAKE_CURRENT_LIST_DIR}/ChartWidget.h
  )

list(APPEND SOURCE_UI_FILES
  ${CMAKE_CURRENT_LIST_DIR}/HistogramWidget.ui
  )

list(APPEND SOURCE_CPP_FILES
  ${CMAKE_CURRENT_LIST_DIR}/HistogramWidget.cpp
  ${CMAKE_CURRENT_LIST_DIR}/HistogramProcessor.cpp
  ${CMAKE_CURRENT_LIST_DIR}/ChartWidget.cpp
  )

list(APPEND CUDA_CPP_FILES
  ${CMAKE_CURRENT_LIST_DIR}/kernel_density_estimation.cu
  )

include_directories(${CMAKE_CURRENT_LIST_DIR})
