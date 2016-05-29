cmake_minimum_required(VERSION 2.8.11)

list(APPEND SOURCE_UI_FILES
    ${CMAKE_CURRENT_LIST_DIR}/TGVL1ThresholdGradientWidget.ui
  )
  
list(APPEND SOURCE_H_FILES
    ${CMAKE_CURRENT_LIST_DIR}/TGVL1ThresholdGradientWidget.h
    ${CMAKE_CURRENT_LIST_DIR}/TGVL1ThresholdGradientProcessor.h
  )

list(APPEND SOURCE_CPP_FILES
    ${CMAKE_CURRENT_LIST_DIR}/TGVL1ThresholdGradientWidget.cpp
    ${CMAKE_CURRENT_LIST_DIR}/TGVL1ThresholdGradientProcessor.cpp
  )

list(APPEND CUDA_CPP_FILES
  ${CMAKE_CURRENT_LIST_DIR}/tgvl1_threshold_gradient.cu
  ${CMAKE_CURRENT_LIST_DIR}/tgvl1_threshold_gradient2.cu
  )

include_directories(${CMAKE_CURRENT_LIST_DIR})
