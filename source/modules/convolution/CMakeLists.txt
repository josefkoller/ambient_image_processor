cmake_minimum_required(VERSION 2.8.11)

list(APPEND SOURCE_UI_FILES
    ${CMAKE_CURRENT_LIST_DIR}/ConvolutionWidget.ui
  )
  
list(APPEND SOURCE_H_FILES
    ${CMAKE_CURRENT_LIST_DIR}/ConvolutionWidget.h
  )

list(APPEND SOURCE_CPP_FILES
    ${CMAKE_CURRENT_LIST_DIR}/ConvolutionWidget.cpp
  )

list(APPEND CUDA_CPP_FILES
  )

include_directories(${CMAKE_CURRENT_LIST_DIR})
