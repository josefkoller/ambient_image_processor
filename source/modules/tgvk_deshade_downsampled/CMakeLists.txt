cmake_minimum_required(VERSION 2.8.11)

list(APPEND SOURCE_UI_FILES
  ${CMAKE_CURRENT_LIST_DIR}/TGVKDeshadeDownsampledWidget.ui
  )
  
list(APPEND SOURCE_H_FILES
  ${CMAKE_CURRENT_LIST_DIR}/TGVKDeshadeDownsampledWidget.h
  ${CMAKE_CURRENT_LIST_DIR}/TGVKDeshadeDownsampledProcessor.h
  ${CMAKE_CURRENT_LIST_DIR}/TGVKDeshadeDownsampledNotMaskedProcessor.h
  )

list(APPEND SOURCE_CPP_FILES
  ${CMAKE_CURRENT_LIST_DIR}/TGVKDeshadeDownsampledWidget.cpp
  ${CMAKE_CURRENT_LIST_DIR}/TGVKDeshadeDownsampledProcessor.cpp
  ${CMAKE_CURRENT_LIST_DIR}/TGVKDeshadeDownsampledNotMaskedProcessor.cpp
  )

list(APPEND CUDA_CPP_FILES
  )

include_directories(${CMAKE_CURRENT_LIST_DIR})
