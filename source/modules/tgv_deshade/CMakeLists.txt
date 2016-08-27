cmake_minimum_required(VERSION 2.8.11)

list(APPEND SOURCE_UI_FILES
  ${CMAKE_CURRENT_LIST_DIR}/TGVDeshadeWidget.ui
  )
  
list(APPEND SOURCE_H_FILES
  ${CMAKE_CURRENT_LIST_DIR}/TGVDeshadeWidget.h
  ${CMAKE_CURRENT_LIST_DIR}/TGVDeshadeProcessor.h
  )

list(APPEND SOURCE_CPP_FILES
  ${CMAKE_CURRENT_LIST_DIR}/TGVDeshadeWidget.cpp
  ${CMAKE_CURRENT_LIST_DIR}/TGVDeshadeProcessor.cpp
  )

list(APPEND CUDA_CPP_FILES
  ${CMAKE_CURRENT_LIST_DIR}/tgv2_l1_deshade.cu
  ${CMAKE_CURRENT_LIST_DIR}/tgv2_l1_deshade_2d.cu
  )

include_directories(${CMAKE_CURRENT_LIST_DIR})
