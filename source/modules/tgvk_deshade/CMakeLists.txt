cmake_minimum_required(VERSION 2.8.11)

list(APPEND SOURCE_UI_FILES
   ${CMAKE_CURRENT_LIST_DIR}/TGVKDeshadeWidget.ui
  )
  
list(APPEND SOURCE_H_FILES
   ${CMAKE_CURRENT_LIST_DIR}/TGVKDeshadeWidget.h
   ${CMAKE_CURRENT_LIST_DIR}/TGVKDeshadeProcessor.h
  )

list(APPEND SOURCE_CPP_FILES
   ${CMAKE_CURRENT_LIST_DIR}/TGVKDeshadeWidget.cpp
   ${CMAKE_CURRENT_LIST_DIR}/TGVKDeshadeProcessor.cpp
  )

list(APPEND CUDA_CPP_FILES
   ${CMAKE_CURRENT_LIST_DIR}/tgvk_l1_deshade.cu
  )

include_directories(${CMAKE_CURRENT_LIST_DIR})
