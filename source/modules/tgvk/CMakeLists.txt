cmake_minimum_required(VERSION 2.8.11)

list(APPEND SOURCE_UI_FILES
   ${CMAKE_CURRENT_LIST_DIR}/TGVKWidget.ui
  )
  
list(APPEND SOURCE_H_FILES
   ${CMAKE_CURRENT_LIST_DIR}/TGVKWidget.h
   ${CMAKE_CURRENT_LIST_DIR}/TGVKProcessor.h
  )

list(APPEND SOURCE_CPP_FILES
   ${CMAKE_CURRENT_LIST_DIR}/TGVKWidget.cpp
   ${CMAKE_CURRENT_LIST_DIR}/TGVKProcessor.cpp
  )

list(APPEND CUDA_CPP_FILES
  ${CMAKE_CURRENT_LIST_DIR}/tgvk_common.cu
  ${CMAKE_CURRENT_LIST_DIR}/tgvk_common_2d.cu
  ${CMAKE_CURRENT_LIST_DIR}/tgvk_l1.cu
  ${CMAKE_CURRENT_LIST_DIR}/tgvk_l1_2d.cu
  )
include_directories(${CMAKE_CURRENT_LIST_DIR})
