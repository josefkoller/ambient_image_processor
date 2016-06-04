cmake_minimum_required(VERSION 2.8.11)

list(APPEND SOURCE_H_FILES
  ${CMAKE_CURRENT_LIST_DIR}/TGVShadingGrowingWidget.h
  ${CMAKE_CURRENT_LIST_DIR}/TGVShadingGrowingProcessor.h
  )

list(APPEND SOURCE_UI_FILES
  ${CMAKE_CURRENT_LIST_DIR}/TGVShadingGrowingWidget.ui
  )

list(APPEND SOURCE_CPP_FILES
  ${CMAKE_CURRENT_LIST_DIR}/TGVShadingGrowingWidget.cpp
  ${CMAKE_CURRENT_LIST_DIR}/TGVShadingGrowingProcessor.cpp
  )

list(APPEND CUDA_CPP_FILES
  ${CMAKE_CURRENT_LIST_DIR}/tgv2_l1_shading_growing.cu
  ${CMAKE_CURRENT_LIST_DIR}/shading_growing.cu
    )
include_directories(${CMAKE_CURRENT_LIST_DIR})
