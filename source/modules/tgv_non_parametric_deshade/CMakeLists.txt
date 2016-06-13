cmake_minimum_required(VERSION 2.8.11)
  
list(APPEND SOURCE_H_FILES
    ${CMAKE_CURRENT_LIST_DIR}/TGVNonParametricDeshadeWidget.h
    ${CMAKE_CURRENT_LIST_DIR}/TGVNonParametricDeshadeProcessor.h
    ${CMAKE_CURRENT_LIST_DIR}/cosine_transform.h
  )

list(APPEND SOURCE_UI_FILES
  ${CMAKE_CURRENT_LIST_DIR}/TGVNonParametricDeshadeWidget.ui
  )

list(APPEND SOURCE_CPP_FILES
    ${CMAKE_CURRENT_LIST_DIR}/TGVNonParametricDeshadeWidget.cpp
    ${CMAKE_CURRENT_LIST_DIR}/TGVNonParametricDeshadeProcessor.cpp
  )

list(APPEND CUDA_CPP_FILES
    ${CMAKE_CURRENT_LIST_DIR}/tgv2_l1_non_parametric_deshade_metrics.cu
    ${CMAKE_CURRENT_LIST_DIR}/tgv2_l1_non_parametric_deshade.cu
  )

include_directories(${CMAKE_CURRENT_LIST_DIR})
