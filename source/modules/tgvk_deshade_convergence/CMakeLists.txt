cmake_minimum_required(VERSION 2.8.11)

list(APPEND SOURCE_UI_FILES
  ${CMAKE_CURRENT_LIST_DIR}/TGVKDeshadeConvergenceWidget.ui
  )
  
list(APPEND SOURCE_H_FILES
  ${CMAKE_CURRENT_LIST_DIR}/TGVKDeshadeConvergenceWidget.h
  ${CMAKE_CURRENT_LIST_DIR}/TGVKDeshadeConvergenceProcessor.h
  )

list(APPEND SOURCE_CPP_FILES
  ${CMAKE_CURRENT_LIST_DIR}/TGVKDeshadeConvergenceWidget.cpp
  ${CMAKE_CURRENT_LIST_DIR}/TGVKDeshadeConvergenceProcessor.cpp
  )

list(APPEND CUDA_CPP_FILES
   ${CMAKE_CURRENT_LIST_DIR}/tgvk_l1_deshade_convergence.cu
  )

include_directories(${CMAKE_CURRENT_LIST_DIR})
