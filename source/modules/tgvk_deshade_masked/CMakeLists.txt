cmake_minimum_required(VERSION 2.8.11)

list(APPEND SOURCE_UI_FILES
  ${CMAKE_CURRENT_LIST_DIR}/TGVKDeshadeMaskedWidget.ui
  )
  
list(APPEND SOURCE_H_FILES
  ${CMAKE_CURRENT_LIST_DIR}/TGVKDeshadeMaskedWidget.h
  ${CMAKE_CURRENT_LIST_DIR}/TGVKDeshadeMaskedProcessor.h
  )

list(APPEND SOURCE_CPP_FILES
  ${CMAKE_CURRENT_LIST_DIR}/TGVKDeshadeMaskedWidget.cpp
  ${CMAKE_CURRENT_LIST_DIR}/TGVKDeshadeMaskedProcessor.cpp
  )

list(APPEND CUDA_CPP_FILES
   ${CMAKE_CURRENT_LIST_DIR}/tgvk_l1_deshade_masked.cu
   ${CMAKE_CURRENT_LIST_DIR}/tgv3_masked_common.cu
   ${CMAKE_CURRENT_LIST_DIR}/tgvk_masked_common.cu
   ${CMAKE_CURRENT_LIST_DIR}/tgv3_masked_common_2d.cu
   ${CMAKE_CURRENT_LIST_DIR}/tgvk_masked_common_2d.cu
   ${CMAKE_CURRENT_LIST_DIR}/tgvk_l1_deshade_masked_2d.cu
  )

include_directories(${CMAKE_CURRENT_LIST_DIR})
