cmake_minimum_required(VERSION 2.8.11)

list(APPEND SOURCE_UI_FILES
  ${CMAKE_CURRENT_LIST_DIR}/TGVDeshadeMaskedWidget.ui
  )
  
list(APPEND SOURCE_H_FILES
  ${CMAKE_CURRENT_LIST_DIR}/TGVDeshadeMaskedWidget.h
  ${CMAKE_CURRENT_LIST_DIR}/TGVDeshadeMaskedProcessor.h
  )

list(APPEND SOURCE_CPP_FILES
  ${CMAKE_CURRENT_LIST_DIR}/TGVDeshadeMaskedWidget.cpp
  ${CMAKE_CURRENT_LIST_DIR}/TGVDeshadeMaskedProcessor.cpp
  )

list(APPEND CUDA_CPP_FILES
  ${CMAKE_CURRENT_LIST_DIR}/tgv2_l1_deshade_masked.cu
  ${CMAKE_CURRENT_LIST_DIR}/tgv_masked_common.cu
  ${CMAKE_CURRENT_LIST_DIR}/tgv2_masked_common.cu
  )

include_directories(${CMAKE_CURRENT_LIST_DIR})
