cmake_minimum_required(VERSION 2.8.11)

list(APPEND SOURCE_UI_FILES
  )
  
list(APPEND SOURCE_H_FILES
  )

list(APPEND SOURCE_CPP_FILES
  )

list(APPEND CUDA_CPP_FILES
  ${CMAKE_CURRENT_LIST_DIR}/tgv_common.cu
  ${CMAKE_CURRENT_LIST_DIR}/tgv_common_2d.cu
  ${CMAKE_CURRENT_LIST_DIR}/tgv1_l1.cu
  ${CMAKE_CURRENT_LIST_DIR}/tgv1_l2.cu
  ${CMAKE_CURRENT_LIST_DIR}/tgv2_common.cu
  ${CMAKE_CURRENT_LIST_DIR}/tgv2_common_2d.cu
  ${CMAKE_CURRENT_LIST_DIR}/tgv2_l1.cu
  ${CMAKE_CURRENT_LIST_DIR}/tgv2_l2.cu
  )

include_directories(${CMAKE_CURRENT_LIST_DIR})
