cmake_minimum_required(VERSION 2.8.11)

list(APPEND SOURCE_UI_FILES
   ${CMAKE_CURRENT_LIST_DIR}/TGV3DeshadeWidget.ui
  )
  
list(APPEND SOURCE_H_FILES
   ${CMAKE_CURRENT_LIST_DIR}/TGV3DeshadeWidget.h
   ${CMAKE_CURRENT_LIST_DIR}/TGV3DeshadeProcessor.h
  )

list(APPEND SOURCE_CPP_FILES
   ${CMAKE_CURRENT_LIST_DIR}/TGV3DeshadeWidget.cpp
   ${CMAKE_CURRENT_LIST_DIR}/TGV3DeshadeProcessor.cpp
  )

list(APPEND CUDA_CPP_FILES
   ${CMAKE_CURRENT_LIST_DIR}/tgv3_l1_deshade.cu
   ${CMAKE_CURRENT_LIST_DIR}/tgv3_l1_deshade_2d.cu
  )

include_directories(${CMAKE_CURRENT_LIST_DIR})
