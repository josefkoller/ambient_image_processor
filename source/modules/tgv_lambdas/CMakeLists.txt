cmake_minimum_required(VERSION 2.8.11)

list(APPEND SOURCE_UI_FILES
  ${CMAKE_CURRENT_LIST_DIR}/TGVLambdasWidget.ui
  )
  
list(APPEND SOURCE_H_FILES
  ${CMAKE_CURRENT_LIST_DIR}/TGVLambdasWidget.h
  ${CMAKE_CURRENT_LIST_DIR}/TGVLambdasProcessor.h
  )

list(APPEND SOURCE_CPP_FILES
  ${CMAKE_CURRENT_LIST_DIR}/TGVLambdasWidget.cpp
  ${CMAKE_CURRENT_LIST_DIR}/TGVLambdasProcessor.cpp
  )

list(APPEND CUDA_CPP_FILES
  ${CMAKE_CURRENT_LIST_DIR}/tgvl2_lambdas.cu
  )

include_directories(${CMAKE_CURRENT_LIST_DIR})
