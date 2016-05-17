cmake_minimum_required(VERSION 2.8.11)

list(APPEND SOURCE_UI_FILES
  ${CMAKE_CURRENT_LIST_DIR}/TGVWidget.ui
  )
  
list(APPEND SOURCE_H_FILES
  ${CMAKE_CURRENT_LIST_DIR}/TGVWidget.h
  ${CMAKE_CURRENT_LIST_DIR}/TGVProcessor.h
  )

list(APPEND SOURCE_CPP_FILES
  ${CMAKE_CURRENT_LIST_DIR}/TGVWidget.cpp
  ${CMAKE_CURRENT_LIST_DIR}/TGVProcessor.cpp
  )

list(APPEND CUDA_CPP_FILES
  ${CMAKE_CURRENT_LIST_DIR}/Image.cuh
  ${CMAKE_CURRENT_LIST_DIR}/thrust_operators.cuh
  ${CMAKE_CURRENT_LIST_DIR}/tgv_kernel.cu
  ${CMAKE_CURRENT_LIST_DIR}/Image.cu
  )

cuda_add_executable(image_tests
  ${CMAKE_CURRENT_LIST_DIR}/image_tests_console.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Image.cuh
  ${CMAKE_CURRENT_LIST_DIR}/Image.cu
  ${CMAKE_CURRENT_LIST_DIR}/thrust_operators.cuh
)
