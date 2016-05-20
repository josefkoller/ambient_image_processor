cmake_minimum_required(VERSION 2.8.11)

list(APPEND SOURCE_H_FILES
  ${CMAKE_CURRENT_LIST_DIR}/ThresholdFilterWidget.h
  ${CMAKE_CURRENT_LIST_DIR}/ThresholdFilterProcessor.h
  )

list(APPEND SOURCE_UI_FILES
  ${CMAKE_CURRENT_LIST_DIR}/ThresholdFilterWidget.ui
  )

list(APPEND SOURCE_CPP_FILES
  ${CMAKE_CURRENT_LIST_DIR}/ThresholdFilterWidget.cpp
  ${CMAKE_CURRENT_LIST_DIR}/ThresholdFilterProcessor.cpp
  )

include_directories(${CMAKE_CURRENT_LIST_DIR})
