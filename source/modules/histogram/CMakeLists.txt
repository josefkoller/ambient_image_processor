cmake_minimum_required(VERSION 2.8.11)

list(APPEND SOURCE_H_FILES
  ${CMAKE_CURRENT_LIST_DIR}/HistogramWidget.h
  ${CMAKE_CURRENT_LIST_DIR}/HistogramProcessor.h
  )

list(APPEND SOURCE_UI_FILES
  ${CMAKE_CURRENT_LIST_DIR}/HistogramWidget.ui
  )

list(APPEND SOURCE_CPP_FILES
  ${CMAKE_CURRENT_LIST_DIR}/HistogramWidget.cpp
  ${CMAKE_CURRENT_LIST_DIR}/HistogramProcessor.cpp
  )

include_directories(${CMAKE_CURRENT_LIST_DIR})
