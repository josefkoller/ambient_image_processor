cmake_minimum_required(VERSION 2.8.11)

list(APPEND SOURCE_H_FILES
  ${CMAKE_CURRENT_LIST_DIR}/ExtractWidget.h
  ${CMAKE_CURRENT_LIST_DIR}/ExtractProcessor.h
  )

list(APPEND SOURCE_UI_FILES
  ${CMAKE_CURRENT_LIST_DIR}/ExtractWidget.ui
  )

list(APPEND SOURCE_CPP_FILES
  ${CMAKE_CURRENT_LIST_DIR}/ExtractWidget.cpp
  ${CMAKE_CURRENT_LIST_DIR}/ExtractProcessor.cpp
  )

include_directories(${CMAKE_CURRENT_LIST_DIR})
