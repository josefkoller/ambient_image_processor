cmake_minimum_required(VERSION 2.8.11)

list(APPEND SOURCE_H_FILES
  ${CMAKE_CURRENT_LIST_DIR}/UnsharpMaskingWidget.h
  ${CMAKE_CURRENT_LIST_DIR}/UnsharpMaskingProcessor.h
  )

list(APPEND SOURCE_UI_FILES
  ${CMAKE_CURRENT_LIST_DIR}/UnsharpMaskingWidget.ui
  )

list(APPEND SOURCE_CPP_FILES
  ${CMAKE_CURRENT_LIST_DIR}/UnsharpMaskingWidget.cpp
  ${CMAKE_CURRENT_LIST_DIR}/UnsharpMaskingProcessor.cpp
  )

include_directories(${CMAKE_CURRENT_LIST_DIR})
