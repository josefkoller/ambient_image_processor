cmake_minimum_required(VERSION 2.8.11)

list(APPEND SOURCE_H_FILES
  ${CMAKE_CURRENT_LIST_DIR}/ShrinkWidget.h
  ${CMAKE_CURRENT_LIST_DIR}/ShrinkProcessor.h
  )

list(APPEND SOURCE_UI_FILES
  ${CMAKE_CURRENT_LIST_DIR}/ShrinkWidget.ui
  )

list(APPEND SOURCE_CPP_FILES
  ${CMAKE_CURRENT_LIST_DIR}/ShrinkWidget.cpp
  ${CMAKE_CURRENT_LIST_DIR}/ShrinkProcessor.cpp
  )

include_directories(${CMAKE_CURRENT_LIST_DIR})
