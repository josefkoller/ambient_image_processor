cmake_minimum_required(VERSION 2.8.11)

list(APPEND SOURCE_H_FILES
  ${CMAKE_CURRENT_LIST_DIR}/LineProfile.h
  ${CMAKE_CURRENT_LIST_DIR}/LineProfileWidget.h
  ${CMAKE_CURRENT_LIST_DIR}/LineProfileProcessor.h
  )

list(APPEND SOURCE_UI_FILES
  ${CMAKE_CURRENT_LIST_DIR}/LineProfileWidget.ui
  )

list(APPEND SOURCE_CPP_FILES
  ${CMAKE_CURRENT_LIST_DIR}/LineProfileWidget.cpp
  ${CMAKE_CURRENT_LIST_DIR}/LineProfileProcessor.cpp
  )

include_directories(${CMAKE_CURRENT_LIST_DIR})
