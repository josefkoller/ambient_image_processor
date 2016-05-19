cmake_minimum_required(VERSION 2.8.11)

list(APPEND SOURCE_H_FILES
  ${CMAKE_CURRENT_LIST_DIR}/ImageInformationWidget.h
  ${CMAKE_CURRENT_LIST_DIR}/ImageInformationProcessor.h
  )

list(APPEND SOURCE_UI_FILES
  ${CMAKE_CURRENT_LIST_DIR}/ImageInformationWidget.ui
  )

list(APPEND SOURCE_CPP_FILES
  ${CMAKE_CURRENT_LIST_DIR}/ImageInformationWidget.cpp
  ${CMAKE_CURRENT_LIST_DIR}/ImageInformationProcessor.cpp
  )

include_directories(${CMAKE_CURRENT_LIST_DIR})
