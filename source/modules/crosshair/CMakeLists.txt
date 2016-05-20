cmake_minimum_required(VERSION 2.8.11)

list(APPEND SOURCE_H_FILES
  ${CMAKE_CURRENT_LIST_DIR}/CrosshairModule.h
  )

list(APPEND SOURCE_UI_FILES
  )

list(APPEND SOURCE_CPP_FILES
  ${CMAKE_CURRENT_LIST_DIR}/CrosshairModule.cpp
  )

include_directories(${CMAKE_CURRENT_LIST_DIR})
