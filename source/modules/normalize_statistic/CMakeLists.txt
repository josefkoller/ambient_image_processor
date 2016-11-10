cmake_minimum_required(VERSION 2.8.11)

list(APPEND SOURCE_H_FILES
  ${CMAKE_CURRENT_LIST_DIR}/NormalizeStatisticWidget.h
  ${CMAKE_CURRENT_LIST_DIR}/NormalizeStatisticProcessor.h
  )

list(APPEND SOURCE_UI_FILES
  ${CMAKE_CURRENT_LIST_DIR}/NormalizeStatisticWidget.ui
  )

list(APPEND SOURCE_CPP_FILES
  ${CMAKE_CURRENT_LIST_DIR}/NormalizeStatisticWidget.cpp
  ${CMAKE_CURRENT_LIST_DIR}/NormalizeStatisticProcessor.cpp
  )

include_directories(${CMAKE_CURRENT_LIST_DIR})
