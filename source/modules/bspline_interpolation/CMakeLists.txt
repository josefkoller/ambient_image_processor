cmake_minimum_required(VERSION 2.8.11)

list(APPEND SOURCE_H_FILES
  ${CMAKE_CURRENT_LIST_DIR}/BSplineInterpolationWidget.h
  ${CMAKE_CURRENT_LIST_DIR}/BSplineInterpolationProcessor.h
  )

list(APPEND SOURCE_UI_FILES
  ${CMAKE_CURRENT_LIST_DIR}/BSplineInterpolationWidget.ui
  )

list(APPEND SOURCE_CPP_FILES
  ${CMAKE_CURRENT_LIST_DIR}/BSplineInterpolationWidget.cpp
  ${CMAKE_CURRENT_LIST_DIR}/BSplineInterpolationProcessor.cpp
  )

include_directories(${CMAKE_CURRENT_LIST_DIR})
