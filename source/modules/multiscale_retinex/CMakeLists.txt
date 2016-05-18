cmake_minimum_required(VERSION 2.8.11)

list(APPEND SOURCE_H_FILES
  ${CMAKE_CURRENT_LIST_DIR}/ShrinkFunctor.h
  ${CMAKE_CURRENT_LIST_DIR}/ShrinkFilter.h
  ${CMAKE_CURRENT_LIST_DIR}/MultiScaleRetinexScale.h
  ${CMAKE_CURRENT_LIST_DIR}/MultiScaleRetinex.h
  ${CMAKE_CURRENT_LIST_DIR}/MultiScaleRetinexWidget.h
  ${CMAKE_CURRENT_LIST_DIR}/MultiScaleRetinexProcessor.h
  )

list(APPEND SOURCE_UI_FILES
  ${CMAKE_CURRENT_LIST_DIR}/MultiScaleRetinexWidget.ui
  ${CMAKE_CURRENT_LIST_DIR}/MultiScaleRetinexScale.ui
  )

list(APPEND SOURCE_CPP_FILES
  ${CMAKE_CURRENT_LIST_DIR}/ShrinkFunctor.cpp
  ${CMAKE_CURRENT_LIST_DIR}/ShrinkFilter.cpp
  ${CMAKE_CURRENT_LIST_DIR}/MultiScaleRetinexScale.cpp
  ${CMAKE_CURRENT_LIST_DIR}/MultiScaleRetinex.cpp
  ${CMAKE_CURRENT_LIST_DIR}/MultiScaleRetinexWidget.cpp
  ${CMAKE_CURRENT_LIST_DIR}/MultiScaleRetinexProcessor.cpp
  )

include_directories(${CMAKE_CURRENT_LIST_DIR})
