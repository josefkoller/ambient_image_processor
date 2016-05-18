cmake_minimum_required(VERSION 2.8.11)

list(APPEND SOURCE_H_FILES
  ${CMAKE_CURRENT_LIST_DIR}/ImageWidget.h
  ${CMAKE_CURRENT_LIST_DIR}/ITKToQImageConverter.h
  ${CMAKE_CURRENT_LIST_DIR}/ITKImageProcessor.h
  ${CMAKE_CURRENT_LIST_DIR}/ImageWidget.h
  ${CMAKE_CURRENT_LIST_DIR}/qcustomplot/qcustomplot.h
  )

list(APPEND SOURCE_UI_FILES
  ${CMAKE_CURRENT_LIST_DIR}/ImageWidget.ui
  )

list(APPEND SOURCE_CPP_FILES
  ${CMAKE_CURRENT_LIST_DIR}/ImageWidget.cpp
  ${CMAKE_CURRENT_LIST_DIR}/ITKToQImageConverter.cpp
  ${CMAKE_CURRENT_LIST_DIR}/ITKImageProcessor.cpp
  ${CMAKE_CURRENT_LIST_DIR}/ImageWidget.cpp
  ${CMAKE_CURRENT_LIST_DIR}/qcustomplot/qcustomplot.cpp
  )

include_directories(${CMAKE_CURRENT_LIST_DIR})
