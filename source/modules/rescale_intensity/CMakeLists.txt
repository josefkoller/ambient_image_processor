cmake_minimum_required(VERSION 2.8.11)

list(APPEND SOURCE_H_FILES
    ${CMAKE_CURRENT_LIST_DIR}/RescaleIntensityWidget.h
    ${CMAKE_CURRENT_LIST_DIR}/RescaleIntensityProcessor.h
  )

list(APPEND SOURCE_UI_FILES
    ${CMAKE_CURRENT_LIST_DIR}/RescaleIntensityWidget.ui
  )

list(APPEND SOURCE_CPP_FILES
    ${CMAKE_CURRENT_LIST_DIR}/RescaleIntensityWidget.cpp
    ${CMAKE_CURRENT_LIST_DIR}/RescaleIntensityProcessor.cpp
  )

include_directories(${CMAKE_CURRENT_LIST_DIR})
