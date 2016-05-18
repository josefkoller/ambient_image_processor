cmake_minimum_required(VERSION 2.8.11)

list(APPEND SOURCE_H_FILES
  ${CMAKE_CURRENT_LIST_DIR}/RegionGrowingSegmentation.h
  ${CMAKE_CURRENT_LIST_DIR}/RegionGrowingSegmentationWidget.h
  ${CMAKE_CURRENT_LIST_DIR}/SegmentsToLabelImageConverter.h
  ${CMAKE_CURRENT_LIST_DIR}/RegionGrowingSegmentationProcessor.h
  )

list(APPEND SOURCE_UI_FILES
  ${CMAKE_CURRENT_LIST_DIR}/RegionGrowingSegmentationWidget.ui
  )

list(APPEND SOURCE_CPP_FILES
  ${CMAKE_CURRENT_LIST_DIR}/RegionGrowingSegmentation.cpp
  ${CMAKE_CURRENT_LIST_DIR}/RegionGrowingSegmentationWidget.cpp
  ${CMAKE_CURRENT_LIST_DIR}/SegmentsToLabelImageConverter.cpp
  ${CMAKE_CURRENT_LIST_DIR}/RegionGrowingSegmentationProcessor.cpp
  )

include_directories(${CMAKE_CURRENT_LIST_DIR})
