cmake_minimum_required(VERSION 2.8.11)

list(APPEND SOURCE_H_FILES
  ${CMAKE_CURRENT_LIST_DIR}/RegionCurvatureEdgeCorrection.h
  ${CMAKE_CURRENT_LIST_DIR}/RegionCurvatureEdgeCorrectionProcessor.h
  ${CMAKE_CURRENT_LIST_DIR}/Vector3.h
  )

list(APPEND SOURCE_UI_FILES
  ${CMAKE_CURRENT_LIST_DIR}/RegionCurvatureEdgeCorrection.ui
  )

list(APPEND SOURCE_CPP_FILES
  ${CMAKE_CURRENT_LIST_DIR}/RegionCurvatureEdgeCorrection.cpp
  ${CMAKE_CURRENT_LIST_DIR}/RegionCurvatureEdgeCorrectionProcessor.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Vector3.cpp
  )

include_directories(${CMAKE_CURRENT_LIST_DIR})
