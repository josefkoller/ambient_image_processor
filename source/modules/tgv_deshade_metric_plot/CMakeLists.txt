cmake_minimum_required(VERSION 2.8.11)
  
list(APPEND SOURCE_H_FILES
      ${CMAKE_CURRENT_LIST_DIR}/TGVDeshadeMetricPlotWidget.h
   ${CMAKE_CURRENT_LIST_DIR}/TGVDeshadeMetricPlotProcessor.h
  #  ${CMAKE_CURRENT_LIST_DIR}/cosine_transform.h
  )

list(APPEND SOURCE_UI_FILES
  ${CMAKE_CURRENT_LIST_DIR}/TGVDeshadeMetricPlotWidget.ui
  )

list(APPEND SOURCE_CPP_FILES
   ${CMAKE_CURRENT_LIST_DIR}/TGVDeshadeMetricPlotWidget.cpp
    ${CMAKE_CURRENT_LIST_DIR}/TGVDeshadeMetricPlotProcessor.cpp
  )

list(APPEND CUDA_CPP_FILES
    ${CMAKE_CURRENT_LIST_DIR}/tgv2_l1_deshaded_metric_plot.cu
  )

include_directories(${CMAKE_CURRENT_LIST_DIR})
