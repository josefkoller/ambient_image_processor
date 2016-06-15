cmake_minimum_required(VERSION 2.8.11)
  
list(APPEND SOURCE_H_FILES
   ${CMAKE_CURRENT_LIST_DIR}/TGVDeshadeIntegralMetricPlotWidget.h
   ${CMAKE_CURRENT_LIST_DIR}/TGVDeshadeIntegralMetricPlotProcessor.h
   ${CMAKE_CURRENT_LIST_DIR}/ParameterValueSet.h

  #  ${CMAKE_CURRENT_LIST_DIR}/cosine_transform.h
  )

list(APPEND SOURCE_UI_FILES
  ${CMAKE_CURRENT_LIST_DIR}/TGVDeshadeIntegralMetricPlotWidget.ui
   ${CMAKE_CURRENT_LIST_DIR}/ParameterValueSet.ui
  )

list(APPEND SOURCE_CPP_FILES
   ${CMAKE_CURRENT_LIST_DIR}/TGVDeshadeIntegralMetricPlotWidget.cpp
   ${CMAKE_CURRENT_LIST_DIR}/TGVDeshadeIntegralMetricPlotProcessor.cpp
   ${CMAKE_CURRENT_LIST_DIR}/ParameterValueSet.cpp
  )

list(APPEND CUDA_CPP_FILES
    ${CMAKE_CURRENT_LIST_DIR}/tgv2_l1_deshaded_integral_metric_plot.cu
  )

include_directories(${CMAKE_CURRENT_LIST_DIR})
