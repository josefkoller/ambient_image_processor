project(ambient)

cmake_minimum_required(VERSION 2.8.11)


#set(CMAKE_CXX_FLAGS "-std=c++11 -fpermissive ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/output)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/output)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/output)

find_package(CUDA REQUIRED)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

SET( CUDA_PROPAGATE_HOST_FLAGS OFF )
set(CUDA_NVCC_FLAGS "--std=c++11")
#set(CUDA_NVCC_FLAGS "--std=c++11;--gpu-architecture=compute_50;--gpu-code=sm_50")
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_50,code=sm_50")
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_VERBOSE_BUILD ON)

add_definitions("-DCUDA_BLOCK_DIMENSION=1024")

#enable openmp
#set(CMAKE_CXX_FLAGS "-fopenmp -lgomp ${CMAKE_CXX_FLAGS}")

find_package(Qt5Core REQUIRED)
include_directories(${Qt5Core_INCLUDE_DIRS})
find_package(Qt5Widgets REQUIRED)
include_directories(${Qt5Widgets_INCLUDE_DIRS})
# Add compiler flags for building executables (-fPIE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Qt5Widgets_EXECUTABLE_COMPILE_FLAGS}")

include_directories(source)

SET(SOURCE_UI_FILES
  source/MainWindow.ui
)

SET(SOURCE_CPP_FILES
  source/MainWindow.cpp
)

SET(SOURCE_H_FILES
  source/MainWindow.h
)

include(source/modules/itk/CMakeLists.txt)
include(source/modules/image_widget/CMakeLists.txt)
include(source/modules/base_module/CMakeLists.txt)
include(source/modules/itk/CMakeLists.txt)
include(source/modules/cuda/CMakeLists.txt)
include(source/modules/non_local_gradient/CMakeLists.txt)
include(source/modules/thrust/CMakeLists.txt)
include(source/modules/region_growing_segmentation/CMakeLists.txt)
include(source/modules/multiscale_retinex/CMakeLists.txt)
include(source/modules/deshade_segmented/CMakeLists.txt)
include(source/modules/line_profile/CMakeLists.txt)
include(source/modules/unsharp_masking/CMakeLists.txt)
include(source/modules/image_information/CMakeLists.txt)
include(source/modules/histogram/CMakeLists.txt)
include(source/modules/spline_interpolation/CMakeLists.txt)
include(source/modules/shrink/CMakeLists.txt)
include(source/modules/extract/CMakeLists.txt)
include(source/modules/bilateral_filter/CMakeLists.txt)
include(source/modules/threshold/CMakeLists.txt)
include(source/modules/crosshair/CMakeLists.txt)
include(source/modules/slice_control/CMakeLists.txt)
include(source/modules/tgv_cuda/CMakeLists.txt)
include(source/modules/tgv_thrust/CMakeLists.txt)
include(source/modules/tgv/CMakeLists.txt)
include(source/modules/image_view/CMakeLists.txt)
include(source/modules/tgvl1_threshold_gradient/CMakeLists.txt)
include(source/modules/manual_multiplicative_deshade/CMakeLists.txt)
include(source/modules/cuda_image_operations/CMakeLists.txt)
include(source/modules/tgv_lambdas/CMakeLists.txt)
include(source/modules/binary_operations/CMakeLists.txt)
include(source/modules/convolution/CMakeLists.txt)
include(source/modules/region_curvature_edge_correction/CMakeLists.txt)
include(source/modules/rescale_intensity/CMakeLists.txt)
include(source/modules/tgv_shading_growing/CMakeLists.txt)
include(source/modules/tgv_deshade/CMakeLists.txt)
include(source/modules/unary_operations/CMakeLists.txt)
include(source/modules/morphological_filter/CMakeLists.txt)
include(source/modules/opencv_file/CMakeLists.txt)
include(source/modules/image_view_control/CMakeLists.txt)
include(source/modules/tgv_non_parametric_deshade/CMakeLists.txt)
include(source/modules/tgv_deshade_metric_plot/CMakeLists.txt)
include(source/modules/tgv_deshade_integral_metric_plot/CMakeLists.txt)
include(source/modules/conjugate_gradient/CMakeLists.txt)
include(source/modules/tgv3/CMakeLists.txt)
include(source/modules/tgv3_deshade/CMakeLists.txt)
include(source/modules/tgvk/CMakeLists.txt)
include(source/modules/tgvk_deshade/CMakeLists.txt)
include(source/modules/tgvk_deshade_convergence/CMakeLists.txt)
include(source/modules/resize/CMakeLists.txt)

QT5_WRAP_UI(SOURCE_UID_FILES ${SOURCE_UI_FILES})

SET(SOURCE_ALL_FILES
  ${SOURCE_UI_FILES}
  ${SOURCE_CPP_FILES}
  ${SOURCE_H_FILES}
)

message(STATUS "source files: ${SOURCE_ALL_FILES}")

SET(ITK_APPLICATIONS
  ambient_application
)
cuda_add_executable(ambient_application
  source/ambient_main.cpp
  ${SOURCE_ALL_FILES}
  ${CUDA_CPP_FILES}
)

target_link_libraries(ambient_application
  Qt5::Core
  Qt5::Widgets
 # -lfftw3
)

#include(source/terminal_programs/CMakeLists.txt)

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})  # import itk after cuda_add_executable, else an invalid nvcc flag is set
add_definitions("-DVCL_CAN_STATIC_CONST_INIT_FLOAT=0")
add_definitions("-DVCL_NEEDS_INLINE_INSTANTIATION=0")

find_package( OpenCV REQUIRED )

foreach(itk_application IN LISTS ITK_APPLICATIONS)
    target_link_libraries(${itk_application}
      ${ITK_LIBRARIES}
      ${OpenCV_LIBS}
    )
endforeach(itk_application)
