project(ambient)

cmake_minimum_required(VERSION 2.8.11)

set(CMAKE_CXX_FLAGS "-std=c++11 -fpermissive ${CMAKE_CXX_FLAGS}")
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/output)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/output)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/output)

find_package(CUDA REQUIRED)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

SET( CUDA_PROPAGATE_HOST_FLAGS OFF )
set(CUDA_NVCC_FLAGS "--std=c++11;--gpu-architecture=compute_50;--gpu-code=sm_50")
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_50,code=sm_50")
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_VERBOSE_BUILD ON)

#enable openmp
#set(CMAKE_CXX_FLAGS "-std=c++11 -fpermissive -fopenmp -lgomp ${CMAKE_CXX_FLAGS}")

find_package(Qt5Core REQUIRED)
include_directories(${Qt5Core_INCLUDE_DIRS})
find_package(Qt5Widgets REQUIRED)
include_directories(${Qt5Widgets_INCLUDE_DIRS})
# Add compiler flags for building executables (-fPIE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Qt5Widgets_EXECUTABLE_COMPILE_FLAGS}")

include_directories(source)

SET(SOURCE_UI_FILES
  source/MainWindow.ui
  source/ImageWidget.ui
  source/segmentation/RegionGrowingSegmentationWidget.ui
  source/retinex/MultiScaleRetinexScale.ui
)

SET(SOURCE_CPP_FILES
  source/MainWindow.cpp
  source/ITKToQImageConverter.cpp
  source/ITKImageProcessor.cpp
  source/ImageWidget.cpp
  source/qcustomplot/qcustomplot.cpp
  source/cuda/CudaImageProcessor.cpp
  source/cuda/ITKRawImageConverter.cpp
  source/cuda/RawImage.cpp
  source/retinex/ShrinkFunctor.cpp
  source/retinex/ShrinkFilter.cpp
  source/segmentation/RegionGrowingSegmentation.cpp
  source/segmentation/RegionGrowingSegmentationWidget.cpp
  source/segmentation/SegmentsToLabelImageConverter.cpp
  source/segmentation/RegionGrowingSegmentationProcessor.cpp
  source/retinex/MultiScaleRetinexScale.cpp
  source/retinex/MultiScaleRetinex.cpp
)

SET(SOURCE_H_FILES
  source/MainWindow.h
  source/ITKToQImageConverter.h
  source/ITKImageProcessor.h
  source/ImageWidget.h
  source/qcustomplot/qcustomplot.h
  source/cuda/CudaImageProcessor.h
  source/cuda/ITKRawImageConverter.h
  source/cuda/RawImage.h
  source/retinex/ShrinkFunctor.h
  source/retinex/ShrinkFilter.h
  source/segmentation/RegionGrowingSegmentation.h
  source/segmentation/RegionGrowingSegmentationWidget.h
  source/segmentation/SegmentsToLabelImageConverter.h
  source/segmentation/RegionGrowingSegmentationProcessor.h
  source/retinex/MultiScaleRetinexScale.h
  source/retinex/MultiScaleRetinex.h
)

include(source/non_local_gradient/CMakeLists.txt)
include(source/deshade_segmented/CMakeLists.txt)
include(source/thrust_tgv/CMakeLists.txt)

QT5_WRAP_UI(SOURCE_UID_FILES ${SOURCE_UI_FILES})

SET(SOURCE_ALL_FILES
  ${SOURCE_UI_FILES}
  ${SOURCE_CPP_FILES}
  ${SOURCE_H_FILES}
)

message(STATUS "source files: ${SOURCE_ALL_FILES}")

cuda_add_executable(ambient_application
  source/ambient_main.cpp
#  source/cuda/CudaImage.cuh
  ${SOURCE_ALL_FILES}
  ${CUDA_CPP_FILES}

  source/cuda/cuda_kernels.cu
)

add_executable(linear_retinex_console
  source/linear_retinex_console.cpp
  source/ITKImageProcessor.h
  source/ITKImageProcessor.cpp
  source/retinex/ShrinkFilter.h
  source/retinex/ShrinkFilter.cpp
  source/retinex/ShrinkFunctor.h
  source/retinex/ShrinkFunctor.cpp
)

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

target_link_libraries(ambient_application
  Qt5::Core
  Qt5::Widgets
  ${ITK_LIBRARIES}
  -lfftw3f
)

target_link_libraries(linear_retinex_console
    ${ITK_LIBRARIES}
)
