cmake_minimum_required(VERSION 2.8.11)

cuda_add_executable(tgv2_deshade_application
  ${CMAKE_CURRENT_LIST_DIR}/tgv2_deshade_main.cpp
)
target_link_libraries(tgv2_deshade_application
  ambient_library
)
list(APPEND ITK_APPLICATIONS
  tgv2_deshade_application
  )

cuda_add_executable(tgvk_deshade_downsampled_application
  ${CMAKE_CURRENT_LIST_DIR}/tgvk_deshade_downsampled_main.cpp
)
target_link_libraries(tgvk_deshade_downsampled_application
  ambient_library
)
list(APPEND ITK_APPLICATIONS
  tgvk_deshade_downsampled_application
  )

cuda_add_executable(tgv2_deshade_convergence_test_application
  ${CMAKE_CURRENT_LIST_DIR}/tgv2_deshade_convergence_test_main.cpp
)
target_link_libraries(tgv2_deshade_convergence_test_application
  ambient_library
)
list(APPEND ITK_APPLICATIONS
  tgv2_deshade_application
  tgv2_deshade_convergence_test_application
  )


include(${CMAKE_CURRENT_LIST_DIR}/image_metrics/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/sqlite_to_gnuplot/CMakeLists.txt)

