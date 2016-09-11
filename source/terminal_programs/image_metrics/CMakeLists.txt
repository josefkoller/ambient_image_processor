cmake_minimum_required(VERSION 2.8.11)

cuda_add_executable(image_metric_to_sqlite
  ${CMAKE_CURRENT_LIST_DIR}/image_metric_to_sqlite_main.cpp
)

target_link_libraries(image_metric_to_sqlite
  libsqlite3.so
  ambient_library
)

list(APPEND ITK_APPLICATIONS
  image_metric_to_sqlite
)
