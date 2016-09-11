cmake_minimum_required(VERSION 2.8.11)

cuda_add_executable(sqlite_to_gnuplot
  ${CMAKE_CURRENT_LIST_DIR}/sqlite_to_gnuplot_main.cpp
)

target_link_libraries(sqlite_to_gnuplot
  libsqlite3.so
  ambient_library
)

list(APPEND ITK_APPLICATIONS
  sqlite_to_gnuplot
)
