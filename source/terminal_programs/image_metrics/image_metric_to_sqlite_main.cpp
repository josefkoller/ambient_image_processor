
#include <iostream>
#include <string>
typedef std::string String;

void perform(const char* database_file_path);
int main(int argc, char *argv[])
{
    std::cout << "started program: " << argv[0] << std::endl;
    if(argc > 1)
    {
        std::cout << "database file path: " << argv[1] << std::endl;
    }
    else
    {
        std::cout << "Usage: database_file_path as the only argument" << std::endl;
        return 1;
    }
    perform(argv[1]);
    std::cout << "finished program: " << argv[0] << std::endl;
    return 0;
}

typedef const unsigned char* SQLString;
struct RunParameter
{
  int id;
  String denoised_image_path;
  String deshaded_image_path;
  String shading_image_path;
  RunParameter(int id, String denoised_image_path, String deshaded_image_path, String shading_image_path)
    : id(id), denoised_image_path(denoised_image_path), deshaded_image_path(deshaded_image_path), shading_image_path(shading_image_path) {}
};

#include <vector>
#include <sqlite3.h>

typedef sqlite3* Database;


Database openDatabase(const char* database_file_path)
{
  sqlite3* database;
  if(sqlite3_open(database_file_path, &database) != SQLITE_OK)
  {
    std::cout << "SQLite ERROR: " << sqlite3_errmsg(database) << std::endl;
    return nullptr;
  }
  return database;
}

std::vector<RunParameter> readRunParameters(Database database)
{
  std::vector<RunParameter> run_parameters;
  std::string query = "select id, output_denoised_file, output_deshaded_file, output_shading_file from run";
  sqlite3_stmt* statement;
  sqlite3_prepare(database, query.c_str(), -1, &statement, NULL);
  while(sqlite3_step(statement) != SQLITE_DONE)
  {
    int id = sqlite3_column_int(statement, 0);
    SQLString denoised_image_path = sqlite3_column_text(statement, 1);
    SQLString deshaded_image_path = sqlite3_column_text(statement, 2);
    SQLString shading_image_path = sqlite3_column_text(statement, 3);
    run_parameters.push_back(RunParameter(
      id,
      reinterpret_cast<const char*>(denoised_image_path),
      reinterpret_cast<const char*>(deshaded_image_path),
      reinterpret_cast<const char*>(shading_image_path) ));
  }
  return run_parameters;
}

struct RunMetrics
{
  float denoised_coefficient_of_variation = 0;
  float denoised_mean_total_variation = 0;
  float deshaded_coefficient_of_variation = 0;
  float deshaded_mean_total_variation = 0;
};

void write_metrics_to_database(RunMetrics run, int run_id, Database database)
{
  char command[512];
  sprintf(command, "update run set denoised_coefficient_of_variation=%f,denoised_mean_total_variation=%f,deshaded_coefficient_of_variation=%f,deshaded_mean_total_variation=%f where id=%d",
  run.denoised_coefficient_of_variation,
  run.denoised_mean_total_variation,
  run.deshaded_coefficient_of_variation,
  run_id);
  sqlite3_stmt* statement;
  if(sqlite3_prepare(database, command, -1, &statement, NULL) != SQLITE_OK)
  {
     std::cout << "  SQL error: " << sqlite3_errmsg(database) << std::endl;
     exit(1);
  }
  sqlite3_step(statement);
}

#include "ITKImage.h"
#include "ImageInformationProcessor.h"
#include "CudaImageOperationsProcessor.h"
RunMetrics calculate_metrics(RunParameter parameter)
{
  RunMetrics metrics;

  auto denoised_image = ITKImage::read(parameter.denoised_image_path);
  metrics.denoised_mean_total_variation = CudaImageOperationsProcessor::tv(denoised_image) / denoised_image.voxel_count;
  metrics.denoised_coefficient_of_variation = ImageInformationProcessor::coefficient_of_variation(denoised_image);

  auto deshaded_image = ITKImage::read(parameter.deshaded_image_path);
  metrics.deshaded_mean_total_variation = CudaImageOperationsProcessor::tv(deshaded_image) / deshaded_image.voxel_count;
  metrics.deshaded_coefficient_of_variation = ImageInformationProcessor::coefficient_of_variation(deshaded_image);

  return metrics;
}


void perform(const char* database_file_path)
{
  Database database = openDatabase(database_file_path);
  std::vector<RunParameter> run_parameters = readRunParameters(database);
  for(RunParameter parameter : run_parameters)
  {
    RunMetrics metrics = calculate_metrics(parameter);
    write_metrics_to_database(metrics, parameter.id, database);
  }
  sqlite3_close(database);
}
