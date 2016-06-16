
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sqlite3.h>

typedef std::string String;
typedef sqlite3* Database;

void perform(int argc, char *argv[]);

int main(int argc, char *argv[])
{
    std::cout << "started program: " << argv[0] << std::endl;
    if(argc > 3)
    {
        std::cout << "database file path: " << argv[1] << std::endl;
        std::cout << "output file path: " << argv[2] << std::endl;
    }
    else
    {
        std::cout << "Usage: database_file_path output_file_path column1 [column2] [column3] ..." << std::endl;
        return 1;
    }
    perform(argc, argv);
    std::cout << "finished program: " << argv[0] << std::endl;
    return 0;
}

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

void writeHeader(int argc, char *argv[], const uint offset, std::ofstream& stream)
{
    const uint column_count = argc - offset;
    stream << "# ";
    for(int i = 0; i < column_count; i++)
    {
        const char* column_name = argv[i + offset];
        stream << column_name << " ";

        std::cout << "column" << i << ": " << column_name << std::endl;
    }
    stream << std::endl;
}

void writeData(const char* database_file_path, int argc, char *argv[], const uint offset, std::ofstream& stream)
{
    Database database = openDatabase(database_file_path);

    // select data
    const uint column_count = argc - offset;
    std::string select_command = "select ";
    for(int i = 0; i < column_count; i++)
    {
        const char* column_name = argv[i + offset];
        select_command += column_name;
        if(i < column_count - 1)
            select_command += ",";
    }
    select_command += " from run";

    char command[512];
    sprintf(command, select_command.c_str());
    sqlite3_stmt* statement;
    if(sqlite3_prepare(database, command, -1, &statement, NULL) != SQLITE_OK)
    {
       std::cout << "  SQL error: " << sqlite3_errmsg(database) << std::endl;
       std::cout << "  SQL command: " << command << std::endl;
       exit(1);
    }

    // write data
    while(sqlite3_step(statement) == SQLITE_ROW)
    {
        for(int i = 0; i < column_count; i++)
        {
            double value = sqlite3_column_double(statement, i);
            stream << value;
            if(i < column_count - 1)
                stream << " ";
        }
        stream << std::endl;
    }

    sqlite3_close(database);
}

void perform(int argc, char *argv[])
{
  const char* database_file_path = argv[1];
  const char* output_file_path = argv[2];

  const uint offset = 3;

  std::ofstream stream;
  stream.open(output_file_path);

  writeHeader(argc, argv, offset, stream);
  writeData(database_file_path, argc, argv, offset, stream);

  stream.close();
}
