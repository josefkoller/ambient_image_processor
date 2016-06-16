# # Gemfile for linear_retinex_to_sql tasks

require 'colorize'

def run_specific configuration_file=""
  if configuration_file == "" and ARGV.length == 0 
    puts "specify a configuration file"
    return
  end

  if configuration_file == ""
    configuration_file = ARGV[0]
  end

  puts "configuration: #{configuration_file}"
  if File.exists? configuration_file
    load configuration_file
    parameter = read_parameter
    
    require 'fileutils.rb'
    FileUtils.cp configuration_file, File.join(parameter[:output_path], "parameter.rb" )
    run parameter
  end
end
def run parameter
  create_sql_database parameter
  optimize parameter
end

def step parameter, key
    if parameter[key][:bind_to]
      parameter[key][:value] = parameter[ parameter[key][:bind_to]][:value]
      return
    end
    if parameter[key][:step_factor]
      parameter[key][:value] *= parameter[key][:step_factor]
      return
    end
    if parameter[key][:count] == 1
      return
    end
    step_size = ( parameter[key][:max] - parameter[key][:min] ) / ( parameter[key][:count] - 1.0 )
    parameter[key][:value] += step_size
end

def optimize parameter
  return unless File.exists? parameter[:output_sql_file_path]
  require 'sqlite3'
  database = SQLite3::Database.new parameter[:output_sql_file_path]
  begin
    output_file_id = 1

    parameter[:alpha0][:count] = 1 if parameter[:alpha0][:bind_to]

    parameter[:alpha1][:value] = parameter[:alpha1][:min]
    parameter[:alpha1][:count].times do |alpha1_index|
      step parameter, :alpha1 if alpha1_index > 0

      parameter[:alpha0][:value] = parameter[:alpha0][:min]
      parameter[:alpha0][:count].times do |alpha0_index|
        step parameter, :alpha0 if alpha0_index > 0 or parameter[:alpha0][:bind_to]
        forward_model database, output_file_id, parameter
        output_file_id += 1
      end
    end
  end
end

def run_program command
  puts command.yellow

  require 'timeout'
  #one day ... maximum
  Timeout::timeout(60*60*24) {
    passed = system command
    throw "System command failed" unless passed
  }
end
def forward_model database, output_file_id, parameter
  command = "#{parameter[:program_path]} #{parameter[:input_image_path]}"
  command += " #{parameter[:lambda]}"
  command += " #{parameter[:alpha0][:value]}"
  command += " #{parameter[:alpha1][:value]}"
  command += " #{parameter[:iteration_count]}"
  command += " #{parameter[:input_mask_path]}"
  file_sulfix = "%04d.#{parameter[:output_format]}" % output_file_id
  output_denoised_file = "#{parameter[:output_denoised_file_path]}_#{file_sulfix}"
  output_shading_file = "#{parameter[:output_shading_file_path]}_#{file_sulfix}"
  output_deshaded_file = "#{parameter[:output_deshaded_file_path]}_#{file_sulfix}"
  command += " #{output_denoised_file}"
  command += " #{output_shading_file}"
  command += " #{output_deshaded_file}"
  begin
    run_program command
    insert_run database, output_file_id, true, output_denoised_file, output_shading_file, output_deshaded_file, parameter
    puts "run #{"%04d" % output_file_id} passed".green
  rescue
    insert_run database, output_file_id, false, output_denoised_file, output_shading_file, output_deshaded_file, parameter
    puts "run #{"%04d" % output_file_id} failed with parameter: alpha0=#{alpha0}, alpha1=#{alpha1}"
    puts "command #{command}".red
  end
end
def insert_run database, output_file_id, passed, output_denoised_file, output_shading_file, output_deshaded_file, parameter
  command = ""
  command += "insert into run (input_file, id, alpha0, alpha1, "
  command += "lambda, iteration_count, passed, output_denoised_file, output_shading_file, output_deshaded_file) values("
  command += "?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"

  values = [parameter[:input_image_path],
            output_file_id.to_s,
            parameter[:alpha0][:value].to_s,
            parameter[:alpha1][:value].to_s,
            parameter[:lambda].to_s,
            parameter[:iteration_count].to_s,
            passed.to_s,
            output_denoised_file,
            output_shading_file,
            output_deshaded_file]
  begin
    database.execute command, values
  rescue
    puts "problem while inserting data:"
    puts "executed command: #{command}"
  end
end
def create_sql_database parameter
  target_sql_file_path = parameter[:output_sql_file_path]
  require 'sqlite3'
  File.delete target_sql_file_path if File.exists? target_sql_file_path

  sql_command = "create table run ("
  sql_command += "\n  input_file varchar(512),"
  sql_command += "\n  id int,"
  sql_command += "\n  alpha0 int,"
  sql_command += "\n  alpha1 int,"
  sql_command += "\n  lambda int,"
  sql_command += "\n  iteration_count int,"
  sql_command += "\n  passed boolean,"
  sql_command += "\n  output_denoised_file varchar(512),"
  sql_command += "\n  output_shading_file varchar(512),"
  sql_command += "\n  output_deshaded_file varchar(512),"
  parameter[:metrics].each_with_index do |metric, index|
    sql_command += "\n  #{metric} double"
    sql_command += index == parameter[:metrics].size - 1 ? ")" : ","
  end

  begin
    database = SQLite3::Database.new target_sql_file_path
  rescue
    puts "problem while creating database"
  end
  begin
    database.execute sql_command
  rescue
    puts "problem while creating table"
    puts "executed command: #{sql_command}"
  end
end

if $0 == __FILE__
  puts 'executing script...'
  run_specific
end

