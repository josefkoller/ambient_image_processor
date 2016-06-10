# # Gemfile for linear_retinex_to_sql tasks

def run_specific
  if ARGV.length == 0 
    puts "specify a configuration file"
    return
  end

  configuration_file = ARGV[0]
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
  puts "running with parameter: #{parameter}"
  create_sql_database parameter
  optimize parameter
end
def optimize parameter
  return unless File.exists? parameter[:output_sql_file_path]
  require 'sqlite3'
  database = SQLite3::Database.new parameter[:output_sql_file_path]
  begin
    if parameter[:alpha0][:count] == 1
      alpha0_step = 0
    else
      alpha0_step = ( parameter[:alpha0][:max] - parameter[:alpha0][:min] ) / ( parameter[:alpha0][:count] - 1.0 )
    end
    if parameter[:alpha1][:count] == 1
      alpha1_step = 0
    else
      alpha1_step = (parameter[:alpha1][:max] - parameter[:alpha1][:min]) / (parameter[:alpha1][:count] - 1.0)
    end
    output_file_id = 1
    for alpha1_index in 1..parameter[:alpha1][:count] do
        alpha1 = parameter[:alpha1][:min] + alpha1_step * (alpha1_index - 1)
        for alpha0_index in 1..parameter[:alpha0][:count] do
            alpha0 = parameter[:alpha0][:min] + alpha0_step * (alpha0_index - 1)
            forward_model database, alpha0, alpha1, output_file_id, parameter
            output_file_id += 1
        end
    end
  end
end
def run_program command
  require 'timeout'
  #one day ... maximum
  Timeout::timeout(60*60*24) {
    passed = system command
    throw "System command failed" unless passed
  }
end
def forward_model database, alpha0, alpha1, output_file_id, parameter
  command = "#{parameter[:program_path]} #{parameter[:input_image_path]}"
  command += " #{parameter[:lambda]} #{alpha0} #{alpha1} #{parameter[:iteration_count]}"
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
    insert_run database, alpha0, alpha1, output_file_id, true, output_denoised_file, output_shading_file, output_deshaded_file, parameter
    puts "run #{"%04d" % output_file_id} passed" 
  rescue
    insert_run database, alpha0, alpha1, output_file_id, false, output_denoised_file, output_shading_file, output_deshaded_file, parameter
    puts "run #{"%04d" % output_file_id} failed with parameter: alpha0=#{alpha0}, alpha1=#{alpha1}"
    puts "command #{command}"
  end
end
def insert_run database, alpha0, alpha1, output_file_id, passed, output_denoised_file, output_shading_file, output_deshaded_file, parameter
  command = ""
  command += "insert into run (input_file, id, alpha0, alpha1, "
  command += "lambda, iteration_count, passed, output_denoised_file, output_shading_file, output_deshaded_file) values("
  command += "?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"

  values = [parameter[:input_image_path],
            output_file_id.to_s,
            alpha0.to_s,
            alpha1.to_s,
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

