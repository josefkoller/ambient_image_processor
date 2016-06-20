#!/usr/bin/ruby
require 'colorize'

def select_minimum parameter
  require 'sqlite3'
  database = SQLite3::Database.new parameter[:output_sql_file_path]
  command = "select #{parameter[:parameter1_name]}"
  command += ",#{parameter[:parameter2_name]}"
  command += " from run where #{parameter[:metric_name]} ="
  command += "(select min(#{parameter[:metric_name]}) from run)"

  minimum_data = database.execute command
  minimum_data = minimum_data[0]
  minimum = { 
    parameter[:parameter1_name].to_sym => minimum_data[0],
    parameter[:parameter2_name].to_sym => minimum_data[1] }

  puts "minimum: #{minimum}".green
  minimum
end

def calculate_next_search_space parameter
  minimum = select_minimum parameter
  minimum.each do |key, value|
    config = parameter[key]
    step = (config[:max] - config[:min]) / config[:count].to_f

    config[:min] = value - step
    config[:max] = value + step
  end
end

def optimize_and_calculate_metric parameter
  load "ruby/optimize.rb"
  run parameter

  command = "build/output/image_metric_to_sqlite #{parameter[:output_sql_file_path]}"
  command += " #{parameter[:entropy_kde_bandwidth]}"
  command += " #{parameter[:entropy_window_from]}"
  command += " #{parameter[:entropy_window_to]}"
  puts command.yellow
  system command
end

def plot parameter, plot_logarithmic=false
  begin
    plot_data_file = "#{parameter[:output_path]}/gnuplot.dat"
    if File.exists? plot_data_file
      plot_data_file += "_next"
    end
    command = "build/output/sqlite_to_gnuplot #{parameter[:output_sql_file_path]}"
    command += " #{plot_data_file}"
    command += " #{parameter[:parameter1_name]}"
    command += " #{parameter[:parameter2_name]}"
    command += " #{parameter[:metric_name]}"
    puts command.yellow
    system command

    command = "gnuplot -e \"filename='#{plot_data_file}';"
    command += " x='#{parameter[:parameter1_name]}';"
    command += " y='#{parameter[:parameter2_name]}';"
    command += " z='#{parameter[:metric_name]}';"
    command += " use_logscale=1;" if plot_logarithmic
    command += "\""
    command += " gnuplot/plot_data.gp"
    puts command.yellow
    pid = spawn command
    Process.detach pid
  rescue => exception
    puts exception.to_s.red
  end
end

if ARGV.length != 1
  puts "arguments: configuration_file".red
  exit
end

# read script input parameter
configuration_file = ARGV[0]

unless File.exists? configuration_file
  puts "configuration file does not exist".red
  exit
end

require 'pp'
load configuration_file
parameter = read_parameter

puts "parameter:".yellow
pp parameter

begin
    optimize_and_calculate_metric parameter
    select_minimum parameter
    plot parameter
=begin
  parameter[:alpha1] = {
    :min => 1,
    :step_factor => 0.1,
    :count => 8
  }
  parameter[:alpha0][:bind_to] = :alpha1
  optimize_and_calculate_metric parameter
  plot parameter, :plot_logarithmic

  minimum = select_minimum parameter
  minimum.each do |key, value|
    config = parameter[key]
    step = value / 10.0
    puts "stepping #{step} with #{key.to_s}".green
    config[:min] = value - step
    config[:max] = value + step
    config[:count] = 10
  end
  parameter[:alpha0][:bind_to] = nil

  # second, estimate the ratio
  parameter[:multiscale_optimization_depth].times do 
    optimize_and_calculate_metric parameter
    calculate_next_search_space parameter
    plot parameter
  end
=end
  
rescue => exception
  puts exception.to_s.red
end

