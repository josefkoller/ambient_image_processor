#!/usr/bin/ruby
require 'colorize'

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

def run_program command
  start = Time.now
  duration = -1

  require 'timeout'
  #one day ... maximum
  Timeout::timeout(60*60*24) {
    passed = system command

    duration = Time.now - start

    throw "System command failed" unless passed
  }
  duration
end

image_path = '~/Documents/tu/master_thesis/test_data/MR/serie14_slice27/serie14_slice27_rescaled.mha'

wrap_size = 32
max_block_dimension = 1024
program_path = 'build/output/tgv2_deshade_application'
lambda_value = 1
alpha0 = 2
alpha1 = 1
iteration_count = 1000
output_denoised_path = '/tmp/denoised.mha'
output_shading_path = '/tmp/shading.mha'
output_deshaded_path = '/tmp/deshaded.mha'

cuda_block_dimension = max_block_dimension

block_dimensions = []
durations = []

begin
  while cuda_block_dimension >= wrap_size do
    command = "#{program_path} #{image_path} #{lambda_value} #{alpha0} #{alpha1} #{iteration_count} #{cuda_block_dimension}"
    command += " #{output_denoised_path} #{output_shading_path} #{output_deshaded_path}"
    duration = run_program command
    puts "block dimension: #{cuda_block_dimension} duration: #{duration} s"

    block_dimensions << cuda_block_dimension
    durations << duration
    cuda_block_dimension -= wrap_size
  end

  min_index = durations.index durations.min
  puts "minimum: "
  puts "block dimension: #{block_dimensions[min_index]} duration: #{durations[min_index]} s".green
rescue => exception
  puts exception.to_s.red
end

