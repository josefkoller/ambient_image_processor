#!/usr/bin/ruby
require 'colorize'

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

data_path = '~/Documents/tu/master_thesis/test_data/MR_volume_rendering/tissue_only/'
data_path = ARGV[0] if ARGV.length > 0

image_path = "#{data_path}/image.mha"
mask_path = "#{data_path}/mask.mha"

wrap_size = 32
max_block_dimension = 1024
program_path = 'build/output/tgv2_deshade_application'
lambda_value = 1
alpha0 = 2
alpha1 = 1
iteration_count = 100
output_denoised_path = '/tmp/denoised.mha'
output_shading_path = '/tmp/shading.mha'
output_deshaded_path = '/tmp/deshaded.mha'

cuda_block_dimension = max_block_dimension

block_dimensions = []
durations = []

begin
  while cuda_block_dimension >= wrap_size do
    command = "#{program_path} #{image_path} #{lambda_value} #{alpha0} #{alpha1} #{iteration_count}"
    command += " #{cuda_block_dimension} #{mask_path}"
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

