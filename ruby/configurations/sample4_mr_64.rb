# tgv2 deshading optimization configuration

def read_parameter
  parameter = {}
  require 'pathname'
  root_directory = (Pathname.new __FILE__).parent.parent.parent
  parameter[:program_path]="#{root_directory}/build/output/tgv2_deshade_application"


  parameter[:input_image_path]="#{root_directory}/data/input/mr14_slice3_64/image.mha"
  parameter[:input_mask_path]="#{root_directory}/data/input/mr14_slice3_64/mask.mha"

  parameter[:iteration_count]=5e3
  parameter[:lambda]=1
  parameter[:alpha0]={ :min=>0.000023, :max=>0.000025, :count=>1e1 }
  parameter[:alpha1]={ :min=>0.000010, :max=>0.000090, :count=>1e1 }

  time = Time.new
  time_string = time.strftime "%Y_%m_%d_%H_%M_%S"

  output_path = "#{root_directory}/data/output"
  output_path = File.join output_path, time_string
  parameter[:output_path] = output_path

  Dir.mkdir output_path
  Dir.mkdir File.join(output_path, 'images')
  
  parameter[:output_sql_file_path] = "#{output_path}/database.sqlite"
  parameter[:output_denoised_file_path] = "#{output_path}/images/denoised"
  parameter[:output_shading_file_path] = "#{output_path}/images/shading"
  parameter[:output_deshaded_file_path] = "#{output_path}/images/deshaded"
  parameter[:output_format] = "mha"
  parameter[:metrics] = ["denoised_coefficient_of_variation", "denoised_mean_total_variation",
                        "deshaded_coefficient_of_variation", "deshaded_mean_total_variation",
                        "deshaded_entropy"]
  parameter
end
