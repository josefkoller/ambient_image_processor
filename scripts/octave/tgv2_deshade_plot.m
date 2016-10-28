% tgv deshade cost function plot

% tgv2 deshade convergence test

close all;
clear all;
more off;

input_file_name = '../data/input/mr_serie14_slice3/00_denoised.mha';
mask_file_name = '../data/input/mr_serie14_slice3/01_mask.mha';

lambda = 1;
check_iteration_count = 10;
metric_file_name = 'metric_file.txt';
application_path = '../build/output/tgv2_deshade_convergence_test_application';


alpha1 = 0.01;

alpha0_values = linspace(0.01, 0.02, 4);

metric_value_index = 2;
metric_values = [];
for alpha0 = alpha0_values
  command = sprintf('%s %s %f %f %f %d %s %s', application_path, ...
              input_file_name, lambda, alpha0, alpha1, check_iteration_count, ...
              mask_file_name, metric_file_name);
  [result, output] = system(command);
  disp(output);
  disp(sprintf('exit code: %d', result));

  if result == 0
    file_data = load(metric_file_name);
    metric_value = file_data(metric_value_index);
    metric_values(length(metric_values)+1) = metric_value;
  end
end

plot(alpha0_values, metric_values, 'x-');