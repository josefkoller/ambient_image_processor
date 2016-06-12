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

metric_values_history = {};

alpha_values = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8];
for alpha_value = alpha_values
  alpha0 = alpha_value;
  alpha1 = alpha_value;
  command = sprintf('%s %s %f %f %f %d %s %s', application_path, ...
              input_file_name, lambda, alpha0, alpha1, check_iteration_count, ...
              mask_file_name, metric_file_name);
  [result, output] = system(command);
  disp(output);
  disp(sprintf('exit code: %d', result));

  if result == 0
    metric_values = load(metric_file_name);
    metric_values_history{length(metric_values_history)+1} = metric_values;
  end
end

#converge_iteration = -1;

colors = ['r', 'b', 'g', 'm'];

for h = 1:length(metric_values_history)
  metric_values = metric_values_history{h};
  for i = 1:length(metric_values)
    metric_value = metric_values(i);

    figure(i);
    
    color = colors(i);
    
    handle = stem(h, metric_value);
    set (handle, "color", color);
    hold on;
  end  
end
hold off;

return;

for h = 1:length(metric_values_history)
  metric_values = metric_values_history{h};
  for i = 1:length(metric_values)
    metric_value = metric_values(i);
    if isnan(metric_value)
      converge_iteration = h - 1;
      break;
    end
  end  
  if converge_iteration > -1
    break;
  end
end

if converge_iteration == -1
  disp('could not find the decade of the step size parameters');
end

return;

# FOUND the decade of alphas, determine the ratio...
alpha1 = 10 ^ -(converge_iteration-1);
alpha_ratios = [1, 1.2, 1.8, 2, 2.2];

metric_values_history = {};
for alpha_ratio=alpha_ratios
  alpha0 = alpha1 * alpha_ratio;
  
  command = sprintf('%s %s %f %f %f %d %s %s', application_path, ...
              input_file_name, lambda, alpha0, alpha1, check_iteration_count, ...
              mask_file_name, metric_file_name);
  [result, output] = system(command);
  disp(output);
  disp(sprintf('exit code: %d', result));

  if result == 0
    metric_values = load(metric_file_name);
    metric_values_history{length(metric_values_history)+1} = metric_values;
  end
end

colors = ['r', 'b', 'g', 'm'];

for h = 1:length(metric_values_history)
  metric_values = metric_values_history{h};
  for i = 1:length(metric_values)
    metric_value = metric_values(i);

    figure(i);
    
    color = colors(i);
    
    handle = stem(h, metric_value);
    set (handle, "color", color);
    hold on;
  end  
end
hold off;





