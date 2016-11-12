import sys
from subprocess import call, Popen, PIPE

print 'TGV DCT Deshade algorithm, evaluator script'

application_path = '../../build/output/tgvk_deshade_downsampled_evaluator_application'
data_root_path = '/home/joe/Documents/tu/master_thesis/documents/05_thesis/figures/'

print 'application: ' + application_path
print 'data root direction: ' + data_root_path

task_list_file = sys.argv[1]
print 'task list file: ' + task_list_file

def executeShellCommandAndReturnStdOutput(command):
  #print(command)
  process = Popen(command, stdin=PIPE, stdout=PIPE)
  (output, error) = process.communicate()
  print(output)
  if error:
    print(error)
    return
  return output.strip()

task_file = open(task_list_file, 'r')
line_counter = 1
for line in task_file:
    line = line.strip()
    line_counter+= 1
    if line.startswith('#') or line == '':
        continue;

    print('line %d: %s' % (line_counter, line))

    parts = line.split(';')
    root_dir = data_root_path + parts[0] + '/'
    input_file = parts[1]
    mask_file = parts[2]
    mode = parts[3]
    line_start = parts[4]
    line_end = parts[5]
    f_downsampling = parts[6]
    lambda_value = parts[7]
    iteration_count = parts[8]
    alpha = parts[9]

    command = [application_path, root_dir, input_file, mask_file, mode, \
      line_start, line_end, f_downsampling, lambda_value, iteration_count, alpha]
    executeShellCommandAndReturnStdOutput(command)
