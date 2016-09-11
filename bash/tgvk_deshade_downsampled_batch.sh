cd $(dirname "$0")/..
data_root_dir=~/Documents/tu/master_thesis/documents/05_thesis/figures/mr_serie15
output_data_root_dir="${data_root_dir}/tgvk"
program=build/output/tgvk_deshade_downsampled_application
input="${data_root_dir}/12_input.mha"
mask="${data_root_dir}/22_mask.mha"
gpu_timeout=3
run_id=0
lambda=2
iterations=10000
perform() {
  run_id=$((run_id+1))
  f_downsampling=$1
  order=$2
  output_prefix="${output_data_root_dir}/output_${run_id}"
  log_output="${output_data_root_dir}/output_${run_id}_program_log.txt"
  sleep $gpu_timeout && (nvidia-smi | tee "${output_data_root_dir}/output_${run_id}_gpu_log.txt") &
  echo "executing: $program $input $f_downsampling $order $lambda $iterations $mask $output_prefix > $log_output"
  $program $input $f_downsampling $order $lambda $iterations $mask $output_prefix 2>&1 | tee $log_output
}

perform 1.00 2
perform 0.50 2
perform 0.25 2

perform 1.00 3
perform 0.50 3
perform 0.25 3

perform 1.00 4
perform 0.50 4
perform 0.25 4
