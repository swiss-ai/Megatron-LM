#!/bin/bash

#SBATCH --account=a-a06
#SBATCH --time=00:59:59
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --environment=/capstor/store/cscs/swissai/a06/containers/NGC-PyTorch/ngc_pt_jan.toml	# Vanilla 25.01 PyTorch NGC Image
#SBATCH --no-requeue

input_folder=$1
output_prefix=$2
MEGATRON_LM_DIR=$3

if [ -z "$input_folder" ] || [ -z "$output_prefix" ] || [ -z "$MEGATRON_LM_DIR" ]; then
    echo "Usage: sbatch $0 <input_folder> <output_prefix> <megatron_lm_dir>"
    exit 1
fi

cd "$MEGATRON_LM_DIR"
export PYTHONPATH="$MEGATRON_LM_DIR:$PYTHONPATH"

# Clean if previous job failed
rm -f "$output_prefix.bin"
rm -f "$output_prefix.idx"

echo "START TIME: $(date) | Merging all tokenized documents in $input_folder and storing them in $output_prefix"
numactl --membind=0-3 python3 "$MEGATRON_LM_DIR/scripts/merge_datasets/merge_datasets.py" --input "$input_folder" --output-prefix "$output_prefix"
echo "FINISH TIME: $(date) | Merged all files in $input_folder ! Stored in $output_prefix"

sleep 60 # NOTE(tj.solergibert) Sleep just to make sure all copies are done before exiting
ls -l "$output_prefix.bin"
ls -l "$output_prefix.idx"
