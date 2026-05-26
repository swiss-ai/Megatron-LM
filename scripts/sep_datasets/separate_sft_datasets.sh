#!/bin/bash

#SBATCH --job-name=split_sft_buckets
#SBATCH --output=logs/split_sft_buckets_%j.out
#SBATCH --error=logs/split_sft_buckets_%j.err
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5-0


mkdir -p logs

SCRIPT_DIR="$SLURM_SUBMIT_DIR"
OUTPUT_DIR="/capstor/store/cscs/swissai/infra01/datasets_tokenized/apertus_sft_datasets/tool_sft_datasets"
MEGATRON_PATH=/capstor/scratch/cscs/dtamayomela/megatron/pre-training/megatron_fixed

echo "Job $SLURM_JOB_ID started at $(date)"
echo "Output dir: $OUTPUT_DIR"

srun --environment="/capstor/scratch/cscs/dtamayomela/containers/nemo.toml" bash -c "\
    export PYTHONPATH=${MEGATRON_PATH} && \
    python -u ${SCRIPT_DIR}/separate_sft_datasets.py \
        --output-dir $OUTPUT_DIR"

echo "Job finished at $(date)"