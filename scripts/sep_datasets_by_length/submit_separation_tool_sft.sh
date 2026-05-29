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
MEGATRON_PATH=/capstor/scratch/cscs/$USER/megatron/pre-training/megatron_fixed

OUTPUT_DIR="/capstor/store/cscs/swissai/infra01/datasets_tokenized/apertus_sft_datasets/tool_sft_datasets"
BASE="/capstor/store/cscs/swissai/infra01/datasets_tokenized/apertus_sft_datasets"

NAMED_DATASETS=(
    "EnvScaler-SFT-Traj-9K:${BASE}/EnvScaler-SFT-Traj-9K_corr/tokenizer_tool_tokens/EnvScaler-SFT-Traj-9K_corr/dump-0/00000_tokens"
    "OpenSeeker-v1-Data:/capstor/store/cscs/swissai/infra01/datasets_tokenized/apertus_sft_datasets/OpenSeeker-v1-Data_corr/Apertus-v1p5-tool_output_toks-think_toks/OpenSeeker-v1-Data_corr/dump-0/00000_tokens"
    "Toucan-1.5M_filtered:${BASE}/Toucan-1.5M_filtered_corr/tokenizer_tool_tokens/Toucan-1.5M_filtered_corr/dump-0/00000_tokens"
)

echo "Job $SLURM_JOB_ID started at $(date)"
echo "Output dir: $OUTPUT_DIR"

srun --environment="/capstor/scratch/cscs/$USER/containers/nemo.toml" bash -c "\
    export PYTHONPATH=${MEGATRON_PATH} && \
    python -u ${SCRIPT_DIR}/split_buckets.py \
        --named-datasets ${NAMED_DATASETS[*]} \
        --bucket-set sft \
        --output-dir ${OUTPUT_DIR}"

echo "Job finished at $(date)"