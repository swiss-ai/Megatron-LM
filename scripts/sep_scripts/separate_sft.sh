#!/bin/bash
#SBATCH --job-name=split_ratio_v1p5
#SBATCH --output=logs/split_ratio_%j.out
#SBATCH --error=logs/split_ratio_%j.err
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5-0

set -euo pipefail
mkdir -p logs

INPUT_DIR="/capstor/store/cscs/swissai/infra01/datasets_tokenized/apertus_sft_datasets/v1p5-mix-v1-28-05-linearised_fix_display_answers_tool_toolheaders_injected/Apertus-v1p5-tool_output_toks-think_toks"
OUTPUT_DIR="/capstor/store/cscs/swissai/infra01/datasets_tokenized/apertus_sft_datasets/v1p5-mix-v1-28-05-linearised_fix_display_answers_tool_toolheaders_injected_split"

TMP_REPO="/capstor/scratch/cscs/$USER/.tmp_pipeline_${SLURM_JOB_ID}"
trap 'rm -rf "$TMP_REPO"' EXIT

git clone --branch dt/change_binary --depth 1 \
    git@github.com:swiss-ai/data-pipeline-pretrain.git \
    "${TMP_REPO}/data-pipeline-pretrain"

git clone --depth 1 \
    git@github.com:swiss-ai/Megatron-LM.git \
    "${TMP_REPO}/Megatron-LM"

PIPELINE_DIR="${TMP_REPO}/data-pipeline-pretrain/examples/separate_binary"
MEGATRON_PATH="${TMP_REPO}/Megatron-LM"


echo "Job ${SLURM_JOB_ID} started at $(date)"
echo "Input dir  : ${INPUT_DIR}"
echo "Output dir : ${OUTPUT_DIR}"


srun --environment="${PIPELINE_DIR}/nemo.toml" bash -c "\
    cd '${MEGATRON_PATH}' && \
    python setup.py build_ext --inplace && \
    export PYTHONPATH='${MEGATRON_PATH}' && \
    python -u ${PIPELINE_DIR}/separate.py \
        --input-dir  '${INPUT_DIR}' \
        --output-dir '${OUTPUT_DIR}' \
        --ratios 0.8 0.2 \
        --seed   42 \
        --min-tokens 262144 \
        --workers ${SLURM_CPUS_PER_TASK}"

echo "Job finished at $(date)"