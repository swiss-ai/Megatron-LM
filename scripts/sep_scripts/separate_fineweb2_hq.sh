#!/bin/bash
#SBATCH --job-name=split_fineweb_test
#SBATCH --output=logs/split_fineweb_%j.out
#SBATCH --error=logs/split_fineweb_%j.err
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5-0

INPUT_DIR="/capstor/store/cscs/swissai/infra01/datasets_tokenized/fineweb-2_0_1-quality_10-100_langs-filterrobots/Apertus-70B-2509/swissai-fineweb-2_0_1-quality_10-filterrobots"
OUTPUT_DIR="/capstor/scratch/cscs/$USER/data/swissai-fineweb-2_0_1-quality_10-filterrobots"

TMP_REPO="/capstor/scratch/cscs/$USER/.tmp_pipeline_${SLURM_JOB_ID}"
trap 'rm -rf "$TMP_REPO"' EXIT

git clone --branch dt/change_binary \
    git@github.com:swiss-ai/data-pipeline-pretrain.git \
    "${TMP_REPO}/data-pipeline-pretrain"
git -C "${TMP_REPO}/data-pipeline-pretrain" checkout e02f6a9d256a0ed73abfdf190c6181262f5a797a

git clone --depth 1 \
    git@github.com:swiss-ai/Megatron-LM.git \
    "${TMP_REPO}/Megatron-LM"

PIPELINE_DIR="${TMP_REPO}/data-pipeline-pretrain/examples/separate_binary"
MEGATRON_PATH="${TMP_REPO}/Megatron-LM"

echo "Job ${SLURM_JOB_ID} started at $(date)"
echo "Input dir  : ${INPUT_DIR}"
echo "Output dir : ${OUTPUT_DIR}"

srun --environment="/capstor/scratch/cscs/$USER/containers/nemo.toml" bash -c "\
    cd '${MEGATRON_PATH}' && \
    python setup.py build_ext --inplace && \
    export PYTHONPATH='${MEGATRON_PATH}' && \
    python -u ${PIPELINE_DIR}/separate.py \
        --input-dir  '${INPUT_DIR}' \
        --output-dir '${OUTPUT_DIR}' \
        --seed 42"

echo "Job finished at $(date)"