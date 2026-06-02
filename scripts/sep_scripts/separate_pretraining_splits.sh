#!/bin/bash
#SBATCH --job-name=split_ratio_v1p5
#SBATCH --output=logs/split_ratio_%j.out
#SBATCH --error=logs/split_ratio_%j.err
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5-0

# finepdfs
INPUT_DIR="/capstor/store/cscs/swissai/infra01/datasets_tokenized/finepdfs-edu-preprocessed/Apertus-70B-2509/finepdfs-edu-preprocessed"
OUTPUT_DIR="/capstor/scratch/cscs/$USER/data/finepdfs-edu-multilingual-preprocessed"

# fineweb
INPUT_DIR_FINEWEB="/capstor/store/cscs/swissai/infra01/datasets_tokenized/fineweb-2_0_1-quality_10-100_langs-filterrobots/Apertus-70B-2509/swissai-fineweb-2_0_1-quality_10-filterrobots/"
OUTPUT_DIR_FINEWEB="/capstor/scratch/cscs/$USER/data/fineweb-2_0_1-quality_10-100_langs-filterrobots"

# finetranslations
INPUT_DIR_TRANSLATIONS="/capstor/store/cscs/swissai/infra01/datasets_tokenized/finetranslations_preprocessed/Apertus-70B-2509/finetranslations"
OUTPUT_DIR_TRANSLATIONS="/capstor/scratch/cscs/$USER/data/finetranslations"

# finepdfs-edu-multilingual
INPUT_DIR_MULTILINGUAL="/capstor/store/cscs/swissai/infra01/datasets_tokenized/finepdfs-edu-multilingual-preprocessed/Apertus-70B-2509/finepdfs-edu-multilingual-preprocessed"
OUTPUT_DIR_MULTILINGUAL="/capstor/scratch/cscs/$USER/data/finepdfs-edu-multilingual-preprocessed"

TMP_REPO="/capstor/scratch/cscs/$USER/.tmp_pipeline_${SLURM_JOB_ID}"
trap 'rm -rf "$TMP_REPO"' EXIT

# Clone full branch history so the target commit is reachable, then pin to it
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
echo "Input dir (finepdfs)       : ${INPUT_DIR}"
echo "Output dir (finepdfs)      : ${OUTPUT_DIR}"
echo "Input dir (fineweb)        : ${INPUT_DIR_FINEWEB}"
echo "Output dir (fineweb)       : ${OUTPUT_DIR_FINEWEB}"
echo "Input dir (translations)   : ${INPUT_DIR_TRANSLATIONS}"
echo "Output dir (translations)  : ${OUTPUT_DIR_TRANSLATIONS}"
echo "Input dir (multilingual)   : ${INPUT_DIR_MULTILINGUAL}"
echo "Output dir (multilingual)  : ${OUTPUT_DIR_MULTILINGUAL}"

# Please, set the path to the toml
srun --environment="/capstor/scratch/cscs/$USER/containers/nemo.toml" bash -c "\
    cd '${MEGATRON_PATH}' && \
    python setup.py build_ext --inplace && \
    export PYTHONPATH='${MEGATRON_PATH}' && \
    echo '--- [1/4] finepdfs-edu' && \
    python -u ${PIPELINE_DIR}/separate.py \
        --input-dir  '${INPUT_DIR}' \
        --output-dir '${OUTPUT_DIR}' \
        --min-tokens 8193 \
        --seed 42 && \
    echo '--- [2/4] fineweb-2_0_1' && \
    python -u ${PIPELINE_DIR}/separate_corr_now.py \
        --input-dir  '${INPUT_DIR_FINEWEB}' \
        --output-dir '${OUTPUT_DIR_FINEWEB}' \
        --seed 42 && \
    echo '--- [3/4] finetranslations' && \
    python -u ${PIPELINE_DIR}/separate_corr_now.py \
        --input-dir  '${INPUT_DIR_TRANSLATIONS}' \
        --output-dir '${OUTPUT_DIR_TRANSLATIONS}' \
        --seed 42 && \
    echo '--- [4/4] finepdfs-edu-multilingual' && \
    python -u ${PIPELINE_DIR}/separate_corr_now.py \
        --input-dir  '${INPUT_DIR_MULTILINGUAL}' \
        --output-dir '${OUTPUT_DIR_MULTILINGUAL}' \
        --seed 42"

echo "Job finished at $(date)"