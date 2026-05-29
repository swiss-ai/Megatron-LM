#!/bin/bash
#SBATCH --job-name=split_ratio
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

INPUT_DIR="/capstor/store/cscs/swissai/infra01/datasets_tokenized/apertus_sft_datasets/tool_sft_datasets"
# OUTPUT_DIR="/capstor/store/cscs/swissai/infra01/datasets_tokenized/apertus_sft_datasets/tool_sft_datasets_split"
OUTPUT_DIR=/capstor/scratch/cscs/dtamayomela/megatron/pre-training/megatron_documentation/scripts/sep_scripts/test


# Comma-separated list of bucket sub-dirs to process
INPUT_FOLDERS="lower_16k"


TMP_REPO="/capstor/scratch/cscs/$USER/.tmp_pipeline_${SLURM_JOB_ID}"
trap 'rm -rf "$TMP_REPO" "$STAGING"' EXIT

git clone --branch dt/change_binary --depth 1 \
    git@github.com:swiss-ai/data-pipeline-pretrain.git \
    "${TMP_REPO}/data-pipeline-pretrain"

git clone --depth 1 \
    git@github.com:swiss-ai/Megatron-LM.git \
    "${TMP_REPO}/Megatron-LM"

PIPELINE_DIR="${TMP_REPO}/data-pipeline-pretrain/examples/separate_binary"
MEGATRON_PATH="${TMP_REPO}/Megatron-LM"

STAGING="${OUTPUT_DIR}/.staging_${SLURM_JOB_ID}"
mkdir -p "$STAGING"

IFS=',' read -ra FOLDERS <<< "$INPUT_FOLDERS"
for folder in "${FOLDERS[@]}"; do
    folder="$(echo "$folder" | xargs)"   # trim whitespace
    src="${INPUT_DIR}/${folder}"
    dump_dir="${STAGING}/${folder}/dump-0"
    mkdir -p "$dump_dir"

    for idx_file in "${src}"/*.idx; do
        [ -e "$idx_file" ] || continue
        stem="$(basename "$idx_file" .idx)"
        bin_file="${src}/${stem}.bin"
        if [ ! -f "$bin_file" ]; then
            echo "WARNING: no .bin for $idx_file, skipping" >&2
            continue
        fi
        # Symlink with the _tokens suffix that separate.py globs for
        ln -s "$idx_file" "${dump_dir}/${stem}_tokens.idx"
        ln -s "$bin_file" "${dump_dir}/${stem}_tokens.bin"
    done
done

echo "Job $SLURM_JOB_ID started at $(date)"
echo "Input dir  : $INPUT_DIR"
echo "Folders    : $INPUT_FOLDERS"
echo "Output dir : $OUTPUT_DIR"
echo "Staging    : $STAGING"


srun --environment="${PIPELINE_DIR}/nemo.toml" bash -c "\
    cd '${MEGATRON_PATH}' && \
    python setup.py build_ext --inplace && \
    export PYTHONPATH='${MEGATRON_PATH}' && \
    python -u ${PIPELINE_DIR}/separate.py \
        --input-dir  '${STAGING}' \
        --output-dir '${OUTPUT_DIR}' \
        --ratios 0.8 0.2 \
        --seed   42 \
        --min-tokens 262_144 \
        --workers ${SLURM_CPUS_PER_TASK}"

echo "Job finished at $(date)"