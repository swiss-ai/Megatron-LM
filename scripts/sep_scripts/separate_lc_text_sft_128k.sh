#!/bin/bash
#SBATCH --job-name=split
#SBATCH --output=logs/split_%j.out
#SBATCH --error=logs/split_%j.err
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5-0

OUTPUT_DIR="/capstor/store/cscs/swissai/infra01/datasets_tokenized/apertus_sft_datasets/sft_long_context_128k"

LC_SYNTH_TEXT_PATHS="\
/capstor/store/cscs/swissai/infra01/long_context/128k/long_context_samples/synthetic_apertus_sft/CWE/Biomed-Enriched_preprocessed/dump-0,\
/capstor/store/cscs/swissai/infra01/long_context/128k/long_context_samples/synthetic_apertus_sft/CWE/dolma3_olmocr_science_pdfs-preprocessed/dump-0,\
/capstor/store/cscs/swissai/infra01/long_context/128k/long_context_samples/synthetic_apertus_sft/CWE/finepdfs-edu-multilingual-preprocessed/dump-0,\
/capstor/store/cscs/swissai/infra01/long_context/128k/long_context_samples/synthetic_apertus_sft/CWE/finepdfs-edu-preprocessed/dump-0,\
/capstor/store/cscs/swissai/infra01/long_context/128k/long_context_samples/synthetic_apertus_sft/CWE/finetranslations/dump-0,\
/capstor/store/cscs/swissai/infra01/long_context/128k/long_context_samples/synthetic_apertus_sft/CWE/institutional-books-1.0-filtered/dump-0,\
/capstor/store/cscs/swissai/infra01/long_context/128k/long_context_samples/synthetic_apertus_sft/CWE/swissai-fineweb-2_0_1-quality_10-filterrobots/dump-0,\
/capstor/store/cscs/swissai/infra01/long_context/128k/long_context_samples/synthetic_apertus_sft/DocOrder/Biomed-Enriched_preprocessed/dump-0,\
/capstor/store/cscs/swissai/infra01/long_context/128k/long_context_samples/synthetic_apertus_sft/DocOrder/dolma3_olmocr_science_pdfs-preprocessed/dump-0,\
/capstor/store/cscs/swissai/infra01/long_context/128k/long_context_samples/synthetic_apertus_sft/DocOrder/finepdfs-edu-multilingual-preprocessed/dump-0,\
/capstor/store/cscs/swissai/infra01/long_context/128k/long_context_samples/synthetic_apertus_sft/DocOrder/finepdfs-edu-preprocessed/dump-0,\
/capstor/store/cscs/swissai/infra01/long_context/128k/long_context_samples/synthetic_apertus_sft/DocOrder/finetranslations/dump-0,\
/capstor/store/cscs/swissai/infra01/long_context/128k/long_context_samples/synthetic_apertus_sft/DocOrder/institutional-books-1.0-filtered/dump-0,\
/capstor/store/cscs/swissai/infra01/long_context/128k/long_context_samples/synthetic_apertus_sft/DocOrder/swissai-fineweb-2_0_1-quality_10-filterrobots/dump-0,\
/capstor/store/cscs/swissai/infra01/long_context/128k/long_context_samples/synthetic_apertus_sft/ArtificialNeedles/dump-0"


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

IFS=',' read -ra DUMP_PATHS <<< "$LC_SYNTH_TEXT_PATHS"
for dump_path in "${DUMP_PATHS[@]}"; do
    [ -d "$dump_path" ] || { echo "WARNING: $dump_path not found, skipping" >&2; continue; }
    subdataset_dir="$(dirname  "$dump_path")"
    subdataset="$(basename     "$subdataset_dir")"
    category="$(basename "$(dirname "$subdataset_dir")")"
    ln -sfn "$subdataset_dir" "${STAGING}/${category}__${subdataset}"
done

echo "Job ${SLURM_JOB_ID} started at $(date)"
echo "Output dir : ${OUTPUT_DIR}"
echo "Staging    : ${STAGING}"
ls -la "${STAGING}"


srun --environment="${PIPELINE_DIR}/nemo.toml" bash -c "\
    cd '${MEGATRON_PATH}' && \
    python setup.py build_ext --inplace && \
    export PYTHONPATH='${MEGATRON_PATH}' && \
    python -u ${PIPELINE_DIR}/separate.py \
        --input-dir  '${STAGING}' \
        --output-dir '${OUTPUT_DIR}' \
        --ratios 0.2 0.8 \
        --seed   42 \
        --min-tokens 262_144 \
        --workers ${SLURM_CPUS_PER_TASK}"

echo "Job finished at $(date)"