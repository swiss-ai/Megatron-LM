#!/bin/bash
#SBATCH --job-name=split_vision_buckets
#SBATCH --output=logs/split_vision_buckets_%j.out
#SBATCH --error=logs/split_vision_buckets_%j.err
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5-0

mkdir -p logs

SCRIPT_DIR="$SLURM_SUBMIT_DIR"
MEGATRON_PATH=/capstor/scratch/cscs/$USER/megatron/pre-training/megatron_fixed

VISION_DIR="/capstor/store/cscs/swissai/infra01/vision-datasets/Apertus1p5_sft_lct_tokenized"
AUDIO_DIR="/capstor/store/cscs/swissai/infra01/audio-datasets/Apertus1p5_sft_lct_tokenized"

# Override output dir by passing it as the first argument: sbatch submit_vision.sh /my/output
OUTPUT_DIR="${1:-${SLURM_SUBMIT_DIR}/apertus_sft_buckets_fixed}"

echo "Job $SLURM_JOB_ID started at $(date)"
echo "Vision input : $VISION_DIR"
echo "Audio input : $AUDIO_DIR"
echo "Output root  : $OUTPUT_DIR"

srun --environment="/capstor/scratch/cscs/$USER/containers/nemo.toml" bash -c "\
    export PYTHONPATH=${MEGATRON_PATH} && \
    echo '=== Processing vision ===' && \
    python -u ${SCRIPT_DIR}/split_buckets.py \
        --input-dirs '${VISION_DIR}' \
        --bucket-set vision \
        --output-dir '${OUTPUT_DIR}/vision'"

srun --environment="/capstor/scratch/cscs/$USER/containers/nemo.toml" bash -c "\
    export PYTHONPATH=${MEGATRON_PATH} && \
    echo '=== Processing audio ===' && \
    python -u ${SCRIPT_DIR}/split_buckets.py \
        --input-dirs '${AUDIO_DIR}' \
        --bucket-set vision \
        --output-dir '${OUTPUT_DIR}/audio'"

echo "Job finished at $(date)"