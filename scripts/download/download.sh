#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=download_finephrase
#SBATCH --output=/capstor/store/cscs/swissai/infra01/datasets/HuggingFaceFW/finephrase/logs/download_%j.out
#SBATCH --error=/capstor/store/cscs/swissai/infra01/datasets/HuggingFaceFW/finephrase/logs/download_%j.err
#SBATCH --time=11:59:59
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --no-requeue
#SBATCH --environment=/capstor/store/cscs/swissai/infra01/containers/data-pipeline-pretrain/data-pipeline-v1.5.toml

set -euo pipefail

DEST_DIR="/capstor/store/cscs/swissai/infra01/datasets/HuggingFaceFW/finephrase"
mkdir -p "${DEST_DIR}/logs"

# Set HF cache to avoid filling home quota
export HF_HOME="/iopsstor/scratch/cscs/$USER/hf_home"
#export HF_HUB_ENABLE_HDF5_TRANSFER=1    # faster downloads if hf_transfer installed

echo "[$(date -Iseconds)] Starting download of HuggingFaceFW/finephrase"

huggingface-cli download \
    --repo-type dataset \
    --local-dir "${DEST_DIR}" \
    HuggingFaceFW/finephrase

echo "[$(date -Iseconds)] Download complete. Files in ${DEST_DIR}:"
du -sh "${DEST_DIR}"