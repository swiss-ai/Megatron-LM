#!/bin/bash
#SBATCH --job-name=tok
#SBATCH --output=scripts/tokenization/logs/tok_%j.out
#SBATCH --error=scripts/tokenization/logs/tok_%j.err
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=450G
#SBATCH --time=0:30:00
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5

HF_PATH="/iopsstor/scratch/cscs/hyukhymenko/sft-1.1-mixes/v1p5-mix-v1-19-04-linearised"
DATA_BASE="/capstor/scratch/cscs/dtamayomela/tokenize_it_data/sft_data/output.parquet"
MEGATRON_PATH="/capstor/scratch/cscs/dtamayomela/megatron/pre-training/megatron_main"
OUTPUT_FOLDER="datasets/sft_data"

srun --environment=/capstor/scratch/cscs/dtamayomela/containers/data-pipeline.toml bash -c "\
    export PYTHONPATH=${MEGATRON_PATH}
    cd ${MEGATRON_PATH}
    python scripts/tokenization/apply_chat_template.py \
        --input $HF_PATH \
        --output $DATA_BASE \
        --tokenizer swiss-ai/Apertus-8B-Instruct-2509 \
        --num-proc 287 \
        --tasks-per-worker 8"

python3 scripts/tokenization/prepare_dumps.py \
  --dataset-folder $(dirname $DATA_BASE) \
  --preprocessing-metadata-folder "$OUTPUT_FOLDER" \
  --n-dumps 1