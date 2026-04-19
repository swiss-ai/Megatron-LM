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

JSON_FILE="/capstor/scratch/cscs/dtamayomela/tokenize_it_data/example.json"
DATA_BASE="/capstor/scratch/cscs/dtamayomela/tokenize_it_data/sft_data/output.parquet"
MEGATRON_PATH="/capstor/scratch/cscs/dtamayomela/megatron/pre-training/megatron_main"
OUTPUT_FOLDER="datasets/sft_data"

srun --environment=/capstor/scratch/cscs/dtamayomela/containers/data-pipeline.toml bash -c "\
    export PYTHONPATH=${MEGATRON_PATH}
    cd ${MEGATRON_PATH}
    python scripts/tokenization/apply_chat_template.py \
        --input-json $JSON_FILE \
        --output $DATA_BASE \
        --tokenizer swiss-ai/Apertus-8B-Instruct-2509"

python3 scripts/tokenization/prepare_dumps.py \
  --dataset-folder $(dirname $DATA_BASE) \
  --preprocessing-metadata-folder "$OUTPUT_FOLDER" \
  --n-dumps 1