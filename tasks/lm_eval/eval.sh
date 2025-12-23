#!/bin/bash

#SBATCH --nodes=1
#SBATCH --account=a139
#SBATCH --partition=normal
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --job-name=megatron_lm_eval
#SBATCH --output=eval_logs/megatron_evals-%j.out
#SBATCH --error=eval_logs/megatron_evals-%j.err
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --dependency=singleton

# export PYTHONUNBUFFERED=1

# NOTE: the eval logs will be created in the directory where the user is calling this file (eval.sh) from


# Evaluate a Megatron checkpoint using lm-evaluation-harness
# Usage: bash ckpt_convert_scripts/eval.sh

# Install lm-evaluation-harness if not already installed

# we use ngc 25.06 container

export HF_DATASETS_TRUST_REMOTE_CODE=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --tasks)
            TASKS="$2"
            shift 2
            ;;
        --results_path)
            JSON_RESULTS_PATH="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --tokenizer_type)
            TOKENIZER_TYPE="$2"
            shift 2
            ;;
        --tokenizer_model)
            TOKENIZER_MODEL="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
done

export HF_DATASETS_CACHE=$SCRATCH/hf_cache_2 # one can have dataset cache mismatch version for hf datasets

MEGATRON_LM_PATH=/users/anowak/open_source/SwissAi-Megatron-LM
export PYTHONPATH=$PYTHONPATH:$MEGATRON_LM_PATH
export ENVIRONMENT_TOML=${ENVIRONMENT_TOML:-megatron_deepep} # tested with ngc 25.06 nvidia container

export MODEL_CHECKPOINT_DIR=${MODEL_DIR:-/capstor/store/cscs/swissai/a139/andres_checkpoints/megatron_checkpoints/Qwen3-7B-A1B} # This will load the newest megatron checkpoint iter in your checkpoints folder
export TASKS=${TASKS:-winogrande}
export JSON_RESULTS_PATH=${JSON_RESULTS_PATH:-$SLURM_SUBMIT_DIR/eval_results.json}
export BATCH_SIZE=${BATCH_SIZE:-32}
export TOKENIZER_TYPE=${TOKENIZER_TYPE:-HuggingFaceTokenizer}
export TOKENIZER_MODEL=${TOKENIZER_MODEL:-swiss-ai/Apertus-70B-2509}


FULL_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}') # get megatron_lm/tasks/lm_eval path

# 2. Get the directory from that path
SCRIPT_DIR=$(dirname "$FULL_PATH")

export SCRIPT_DIR
echo "Submission Dir: $SLURM_SUBMIT_DIR"
echo "Original Script Dir: $SCRIPT_DIR"

# NOTE: don't know how to work with the dype (bf16), because megatron seems to want this before laoding the checkpoint at force?

# '' everything is treated as a literal string (so no variable expansion until after srun is called), vs "" it is expanded instead
srun --export=ALL --mpi=pmix -l --environment=${ENVIRONMENT_TOML} -u bash -lc "
echo 'Installing lm-eval 0.4.9.2'
pip install lm-eval==0.4.9.2

echo 'Running eval'

torchrun --nproc_per_node=1 $SCRIPT_DIR/eval.py \
    --load ${MODEL_CHECKPOINT_DIR} \
    --bf16 \
    --tasks ${TASKS} \
    --batch-size ${BATCH_SIZE} \
    --output-path ${JSON_RESULTS_PATH} \
    --tokenizer-type ${TOKENIZER_TYPE} \
    --tokenizer-model ${TOKENIZER_MODEL}
"


# Everything else (tokenizer, model config, TP/PP sizes, etc.) is loaded from the checkpoint automatically!
