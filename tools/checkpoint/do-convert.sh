#!/bin/bash

# runs the following checkpoint conversions: 
#   - torch_dist           ---> torch ,  if CKPT_IS_TORCH_DIST=true.
#   - core (torch backend) ---> HF    ,  always.


MEGATRON_LM_DIR=$(dirname $(dirname $( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P ))) # Grandparent of current file location.
CKPT_PATH=/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/Meg-Runs/main-runs-v1/apertus3-8b-128-nodes/checkpoints/

ITERATIONS=300000
ITER_STRING=$(($ITERATIONS/1000))k
# [torch_dist -> torch] dependencies
CKPT_IS_TORCH_DIST=true
TORCH_DIST_SCRIPT=$MEGATRON_LM_DIR/scripts/conversion/torchdist_2_torch.py
TORCH_CKPT_SAVE_PATH=/iopsstor/scratch/cscs/$USER/Meg-Checkpoints/apertus3-8b-128n-$ITER_STRING
# [core (torch) --> HF] dependencies
HF_SAVE_DIR=/capstor/store/cscs/swissai/a06/main_run_megatron/hf-checkpoints/
SAVE_DIR=$HF_SAVE_DIR/apertus3-8b-128n-$ITER_STRING
mkdir -p $HF_SAVE_DIR
LOADER=core
SAVER=llama_hf


# Run torch_dist --> torch
if [[ "$CKPT_IS_TORCH_DIST" == true ]]; then
    LOAD_DIR=$TORCH_CKPT_SAVE_PATH/torch
    echo "Running torch_dist --> torch conversion..."
    CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun $TORCH_DIST_SCRIPT \
    --bf16 \
    --load $CKPT_PATH \
    --ckpt-convert-save $TORCH_CKPT_SAVE_PATH \
    --ckpt-step $ITERATIONS
else
    LOAD_DIR=$CKPT_PATH
    echo "Skipping torch_dist --> torch conversion..."
fi


# Run core --> HF
echo "Running core --> HF conversion..."
python $MEGATRON_LM_DIR/tools/checkpoint/convert.py \
    --model-type GPT \
    --loader  $LOADER \
    --saver $SAVER \
    --load-dir $LOAD_DIR \
    --save-dir $SAVE_DIR \
    --hf-tokenizer alehc/swissai-tokenizer
