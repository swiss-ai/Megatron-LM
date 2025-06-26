#!/bin/bash

# runs the following checkpoint conversions: 
#   - torch_dist           ---> torch ,  if CKPT_IS_TORCH_DIST=true.
#   - core (torch backend) ---> HF    ,  always.


MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM
CKPT_PATH=/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/Meg-Runs/main-runs-v1/apertus3-1b-21-nodes/checkpoints/
# CKPT_PATH=/capstor/scratch/cscs/asolergi/main_run_70B_megatron/Megatron-LM/logs/Meg-Runs/main-runs-v1/apertus3-70b-512-nodes-1e-5lr/checkpoints-512-noOverlap/
export PYTHONPATH=$MEGATRON_LM_DIR:$PYTHONPATH

# [torch_dist -> torch] dependencies
CKPT_IS_TORCH_DIST=true
TORCH_DIST_SCRIPT=$MEGATRON_LM_DIR/scripts/conversion/torchdist_2_torch.py
TORCH_CKPT_SAVE_PATH=/iopsstor/scratch/cscs/$USER/Meg-Checkpoints/test-logits-ap3-1b-21n-it2040000
# TORCH_CKPT_SAVE_PATH=/iopsstor/scratch/cscs/$USER/Meg-Checkpoints/test-logits-ap3-70b-512n-it630000
# [core (torch) --> HF] dependencies
HF_SAVE_DIR=/iopsstor/scratch/cscs/$USER/Meg-Checkpoints/hf-checkpoints
SAVE_DIR=$HF_SAVE_DIR/test-logits-ap3-1b-21n-it2040000
# SAVE_DIR=$HF_SAVE_DIR/test-logits-ap3-70b-512n-it630000
mkdir -p $HF_SAVE_DIR
LOADER=core
SAVER=swissai_hf


# Run torch_dist --> torch
if [[ "$CKPT_IS_TORCH_DIST" == true ]]; then
    LOAD_DIR=$TORCH_CKPT_SAVE_PATH/torch
    echo "Running torch_dist --> torch conversion..."
    CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun $TORCH_DIST_SCRIPT \
    --bf16 \
    --load $CKPT_PATH \
    --ckpt-convert-save $TORCH_CKPT_SAVE_PATH \
    --ckpt-step 670000 \
    # --pipeline-model-parallel-size=4
# --pipeline-model-parallel-size=4  #for the 70b only (same as !!!! --nproc-per-node=4 !!!! right after torchrun)
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
    --test-logits \
    #\ --hf-tokenizer .....