#!/bin/bash

# runs the following checkpoint conversions: 
#   - torch_dist           ---> torch ,  if CKPT_IS_TORCH_DIST=true.
#   - core (torch backend) ---> HF    ,  always.


MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/distillation/Megatron-LM-QAT/
export PYTHONPATH=$MEGATRON_LM_DIR:$PYTHONPATH

# CKPT_PATH=/iopsstor/scratch/cscs/schlag/experiments/merge-for-v2/Megatron-LM/logs/Meg-Runs/apertus2_baselines/apertus2-1b-21n-4096sl-504gbsz-ademamix-wsd-xielu-crossDocAttn-goldfish-beta2-qkNorm-untie/checkpoints
CKPT_PATH=/capstor/scratch/cscs/dhia680/qat-distill/Megatron-LM-QAT/logs/Meg-Runs/ap1b-1n/checkpoints/    # HEREEEE

# [torch_dist -> torch] dependencies
CKPT_IS_TORCH_DIST=true
TORCH_DIST_SCRIPT=$MEGATRON_LM_DIR/scripts/conversion/torchdist_2_torch.py
# TORCH_CKPT_SAVE_PATH=/iopsstor/scratch/cscs/$USER/Meg-Checkpoints/test2-apertus2-1b-21n
TORCH_CKPT_SAVE_PATH=/capstor/scratch/cscs/dhia680/qat-distill/Megatron-LM-QAT/logs/Meg-Runs/ap1b-1n/torch-checkpoints/ap1b-1n   # HEREEEE
# [core (torch) --> HF] dependencies
# HF_SAVE_DIR=/iopsstor/scratch/cscs/$USER/Meg-Checkpoints/hf-checkpoints
HF_SAVE_DIR=/iopsstor/scratch/cscs/dhia680/distillation/converted-hf
# SAVE_DIR=$HF_SAVE_DIR/test2-apertus2-1b-21n
SAVE_DIR=$HF_SAVE_DIR/ap1b-1n   # HEREEEE
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
    --ckpt-convert-save $TORCH_CKPT_SAVE_PATH
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


cp -r $SAVE_DIR /capstor/store/cscs/swissai/a140/checkpoints/hf-debug-ckpts/