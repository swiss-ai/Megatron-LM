# Make sure to set your WANDB_API_KEY.
export LOGS_ROOT=$SCRATCH/eval-logs/main-v1
# export MODEL=/capstor/store/cscs/swissai/a06/main_run_megatron/Megatron-LM/logs/Meg-Runs/main-runs-v1/apertus3-1b-21-nodes/checkpoints/
export MODEL=/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/Meg-Runs/main-runs-v1/apertus3-3b-64-nodes/checkpoints/
export NAME=Apertus3-3B
export ARGS="--convert-to-hf --size 3 --wandb-entity epflmlo-epfl --wandb-project swissai-eval-main-v1.1 --wandb-id $NAME --bs auto --tokens-per-iter 2097152 --tasks swissai_eval"

# ITS="170000,220000,265000"
# ITS="300000,400000,500000,600000"
# ITS="700000,800000"
ITS="900000,990000"
bash scripts/evaluation/submit_evaluation.sh $MODEL $ARGS --iterations $ITS
