# Make sure to set your WANDB_API_KEY.
export LOGS_ROOT=$SCRATCH/eval-logs/main-v1
# export MODEL=/capstor/store/cscs/swissai/a06/main_run_megatron/Megatron-LM/logs/Meg-Runs/main-runs-v1/apertus3-8b-128-nodes/checkpoints/
export MODEL=/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/Meg-Runs/main-runs-v1/apertus3-8b-128-nodes/checkpoints/
export NAME=Apertus3-8B
export ARGS="--convert-to-hf --size 8 --wandb-entity epflmlo-epfl --wandb-project swissai-eval-main-v1 --wandb-id $NAME --bs auto --tokens-per-iter 4194304 --tasks swissai_eval"

ITS="162000,192000,222000,262000"
bash scripts/evaluation/submit_evaluation.sh $MODEL $ARGS --iterations $ITS
