# Make sure to set your WANDB_API_KEY.
export LOGS_ROOT=$SCRATCH/eval-logs/main-v1
# export MODEL=/capstor/store/cscs/swissai/a06/main_run_megatron/Megatron-LM/logs/Meg-Runs/main-runs-v1/apertus3-1b-21-nodes/checkpoints/
export MODEL=/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/Meg-Runs/main-runs-v1/apertus3-1b-21-nodes/checkpoints/
export NAME=Apertus3-1.5B
export ARGS="--convert-to-hf --size 1 --wandb-entity epflmlo-epfl --wandb-project swissai-eval-main-v1 --wandb-id $NAME --bs auto --tokens-per-iter 2064384 --tasks swissai_eval"

# those were in capstor at time of submitting
# bash scripts/evaluation/submit_evaluation.sh $MODEL $ARGS --iterations "60000,140000,220000,300000,380000"
# on iopsstor
bash scripts/evaluation/submit_evaluation.sh $MODEL $ARGS --iterations "460000,510000"
