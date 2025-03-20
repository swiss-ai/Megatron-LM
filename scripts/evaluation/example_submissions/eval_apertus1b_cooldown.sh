# Make sure to set your WANDB_API_KEY.
export LOGS_ROOT=$SCRATCH/eval-logs/main-v1
export MODEL=/iopsstor/scratch/cscs/ahgele/main_run_megatron/Megatron-LM/logs/Meg-Runs/main-runs-v1-cooldown/apertus3-1b-21-nodes-fwedu-100bt/checkpoints/
export NAME=Apertus3-1.5B-cooldown-fwedu-100B
export ARGS="--convert-to-hf --size 1 --wandb-entity epflmlo-epfl --wandb-project swissai-eval-main-v1.1-cooldown --wandb-id $NAME --bs auto --tokens-per-iter 2064384 --tasks swissai_eval"

bash scripts/evaluation/submit_evaluation.sh $MODEL $ARGS --iterations "850000"
