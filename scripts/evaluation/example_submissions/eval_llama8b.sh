# Make sure to set your WANDB_API_KEY.
export LOGS_ROOT=$SCRATCH/eval-logs/main-v1
export MODEL=meta-llama/Llama-3.1-8B
export TOKENIZER=$MODEL
export NAME=Llama3.1-8B
export ARGS="--size 8 --wandb-entity epflmlo-epfl --wandb-project swissai-eval-main-v1 --wandb-id $NAME --bs auto --consumed-tokens 15000000000000 --tasks swissai_eval"

bash scripts/evaluation/submit_evaluation.sh $MODEL $ARGS
