# Make sure to set your WANDB_API_KEY.
export LOGS_ROOT=$SCRATCH/eval-logs/main-v1
export MODEL=Qwen/Qwen2.5-72B
export TOKENIZER=$MODEL
export NAME=Qwen2.5-72B
export ARGS="--size 70 --wandb-entity andreas-marfurt --wandb-project swissai-eval-main-v1.2 --wandb-id $NAME --bs auto --consumed-tokens 18000000000000 --tasks swissai_eval"

bash scripts/evaluation/submit_evaluation.sh $MODEL $ARGS
