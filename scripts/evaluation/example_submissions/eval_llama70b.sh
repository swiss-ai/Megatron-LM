# Make sure to set your WANDB_API_KEY.
export LOGS_ROOT=$SCRATCH/eval-logs/main-v1
export MODEL=meta-llama/Llama-3.1-70B
export TOKENIZER=$MODEL
export NAME=Llama3.1-70B
export ARGS="--size 70 --wandb-entity andreas-marfurt --wandb-project swissai-eval-main-v1.2 --wandb-id $NAME --bs auto --consumed-tokens 15000000000000 --tasks swissai_eval"

bash scripts/evaluation/submit_evaluation.sh $MODEL $ARGS
