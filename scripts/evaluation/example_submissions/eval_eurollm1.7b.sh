# Make sure to set your WANDB_API_KEY.
export LOGS_ROOT=$SCRATCH/eval-logs/main-v1
export MODEL=utter-project/EuroLLM-1.7B
export TOKENIZER=$MODEL
export NAME=EuroLLM-1.7B
export ARGS="--size 1 --wandb-entity epflmlo-epfl --wandb-project swissai-eval-main-v1.1 --wandb-id $NAME --bs auto --consumed-tokens 4000000000000 --tasks swissai_eval"

bash scripts/evaluation/submit_evaluation.sh $MODEL $ARGS
