# Make sure to set your WANDB_API_KEY.
export LOGS_ROOT=$SCRATCH/eval-logs/main-v1
export MODEL=allenai/OLMo-2-1124-7B
export TOKENIZER=$MODEL
export NAME=OLMo-2-1124-7B
export ARGS="--size 8 --wandb-entity epflmlo-epfl --wandb-project swissai-eval-main-v1 --wandb-id $NAME --bs auto --tokens-per-iter 4194304 --tasks swissai_eval"

ITS="50000,251000,502000,753000,928646"
bash scripts/evaluation/submit_evaluation.sh $MODEL $ARGS --iterations $ITS --revisions stage1-step50000-tokens210B,stage1-step251000-tokens1053B,stage1-step502000-tokens2106B,stage1-step753000-tokens3159B,stage1-step928646-tokens3896B

# note that it doesnt include the cooldown (their 2nd stage) yet, neither the final model