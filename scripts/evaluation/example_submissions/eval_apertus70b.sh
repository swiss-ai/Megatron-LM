# Make sure to set your WANDB_API_KEY.
export LOGS_ROOT=$SCRATCH/eval-logs/main-v1
export MODEL=/iopsstor/scratch/cscs/schlag/main_run_70B_megatron/Megatron-LM/logs/Meg-Runs/main-runs-v1/apertus3-70b-512-nodes-1e-5lr/checkpoints/
export NAME=Apertus3-70B
export CONTAINER_PATH=/iopsstor/scratch/cscs/ahernnde/ncg_pt.toml
export ARGS="--convert-to-hf --size 70 --wandb-entity epflmlo-epfl --wandb-project swissai-eval-main-v1.1 --wandb-id $NAME --bs auto --tokens-per-iter 8388608 --tasks swissai_eval"

ITS="53000"
bash scripts/evaluation/submit_evaluation.sh $MODEL $ARGS --iterations $ITS
