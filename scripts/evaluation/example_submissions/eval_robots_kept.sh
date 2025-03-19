# Make sure to set your WANDB_API_KEY.
export LOGS_ROOT=$SCRATCH/eval-logs/main-v1
export MODEL=/iopsstor/scratch/cscs/dfan/Megatron-LM/logs/Meg-Runs/robotstxt_ablation/apertus-1b-21n-4096sl-504gbsz-fw-edu-robots-kept/checkpoints/
export NAME=robots-kept-$(date '+%Y-%m-%d_%H-%M-%S')
export ARGS="--convert-to-hf --size 1 --wandb-entity meta-robots --wandb-project robots-txt-ablation-eval-correct --wandb-id $NAME --bs 64 --tokens-per-iter 2064384 --tasks scripts/evaluation/english_eval"

bash scripts/evaluation/submit_evaluation.sh $MODEL $ARGS --iterations "10000,20000,30000,40000,48441"
