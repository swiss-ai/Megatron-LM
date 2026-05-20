# Make sure to set your WANDB_API_KEY.
export LOGS_ROOT=$SCRATCH/eval-logs/main-v1
ROOT_DIR=/capstor/store/cscs/swissai/infra01/users/dfan/hf-checkpoints/meta-data
CKPT_NAME=meta-0-standard
# export MODEL=/iopsstor/scratch/cscs/dfan/Megatron-LM/logs/Meg-Runs/robotstxt_ablation_runs/apertus-1b-21n-4096sl-504gbsz-fw-edu-robots-filtered/checkpoints/ 
# export MODEL=/iopsstor/scratch/cscs/dfan/Megatron-LM/logs/Meg-Runs/robotstxt_ablation_runs/apertus-1b-21n-4096sl-504gbsz-fw-edu-robots-filtered-plus-Top1-domains/checkpoints/
# export MODEL=/iopsstor/scratch/cscs/dfan/Megatron-LM/logs/Meg-Runs/robotstxt_ablation_runs/apertus-1b-21n-4096sl-504gbsz-fw-edu-robots-filtered-plus-Top5-domains/checkpoints/
# export MODEL=/iopsstor/scratch/cscs/dfan/Megatron-LM/logs/Meg-Runs/robotstxt_ablation_runs/apertus-1b-21n-4096sl-504gbsz-fw-edu-robots-filtered-plus-Top10-domains/checkpoints/
# export MODEL=/iopsstor/scratch/cscs/dfan/Megatron-LM/logs/Meg-Runs/meta_data_conditioning_masking_out/apertus-1b-21n-4096sl-504gbsz-fw-edu-url-0.9_non_url-0.1-APPEND/checkpoints/
# export MODEL=/iopsstor/scratch/cscs/dfan/Megatron-LM/logs/Meg-Runs/meta_data_conditioning_masking_out/apertus-1b-21n-4096sl-504gbsz-fw-edu-0.9-5-meta-tokens-correct/checkpoints/
LM_EVAL_BRANCH=std-5shot
ADD_BOS=false  # set to true to prepend BOS token
export MODEL=$ROOT_DIR/$CKPT_NAME/
export NAME=std-5shots-$(date '+%Y-%m-%d_%H-%M-%S')
BOS_FLAG=$( [ "$ADD_BOS" = true ] && echo "--add-bos-token" || echo "" )
export ARGS="--convert-to-hf --size 1 --partition debug $BOS_FLAG --wandb-entity meta-robots --wandb-project meta_eval_bos --wandb-id $NAME --bs 32 --tokens-per-iter 2064384 --tasks scripts/evaluation/english_eval --lm-eval-branch $LM_EVAL_BRANCH"

bash scripts/evaluation/submit_evaluation.sh $MODEL $ARGS --iterations "48441"
