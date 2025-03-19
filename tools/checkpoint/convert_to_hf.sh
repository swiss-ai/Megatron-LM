MEGATRON_LM_DIR=/iopsstor/scratch/cscs/dfan/Megatron-LM-eval
TORCH_NODIST_PATH=$(mktemp -d -p $SCRATCH/.tmp)
# CHECKPOINT_PATH=/iopsstor/scratch/cscs/dfan/Megatron-LM/logs/Meg-Runs/robotstxt_ablation_runs/apertus-1b-21n-4096sl-504gbsz-fw-edu-robots-filtered/checkpoints/ 
# CHECKPOINT_PATH=/iopsstor/scratch/cscs/dfan/Megatron-LM/logs/Meg-Runs/robotstxt_ablation_runs/apertus-1b-21n-4096sl-504gbsz-fw-edu-robots-kept/checkpoints/
CHECKPOINT_PATH=/iopsstor/scratch/cscs/dfan/Megatron-LM/logs/Meg-Runs/robotstxt-1b-cooldown/apertus3-1b-21-nodes-100bt-NewsDomains/checkpoints
ITERATIONS=$(cat $CHECKPOINT_PATH/latest_checkpointed_iteration.txt)
HF_CKPT_PATH=/iopsstor/scratch/cscs/dfan/hf-checkpoints/robotstxt-filtered-plus-News
TOKENIZER=alehc/swissai-tokenizer
REPOS_PATH=$(mktemp -d -p $SCRATCH/.tmp)

export CUDA_DEVICE_MAX_CONNECTIONS=1

export PYTHONPATH=$MEGATRON_LM_DIR:$PYTHONPATH

REPOS_PATH=$(mktemp -d -p $SCRATCH/.tmp)
cd $REPOS_PATH
git clone https://github.com/swiss-ai/transformers.git
cd transformers
git checkout swissai-model
python -m pip install -e .

cd $MEGATRON_LM_DIR
torchrun scripts/conversion/torchdist_2_torch.py --bf16 --load=$CHECKPOINT_PATH --ckpt-step=$ITERATIONS --ckpt-convert-save=$TORCH_NODIST_PATH

python tools/checkpoint/convert.py --model-type=GPT --loader=core --saver=llama_hf --load-dir=$TORCH_NODIST_PATH/torch --save-dir=$HF_CKPT_PATH --hf-tokenizer=$TOKENIZER
