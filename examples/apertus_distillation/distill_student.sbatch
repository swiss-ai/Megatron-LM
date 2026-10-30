#!/bin/bash
# Train a student from a teacher cache with pretrain_gpt_distill.py: a run of the
# distillation grid or of the KD-target ablation (README.md).
#
#   RUN=1p5b-sample CACHE_DIR=/path/to/sample-cache sbatch --nodes=8 distill_student.sbatch
#
# KD_WEIGHT=0 trains on the labels alone, from the same data; TAIL=true adds the tail
# bucket on a top-k cache. Validation reads EVAL_ITERS global batches of VALID_CACHE_DIR
# from VALID_START, by default the first shard after the training sequences, so every
# evaluation sees the same held-out data. The student starts from random weights, or
# from INIT (a Megatron checkpoint directory, e.g. a pruned model). Resubmitting the same
# RUN resumes from its last checkpoint.
#SBATCH --job-name=apertus-1p5-student
#SBATCH --account=infra01
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus_distillation/logs/slurm/%x-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus_distillation/logs/slurm/%x-%j.err
set -euo pipefail

RUN=${RUN:?set RUN to a run name}
CACHE_DIR=${CACHE_DIR:?set CACHE_DIR to the training cache}
VALID_CACHE_DIR=${VALID_CACHE_DIR:-$CACHE_DIR}
SIZE=${SIZE:-1p5b}                    # 4b, 1p5b or 0p5b
INIT=${INIT:-}                        # optional checkpoint to start from
TRAIN_TOKENS=${TRAIN_TOKENS:-30000000000}
KD_WEIGHT=${KD_WEIGHT:-0.9}
TAIL=${TAIL:-false}
GBS=${GBS:-512}                       # sequences of 8,192 tokens per step
MBS=${MBS:-1}
WARMUP_TOKENS=${WARMUP_TOKENS:-1500000000}
DECAY_FRACTION=${DECAY_FRACTION:-0.2}  # WSD: linear decay over the last 20% of steps
EVAL_ITERS=${EVAL_ITERS:-4}
EVAL_INTERVAL=${EVAL_INTERVAL:-500}
SAVE_INTERVAL=${SAVE_INTERVAL:-2000}
EXIT_MINUTES=${EXIT_MINUTES:-690}

WORKSPACE=${WORKSPACE:-/iopsstor/scratch/cscs/$USER/apertus_distillation}
MEGATRON_LM_DIR=${MEGATRON_LM_DIR:-$WORKSPACE/Megatron-LM-8B}
VENV=${VENV:-$WORKSPACE/.venv-stage3}
CONTAINER_ENV=${CONTAINER_ENV:-/capstor/store/cscs/swissai/infra01/containers/ngc_25-12-alps2.toml}
TOKENIZER=${TOKENIZER:-/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok_instruct}
RUN_DIR=${RUN_DIR:-$WORKSPACE/students/$RUN}
SEQ_LEN=8192

# Shapes of the pruned students (clade_plan.md, Step 3): width only, all 32 layers,
# 128-dimensional heads. The 0.5B ties its head to the input embedding.
case $SIZE in
    4b)   MODEL="--hidden-size 3072 --ffn-hidden-size 10240 --num-attention-heads 32 --num-query-groups 8 --untie-embeddings-and-output-weights"; LR=${LR:-2e-4} ;;
    1p5b) MODEL="--hidden-size 2048 --ffn-hidden-size 4864 --num-attention-heads 16 --num-query-groups 4 --untie-embeddings-and-output-weights"; LR=${LR:-3e-4} ;;
    0p5b) MODEL="--hidden-size 1024 --ffn-hidden-size 3360 --num-attention-heads 8 --num-query-groups 2"; LR=${LR:-6e-4} ;;
    *) echo "unknown SIZE $SIZE" >&2; exit 1 ;;
esac
MIN_LR=${MIN_LR:-$(python3 -c "print($LR / 10)")}

TRAIN_ITERS=$((TRAIN_TOKENS / (GBS * SEQ_LEN)))
WARMUP_ITERS=$((WARMUP_TOKENS / (GBS * SEQ_LEN)))
DECAY_ITERS=$(python3 -c "print(int($TRAIN_ITERS * $DECAY_FRACTION))")
SEQS_PER_FILE=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["seqs_per_file"])' \
    "$VALID_CACHE_DIR/manifest.json")
VALID_START=${VALID_START:-$(( (TRAIN_ITERS * GBS + SEQS_PER_FILE - 1) / SEQS_PER_FILE * SEQS_PER_FILE ))}

DISTILL="--distill-cache-dir $CACHE_DIR --distill-kd-weight $KD_WEIGHT \
  --distill-valid-cache-dir $VALID_CACHE_DIR --distill-valid-start $VALID_START"
if [ "$TAIL" = true ]; then DISTILL="$DISTILL --distill-tail"; fi
LOAD="--load $RUN_DIR/checkpoints"
if [ -n "$INIT" ] && [ ! -f "$RUN_DIR/checkpoints/latest_checkpointed_iteration.txt" ]; then
    LOAD="--load $INIT --finetune --no-load-optim --no-load-rng"
fi
LOGGING="--log-interval 10 --tensorboard-dir $RUN_DIR/tensorboard"
if [ -n "${WANDB_PROJECT:-}" ]; then
    LOGGING="$LOGGING --wandb-project $WANDB_PROJECT --wandb-exp-name $RUN --wandb-save-dir $RUN_DIR/wandb"
else
    export WANDB_MODE=disabled
fi

mkdir -p "$RUN_DIR/triggers" "$WORKSPACE/logs/slurm"
echo "[$(date)] $RUN: $SIZE, $TRAIN_ITERS steps of $GBS x $SEQ_LEN, kd_weight=$KD_WEIGHT tail=$TAIL"
echo "  train: $CACHE_DIR; valid: $VALID_CACHE_DIR from sequence $VALID_START"

ARGS="--num-layers 32 $MODEL --kv-channels 128 --group-query-attention \
  --max-position-embeddings $SEQ_LEN --seq-length $SEQ_LEN \
  --position-embedding-type rope --rotary-base 500000 --use-rope-scaling --rope-scaling-factor 8 \
  --normalization RMSNorm --norm-epsilon 0.00001 --xielu --qk-layernorm --qknorm-impl apex \
  --disable-bias-linear --make-vocab-size-divisible-by 128 --attention-dropout 0 --hidden-dropout 0 \
  --tokenizer-type HuggingFaceTokenizer --tokenizer-model $TOKENIZER \
  --bf16 --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 \
  --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather \
  --micro-batch-size $MBS --global-batch-size $GBS --train-iters $TRAIN_ITERS \
  --optimizer adam --adam-beta1 0.9 --adam-beta2 0.95 --weight-decay 0.1 --clip-grad 0.1 \
  --lr $LR --min-lr $MIN_LR --lr-warmup-iters $WARMUP_ITERS \
  --lr-decay-style WSD --lr-wsd-decay-style linear --lr-wsd-decay-iters $DECAY_ITERS \
  --reset-position-ids --use-packed-seq-params --no-create-attention-mask-in-dataloader \
  --calculate-per-token-loss --dataloader-type external --num-workers 2 \
  --eval-iters $EVAL_ITERS --eval-interval $EVAL_INTERVAL \
  --save $RUN_DIR/checkpoints $LOAD --save-interval $SAVE_INTERVAL --ckpt-format torch_dist \
  --exit-duration-in-mins $EXIT_MINUTES --trigger-path $RUN_DIR/triggers \
  $LOGGING $DISTILL"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-29519}
export WORLD_SIZE=$SLURM_NPROCS
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=8
export PYTHONNOUSERSITE=1
export PYTHONPATH=$MEGATRON_LM_DIR
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
srun --cpus-per-task "$SLURM_CPUS_PER_TASK" --mpi=pmix --environment="$CONTAINER_ENV" \
    --network=disable_rdzv_get --kill-on-bad-exit=1 -lu bash -c "
    source '$VENV/bin/activate' || exit 1
    cd '$MEGATRON_LM_DIR'
    RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID python3 pretrain_gpt_distill.py $ARGS
    "
echo "[$(date)] done"
