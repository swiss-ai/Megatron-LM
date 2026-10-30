# Flags shared by the Apertus 1.5 8B cache launchers: the stage-3 teacher and the
# data flags of its stage-3 run (SFT packing and masks, Goldfish, modality weights).
# Source it after setting the variables below; it defines TEACHER_ARGS and
# link_teacher, and leaves the data blend and the distill-* flags to the caller.

TEACHER=${TEACHER:-/capstor/store/cscs/swissai/infra01/apertus_checkpoints/v1p5/megatron/8B/pretraining/stage3_iter_0478000}
TEACHER_STEP=${TEACHER_STEP:-478000}
TOKENIZER=${TOKENIZER:-/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok_instruct}
SEQ_LEN=${SEQ_LEN:-8192}
MBS=${MBS:-8}
# The Stage P cache: 24,051,712 sequences (197B tokens), whole shards of 8,192 sequences and
# of 1,024, so a preflight with small shards reads the same data order. This number sizes
# every dataset of the blend, so a held-out set must use the same one.
TARGET_TOKENS=${TARGET_TOKENS:-197031624704}

# Megatron finds checkpoints by step; link the teacher under that name in $1.
link_teacher() {
    mkdir -p "$1"
    ln -sfn "$TEACHER" "$1/iter_$(printf %07d "$TEACHER_STEP")"
    echo "$TEACHER_STEP" > "$1/latest_checkpointed_iteration.txt"
}

teacher_args() {
    local load_dir=$1
    echo "--load $load_dir --ckpt-format torch_dist --ckpt-step $TEACHER_STEP \
  --no-load-optim --no-load-rng --finetune --no-initialization --exit-on-missing-checkpoint \
  --use-checkpoint-args --no-use-tokenizer-model-from-checkpoint-args --ckpt-fully-parallel-load \
  --num-layers 32 --hidden-size 4096 --ffn-hidden-size 21504 \
  --num-attention-heads 32 --group-query-attention --num-query-groups 8 \
  --max-position-embeddings $SEQ_LEN --seq-length $SEQ_LEN \
  --position-embedding-type rope --rotary-base 500000 --use-rope-scaling --rope-scaling-factor 8 \
  --normalization RMSNorm --norm-epsilon 0.00001 --xielu --qk-layernorm --qknorm-impl apex \
  --untie-embeddings-and-output-weights --disable-bias-linear \
  --make-vocab-size-divisible-by 128 --attention-dropout 0 --hidden-dropout 0 \
  --bf16 --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --micro-batch-size $MBS \
  --tokenizer-type HuggingFaceTokenizer --tokenizer-model $TOKENIZER \
  --ap-sft --ap-sft-mask-special-tokens --ap-sft-pack-samples --ap-sft-packing-strategy bfd \
  --calculate-per-token-loss \
  --split 100,0,0 --reset-position-ids --use-packed-seq-params \
  --no-create-attention-mask-in-dataloader --eod-mask-loss \
  --num-workers 2 --num-dataset-builder-threads 4 \
  --goldfish-loss --goldfish-k 50 --goldfish-h 50 \
  --loss-mask-token-ids 131082 131083 131084 --audio-weight 0 --vision-weight 0 \
  --eval-iters 0 --eval-interval 1000000000 --log-interval 1 --distill-vocab-size 131072"
}

# Launch get_teacher_logits.py on every allocated GPU with the given arguments.
run_writer() {
    export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    export MASTER_PORT=${MASTER_PORT:-29517}
    export WORLD_SIZE=$SLURM_NPROCS
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    export OMP_NUM_THREADS=8
    export PYTHONNOUSERSITE=1
    export PYTHONPATH=$MEGATRON_LM_DIR
    export HF_HUB_OFFLINE=1
    export TOKENIZERS_PARALLELISM=false
    export WANDB_MODE=disabled
    srun --cpus-per-task "$SLURM_CPUS_PER_TASK" --mpi=pmix --environment="$CONTAINER_ENV" \
        --network=disable_rdzv_get --kill-on-bad-exit=1 -lu bash -c "
        source '$VENV/bin/activate' || exit 1
        cd '$MEGATRON_LM_DIR'
        RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID python3 get_teacher_logits.py $*
        "
}
