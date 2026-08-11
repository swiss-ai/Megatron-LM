#!/bin/bash

#SBATCH --account=infra01
#SBATCH --time=01:00:00
#SBATCH --job-name=sft-packing-init
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM/logs/slurm/init/%x-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM/logs/slurm/init/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=460000
#SBATCH --no-requeue

echo "START TIME: $(date)"

################ How To ################

# This script is to prepare training with SFT in megatron using the ApertusSFTDataset.
# Given the datasets, it initialized them and packs samples. After initialization, each dataset will print packing
# statistics to the stdout. THis includes the available number of packed samples for one epoch.
# The user can then use these counts to make informed decisions about configuring sft training:
#
# Ex. If want to use all samples once (one epoch) use --train-samples=X for Megatron training with X being the sum for all
#     packed sample counts of all datasets involved. Also dont specify fixed dataset ratios ex. 0.1 /path/to/dset
#
# Ex2:  you can use specific ratios 0.1 & 0.9 for example and different number of iters / samples. In this case the datasets
#       will need more samples (megatron requests 0.5% sample buffer) and ratios might not reflect perfect weighted distribution by num_samples.
#
# In any case the indices precomputed here dont need to be reused but will be if training settings exactly match.
#
# Cache reuse: the requested train sample count (--train-samples, or --train-iters * --global-batch-size)
# is part of the index cache hash. Since training is normally launched with --train-samples set to the
# count this script reports, run this script TWICE for cache reuse: once to learn the count, then again
# with TRAIN_SAMPLES set to it. Otherwise training silently re-packs at launch (correct, just slower).
#
# Dataset types: entries in DATASETS may carry the explicit markers used at training time,
# e.g. "sft:/data/dolly" or "pretrain:/data/fineweb". With --ap-sft set (required here),
# unmarked entries are treated as SFT. Use markers when your training run mixes SFT and
# pretraining data in one blend, and use the SAME markers here so the cache hash matches.
# Only the SFT entries are packed; pretrain entries are built as regular GPT datasets.

################ Configs ################

# Runtime/container controls
CONTAINER_ENV=/iopsstor/scratch/cscs/ahernnde/ncg_new_v2.toml
SRUN_EXTRA_PARAMS=()  # Example: SRUN_EXTRA_PARAMS=(--mpi=mpix)

# Parallelism settings do NOT affect the packing index or its cache hash —
# a single GPU works regardless of the training run's topology. These are
# only configurable so the script can run in unusual container setups.
TP=1
PP=1
EP=1
GPUS_PER_NODE=4

# Calculate required world size: TP * PP * EP
REQUIRED_GPUS=$((TP * PP * EP))
if [ $REQUIRED_GPUS -gt $GPUS_PER_NODE ]; then
    echo "ERROR: Need at least $REQUIRED_GPUS GPUs (TP=$TP * PP=$PP * EP=$EP)"
    echo "Adjust --gpus-per-node or request more nodes"
    exit 1
fi

# Dataset configuration (must match your training, including any sft:/pretrain: markers)
DATASETS=(
    /capstor/store/cscs/swissai/infra01/vision-datasets/merged/image_sft_v1_fixed
)

# Training parameters that affect index/cache identity (MUST match training if cache reuse is desired)
SEQ_LEN=8192
GBS=240
TRAIN_ITERS=10000  # Used only when TRAIN_SAMPLES is empty; requested sample count = TRAIN_ITERS * GBS
TRAIN_SAMPLES=     # Optional: set to the reported packed sample count on a second run for cache reuse
RANDOM_SEED=28
TOKENIZER_MODEL=/capstor/store/cscs/swissai/infra01/MLLM/llama3_vision_instruct_emu3_tokenizer
# This branch currently uses dataset default behavior (True) and does not expose a CLI switch.
ADD_EXTRA_TOKEN_TO_SEQUENCE=true

# Megatron source and dataset cache
MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM
DATASET_CACHE_DIR=/iopsstor/scratch/cscs/$USER/datasets/cache

# Set up ENV
cd $MEGATRON_LM_DIR
export PYTHONPATH=$MEGATRON_LM_DIR:$PYTHONPATH
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
MASTER_ADDR=$(hostname)
MASTER_PORT=25678
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK/SLURM_GPUS_PER_NODE))

ulimit -c 0

# Print configuration
echo "========================================"
echo "SFT Dataset Packing Initialization"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo ""
echo "Parallelism Settings:"
echo "  TP: $TP"
echo "  PP: $PP"
echo "  EP: $EP"
echo "  World Size: $REQUIRED_GPUS"
echo ""
echo "Data Configuration:"
echo "  Sequence Length: $SEQ_LEN"
echo "  Global Batch Size: $GBS"
echo "  Random Seed: $RANDOM_SEED"
echo "  add_extra_token_to_sequence: $ADD_EXTRA_TOKEN_TO_SEQUENCE"
echo "  Tokenizer: $TOKENIZER_MODEL"
echo "  Datasets: ${DATASETS[@]}"
echo "========================================"
echo ""

# Network size args (only needed for validation, don't affect packing)
NETWORK_SIZE_ARGS=(
    --num-layers 28
    --hidden-size 3072
    --ffn-hidden-size 8192
    --num-attention-heads 24
    --group-query-attention
    --num-query-groups 8
    --max-position-embeddings $SEQ_LEN
    --position-embedding-type rope
    --rotary-base 500000
    --use-rope-scaling
    --rope-scaling-factor 32
    --make-vocab-size-divisible-by 128
    --normalization RMSNorm
    --swiglu
)

# Parallelism args (do not affect the packing index; any topology works)
DISTRIBUTED_ARGS=(
    --tensor-model-parallel-size $TP
    --pipeline-model-parallel-size $PP
    --expert-model-parallel-size $EP
    --context-parallel-size 1
)

# Tokenizer args (MUST match training, as sft dataset needs special values in a tokenizer during initialization)
TOKENIZER_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model $TOKENIZER_MODEL
)

# Data args (MUST match training)
DATA_ARGS=(
    --data-path ${DATASETS[@]}
    --data-cache-path $DATASET_CACHE_DIR
    --split 100,0,0
    --seq-length $SEQ_LEN
    --num-workers 1
    --num-dataset-builder-threads 1
    --ap-sft
    --ap-sft-pack-samples
    --ap-sft-mask-special-tokens
)

# Training args (for sample calculation)
TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size $GBS
    --calculate-per-token-loss
)
if [ -n "$TRAIN_SAMPLES" ]; then
    TRAINING_ARGS+=(--train-samples $TRAIN_SAMPLES)
else
    TRAINING_ARGS+=(--train-iters $TRAIN_ITERS)
fi

# Other required args
MISC_ARGS=(
    --bf16
    --seed $RANDOM_SEED
    --trigger-path /tmp
)

TORCHRUN_ARGS=(
    --nproc-per-node $REQUIRED_GPUS
    --nnodes 1
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT
    --rdzv_backend c10d
    --max_restarts 0
    --tee 3
)

CMD_PREFIX="numactl --membind=0-3"

INIT_CMD="torchrun ${TORCHRUN_ARGS[@]} $MEGATRON_LM_DIR/tools/initialize_sft_dataset.py \
    ${NETWORK_SIZE_ARGS[@]} \
    ${DISTRIBUTED_ARGS[@]} \
    ${TOKENIZER_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MISC_ARGS[@]}"

# Run initialization
srun -lu \
    --environment $CONTAINER_ENV \
    "${SRUN_EXTRA_PARAMS[@]}" \
    --cpus-per-task $SLURM_CPUS_PER_TASK \
    --wait 60 \
    bash -c "$CMD_PREFIX $INIT_CMD"

EXIT_CODE=$?

echo ""
echo "========================================"
echo "Initialization completed with exit code: $EXIT_CODE"
echo "========================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
echo "NEXT STEPS:"
echo "1. Check the output above for 'Training dataset: N packed samples available'"
echo "2. Use this value for --train-samples in your actual training script"
echo "3. Ensure init and train use identical packing-related args if you want to reuse cached indices"
echo ""
    echo "Output saved to: /iopsstor/scratch/cscs/$USER/Megatron-LM/logs/slurm/init/sft-packing-init-$SLURM_JOB_ID.out"
fi

echo "END TIME: $(date)"

exit $EXIT_CODE
