#!/bin/bash

#SBATCH --account=infra01
#SBATCH --time=00:59:59
#SBATCH --job-name=apertus-vocab-extension-test
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM/logs/slurm/test/%x-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM/logs/slurm/test/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --no-requeue

# Validates that loader_apertus.extend_embedding_table preserves the original
# model's behavior on token IDs within the original vocabulary.
#
# Two HuggingFace copies of the model are loaded: one untouched, one with the
# embedding/lm_head tables extended via the same code path used by
# tools/checkpoint/loader_apertus.py during conversion. Logits on random
# in-vocab tokens are compared.
#
# Configured for the 70B model: bf16 + device_map="auto" sharding across 4 GPUs
# (~280 GB total weights for the two model copies, fits in 4*80GB).
#
# Usage:
#   sbatch submit_test_vocab_extension.sh
#
# Edit MODEL_DIR and EXTEND_VOCAB_TO below for the model under test.
# The script fits an 8B model in bf16 on a single GPU; for 70B+ either run on
# CPU (drop --bf16, set --device cpu) or shard with device_map="auto" by
# editing the test script.

echo "START TIME: $(date)"

################ Configs ################
MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM

# HuggingFace Apertus checkpoint to validate against.
MODEL_DIR=/iopsstor/scratch/cscs/nirmiger/hf/Apertus-70B-2509

# Target vocab size after extension. Should match what you would pass as
# --extend-vocab-to to convert.py.
EXTEND_VOCAB_TO=262144

# Test parameters.
BATCH_SIZE=2
SEQ_LEN=32
SEED=0
PRECISION_FLAG="--bf16"   # bf16 required for 70B to fit; allows ~1e-2 logit tolerance
DEVICE="auto"             # 'auto' shards across all visible GPUs; 'cuda' uses one GPU
#########################################

# Output directory for this job's results (test stdout/err already go to slurm files;
# this is just for any auxiliary artifacts you may want to drop alongside).
RESULTS_DIR=$MEGATRON_LM_DIR/logs/slurm/test/$SLURM_JOB_NAME-$SLURM_JOB_ID
mkdir -p $RESULTS_DIR

echo "[$(date)] MODEL_DIR=$MODEL_DIR"
echo "[$(date)] EXTEND_VOCAB_TO=$EXTEND_VOCAB_TO"
echo "[$(date)] Results dir: $RESULTS_DIR"
echo "[$(date)] Slurm logs: /iopsstor/scratch/cscs/$USER/Megatron-LM/logs/slurm/test/$SLURM_JOB_NAME-$SLURM_JOB_ID.{out,err}"

ulimit -c 0

srun --output=$RESULTS_DIR/test.out --error=$RESULTS_DIR/test.err \
    --cpus-per-task=$SLURM_CPUS_PER_TASK \
    --environment=/capstor/store/cscs/swissai/infra01/containers/ngc_25-12-alps2.toml \
    --mpi=pmix \
    --network=disable_rdzv_get \
    -lu \
    bash -c "
        cd $MEGATRON_LM_DIR && \
        python scripts/conversion/test_apertus_vocab_extension.py \
            --model-dir $MODEL_DIR \
            --extend-vocab-to $EXTEND_VOCAB_TO \
            --batch-size $BATCH_SIZE \
            --seq-len $SEQ_LEN \
            --seed $SEED \
            --device $DEVICE \
            $PRECISION_FLAG
    "

EXIT_CODE=$?

echo "[$(date)] Test finished with exit code $EXIT_CODE"
echo "END TIME: $(date)"

exit $EXIT_CODE
