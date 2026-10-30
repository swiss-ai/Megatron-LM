#!/bin/bash
# Write the pruning calibration set: NUM_SEQUENCES sequences of 8,192 tokens with the
# shares of the pruning blend (stage_p_blends.sh) that the training cache never reads.
# The writer builds the datasets of the training blend, with the same seed and size, and
# reads on after the sequences of that cache. A rerun of the same command resumes.
#
#   CACHE_DIR=/path/to/calibration BLEND=/path/to/pruning/blend.txt \
#     TRAIN_BLEND=/path/to/distill/blend.txt sbatch calibration_set_8b.sbatch
#
# MODE=topk (default) also stores the teacher's top-256 at every position with loss,
# so pruned candidates can be compared with the teacher on the same sequences;
# MODE=tokens stores the data only.
#SBATCH --job-name=apertus-1p5-8b-calibration
#SBATCH --account=infra01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus_distillation/logs/slurm/%x-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus_distillation/logs/slurm/%x-%j.err
set -euo pipefail

CACHE_DIR=${CACHE_DIR:?set CACHE_DIR to the calibration set directory}
NUM_SEQUENCES=${NUM_SEQUENCES:-1024}
BLEND=${BLEND:?set BLEND to the pruning blend}
TRAIN_BLEND=${TRAIN_BLEND:?set TRAIN_BLEND to the blend of the training cache}
MODE=${MODE:-topk}
SEED=${SEED:-41}                    # the training cache's seed: it fixes each dataset's order

WORKSPACE=${WORKSPACE:-/iopsstor/scratch/cscs/$USER/apertus_distillation}
MEGATRON_LM_DIR=${MEGATRON_LM_DIR:-$WORKSPACE/Megatron-LM-8B}
VENV=${VENV:-$WORKSPACE/.venv-stage3}
CONTAINER_ENV=${CONTAINER_ENV:-/capstor/store/cscs/swissai/infra01/containers/ngc_25-12-alps2.toml}
DATASET_CACHE_DIR=${DATASET_CACHE_DIR:-$WORKSPACE/datasets/cache}
source "$MEGATRON_LM_DIR/examples/apertus_distillation/teacher_8b_args.sh"

mkdir -p "$CACHE_DIR"
LOAD_DIR=$CACHE_DIR/.teacher-load
link_teacher "$LOAD_DIR"
# One shard per rank, in chunks of 32 sequences.
SEQS_PER_CHUNK=32
SEQS_PER_FILE=$(( NUM_SEQUENCES / SLURM_NPROCS / SEQS_PER_CHUNK * SEQS_PER_CHUNK ))
SEQS_PER_FILE=$(( SEQS_PER_FILE < SEQS_PER_CHUNK ? SEQS_PER_CHUNK : SEQS_PER_FILE ))
echo "[$(date)] $NUM_SEQUENCES calibration sequences, mode=$MODE, $SEQS_PER_FILE per shard -> $CACHE_DIR"
run_writer "$(teacher_args "$LOAD_DIR") --seed $SEED \
  --data-args-path $TRAIN_BLEND --data-cache-path $DATASET_CACHE_DIR \
  --distill-heldout-blend $BLEND --distill-heldout-after $((TARGET_TOKENS / SEQ_LEN)) \
  --distill-cache-dir $CACHE_DIR --distill-cache-num-samples $NUM_SEQUENCES \
  --distill-cache-mode $MODE --distill-topk 256 --distill-prob-dtype fp32 \
  --distill-seqs-per-file $SEQS_PER_FILE --distill-seqs-per-chunk $SEQS_PER_CHUNK \
  --distill-self-check"
echo "[$(date)] done"
