#!/bin/bash
# Write a teacher cache from the Apertus 1.5 8B stage-3 checkpoint over the frozen blend
# of stage_p_blends.sh. launch_stage_p.sh runs this with the settings of the Stage P cache.
#
#   BLEND=/path/to/blend.txt CACHE_DIR=/path/to/cache sbatch --nodes=32 cache_teacher_8b.sbatch
#
# TARGET_TOKENS fixes the data order: set it to the largest cache you may want. A
# rerun with the same settings resumes after the completed files (progress.json),
# on any number of nodes. STOP_TOKENS writes only the first shards, e.g. a prefix for
# an ablation; a later run without it continues the same order. AUTO_REQUEUE=true
# submits the next job at the end of every job that exits cleanly with new shards,
# until the cache (or its STOP_TOKENS prefix) is complete; a job that fails or adds
# no shard is not resubmitted, and running the same command again continues the cache.
# Check the result with examples/apertus_distillation/verify_teacher_cache.py.
#SBATCH --job-name=apertus-1p5-8b-teacher-cache
#SBATCH --account=infra01
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus_distillation/logs/slurm/%x-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus_distillation/logs/slurm/%x-%j.err
set -euo pipefail

CACHE_DIR=${CACHE_DIR:?set CACHE_DIR to the cache directory}
BLEND=${BLEND:?set BLEND to the blend file of the data mix}
STOP_TOKENS=${STOP_TOKENS:-}        # optional: write only the shards covering these first tokens
MODE=${MODE:-topk}                  # topk, sample or tokens
SLOTS=${SLOTS:-256}                 # top-k size, or draws per position in sample mode
SEQS_PER_FILE=${SEQS_PER_FILE:-8192}  # sequences per shard (~37 GB at fp32 top-256; ~2.9k shards for 197B)
SEQS_PER_CHUNK=${SEQS_PER_CHUNK:-8}   # sequences per compressed chunk; students read whole chunks
SEED=${SEED:-41}                    # data order; 41 is the teacher's stage-3 seed
EXIT_MINUTES=${EXIT_MINUTES:-700}    # stop at a shard boundary before the 12 h limit
# Over one round (~40 min): ranks with no shard in a short last round wait in a collective.
TIMEOUT_MINUTES=${TIMEOUT_MINUTES:-120}
AUTO_REQUEUE=${AUTO_REQUEUE:-false}
HELD_OUT=${HELD_OUT:-}               # writer arguments of a cache that reads on after another

WORKSPACE=${WORKSPACE:-/iopsstor/scratch/cscs/$USER/apertus_distillation}
MEGATRON_LM_DIR=${MEGATRON_LM_DIR:-$WORKSPACE/Megatron-LM-8B}
VENV=${VENV:-$WORKSPACE/.venv-stage3}
CONTAINER_ENV=${CONTAINER_ENV:-/capstor/store/cscs/swissai/infra01/containers/ngc_25-12-alps2.toml}
DATASET_CACHE_DIR=${DATASET_CACHE_DIR:-$WORKSPACE/datasets/cache}
source "$MEGATRON_LM_DIR/examples/apertus_distillation/teacher_8b_args.sh"

completed_shards() {
    python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["completed_files"])' \
        "$CACHE_DIR/progress.json" 2>/dev/null || echo 0
}
START=$(completed_shards)

LOAD_DIR=$CACHE_DIR/.teacher-load
link_teacher "$LOAD_DIR"

NUM_SAMPLES=$((TARGET_TOKENS / SEQ_LEN))
STOP_ARGS=
if [ -n "$STOP_TOKENS" ]; then
    STOP_FILES=$(( (STOP_TOKENS / SEQ_LEN + SEQS_PER_FILE - 1) / SEQS_PER_FILE ))
    STOP_ARGS="--distill-cache-stop-files $STOP_FILES"
fi
echo "[$(date)] $SLURM_NPROCS ranks, $NUM_SAMPLES sequences, mode=$MODE slots=$SLOTS $STOP_ARGS -> $CACHE_DIR"
run_writer "$(teacher_args "$LOAD_DIR") --seed $SEED $STOP_ARGS $HELD_OUT \
  --data-args-path $BLEND --data-cache-path $DATASET_CACHE_DIR \
  --distill-cache-dir $CACHE_DIR --distill-cache-num-samples $NUM_SAMPLES \
  --distill-cache-mode $MODE --distill-topk $SLOTS --distill-samples $SLOTS --distill-prob-dtype fp32 \
  --distill-seqs-per-file $SEQS_PER_FILE --distill-seqs-per-chunk $SEQS_PER_CHUNK \
  --distill-exit-duration-in-mins $EXIT_MINUTES --distill-self-check \
  --distributed-timeout-minutes $TIMEOUT_MINUTES"

DONE=$(completed_shards)
TOTAL=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["num_files"])' \
    "$CACHE_DIR/manifest.json")
echo "[$(date)] $DONE/$TOTAL shards complete"
GOAL=$TOTAL
if [ -n "$STOP_TOKENS" ] && [ "$STOP_FILES" -lt "$TOTAL" ]; then GOAL=$STOP_FILES; fi
# Reached only if the writer exited cleanly (set -e).
if [ "$AUTO_REQUEUE" = true ] && [ "$DONE" -lt "$GOAL" ]; then
    if [ "$DONE" -gt "$START" ]; then
        sbatch --nodes="$SLURM_JOB_NUM_NODES" --export=ALL \
            "$MEGATRON_LM_DIR/examples/apertus_distillation/cache_teacher_8b.sbatch"
    else
        echo "[$(date)] no new shards in this job, not resubmitting"
    fi
fi
echo "[$(date)] done"
