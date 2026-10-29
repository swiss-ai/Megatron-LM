#!/bin/bash
# KD-target ablation: the same student, data, steps and schedule with four targets.
#
#   ce        labels only (KD_WEIGHT=0)
#   top256    top-256 teacher probabilities, the Apertus-v1.1 loss
#   tail      top-256 plus the teacher's remaining mass as one tail bucket
#   sample64  64 draws per position from the teacher's full distribution
#
# Writes a top-256 and a 64-sample cache over the first 448 shards (~30B tokens) of the
# Stage P data order (TARGET_TOKENS of cache_teacher_8b.sbatch), so the winner's cache
# can later be extended to the whole Stage P run. Students train on shards 0..446 and
# are all validated on shard 447 of the sample cache: KL from the teacher (unbiased),
# loss on the labels, top-1 confidence and accuracy. The students start once both
# caches are complete.
#
#   CACHE_ROOT=/iopsstor/scratch/cscs/$USER/apertus_distillation/caches \
#     bash examples/apertus_distillation/kd_target_ablation.sh
#
# Size: ~17 TB for the top-256 prefix at fp32, ~1 TB for the sampled one.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
CACHE_ROOT=${CACHE_ROOT:?set CACHE_ROOT to the directory for both caches}
SIZE=${SIZE:-1p5b}
CACHE_NODES=${CACHE_NODES:-32}
STUDENT_NODES=${STUDENT_NODES:-8}
SHARD_TOKENS=$((8192 * 8192))      # sequences per shard x tokens per sequence
TRAIN_SHARDS=${TRAIN_SHARDS:-447}  # 447 shards = 7,152 steps of 512 x 8,192 tokens

TOPK=$CACHE_ROOT/stagep-top256
SAMPLE=$CACHE_ROOT/stagep-sample64
STOP_TOKENS=$(((TRAIN_SHARDS + 1) * SHARD_TOKENS))
cache() {
    CACHE_DIR=$1 MODE=$2 SLOTS=$3 STOP_TOKENS=$STOP_TOKENS \
        sbatch --parsable --nodes="$CACHE_NODES" "$HERE/cache_teacher_8b.sbatch"
}
topk_job=$(cache "$TOPK" topk 256)
sample_job=$(cache "$SAMPLE" sample 64)
echo "caches: top-256 job $topk_job -> $TOPK; 64 samples job $sample_job -> $SAMPLE"

# One job writes each prefix (~4 h on 32 nodes). A student that starts on an incomplete
# cache stops at once and reports how many sequences are missing; resubmit it after a
# rerun of the cache job.
student() {
    local run=$1 cache_dir=$2 kd_weight=$3 tail=$4
    RUN=kd-ablation-$SIZE-$run SIZE=$SIZE CACHE_DIR=$cache_dir VALID_CACHE_DIR=$SAMPLE \
        KD_WEIGHT=$kd_weight TAIL=$tail TRAIN_TOKENS=$((TRAIN_SHARDS * SHARD_TOKENS)) \
        sbatch --parsable --nodes="$STUDENT_NODES" --dependency=afterok:$topk_job:$sample_job \
        "$HERE/distill_student.sbatch"
}
for spec in "ce $SAMPLE 0 false" "top256 $TOPK 0.9 false" "tail $TOPK 0.9 true" \
        "sample64 $SAMPLE 0.9 false"; do
    set -- $spec
    echo "student $1: job $(student "$@")"
done
