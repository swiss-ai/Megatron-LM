#!/bin/bash
# Start or resume the Stage P top-256 cache: the same command does both.
#
#   bash examples/apertus_distillation/launch_stage_p.sh
#
# Each job stops at a shard boundary before its time limit and submits the next one.
# After a failure or a stop nothing is resubmitted: run this command again, on any number
# of nodes. It continues after the last finished shards, drops half-written ones, and
# the manifest refuses any change of settings. It first checks the blends and the input
# files, and refuses while a job of this cache is queued or running: two writers on one
# cache would overwrite each other's shards. Everything the run produces stays under
# CACHE_ROOT: the cache, the dataset index and the job logs.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
CACHE_ROOT=${CACHE_ROOT:-/capstor/store/cscs/swissai/infra01/apertus_1p5_distillation_cache}
NODES=${NODES:-32}
export SBATCH_JOB_NAME=stagep-top256  # also names the jobs that follow
export CACHE_DIR=$CACHE_ROOT/distill/stagep-top256
export BLEND=$CACHE_ROOT/distill/blend.txt
export MODE=topk SLOTS=256 SEQS_PER_FILE=8192 SEQS_PER_CHUNK=8 AUTO_REQUEUE=true
export TMPDIR=/tmp  # short: DataLoader workers put their sockets here
export DATASET_CACHE_DIR=$CACHE_ROOT/dataset-index
export SBATCH_OUTPUT=$CACHE_ROOT/logs/%x-%j.out SBATCH_ERROR=$CACHE_ROOT/logs/%x-%j.err

if squeue -h -u "$USER" -n "$SBATCH_JOB_NAME" -o '%i %T' | grep .; then
    echo "a job of this cache is still in the queue; wait for it or cancel it" >&2
    exit 1
fi
mkdir -p "$CACHE_ROOT/logs" "$DATASET_CACHE_DIR"
python3 "$HERE/check_stage_p.py" blends --root "$CACHE_ROOT"
completed=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["completed_files"])' \
    "$CACHE_DIR/progress.json" 2>/dev/null || echo 0)
echo "$completed shards complete in $CACHE_DIR; submitting on $NODES nodes"
sbatch --nodes="$NODES" "$HERE/cache_teacher_8b.sbatch"
