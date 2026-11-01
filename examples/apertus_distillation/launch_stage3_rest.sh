#!/bin/bash
# Start or resume the cache of the rest of Stage 3: the same command does both.
#
#   bash examples/apertus_distillation/launch_stage3_rest.sh
#
# Run preflight_stage3_rest.sbatch first: it freezes distill-rest/blend.txt and checks it.
# The job runs on preemptable. Slurm requeues it when it is preempted, and it resumes after
# the last finished shards. A job that fails or adds no shard is not resubmitted: run this
# command again. It refuses while a job of this cache is queued or running, because two
# writers on one cache would overwrite each other's shards.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/stage3_rest_env.sh"
NODES=${NODES:-32}
export SBATCH_JOB_NAME=stage3-rest-top256  # also names the jobs that follow
export CACHE_DIR=$CACHE_ROOT/distill-rest/stage3-rest-top256
export SEQS_PER_FILE=$SEQS_PER_SHARD AUTO_REQUEUE=true
export SBATCH_PARTITION=${SBATCH_PARTITION:-preemptable}
export SBATCH_REQUEUE=1 SBATCH_OPEN_MODE=append  # keep the log of the earlier attempts
export SBATCH_TIMELIMIT=24:00:00 EXIT_MINUTES=1400
export SBATCH_OUTPUT=$CACHE_ROOT/logs/%x-%j.out SBATCH_ERROR=$CACHE_ROOT/logs/%x-%j.err

if squeue -h -u "$USER" -n "$SBATCH_JOB_NAME" -o '%i %T' | grep .; then
    echo "a job of this cache is still in the queue; wait for it or cancel it" >&2
    exit 1
fi
if [ ! -f "$CACHE_ROOT/distill-rest/blend.txt" ]; then
    echo "no frozen blend: run preflight_stage3_rest.sbatch first" >&2
    exit 1
fi
python3 "$HERE/check_stage_p.py" blends --root "$CACHE_ROOT"
completed=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["completed_files"])' \
    "$CACHE_DIR/progress.json" 2>/dev/null || echo 0)
echo "$completed shards complete in $CACHE_DIR; submitting on $NODES nodes to $SBATCH_PARTITION"
sbatch --nodes="$NODES" "$HERE/cache_teacher_8b.sbatch"
