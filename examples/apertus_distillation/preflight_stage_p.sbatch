#!/bin/bash
# The check workflow before the Stage P run, on one debug node. Run check_stage_p.py blends
# first (launch_stage_p.sh does it too). This job then
#   1. runs the unit tests in the container;
#   2. starts the real cache launcher on the frozen blend, in small shards in a scratch
#      folder, stops it in the middle of a shard as a failure would, and reruns it;
#   3. checks that cache: settings, data mix, every dataset present, no repeats;
#   4. writes the real pruning set, separate from the cache, and checks it, including
#      that it shares no sequence with step 2.
# After the first shards of the real run: check_stage_p.py same (see data_mix.md).
# It is safe to resubmit: every step continues where it stopped.
#SBATCH --job-name=stage-p-preflight
#SBATCH --account=infra01
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=01:20:00
#SBATCH --no-requeue
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus_distillation/logs/slurm/%x-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus_distillation/logs/slurm/%x-%j.err
set -euo pipefail
CACHE_ROOT=${CACHE_ROOT:-/capstor/store/cscs/swissai/infra01/apertus_1p5_distillation_cache}
WORKSPACE=${WORKSPACE:-/iopsstor/scratch/cscs/$USER/apertus_distillation}
SRC=$WORKSPACE/Megatron-LM-8B
HERE=$SRC/examples/apertus_distillation
CONTAINER_ENV=${CONTAINER_ENV:-/capstor/store/cscs/swissai/infra01/containers/ngc_25-12-alps2.toml}
RUN=$WORKSPACE/distill_cache_smoke/preflight
mkdir -p "$RUN"
step() { echo; echo "===== $(date +%T) $*"; }
in_container() {
    srun --ntasks=1 --nodes=1 --network=disable_rdzv_get --environment="$CONTAINER_ENV" bash -c "
    source '$WORKSPACE/.venv-stage3/bin/activate'
    export PYTHONNOUSERSITE=1 PYTHONPATH='$SRC' OMP_NUM_THREADS=8
    cd '$SRC' && $1"
}
check() { in_container "python3 '$HERE/check_stage_p.py' $* --root '$CACHE_ROOT'"; }

step "unit tests"
in_container "python3 -m pytest --noconftest -q -p no:cacheprovider tests/unit_tests/distillation | tail -3"

CACHE=$RUN/stagep
export TMPDIR=/tmp AUTO_REQUEUE=false
write_cache() {
    CACHE_DIR=$CACHE BLEND=$CACHE_ROOT/distill/blend.txt SEQS_PER_FILE=1024 \
        STOP_TOKENS=$((4 * 1024 * 8192)) bash "$HERE/cache_teacher_8b.sbatch"
}
if [ ! -f "$RUN/stopped" ]; then
    step "cache launcher, to be stopped in the middle of a shard"
    write_cache > "$RUN/first.log" 2>&1 &
    writer=$!
    until grep -q '> writing shards' "$RUN/first.log"; do
        kill -0 "$writer" 2>/dev/null || { tail -30 "$RUN/first.log"; echo "the writer ended early"; exit 1; }
        sleep 15
    done
    sleep 120
    pkill -TERM -P "$writer" srun
    for _ in $(seq 36); do kill -0 "$writer" 2>/dev/null || break; sleep 5; done
    if kill -0 "$writer" 2>/dev/null; then echo "the writer did not stop within 3 min"; exit 1; fi
    wait "$writer" || true
    echo "stopped; $(find "$CACHE/files" -name '*.tmp' | wc -l) half-written shards left"
    touch "$RUN/stopped"
fi

step "the same command again must finish the cache"
write_cache
in_container "python3 '$HERE/verify_teacher_cache.py' '$CACHE'"
check cache "'$CACHE'"

step "pruning set"
CACHE_DIR=$CACHE_ROOT/pruning/calibration-top256 BLEND=$CACHE_ROOT/pruning/blend.txt \
    TRAIN_BLEND=$CACHE_ROOT/distill/blend.txt bash "$HERE/calibration_set_8b.sbatch"
in_container "python3 '$HERE/verify_teacher_cache.py' '$CACHE_ROOT/pruning/calibration-top256'"
check pruning "'$CACHE_ROOT/pruning/calibration-top256'" --against "'$CACHE'"
echo "PREFLIGHT PASSED"
