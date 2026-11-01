#!/bin/bash
# The check workflow before the rest of Stage 3, on one node of preemptable. If the job is
# preempted Slurm requeues it, and every step picks up where it stopped.
#
#   sbatch examples/apertus_distillation/preflight_stage3_rest.sbatch
#
# It freezes distill-rest/blend.txt and checks it against what Stage P and the pruning set
# left. It writes the pruning set again and compares it with the real one, because the
# skip relies on finding exactly those sequences. Then it writes the first sequences of
# the rest with the settings of the real run and checks them.
#
# After the first shards of the real run: check_stage_p.py same, against $RUN/rest.
#SBATCH --job-name=stage3-rest-preflight
#SBATCH --account=infra01
#SBATCH --partition=preemptable
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/capstor/store/cscs/swissai/infra01/apertus_1p5_distillation_cache/logs/%x-%j.out
#SBATCH --error=/capstor/store/cscs/swissai/infra01/apertus_1p5_distillation_cache/logs/%x-%j.err
set -euo pipefail
WORKSPACE=${WORKSPACE:-/iopsstor/scratch/cscs/$USER/apertus_distillation}
SRC=$WORKSPACE/Megatron-LM-8B
HERE=$SRC/examples/apertus_distillation
CONTAINER_ENV=${CONTAINER_ENV:-/capstor/store/cscs/swissai/infra01/containers/ngc_25-12-alps2.toml}
source "$HERE/stage3_rest_env.sh"
RUN=$CACHE_ROOT/preflight/rest-run
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

step "the frozen rest"
in_container "python3 '$HERE/stage3_rest_blend.py' --root '$CACHE_ROOT'"
check rest

step "the pruning set again: the skip must find exactly its sequences"
(unset HELD_OUT TARGET_TOKENS SEQS_PER_CHUNK
 CACHE_DIR=$RUN/pruning-again BLEND=$CACHE_ROOT/pruning/blend.txt \
     TRAIN_BLEND=$CACHE_ROOT/distill/blend.txt bash "$HERE/calibration_set_8b.sbatch")
check same "'$RUN/pruning-again'" --against "'$CACHE_ROOT/pruning/calibration-top256'"

step "the first sequences of the rest, with the settings of the real run"
export AUTO_REQUEUE=false
CACHE_DIR=$RUN/rest SEQS_PER_FILE=1024 STOP_TOKENS=$((4 * 1024 * 8192)) \
    bash "$HERE/cache_teacher_8b.sbatch"
in_container "python3 '$HERE/verify_teacher_cache.py' '$RUN/rest'"
check rest-cache "'$RUN/rest'"
echo "PREFLIGHT PASSED"
