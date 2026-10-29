#!/bin/bash
# Write a teacher cache from the Apertus 1.5 8B stage-3 checkpoint over the stage-3
# data mix, blended by size as in the teacher's own cooldown.
#
#   CACHE_DIR=/path/to/cache TARGET_TOKENS=197000000000 sbatch --nodes=32 cache_teacher_8b.sbatch
#
# TARGET_TOKENS fixes the data order: set it to the largest cache you may want. A
# rerun with the same settings resumes after the completed files (progress.json),
# on any number of nodes; AUTO_REQUEUE=true chains jobs until the cache is complete.
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
TARGET_TOKENS=${TARGET_TOKENS:-197000000000}
MODE=${MODE:-topk}                  # topk, sample or tokens
SLOTS=${SLOTS:-256}                 # top-k size, or draws per position in sample mode
SEQS_PER_FILE=${SEQS_PER_FILE:-8192}  # sequences per shard (~37 GB at fp32 top-256; ~2.9k shards for 197B)
SEQS_PER_CHUNK=${SEQS_PER_CHUNK:-32}   # sequences per compressed chunk inside a shard
SEED=${SEED:-41}                    # data order; 41 is the teacher's stage-3 seed
EXIT_MINUTES=${EXIT_MINUTES:-700}    # stop at a shard boundary before the 12 h limit
AUTO_REQUEUE=${AUTO_REQUEUE:-false}

WORKSPACE=${WORKSPACE:-/iopsstor/scratch/cscs/$USER/apertus_distillation}
MEGATRON_LM_DIR=${MEGATRON_LM_DIR:-$WORKSPACE/Megatron-LM-8B}
VENV=${VENV:-$WORKSPACE/.venv-stage3}
CONTAINER_ENV=${CONTAINER_ENV:-/capstor/store/cscs/swissai/infra01/containers/ngc_25-12-alps2.toml}
DATASET_CACHE_DIR=${DATASET_CACHE_DIR:-$WORKSPACE/datasets/cache}
TEXT_DATA_DIR=${TEXT_DATA_DIR:-/capstor/store/cscs/swissai/infra01/apertus_1p5/apertus_dataset/prepare-8b/text_stage_3}
source "$MEGATRON_LM_DIR/examples/apertus_distillation/teacher_8b_args.sh"

if [ "$AUTO_REQUEUE" = true ]; then
    sbatch --dependency=singleton --job-name="$SLURM_JOB_NAME" --nodes="$SLURM_JOB_NUM_NODES" \
        --export=ALL "$MEGATRON_LM_DIR/examples/apertus_distillation/cache_teacher_8b.sbatch"
fi

LOAD_DIR=$CACHE_DIR/.teacher-load
link_teacher "$LOAD_DIR"

# The stage-3 blend: every prefix, no weights, SFT prefixes tagged, one corrupt shard excluded.
DATA_PATH=$(find -L "$TEXT_DATA_DIR" \
    /capstor/store/cscs/swissai/infra01/audio-datasets/Apertus1p5_cooldown_tokenized \
    /capstor/store/cscs/swissai/infra01/vision-datasets/Apertus1p5_cooldown_tokenized/8B \
    -name '*.bin' | while read -r f; do
        [[ "$(readlink -f "$f")" == */swiss-caselaw-preprocessed/dump-1/00019_tokens.bin ]] || echo "$f"
    done | sed 's/\.bin$//' | sort | \
    awk '{ if ($0 ~ /apertus_sft/) print "sft:" $0; else print "pretrain:" $0 }' | tr '\n' ' ')

NUM_SAMPLES=$((TARGET_TOKENS / SEQ_LEN))
echo "[$(date)] $SLURM_NPROCS ranks, $NUM_SAMPLES sequences, mode=$MODE slots=$SLOTS -> $CACHE_DIR"
run_writer "$(teacher_args "$LOAD_DIR") --seed $SEED \
  --data-path $DATA_PATH --data-cache-path $DATASET_CACHE_DIR \
  --distill-cache-dir $CACHE_DIR --distill-cache-num-samples $NUM_SAMPLES \
  --distill-cache-mode $MODE --distill-topk $SLOTS --distill-samples $SLOTS \
  --distill-seqs-per-file $SEQS_PER_FILE --distill-seqs-per-chunk $SEQS_PER_CHUNK \
  --distill-exit-duration-in-mins $EXIT_MINUTES --distill-self-check"

if [ "$AUTO_REQUEUE" = true ] && python3 - "$CACHE_DIR" <<'EOF'
import json, sys
progress = json.load(open(sys.argv[1] + '/progress.json'))
manifest = json.load(open(sys.argv[1] + '/manifest.json'))
sys.exit(0 if progress['completed_files'] >= manifest['num_files'] else 1)
EOF
then
    echo "[$(date)] cache complete, cancelling the queued follow-up job"
    scancel --jobname "$SLURM_JOB_NAME" --state=PENDING
fi
echo "[$(date)] done"
