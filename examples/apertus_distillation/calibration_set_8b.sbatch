#!/bin/bash
# Write the pruning calibration set: NUM_SEQUENCES sequences of 8,192 tokens from
# the stage-3 data, with fixed shares of text, image-text and speech (by default
# 50 / 35 / 15). Inside each group datasets are weighted by size.
#
#   CACHE_DIR=/path/to/calibration sbatch calibration_set_8b.sbatch
#
# MODE=topk (default) also stores the teacher's top-256 at every position with loss,
# so pruned candidates can be compared with the teacher on the same sequences;
# MODE=tokens stores the data only. SEED differs from the stage-3 seed, so the
# calibration draw is a different shuffle from the training cache.
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
MODE=${MODE:-topk}
SEED=${SEED:-4242}
TEXT_SHARE=${TEXT_SHARE:-0.50}
VISION_SHARE=${VISION_SHARE:-0.35}
AUDIO_SHARE=${AUDIO_SHARE:-0.15}

WORKSPACE=${WORKSPACE:-/iopsstor/scratch/cscs/$USER/apertus_distillation}
MEGATRON_LM_DIR=${MEGATRON_LM_DIR:-$WORKSPACE/Megatron-LM-8B}
VENV=${VENV:-$WORKSPACE/.venv-stage3}
CONTAINER_ENV=${CONTAINER_ENV:-/capstor/store/cscs/swissai/infra01/containers/ngc_25-12-alps2.toml}
DATASET_CACHE_DIR=${DATASET_CACHE_DIR:-$WORKSPACE/datasets/cache}
TEXT_DATA_DIR=${TEXT_DATA_DIR:-/capstor/store/cscs/swissai/infra01/apertus_1p5/apertus_dataset/prepare-8b/text_stage_3}
VISION_DATA_DIR=${VISION_DATA_DIR:-/capstor/store/cscs/swissai/infra01/vision-datasets/Apertus1p5_cooldown_tokenized/8B}
AUDIO_DATA_DIR=${AUDIO_DATA_DIR:-/capstor/store/cscs/swissai/infra01/audio-datasets/Apertus1p5_cooldown_tokenized}
source "$MEGATRON_LM_DIR/examples/apertus_distillation/teacher_8b_args.sh"

mkdir -p "$CACHE_DIR"
BLEND=$CACHE_DIR/blend.txt
if [ ! -f "$BLEND" ]; then
    python3 "$MEGATRON_LM_DIR/examples/apertus_distillation/calibration_blend.py" --output "$BLEND" \
        --group text "$TEXT_SHARE" "$TEXT_DATA_DIR" \
        --group vision "$VISION_SHARE" "$VISION_DATA_DIR" \
        --group audio "$AUDIO_SHARE" "$AUDIO_DATA_DIR" \
        --exclude '*/swiss-caselaw-preprocessed/dump-1/00019_tokens'
fi

LOAD_DIR=$CACHE_DIR/.teacher-load
link_teacher "$LOAD_DIR"
# One shard per rank, in chunks of 32 sequences.
SEQS_PER_CHUNK=32
SEQS_PER_FILE=$(( NUM_SEQUENCES / SLURM_NPROCS / SEQS_PER_CHUNK * SEQS_PER_CHUNK ))
SEQS_PER_FILE=$(( SEQS_PER_FILE < SEQS_PER_CHUNK ? SEQS_PER_CHUNK : SEQS_PER_FILE ))
echo "[$(date)] $NUM_SEQUENCES calibration sequences, mode=$MODE, $SEQS_PER_FILE per shard -> $CACHE_DIR"
run_writer "$(teacher_args "$LOAD_DIR") --seed $SEED \
  --data-args-path $BLEND --data-cache-path $DATASET_CACHE_DIR \
  --distill-cache-dir $CACHE_DIR --distill-cache-num-samples $NUM_SEQUENCES \
  --distill-cache-mode $MODE --distill-topk 256 \
  --distill-seqs-per-file $SEQS_PER_FILE --distill-seqs-per-chunk $SEQS_PER_CHUNK \
  --distill-self-check"
echo "[$(date)] done"
