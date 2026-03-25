#!/bin/bash

#SBATCH --account=infra01
#SBATCH --time=12:00:00
#SBATCH --job-name=remove-caption
#SBATCH --partition=normal
#SBATCH --reservation=PA-2338-RL
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM/logs/slurm/remove_caption/remove_caption_%x-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM/logs/slurm/remove_caption/remove_caption_%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=460000
#SBATCH --environment=/iopsstor/scratch/cscs/ahernnde/ncg_new_v2.toml
#SBATCH --no-requeue

echo "START TIME: $(date)"

################ Defaults ################
INPUT=""
OUTPUT_PREFIX=""
IMAGE_END_TOKEN_ID=""
EOD_TOKEN_ID=""
WORKERS=32
LOG_INTERVAL=500

################ Parse Arguments ################
while [[ $# -gt 0 ]]; do
    case $1 in
        --input)
            INPUT="$2"
            shift 2
            ;;
        --output-prefix)
            OUTPUT_PREFIX="$2"
            shift 2
            ;;
        --image-end-token-id)
            IMAGE_END_TOKEN_ID="$2"
            shift 2
            ;;
        --eod-token-id)
            EOD_TOKEN_ID="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --log-interval)
            LOG_INTERVAL="$2"
            shift 2
            ;;
        --)
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: sbatch $0 -- --input <prefix> --output-prefix <prefix> --image-end-token-id <id> --eod-token-id <id> [--workers N] [--log-interval N]"
            exit 1
            ;;
    esac
done

if [ -z "$INPUT" ] || [ -z "$OUTPUT_PREFIX" ] || [ -z "$IMAGE_END_TOKEN_ID" ] || [ -z "$EOD_TOKEN_ID" ]; then
    echo "ERROR: --input, --output-prefix, --image-end-token-id, and --eod-token-id are required."
    echo "Usage: sbatch $0 -- --input <prefix> --output-prefix <prefix> --image-end-token-id <id> --eod-token-id <id> [--workers N] [--log-interval N]"
    exit 1
fi

################ Setup ################
MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM
export PYTHONPATH="$MEGATRON_LM_DIR:$PYTHONPATH"

# Create log directory
mkdir -p /iopsstor/scratch/cscs/$USER/Megatron-LM/logs/slurm/remove_caption

cd $MEGATRON_LM_DIR

echo "Input:              $INPUT"
echo "Output prefix:      $OUTPUT_PREFIX"
echo "Image-end token ID: $IMAGE_END_TOKEN_ID"
echo "EOD token ID:       $EOD_TOKEN_ID"
echo "Workers:            $WORKERS"
echo "Log interval:       $LOG_INTERVAL"

################ Run ################
srun python $MEGATRON_LM_DIR/scripts/remove_caption/remove_caption.py \
    --input "$INPUT" \
    --output-prefix "$OUTPUT_PREFIX" \
    --image-end-token-id "$IMAGE_END_TOKEN_ID" \
    --eod-token-id "$EOD_TOKEN_ID" \
    --workers "$WORKERS" \
    --log-interval "$LOG_INTERVAL"

echo "END TIME: $(date)"
