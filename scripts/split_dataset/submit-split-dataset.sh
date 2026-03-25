#!/bin/bash

#SBATCH --account=infra01
#SBATCH --time=04:00:00
#SBATCH --job-name=split-dataset
#SBATCH --partition=normal
#SBATCH --reservation=PA-2338-RL
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM/logs/slurm/split-dataset/split-dataset_%x-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM/logs/slurm/split-dataset/split-dataset_%x-%j.err
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
RATIOS="0.5 0.5"
MULTIMODAL=false

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
        --ratios)
            RATIOS=""
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                RATIOS="$RATIOS $1"
                shift
            done
            RATIOS="${RATIOS# }"  # trim leading space
            ;;
        --multimodal)
            MULTIMODAL=true
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: sbatch $0 -- --input <prefix> --output-prefix <prefix> [--ratios 0.8 0.2] [--multimodal]"
            exit 1
            ;;
    esac
done

if [ -z "$INPUT" ] || [ -z "$OUTPUT_PREFIX" ]; then
    echo "ERROR: --input and --output-prefix are required."
    echo "Usage: sbatch $0 -- --input <prefix> --output-prefix <prefix> [--ratios 0.8 0.2] [--multimodal]"
    exit 1
fi

################ Setup ################
MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM
export PYTHONPATH="$MEGATRON_LM_DIR:$PYTHONPATH"

# Create log directory
mkdir -p /iopsstor/scratch/cscs/$USER/Megatron-LM/logs/slurm/split-dataset

cd $MEGATRON_LM_DIR

echo "Input:         $INPUT"
echo "Output prefix: $OUTPUT_PREFIX"
echo "Ratios:        $RATIOS"
echo "Multimodal:    $MULTIMODAL"

################ Run ################
SPLIT_ARGS=(
    --input "$INPUT"
    --output-prefix "$OUTPUT_PREFIX"
    --ratios $RATIOS
)

if [ "$MULTIMODAL" = true ]; then
    SPLIT_ARGS+=(--multimodal)
fi

srun python $MEGATRON_LM_DIR/scripts/split_dataset/split_dataset_fast.py "${SPLIT_ARGS[@]}"

echo "END TIME: $(date)"
