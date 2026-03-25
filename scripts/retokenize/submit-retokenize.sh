#!/bin/bash

#SBATCH --account=infra01
#SBATCH --time=12:00:00
#SBATCH --job-name=retokenize
#SBATCH --partition=normal
#SBATCH --reservation=PA-2338-RL
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM/logs/slurm/retokenize/retokenize_%x-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM/logs/slurm/retokenize/retokenize_%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=460000
#SBATCH --environment=/iopsstor/scratch/cscs/ahernnde/ncg_new_v2.toml
#SBATCH --no-requeue

echo "START TIME: $(date)"

################ Defaults ################
#INPUT="/users/rkreft/scratch/ap1p5_tokenized_ipsstor/merged/llavaOv1_5_Midtrain_imgonly_apertus8b_emu3p5_merged"
INPUT="/users/rkreft/scratch/ap1p5_tokenized_ipsstor/merged/llavaOv1_5_Midtrain_paired_apertus8b_emu3p5_merged"

#OUTPUT_PREFIX="/users/rkreft/scratch/llama3_emu3p5_data/llavaOv_1_5_Midtrain_img_only_merged"
OUTPUT_PREFIX="/users/rkreft/scratch/llama3_emu3p5_data/llavaOv_1_5_Midtrain_paired_merged"

SOURCE_TOKENIZER=/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5
TARGET_TOKENIZER=/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/llama_emu3.5_base
MULTIMODAL=false
UNORDERED=false
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
        --source-tokenizer)
            SOURCE_TOKENIZER="$2"
            shift 2
            ;;
        --target-tokenizer)
            TARGET_TOKENIZER="$2"
            shift 2
            ;;
        --multimodal)
            MULTIMODAL=true
            shift
            ;;
        --unordered)
            UNORDERED=true
            shift
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --log-interval)
            LOG_INTERVAL="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: sbatch $0 -- --input <prefix> --output-prefix <prefix> [--source-tokenizer <path>] [--target-tokenizer <path>] [--multimodal] [--unordered] [--workers N] [--log-interval N]"
            exit 1
            ;;
    esac
done

if [ -z "$INPUT" ] || [ -z "$OUTPUT_PREFIX" ]; then
    echo "ERROR: --input and --output-prefix are required."
    echo "Usage: sbatch $0 -- --input <prefix> --output-prefix <prefix> [--source-tokenizer <path>] [--target-tokenizer <path>] [--multimodal] [--unordered] [--workers N] [--log-interval N]"
    exit 1
fi

################ Setup ################
MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM
export PYTHONPATH="$MEGATRON_LM_DIR:$PYTHONPATH"

# Create log directory
mkdir -p /iopsstor/scratch/cscs/$USER/Megatron-LM/logs/slurm/retokenize

cd $MEGATRON_LM_DIR

echo "Input:            $INPUT"
echo "Output prefix:    $OUTPUT_PREFIX"
echo "Source tokenizer: $SOURCE_TOKENIZER"
echo "Target tokenizer: $TARGET_TOKENIZER"
echo "Multimodal:       $MULTIMODAL"
echo "Unordered:        $UNORDERED"
echo "Workers:          $WORKERS"

################ Run ################
RETOKENIZE_ARGS=(
    --input "$INPUT"
    --output-prefix "$OUTPUT_PREFIX"
    --source-tokenizer "$SOURCE_TOKENIZER"
    --target-tokenizer "$TARGET_TOKENIZER"
    --workers "$WORKERS"
    #--log-interval "$LOG_INTERVAL"
)

if [ "$MULTIMODAL" = true ]; then
    RETOKENIZE_ARGS+=(--multimodal)
fi

if [ "$UNORDERED" = true ]; then
    RETOKENIZE_ARGS+=(--unordered)
fi

srun python $MEGATRON_LM_DIR/scripts/retokenize/retokenize.py "${RETOKENIZE_ARGS[@]}"

echo "END TIME: $(date)"
