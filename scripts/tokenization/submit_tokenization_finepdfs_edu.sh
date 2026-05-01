#!/bin/bash

# ⚠️ WARNING ⚠️
# Make sure to prepare the dumps before tokenizing the data!
# Check scripts/tokenization/prepare_dumps.py
# ⚠️ WARNING ⚠️

# --- Global Configurations ---
NUMBER_OF_DATATROVE_TASKS=5
TOKENIZER=swiss-ai/Apertus-70B-2509
TOKENIZER_NAME=Apertus-70B-2509
COLUMN_KEY=text
MEGATRON_LM_DIR=/users/$USER/Megatron-LM
PATH_TO_OUTPUT_BASE=/capstor/store/cscs/swissai/infra01/datasets_tokenized/

MAX_SUBMISSIONS=500
CURRENT_SUBMISSION_COUNT=0

REHYDRATE=True  # Set to True or False
if [ "$REHYDRATE" = "True" ]; then
  REHYDRATE_FLAG="--rehydrate"
else
  REHYDRATE_FLAG=""
fi

# ==========================================
# RESUME LOGIC
# ==========================================
# Get the start pattern from the first command line argument (if provided)
START_AT_PATTERN="$1"
FOUND_START_POINT=false

# If no argument is provided, start immediately
if [ -z "$START_AT_PATTERN" ]; then
    FOUND_START_POINT=true
    echo "No resume point specified. Processing all datasets..."
else
    echo "Skipping datasets until matching pattern: '$START_AT_PATTERN'..."
fi
# ==========================================


# Define the directory where your metadata folders (finepdfs-edu-*) are located
BASE_METADATA_DIR=$PATH_TO_OUTPUT_BASE

for dataset_path in "$BASE_METADATA_DIR"/finepdfs-edu-*; do

    # Check if directory exists to avoid errors if glob fails
    [ -d "$dataset_path" ] || continue

    # Extract the folder name (e.g., "finepdfs-edu-math")
    DATASET_NAME=$(basename "$dataset_path")

    # --- RESUME CHECK ---
    # If we haven't found the start point yet, check if this is the one
    if [ "$FOUND_START_POINT" = "false" ]; then
        if [[ "$DATASET_NAME" == *"$START_AT_PATTERN"* ]]; then
            echo "✅ Found resume point at: $DATASET_NAME"
            FOUND_START_POINT=true
        else
            # Skip this iteration without printing too much noise
            # echo "Skipping $DATASET_NAME..." 
            continue
        fi
    fi
    # --------------------
    
    echo "=================================================="
    echo "Processing Dataset: $DATASET_NAME"
    echo "=================================================="

    # --- dynamic Paths (Updated per dataset) ---
    PATH_TO_PREPROCESSING_METADATA=$dataset_path
    PATH_TO_DATATROVE_LOGGING_DIR=$MEGATRON_LM_DIR/logs/datatrove
    PATH_TO_SLURM_LOGGING_DIR=$MEGATRON_LM_DIR/logs/slurm/tokenization-$TOKENIZER_NAME-$DATASET_NAME
    
    # Output path includes the specific dataset name
    DATASET_OUTPUT_FOLDER_NAME=$PATH_TO_OUTPUT_BASE/$TOKENIZER_NAME/$DATASET_NAME
    CSV_RESULTS_FILE=$PATH_TO_PREPROCESSING_METADATA/tokenize-$TOKENIZER_NAME-$DATASET_NAME.csv

    # Create directories for this specific dataset
    mkdir -p $DATASET_OUTPUT_FOLDER_NAME
    mkdir -p $PATH_TO_SLURM_LOGGING_DIR
    mkdir -p $PATH_TO_PREPROCESSING_METADATA/completed-dumps
    
    # Update the symlink in the metadata folder to point to the new output
    ln -sfn $DATASET_OUTPUT_FOLDER_NAME $PATH_TO_PREPROCESSING_METADATA/tokenized-dir-link

    # Create CSV Header for this dataset
    echo "slurm_job_id,node,start,end,paths_file,output_folder,dataset_total_size,processed_total_size,number_of_workers_per_node,time,bw,total_tokens_processed,throughput (Million Tokens/Second/Node)" > $CSV_RESULTS_FILE

    # --- INNER LOOP: Iterate through dumps for this specific dataset ---
    for paths_file in "$PATH_TO_PREPROCESSING_METADATA/dumps"/*; do
        # Extract dump ID using regex
        dump=$(grep -oP '(?<=paths_file_)\d+(?=\.txt)' <<< $paths_file)
        
        output_folder=$DATASET_OUTPUT_FOLDER_NAME/dump-$dump
        logging_dir=$PATH_TO_DATATROVE_LOGGING_DIR/$TOKENIZER_NAME/$DATASET_NAME/dump-$dump
        
        # Submit the job
        sbatch \
            --job-name=tok-$DATASET_NAME-$dump \
            --output=$PATH_TO_SLURM_LOGGING_DIR/R-%x-%j.out \
            --error=$PATH_TO_SLURM_LOGGING_DIR/R-%x-%j.err \
            --reservation=SD-69241-apertus-1-5 \
            $MEGATRON_LM_DIR/scripts/tokenization/tokenize.sh \
            $PATH_TO_PREPROCESSING_METADATA/raw-dataset-link \
            $output_folder \
            $TOKENIZER \
            $logging_dir \
            $CSV_RESULTS_FILE \
            $paths_file \
            $NUMBER_OF_DATATROVE_TASKS \
            $MEGATRON_LM_DIR \
            $COLUMN_KEY \
            $REHYDRATE_FLAG
          
        echo "Submitted dump $dump for $DATASET_NAME"
        
        # ### ADDED: Counter and Break Logic ###
        ((CURRENT_SUBMISSION_COUNT++))
        
        if [ "$CURRENT_SUBMISSION_COUNT" -ge "$MAX_SUBMISSIONS" ]; then
            echo ""
            echo "🛑 Reached limit of $MAX_SUBMISSIONS submissions. Stopping script."
            break 2
        fi
        # ######################################
        
    done
    
    echo "Finished submitting jobs for $DATASET_NAME"
    echo ""
done