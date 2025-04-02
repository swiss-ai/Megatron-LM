#!/bin/bash

MEGATRON_LM_DIR="$SCRATCH/Megatron-LM"
ERROR_ANALYZER="$MEGATRON_LM_DIR/megatron/training/utils_slurm.py"
DEBUG_DIR="$SCRATCH/debug"
REPORTS_DIR="$DEBUG_DIR/reports"
# Modify these eventually
ERR_FILES_DIR="/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/slurm/training"
FILE_PATTERN="ap3-3b"  # Leave empty to analyze all .err files in the directory


# Ensure directories exist
mkdir -p "$REPORTS_DIR"

# Find error files based on the pattern
if [ -z "$FILE_PATTERN" ]; then
    mapfile -t ERROR_FILES < <(find "$ERR_FILES_DIR" -name "*.err" -type f)
else
    mapfile -t ERROR_FILES < <(find "$ERR_FILES_DIR" -name "*${FILE_PATTERN}*.err" -type f)
fi

# Check if any files were found
if [ ${#ERROR_FILES[@]} -eq 0 ]; then
    echo "No matching .err files found in $ERR_FILES_DIR"
    exit 1
fi

echo "Found ${#ERROR_FILES[@]} error files for analysis."

# Iterate over each error file
for ERROR_FILE in "${ERROR_FILES[@]}"; do
    BASENAME=$(basename "$ERROR_FILE" .err)
    ANALYSIS_OUTPUT="${REPORTS_DIR}/${BASENAME}_analysis_$(date +"%Y%m%d_%H%M%S").txt"
    echo "------------------------------------------------------------------------"
    echo "Analyzing: $ERROR_FILE"
    if [ -f "$ERROR_ANALYZER" ]; then
        python "$ERROR_ANALYZER" "$ERROR_FILE" --output "$ANALYSIS_OUTPUT"
        [ -f "$ANALYSIS_OUTPUT" ] && echo "Analysis saved to: $ANALYSIS_OUTPUT" || echo "Error: Analysis failed for $ERROR_FILE"
    else
        echo "Error analyzer script not found at: $ERROR_ANALYZER"
        exit 1
    fi
done

echo "Error analysis completed for all files."