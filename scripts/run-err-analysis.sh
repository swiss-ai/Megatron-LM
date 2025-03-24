#!/bin/bash

# Variables
MEGATRON_LM_DIR="$SCRATCH/Megatron-LM"
ERROR_ANALYZER="$MEGATRON_LM_DIR/megatron/training/utils_slurm.py"
DEBUG_DIR="$SCRATCH/debug"

# List of error files
ERROR_FILES=(
    "/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/slurm/training/ap3-3b-224837.err"
    "/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/slurm/training/ap3-3b-224854.err"
    "/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/slurm/training/ap3-3b-225803.err"
    "/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/slurm/training/ap3-3b-226038.err"
    "/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/slurm/training/ap3-3b-227908.err"
    "/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/slurm/training/ap3-3b-228210.err"
    "/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/slurm/training/ap3-3b-230204.err"
    "/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/slurm/training/ap3-3b-232123.err"
    "/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/slurm/training/ap3-3b-232197.err"
    "/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/slurm/training/ap3-3b-232294.err"
    "/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/slurm/training/ap3-3b-232295.err"
    "/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/slurm/training/ap3-3b-232306.err"
    "/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/slurm/training/ap3-3b-233791.err"
    "/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/slurm/training/ap3-3b-233811.err"
    "/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/slurm/training/ap3-3b-234033.err"
    "/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/slurm/training/ap3-3b-234046.err"
    "/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/slurm/training/ap3-3b-238151.err"
    "/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/slurm/training/ap3-3b-238392.err"
    "/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/slurm/training/ap3-3b-239144.err"
    "/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/slurm/training/ap3-3b-239211.err"
    "/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/slurm/training/ap3-3b-239250.err"
    "/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/slurm/training/ap3-3b-239258.err"
    "/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/slurm/training/ap3-3b-239266.err"
    "/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/slurm/training/ap3-3b-239881.err"
    "/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/slurm/training/ap3-3b-251376.err"
    "/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/slurm/training/ap3-3b-251387.err"
    "/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/slurm/training/ap3-3b-254871.err"
    "/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/slurm/training/ap3-3b-257480.err"
    "/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/slurm/training/ap3-3b-257490.err"
    "/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/slurm/training/ap3-3b-257510.err"
    "/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/slurm/training/ap3-3b-259646.err"
    "/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/slurm/training/ap3-3b-259789.err"
    "/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/slurm/training/ap3-3b-260329.err"
    "/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/slurm/training/ap3-3b-260833.err"
    "/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/slurm/training/ap3-3b-260859.err"
    "/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/slurm/training/ap3-3b-261222.err"
    "/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/slurm/training/ap3-3b-261294.err"
    "/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/slurm/training/ap3-3b-262305.err"
    "/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/slurm/training/ap3-3b-262320.err"
    "/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/slurm/training/ap3-3b-267065.err"
    "/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/slurm/training/ap3-3b-272340.err"
    "/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/slurm/training/ap3-3b-272896.err"
    "/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/slurm/training/ap3-3b-274114.err"
    "/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/slurm/training/ap3-3b-279611.err"
    "/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/slurm/training/ap3-3b-279814.err"
    "/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/slurm/training/ap3-3b-281014.err"
    "/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/slurm/training/ap3-3b-281384.err"
    "/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/slurm/training/ap3-3b-284702.err"
    "/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/slurm/training/ap3-3b-286808.err"
    "/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/slurm/training/ap3-3b-290161.err"
    "/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/slurm/training/ap3-3b-290365.err"
    "/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/slurm/training/ap3-3b-291209.err"
    "/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/slurm/training/ap3-3b-291907.err"
    "/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/slurm/training/ap3-3b-292003.err"
    "/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/slurm/training/ap3-3b-293335.err"
)

# Ensure the debug directory exists
mkdir -p "$DEBUG_DIR"

# Iterate over each error file
for ERROR_FILE in "${ERROR_FILES[@]}"; do
    # Check if the error file exists
    if [ ! -f "$ERROR_FILE" ]; then
        echo "Error: Specified .err file does not exist: $ERROR_FILE"
        continue
    fi

    # Generate a unique output file name
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S") # Generate a unique timestamp
    ANALYSIS_OUTPUT="${DEBUG_DIR}/reports/error_analysis_${TIMESTAMP}.txt"

    # Run the error analysis
    echo "Running error analysis on file: $ERROR_FILE"
    if [ -f "$ERROR_ANALYZER" ]; then
        python "$ERROR_ANALYZER" "$ERROR_FILE" --output "$ANALYSIS_OUTPUT"
        if [ -f "$ANALYSIS_OUTPUT" ]; then
            echo "Error analysis saved to: $ANALYSIS_OUTPUT"
        else
            echo "Error: Analysis output not generated."
        fi
    else
        echo "Error analyzer script not found at: $ERROR_ANALYZER"
        exit 1
    fi
done

echo "Error analysis completed for all files."