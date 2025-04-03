TOKENIZER_NAME=mistralai/Mistral-Nemo-Base-2407
DATASET_NAME=fineweb-edu-News
TopK=5

MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM
PATH_TO_SLURM_LOGGING_DIR=$MEGATRON_LM_DIR/logs/slurm/merge-$TOKENIZER_NAME-$DATASET_NAME

PATH_TO_INPUT_FOLDER=/iopsstor/scratch/cscs/$USER/data/robots-txt/AllNewsOptout
PATH_TO_OUTPUT_FOLDER=/iopsstor/scratch/cscs/$USER/data/robots-txt/AllNewsOptout-merged  # /iopsstor/scratch/cscs/$USER/datasets



# Set base directories
CLUSTERED_DIR="${PATH_TO_INPUT_FOLDER}-clustered"

# Create all 180 directories in a single batch (avoiding redundant `mkdir -p` calls)
mkdir -p "$CLUSTERED_DIR"/dir_{1..100}

# Initialize directory counter
dir_counter=1

# Find all .bin and .idx files in order
find "$PATH_TO_INPUT_FOLDER" -maxdepth 1 -type f \( -name "*.bin" -o -name "*.idx" \) | awk -F'/' '{print $NF}' | awk -F'.' '{print $1}' | uniq > /tmp/ordered_prefixes.txt

# Use ordered bucketing (no sorting needed)
while read -r prefix; do
    target_dir="$CLUSTERED_DIR/dir_$dir_counter"

    # Ensure target directory exists
    mkdir -p "$target_dir"

    # Move both .bin and .idx files for the prefix (if they exist)
    mv "$PATH_TO_INPUT_FOLDER/$prefix.bin" "$target_dir/" 2>/dev/null &
    mv "$PATH_TO_INPUT_FOLDER/$prefix.idx" "$target_dir/" 2>/dev/null &

    # Increment bucket (loop back after 100)
    ((dir_counter = (dir_counter % 100) + 1))

done < /tmp/ordered_prefixes.txt

# Wait for all background move jobs to finish
wait


mkdir -p $PATH_TO_SLURM_LOGGING_DIR
mkdir -p $PATH_TO_OUTPUT_FOLDER


PATH_TO_INPUT_FOLDER_NEW=$PATH_TO_INPUT_FOLDER-clustered

for dump_folder in "$PATH_TO_INPUT_FOLDER_NEW"/*; do
    dump_prefix=$(basename $dump_folder)
    sbatch --job-name=merge-$DATASET_NAME-$dump_prefix --output=$PATH_TO_SLURM_LOGGING_DIR/R-%x-%j.out --error=$PATH_TO_SLURM_LOGGING_DIR/R-%x-%j.err $MEGATRON_LM_DIR/scripts/merge_datasets/merge.sh $dump_folder $PATH_TO_OUTPUT_FOLDER/$dump_prefix-merged $MEGATRON_LM_DIR
done