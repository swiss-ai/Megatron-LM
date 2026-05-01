DATA_BASE="/capstor/store/cscs/swissai/infra01/datasets/swiss-ai/legal-mc4-preprocessed/data/"

OUTPUT_FOLDER="/iopsstor/scratch/cscs/snajemmeyer/tokenized_datasets_apertus_1_5/swiss-ai/legal-mc4-preprocessed"

python3 scripts/tokenization/prepare_dumps.py \
  --dataset-folder "$DATA_BASE" \
  --preprocessing-metadata-folder "$OUTPUT_FOLDER" \
  --n-dumps 10
  



