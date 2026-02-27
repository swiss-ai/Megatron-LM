DATA_BASE="/capstor/store/cscs/swissai/infra01/datasets/swiss-ai/Biomed-Enriched_preprocessed/data"
OUTPUT_FOLDER="datasets/Biomed-Enriched_preprocessed"


python3 scripts/tokenization/prepare_dumps.py \
  --dataset-folder "$DATA_BASE" \
  --preprocessing-metadata-folder "$OUTPUT_FOLDER" \
  --n-dumps 2