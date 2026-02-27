DATA_BASE="/capstor/store/cscs/swissai/infra01/datasets/swiss-ai/dolma3_olmocr_science_pdfs-preprocessed/data"
OUTPUT_FOLDER="datasets/dolma3_olmocr_science_pdfs-preprocessed"


python3 scripts/tokenization/prepare_dumps.py \
  --dataset-folder "$DATA_BASE" \
  --preprocessing-metadata-folder "$OUTPUT_FOLDER" \
  --n-dumps 2