DATA_BASE="/capstor/scratch/cscs/snajemmeyer/SYNTH-preprocessed/data/output"

OUTPUT_FOLDER="datasets/PleIAs-SYNTH"

python3 scripts/tokenization/prepare_dumps.py \
  --dataset-folder "$DATA_BASE" \
  --preprocessing-metadata-folder "$OUTPUT_FOLDER" \
  --n-dumps 20
  



