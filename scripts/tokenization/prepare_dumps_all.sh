# processed ones
for ds in finepdfs-edu-preprocessed finepdfs-edu-multilingual-preprocessed; do # eurlex_resources-preprocessed; do
    python3 scripts/tokenization/prepare_dumps.py \
        --dataset-folder "/capstor/store/cscs/swissai/infra01/datasets/swiss-ai/${ds}/data/" \
        --preprocessing-metadata-folder "/capstor/store/cscs/swissai/infra01/datasets_tokenized/${ds}/" \
        --n-dumps 15
done

# not processed ones
python3 scripts/tokenization/prepare_dumps.py \
    --dataset-folder    "/capstor/store/cscs/swissai/infra01/users/vsabolce/apertus_moe/datasets/swissai-fineweb-2_0_1-quality_10-filterrobots/data/output/" \
    --preprocessing-metadata-folder "/iopsstor/scratch/cscs/snajemmeyer/tokenized_datasets_apertus_1_5/swiss-ai/swissai-fineweb-2_0_1-quality_10-filterrobots" \
    --n-dumps 150

python3 scripts/tokenization/prepare_dumps.py \
    --dataset-folder    "/capstor/scratch/cscs/snajemmeyer/SYNTH-preprocessed/data/" \
    --preprocessing-metadata-folder "/iopsstor/scratch/cscs/snajemmeyer/tokenized_datasets_apertus_1_5/swiss-ai/SYNTH-preprocessed" \
    --n-dumps 25


python3 scripts/tokenization/prepare_dumps.py \
    --dataset-folder    "/capstor/store/cscs/swissai/infra01/datasets/swiss-ai/finetranslations_preprocessed" \
    --preprocessing-metadata-folder "/capstor/store/cscs/swissai/infra01/datasets_tokenized/finetranslations_preprocessed" \
    --n-dumps 20

python3 scripts/tokenization/prepare_dumps.py \
    --dataset-folder    "/capstor/scratch/cscs/snajemmeyer/Nemotron-CC-v2.1-preprocessed/data/" \
    --preprocessing-metadata-folder "/capstor/store/cscs/swissai/infra01/datasets_tokenized/Nemotron-CC-v2.1-preprocessed" \
    --n-dumps 20

#python3 scripts/tokenization/prepare_dumps.py \
#    --dataset-folder    "/capstor/scratch/cscs/snajemmeyer/finepdfs-edu-multilingual-preprocessed" \
#    --preprocessing-metadata-folder "/iopsstor/scratch/cscs/snajemmeyer/tokenized_datasets_apertus_1_5/swiss-ai/finepdfs-edu-multilingual-preprocessed" \
#    --n-dumps 20

python3 scripts/tokenization/prepare_dumps.py \
    --dataset-folder    "/capstor/store/cscs/swissai/infra01/datasets/swiss-ai/swiss-caselaw-preprocessed/data/" \
    --preprocessing-metadata-folder "/capstor/store/cscs/swissai/infra01/datasets_tokenized/swiss-caselaw-preprocessed" \
    --n-dumps 5