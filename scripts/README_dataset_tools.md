# Dataset Tools

Tools for manipulating already-tokenized Megatron `.bin`/`.idx` datasets.

---

## SFT Packing Preflight

Counts Apertus SFT packed samples before launching training. The tool reads
Megatron indexed dataset prefixes, runs the same BFD packer used by
`ApertusSFTDataset`, and reports the full-batch `TRAINING_STEPS` value.

```bash
python tools/preflight_sft_packing.py \
    --data-path-file /data/sft_prefixes.txt \
    --seq-length 8192 \
    --global-batch-size 1024
```

The prefix file contains one path and one integer epoch count per line. If the
path is a directory, all `*.idx` files in it are counted. Fractional epochs are
not supported yet. Blank lines and comment lines are ignored.

```text
# text x2
/data/text/dump-0 2

# vision x1
/data/vision 1

# audio x2
/data/audio 2
```

Use the printed `floor(total_packed_samples / global_batch_size)` value for
finite `--dataloader-type single` runs so the final partial global batch is not
requested.

---

## Retokenize

Converts a dataset from one HuggingFace tokenizer to another. Each sequence is decoded back to text with the source tokenizer, then re-encoded with the target tokenizer. BOS/EOS tokens are handled automatically: if a sequence starts with the source BOS or ends with the source EOS, those are stripped before decoding and replaced with the corresponding target BOS/EOS after encoding. For multimodal datasets, non-text sequences (mode != 0) are copied as-is.

```bash
python scripts/retokenize/retokenize.py \
    --input /data/my_dataset \
    --output-prefix /data/my_dataset_newtok \
    --source-tokenizer meta-llama/Llama-2-7b-hf \
    --target-tokenizer mistralai/Mistral-7B-v0.1 \
    --workers 8
```

A verification step runs after finalization, checking that the output has the same number of documents and sequences as the input.

---

## Split

Splits a dataset into parts at document boundaries according to the given ratios. Document order is preserved -- part 0 gets the first N documents, part 1 gets the next, etc. No shuffling is performed.

```bash
python scripts/split_dataset/split_dataset.py \
    --input /data/my_dataset \
    --output-prefix /data/my_dataset_split \
    --ratios 0.8 0.2
```

Produces `/data/my_dataset_split_part0` (80% of docs) and `/data/my_dataset_split_part1` (20% of docs).

---

## Merge

Concatenates multiple datasets into one. Order is preserved exactly as specified on the command line -- all sequences and documents from the first input come first, then the second, and so on.

```bash
python scripts/merge_dataset/merge_dataset.py \
    --inputs /data/dataset_a /data/dataset_b /data/dataset_c \
    --output-prefix /data/merged
```

All inputs must share the same dtype. The merge uses efficient binary copy (`add_index`) so it is fast even for large datasets.

---

All three tools support `--multimodal` for multimodal datasets.
