# Dataset Tools

Tools for manipulating already-tokenized Megatron `.bin`/`.idx` datasets.

---

## SFT Packing Preflight

Counts packed Apertus SFT samples and reports the number of complete
global-batch training steps. It uses the same packing logic as
`ApertusSFTDataset`, requires no GPU or Megatron initialization, and does not
create index caches. The first import may compile small C helper libraries next
to the Megatron sources.

Both packing strategies are supported via `--packing-strategy` (required, must
match the training run's `--ap-sft-packing-strategy`):

- `bfd` (Best-Fit Decreasing): order-independent, so the count is exact
  regardless of seed. If training sets `--max-docs-per-bin-sft`, pass the same
  value as `--max-docs-per-bin`.
- `greedy`: packs documents in the seeded shuffle order used at training time.
  Pass the training run's `--seed` (default 1234) or the count will differ.

```bash
python tools/preflight_sft_packing.py \
    --data-path-file tools/example_sft_prefixes.txt \
    --packing-strategy bfd \
    --seq-length 8192 \
    --global-batch-size 1024
```

The prefix file contains one path and one integer epoch count per line (see
`tools/example_sft_prefixes.txt`). If the path is a directory, all `*.idx`
files in it are counted. Fractional epochs are not supported yet. Blank lines
and comment lines are ignored.

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

The preflight count matches training when all of the following are true:

- SFT datasets only: the tool knows nothing about `sft:`/`pretrain:`
  dataset-type markers — every entry is packed as an SFT dataset, and a
  marker-prefixed entry fails with a file-not-found error. For mixed
  SFT and pretraining blends, use the SFT Packing Initialization tool below.
- The whole dataset lands in the train split (`--split 100,0,0`).
- Each counted `.idx` file appears as its **own** `--data-path` entry in
  training. Megatron builds one independently-shuffled dataset per blend
  prefix, which is what the per-file counting here replicates. If you merge
  files into one dataset (see Merge below), count the merged prefix instead —
  packing over the union gives different counts than the per-file sum.
- For `--ap-sft-load-loss-mask` datasets, each stored document is
  `[tokens, loss_mask]` (twice the token length) and training is launched with
  a doubled `--seq-length`. Pass that same doubled value here, or the counts
  will be about twice as high.
- The epochs column is a counting convenience: it multiplies a dataset's
  per-epoch sample count. How those epochs are realized in training depends on
  your blend weights and `--train-samples`; with weighted blends Megatron
  requests a ~0.5% sample surplus and tiles/truncates per dataset, so exact
  integer-epoch consumption per dataset is not guaranteed.

---

## SFT Packing Initialization

Builds dataset indices through the full `BlendedMegatronDatasetBuilder`
pipeline and exits before training. How the two tools compare:

| | Preflight | Initialization |
|---|---|---|
| Needs GPU / Megatron initialization | No | Yes (`torchrun`; one GPU is enough) |
| Writes the index cache | No | Yes — `document_index`, `sample_index`, `shuffle_index` to `--data-cache-path`, reused at training launch |
| Packing strategies | `greedy` and `bfd` | `greedy` and `bfd` |
| Blend weights and splits | No — unweighted, assumes `--split 100,0,0` | Yes, exactly as training |
| `sft:`/`pretrain:` markers (mixed blends) | No — every entry is treated as an SFT dataset; a marker-prefixed entry fails with a file-not-found error | Yes, exactly as training: `pretrain:` entries build as regular `GPTDataset`s, only SFT entries are packed |

**For mixed SFT and pretraining blends, use the initialization tool.** The
preflight can still count the SFT portion of such a blend (list only the bare
SFT prefixes), but its training-step count then ignores the pretraining
samples, which follow a different fixed-length chunking formula.

Use the preflight for quick step-count math; use the initialization when you
want the packing done ahead of the training job. Cached indices are reused by
training only if the cache-relevant args match: seed, seq-length, split,
data-path, tokenizer, packing settings (`--ap-sft-packing-strategy`,
`--ap-sft-load-loss-mask`, `--max-docs-per-bin-sft`), and the requested train
sample count (`--train-samples`, or `--train-iters * --global-batch-size` when
`--train-samples` is not given). TP/PP/EP and model architecture args do not
affect the cache hash. Exception: blends with multiple prefixes and no
explicit weights request all samples from every dataset, so their hash ignores
the requested count (the two-run step below is then unnecessary); weighted
blends additionally scale the per-dataset count by
`--mid-level-dataset-surplus` (default 0.005).

To prebuild an index that training can reuse, first run the initialization to
learn the sample count, then rerun it with `--train-samples <count>`. This
second initialization is optional: if the cache-relevant sample count differs,
training rebuilds the index at launch.

On Slurm, submit `tools/submit-sft-packing-init.sh` after editing its config
section:

```bash
sbatch tools/submit-sft-packing-init.sh
```

Or run `tools/initialize_sft_dataset.py` directly with the same data/training
arguments as `pretrain_gpt.py` plus `--ap-sft --ap-sft-pack-samples` (see the
module docstring for a single-GPU example). The reported packed sample count
per epoch is what you set `--train-samples` to for one-epoch training.

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

The retokenize, split, and merge tools support `--multimodal` for multimodal datasets.
