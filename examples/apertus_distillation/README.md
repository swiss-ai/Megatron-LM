# Teacher caches for Apertus 1.5 distillation

`get_teacher_logits.py` (repository root) runs a frozen teacher over the regular
Megatron data pipeline and stores its next-token distribution for offline
distillation. It reuses the data flags of the teacher's own training run, so the
cache carries the same masks: assistant-only loss on SFT samples, zero weight on
image and audio targets, masked speech task markers, Goldfish.

## What is stored

For every position with loss, either:

- `--distill-cache-mode topk`: the `--distill-topk` most likely tokens and their
  probabilities (Apertus-v1.1 used 256), or
- `--distill-cache-mode sample`: `--distill-samples` draws at T=1, stored as each
  distinct token with count / draws. This is an unbiased estimate of the full
  distribution and needs far less space.

`--distill-vocab-size 131072` restricts the teacher to the text vocabulary and
renormalizes over it, matching a student with a text-only output layer. Each file
also stores the tokens, labels, loss mask and position ids, so a student reads
everything it needs from the cache. The layout is documented in
`megatron/training/distillation/cache_format.py`.

## Running

```bash
CACHE_DIR=/path/to/cache TARGET_TOKENS=197000000000 \
  sbatch --nodes=32 examples/apertus_distillation/cache_teacher_8b.sbatch
python examples/apertus_distillation/verify_teacher_cache.py /path/to/cache --files 50
```

- **Resume:** rerun with the same settings, on any number of nodes; the writer
  continues after the files recorded in `progress.json`. The manifest refuses
  runs whose settings differ.
- **Size:** `TARGET_TOKENS` fixes the data order. To grow a cache later, start it
  with the largest size and stop early.
- **Layout:** few, large shard files. Shard `f` holds sequences
  `f * seqs_per_file ...` of the data order, whatever the number of GPUs. It is
  an uncompressed zip of gzipped chunks of `--distill-seqs-per-chunk` sequences
  plus an `index.json`, so readers load one chunk at a time. The launcher uses
  8,192 sequences per shard: about 2,900 files of ~37 GB for a 197B-token cache.
- **Time limit:** `--distill-exit-duration-in-mins` stops at a shard boundary
  before the job's time limit, so large shards cost no lost work.
- **Check:** `--distill-self-check` compares the chunked scoring with the model's
  own output layer on the first micro-batch.

## Tests

```bash
pytest --noconftest tests/unit_tests/distillation
```
