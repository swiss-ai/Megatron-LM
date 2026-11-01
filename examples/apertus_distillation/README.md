# Teacher caches for Apertus 1.5 distillation

`get_teacher_logits.py` (repository root) runs a frozen teacher over the regular
Megatron data pipeline and stores its next-token distribution for offline
distillation; `pretrain_gpt_distill.py` trains a student from such a cache. It reuses the data flags of the teacher's own training run, so the
cache carries the same masks: assistant-only loss on SFT samples, zero weight on
image and audio targets, masked speech task markers, Goldfish.

## What is stored

For every position with loss, either:

- `--distill-cache-mode topk`: the `--distill-topk` most likely tokens and their
  probabilities (Apertus-v1.1 used 256), or
- `--distill-cache-mode sample`: `--distill-samples` draws at T=1, stored as each
  distinct token with count / draws. This is an unbiased estimate of the full
  distribution and needs far less space.

Both modes also store, per position, the entropy of the teacher's full
distribution and its log-probability of the label, exactly. They turn a
cross-entropy against the teacher into a KL divergence, and give the teacher's own
loss on the data.

`--distill-vocab-size 131072` restricts the teacher to the text vocabulary and
renormalizes over it, matching a student with a text-only output layer. Each file
also stores the tokens, labels, loss mask and position ids, so a student reads
everything it needs from the cache. The layout is documented in
`megatron/training/distillation/cache_format.py`.

## Running

```bash
export CACHE_ROOT=/path/to/caches
bash examples/apertus_distillation/stage_p_blends.sh       # freeze distill/ and pruning/ blends
python3 examples/apertus_distillation/check_stage_p.py blends --root $CACHE_ROOT
sbatch examples/apertus_distillation/preflight_stage_p.sbatch   # also writes the pruning set
bash examples/apertus_distillation/launch_stage_p.sh       # start or resume the cache
python examples/apertus_distillation/verify_teacher_cache.py /path/to/cache --files 50
```

- **Resume:** run `launch_stage_p.sh` again, on any number of nodes; the writer
  continues after the files recorded in `progress.json` and drops half-written
  ones. The manifest refuses runs whose settings differ, and the script refuses
  while a job of the cache is in the queue.
- **Separate sets:** the pruning set uses its own blend and folder. It reads the
  sequences of each dataset after those the cache reaches, so the two share no
  sequences.
- **Size:** `TARGET_TOKENS` fixes the data order. To grow a cache later, start it
  with the largest size and stop early.
- **The rest of Stage 3:** Stage P reads every dataset for under one pass, and a
  finished cache can't grow. `preflight_stage3_rest.sbatch` freezes what Stage P and
  the pruning set leave, and `launch_stage3_rest.sh` starts or resumes a separate
  cache of it on `preemptable`, where Slurm requeues the job after a preemption.
- **Layout:** few, large shard files. Shard `f` holds sequences
  `f * seqs_per_file ...` of the data order, whatever the number of GPUs. It is
  an uncompressed zip of gzipped chunks of `--distill-seqs-per-chunk` sequences
  plus an `index.json`, so readers load one chunk at a time. The launcher uses
  8,192 sequences per shard: about 2,900 files of ~37 GB for a 197B-token cache.
- **Prefix:** `STOP_TOKENS` (`--distill-cache-stop-files`) writes only the first
  shards, e.g. for an ablation; a later run without it continues the same order.
- **Time limit:** `--distill-exit-duration-in-mins` stops at a shard boundary
  before the job's time limit, so large shards cost no lost work.
- **Check:** `--distill-self-check` compares the chunked scoring with the model's
  own output layer on the first micro-batch.

## Training a student

```bash
RUN=1p5b-sample CACHE_DIR=/path/to/cache SIZE=1p5b TRAIN_TOKENS=30000000000 \
  sbatch --nodes=8 examples/apertus_distillation/distill_student.sbatch
```

The student reads tokens, masks and teacher rows from the cache instead of the GPT
datasets. Step `s` uses sequences `s * G ... (s + 1) * G - 1` of the cache for a
global batch of `G`, and data-parallel rank `r` reads the contiguous block `r` of
it, so ranks read whole chunks and a run resumes on any number of GPUs. The loss is
`KD_WEIGHT * KD + (1 - KD_WEIGHT) * CE` per position with loss, weighted by the
cache's loss mask (`megatron/training/distillation/losses.py`):

- CE: cross-entropy on the label over the student's whole vocabulary.
- KD on a sample cache: cross-entropy against the sampled frequencies, an
  unbiased estimate of the cross-entropy against the teacher's full distribution.
- KD on a top-k cache: the Apertus-v1.1 loss `-sum p log q`, which pushes the
  student's mass outside the stored ids to zero; with `TAIL=true`, plus
  `-(1 - sum p) log(1 - sum q)`, which matches it instead.

Both are computed in fp32 blocks of `--distill-loss-chunk` positions and
recomputed in the backward pass. Validation reads the same held-out sequences at
every evaluation and reports the KL from the teacher (unbiased on a sample cache, a
lower bound on a top-k one), the teacher's and the student's loss on the labels,
and the student's top-1 confidence and accuracy. The student supports TP = PP =
CP = 1 and a fixed global batch.

## KD-target ablation

`kd_target_ablation.sh` writes a top-256 and a 64-sample cache over the same first
448 shards (~30B tokens) of the Stage P order, then trains the same 1.5B student
four times: labels only, top-256 with the v1.1 loss, top-256 with the tail
bucket, and 64 samples. All four are validated on shard 447 of the sample cache.
The winner's cache continues to the full Stage P run.

## Tests

```bash
pytest --noconftest tests/unit_tests/distillation
```
