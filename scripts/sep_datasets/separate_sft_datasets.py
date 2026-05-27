#!/usr/bin/env python3
"""
Split SFT indexed datasets into token-length buckets.

Buckets:
    lt_8k    : doc_tokens < 8 192
    8k_16k   : 8 192 <= doc_tokens < 16 384
    gt_65k   : doc_tokens >= 65 536

Output layout:
    <output_dir>/lt_8k/<source_name>.{bin,idx}
    <output_dir>/8k_16k/<source_name>.{bin,idx}
    <output_dir>/gt_65k/<source_name>.{bin,idx}

Usage:
    python split_sft_buckets.py --output-dir tool_datasets
    python split_sft_buckets.py --output-dir tool_datasets --dry-run
"""

import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np

from megatron.core.datasets.indexed_dataset import IndexedDataset, IndexedDatasetBuilder

logger = logging.getLogger(__name__)


DATASETS = {
    "EnvScaler-SFT-Traj-9K":
        "/capstor/store/cscs/swissai/infra01/datasets_tokenized/apertus_sft_datasets/EnvScaler-SFT-Traj-9K_corr/tokenizer_tool_tokens/EnvScaler-SFT-Traj-9K_corr/dump-0/00000_tokens",
    "OpenSeeker-v1-Data":
        "/capstor/store/cscs/swissai/infra01/datasets_tokenized/apertus_sft_datasets/OpenSeeker-v1-Data_corr/tokenizer_tool_tokens/OpenSeeker-v1-Data_corr/dump-0/00000_tokens",
    "Toucan-1.5M_filtered":
        "/capstor/store/cscs/swissai/infra01/datasets_tokenized/apertus_sft_datasets/Toucan-1.5M_filtered_corr/tokenizer_tool_tokens/Toucan-1.5M_filtered_corr/dump-0/00000_tokens",
}




# Half-open intervals [lo, hi) on per-document token count.
# gt_65k uses a very large upper bound as a sentinel.
BUCKETS = {
    "lt_8k":   (1,      8_192),
    "8k_16k":  (8_192,  16_384),
    "16k_262k":  (16_384, 262_144),
}


def fmt_tokens(n: int) -> str:
    if n >= 1e9:  return f"{n / 1e9:.3f}B"
    if n >= 1e6:  return f"{n / 1e6:.1f}M"
    if n >= 1e3:  return f"{n / 1e3:.1f}K"
    return str(n)


def compute_doc_token_counts(ds: IndexedDataset) -> np.ndarray:
    """Return an int64 array with the token count for each document."""
    starts = ds.document_indices[:-1].astype(np.intp)
    return np.add.reduceat(ds.sequence_lengths.astype(np.int64), starts)


def split_source(source_name: str, prefix: str, output_dir: str, dry_run: bool):
    """Read one source shard, fan documents into bucket output files."""

    if not IndexedDataset.exists(prefix):
        logger.error("[%s] dataset not found at prefix: %s", source_name, prefix)
        return

    t0 = time.perf_counter()
    ds = IndexedDataset(prefix)
    doc_counts = compute_doc_token_counts(ds)
    n_docs = len(doc_counts)
    logger.info("[%s] loaded %d documents (%.1fs)", source_name, n_docs,
                time.perf_counter() - t0)

    # Per-bucket stats summary (logged even in dry-run)
    for bucket_name, (lo, hi) in BUCKETS.items():
        mask = (doc_counts >= lo) & (doc_counts < hi)
        n = int(mask.sum())
        toks = int(doc_counts[mask].sum())
        logger.info("  %-10s  %7d docs  %s tokens", bucket_name, n, fmt_tokens(toks))

    if dry_run:
        return

    dtype = ds.index.dtype
    doc_idx = ds.document_indices   # shape (n_docs + 1,)

    for bucket_name, (lo, hi) in BUCKETS.items():
        mask = (doc_counts >= lo) & (doc_counts < hi)
        selected_doc_ids = np.where(mask)[0]

        if len(selected_doc_ids) == 0:
            logger.info("[%s/%s] empty bucket — skipping", source_name, bucket_name)
            continue

        out_prefix = os.path.join(output_dir, bucket_name, source_name)
        os.makedirs(os.path.dirname(out_prefix), exist_ok=True)

        builder = IndexedDatasetBuilder(out_prefix + ".bin", dtype=dtype)
        n_written = 0
        n_tokens  = 0

        # Read contiguous runs of doc IDs for sequential I/O
        i = 0
        ids = selected_doc_ids
        while i < len(ids):
            run_start = ids[i]
            run_end   = run_start
            while i + 1 < len(ids) and ids[i + 1] == run_end + 1:
                i += 1
                run_end = ids[i]
            i += 1

            seq_start = int(doc_idx[run_start])
            seq_end   = int(doc_idx[run_end + 1])
            all_seqs  = ds[seq_start:seq_end]

            for d in range(run_start, run_end + 1):
                lo_seq = int(doc_idx[d])     - seq_start
                hi_seq = int(doc_idx[d + 1]) - seq_start
                seqs    = all_seqs[lo_seq:hi_seq]
                lengths = [len(s) for s in seqs]
                data    = np.concatenate(seqs)
                builder.add_document(data, lengths)
                n_written += 1
                n_tokens  += sum(lengths)

        builder.finalize(out_prefix + ".idx")
        logger.info("[%s/%s] wrote %d docs, %s tokens → %s",
                    source_name, bucket_name, n_written,
                    fmt_tokens(n_tokens), out_prefix)

    del ds
    logger.info("[%s] done in %.1fs\n", source_name, time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser(description="Split SFT datasets into token-length buckets.")
    ap.add_argument("--output-dir", required=True,
                    help="Root output directory (e.g. tool_datasets)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print statistics only; do not write any files.")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    output_dir = str(Path(args.output_dir).resolve())
    if not args.dry_run:
        os.makedirs(output_dir, exist_ok=True)

    if args.dry_run:
        logger.info("DRY RUN — no files will be written.\n")

    logger.info("Buckets:")
    for bname, (lo, hi) in BUCKETS.items():
        logger.info("  %-10s  [%d, %s)", bname, lo,
                    str(hi) if hi < 1_000_000_000 else "∞")
    logger.info("")

    for source_name, prefix in DATASETS.items():
        logger.info("=" * 70)
        logger.info("Processing: %s", source_name)
        split_source(source_name, prefix, output_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()