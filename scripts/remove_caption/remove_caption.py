"""Remove captions from a Megatron .bin/.idx dataset.

For each document (single-sequence), finds the last occurrence of the image-end
token, truncates everything after it, and appends an EOD token. Documents
without any image-end token are dropped (with a warning).

Usage:
    python scripts/remove_caption/remove_caption.py \
        --input my_dataset \
        --output-prefix my_dataset_no_caption \
        --image-end-token-id 128258 \
        --eod-token-id 128001 \
        --workers 8
"""

import argparse
import multiprocessing
import os
import sys

import numpy
import torch

from megatron.core.datasets.indexed_dataset import (
    IndexedDataset,
    IndexedDatasetBuilder,
    get_bin_path,
    get_idx_path,
)


def get_args():
    parser = argparse.ArgumentParser(
        description="Remove captions from a Megatron indexed dataset."
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input dataset prefix (no .bin/.idx extension)",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        required=True,
        help="Output dataset prefix (no extension)",
    )
    parser.add_argument(
        "--image-end-token-id",
        type=int,
        required=True,
        help="Token ID that marks the end of an image",
    )
    parser.add_argument(
        "--eod-token-id",
        type=int,
        required=True,
        help="End-of-document token ID to append after truncation",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of multiprocessing workers (default: 1)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Log progress every N documents (default: 100)",
    )

    args = parser.parse_args()

    assert os.path.exists(get_idx_path(args.input)), f"Missing index file: {get_idx_path(args.input)}"
    assert os.path.exists(get_bin_path(args.input)), f"Missing data file: {get_bin_path(args.input)}"

    output_dir = os.path.dirname(args.output_prefix)
    if output_dir:
        assert os.path.isdir(output_dir), f"Output directory does not exist: {output_dir}"

    return args


# ── Worker globals (initialized once per process) ──

_worker_dataset = None
_worker_image_end_token_id = None
_worker_eod_token_id = None


def _worker_init(input_prefix, image_end_token_id, eod_token_id):
    """Initialize per-worker dataset (mmap) and token IDs."""
    global _worker_dataset, _worker_image_end_token_id, _worker_eod_token_id
    _worker_dataset = IndexedDataset(input_prefix, multimodal=False)
    _worker_image_end_token_id = image_end_token_id
    _worker_eod_token_id = eod_token_id


def _process_document(doc_idx):
    """Process a single document: truncate after last image-end token, append EOD.

    Returns (doc_idx, truncated_tokens) or (doc_idx, None) if no image-end token found.
    """
    dataset = _worker_dataset
    doc_indices = dataset.document_indices
    seq_start = int(doc_indices[doc_idx])
    seq_end = int(doc_indices[doc_idx + 1])

    # Collect all tokens for this document (typically one sequence per doc)
    all_tokens = []
    for seq_idx in range(seq_start, seq_end):
        all_tokens.extend(dataset[seq_idx].tolist())

    # Find the last occurrence of the image-end token
    last_pos = -1
    for j, tok in enumerate(all_tokens):
        if tok == _worker_image_end_token_id:
            last_pos = j

    if last_pos == -1:
        return (doc_idx, None)

    # Truncate after the last image-end token and append EOD
    truncated = all_tokens[: last_pos + 1]
    truncated.append(_worker_eod_token_id)
    return (doc_idx, truncated)


def main():
    args = get_args()

    # Get source dataset info
    source_dataset = IndexedDataset(args.input, multimodal=False)
    num_docs = len(source_dataset.document_indices) - 1
    num_seqs = len(source_dataset.index)
    source_dtype = source_dataset.index.dtype
    del source_dataset

    print(f"Input:              {args.input}")
    print(f"  Sequences:        {num_seqs}")
    print(f"  Documents:        {num_docs}")
    print(f"  Dtype:            {source_dtype}")
    print(f"Image-end token ID: {args.image_end_token_id}")
    print(f"EOD token ID:       {args.eod_token_id}")
    print(f"Workers:            {args.workers}")
    print()

    builder = IndexedDatasetBuilder(
        get_bin_path(args.output_prefix), dtype=source_dtype, multimodal=False
    )

    if args.workers > 1:
        pool = multiprocessing.Pool(
            args.workers,
            initializer=_worker_init,
            initargs=(args.input, args.image_end_token_id, args.eod_token_id),
        )
        doc_iter = pool.imap(_process_document, range(num_docs))
    else:
        _worker_init(args.input, args.image_end_token_id, args.eod_token_id)
        doc_iter = map(_process_document, range(num_docs))

    docs_written = 0
    docs_skipped = 0

    for doc_idx, tokens in doc_iter:
        if tokens is None:
            docs_skipped += 1
            if docs_skipped <= 10:
                print(f"  WARNING: Document {doc_idx} has no image-end token, skipping.")
            elif docs_skipped == 11:
                print("  WARNING: Suppressing further skip warnings...")
            continue

        builder.add_item(torch.tensor(tokens))
        builder.end_document()
        docs_written += 1

        if (docs_written + docs_skipped) % args.log_interval == 0:
            print(f"  Processed {docs_written + docs_skipped}/{num_docs} documents "
                  f"({docs_written} written, {docs_skipped} skipped)...")

    if args.workers > 1:
        pool.close()
        pool.join()

    builder.finalize(get_idx_path(args.output_prefix))

    print(f"\nCaption removal complete -> {args.output_prefix}")
    print(f"  Documents written: {docs_written}")
    print(f"  Documents skipped: {docs_skipped} (no image-end token)")

    # ── Verification ──
    print("\nVerification:")
    output_dataset = IndexedDataset(args.output_prefix, multimodal=False)
    out_num_docs = len(output_dataset.document_indices) - 1
    out_num_seqs = len(output_dataset.index)
    del output_dataset

    doc_match = out_num_docs == docs_written
    seq_match = out_num_seqs == docs_written
    print(f"  Documents: {out_num_docs} (expected {docs_written}) — {'PASS' if doc_match else 'FAIL'}")
    print(f"  Sequences: {out_num_seqs} (expected {docs_written}) — {'PASS' if seq_match else 'FAIL'}")

    if not (doc_match and seq_match):
        sys.exit(1)


if __name__ == "__main__":
    main()
