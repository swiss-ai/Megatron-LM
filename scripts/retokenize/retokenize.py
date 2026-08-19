"""Retokenize a Megatron .bin/.idx dataset from one tokenizer to another.

Decodes each document with the source tokenizer and re-encodes with the target
tokenizer, preserving document boundaries. Supports multiprocessing and
multimodal datasets (non-text sequences are copied as-is).

Usage:
    python scripts/retokenize/retokenize.py \
        --input my_dataset \
        --output-prefix my_dataset_newtok \
        --source-tokenizer meta-llama/Llama-2-7b-hf \
        --target-tokenizer mistralai/Mistral-7B-v0.1 \
        --workers 8 \
        --log-interval 500
"""

import argparse
import multiprocessing
import os
import sys

import numpy
import torch

from megatron.core.datasets.indexed_dataset import (
    DType,
    IndexedDataset,
    IndexedDatasetBuilder,
    get_bin_path,
    get_idx_path,
)


def get_args():
    parser = argparse.ArgumentParser(
        description="Retokenize a Megatron indexed dataset with a different tokenizer."
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
        "--source-tokenizer",
        type=str,
        required=True,
        help="HuggingFace model name/path for the source tokenizer",
    )
    parser.add_argument(
        "--target-tokenizer",
        type=str,
        required=True,
        help="HuggingFace model name/path for the target tokenizer",
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
    parser.add_argument(
        "--multimodal",
        action="store_true",
        help="Whether the dataset is multimodal",
    )

    args = parser.parse_args()

    assert os.path.exists(get_idx_path(args.input)), f"Missing index file: {get_idx_path(args.input)}"
    assert os.path.exists(get_bin_path(args.input)), f"Missing data file: {get_bin_path(args.input)}"

    output_dir = os.path.dirname(args.output_prefix)
    if output_dir:
        assert os.path.isdir(output_dir), f"Output directory does not exist: {output_dir}"

    return args


# ── Worker globals (initialized once per process) ──

_worker_source_tokenizer = None
_worker_target_tokenizer = None
_worker_source_bos_id = None
_worker_source_eos_id = None
_worker_target_bos_id = None
_worker_target_eos_id = None
_worker_dataset = None
_worker_multimodal = None


def _worker_init(input_prefix, source_tok_name, target_tok_name, multimodal):
    """Initialize per-worker tokenizers and dataset (mmap)."""
    global _worker_source_tokenizer, _worker_target_tokenizer, _worker_dataset, _worker_multimodal
    global _worker_source_bos_id, _worker_source_eos_id, _worker_target_bos_id, _worker_target_eos_id
    from transformers import AutoTokenizer

    _worker_source_tokenizer = AutoTokenizer.from_pretrained(source_tok_name)
    _worker_target_tokenizer = AutoTokenizer.from_pretrained(target_tok_name)
    _worker_source_bos_id = _worker_source_tokenizer.bos_token_id
    _worker_source_eos_id = _worker_source_tokenizer.eos_token_id
    _worker_target_bos_id = _worker_target_tokenizer.bos_token_id
    _worker_target_eos_id = _worker_target_tokenizer.eos_token_id
    _worker_dataset = IndexedDataset(input_prefix, multimodal=multimodal)
    _worker_multimodal = multimodal


def _retokenize_sequence(token_ids):
    """Retokenize a single text sequence, preserving BOS/EOS boundaries.

    1. Detect and strip source BOS (first token) and EOS (last token).
    2. Decode the interior tokens with the source tokenizer.
    3. Re-encode with the target tokenizer (no special tokens).
    4. Re-attach target BOS / EOS in the same positions.
    """
    had_bos = False
    had_eos = False

    if len(token_ids) > 0 and _worker_source_bos_id is not None and token_ids[0] == _worker_source_bos_id:
        had_bos = True
        token_ids = token_ids[1:]

    if len(token_ids) > 0 and _worker_source_eos_id is not None and token_ids[-1] == _worker_source_eos_id:
        had_eos = True
        token_ids = token_ids[:-1]

    text = _worker_source_tokenizer.decode(token_ids)
    new_ids = _worker_target_tokenizer.encode(text, add_special_tokens=False)

    if had_bos and _worker_target_bos_id is not None:
        new_ids = [_worker_target_bos_id] + new_ids

    if had_eos and _worker_target_eos_id is not None:
        new_ids = new_ids + [_worker_target_eos_id]

    return new_ids


def _retokenize_document(doc_idx):
    """Retokenize a single document. Returns (doc_idx, list_of_token_arrays, list_of_modes)."""
    dataset = _worker_dataset
    doc_indices = dataset.document_indices
    seq_start = int(doc_indices[doc_idx])
    seq_end = int(doc_indices[doc_idx + 1])

    new_tokens_list = []
    modes_list = []

    for seq_idx in range(seq_start, seq_end):
        if _worker_multimodal:
            tokens, mode = dataset[seq_idx]
        else:
            tokens = dataset[seq_idx]
            mode = 0

        if _worker_multimodal and mode != 0:
            # Non-text modality: copy as-is
            new_tokens_list.append(tokens.tolist())
        else:
            new_ids = _retokenize_sequence(tokens.tolist())
            new_tokens_list.append(new_ids)

        modes_list.append(int(mode))

    return (doc_idx, new_tokens_list, modes_list)


def main():
    args = get_args()

    from transformers import AutoTokenizer

    # Load tokenizers to determine dtype and log BOS/EOS mapping
    source_tokenizer = AutoTokenizer.from_pretrained(args.source_tokenizer)
    target_tokenizer = AutoTokenizer.from_pretrained(args.target_tokenizer)
    output_dtype = DType.optimal_dtype(target_tokenizer.vocab_size)

    print(f"BOS/EOS mapping:")
    print(f"  Source BOS: {source_tokenizer.bos_token!r} (id={source_tokenizer.bos_token_id})")
    print(f"  Source EOS: {source_tokenizer.eos_token!r} (id={source_tokenizer.eos_token_id})")
    print(f"  Target BOS: {target_tokenizer.bos_token!r} (id={target_tokenizer.bos_token_id})")
    print(f"  Target EOS: {target_tokenizer.eos_token!r} (id={target_tokenizer.eos_token_id})")

    del source_tokenizer, target_tokenizer

    # Get source dataset info
    source_dataset = IndexedDataset(args.input, multimodal=args.multimodal)
    num_docs = len(source_dataset.document_indices) - 1
    num_seqs = len(source_dataset.index)
    print(f"Input: {args.input}")
    print(f"  Sequences: {num_seqs}")
    print(f"  Documents: {num_docs}")
    print(f"  Output dtype: {output_dtype.__name__}")
    del source_dataset

    builder = IndexedDatasetBuilder(
        get_bin_path(args.output_prefix), dtype=output_dtype, multimodal=args.multimodal
    )

    if args.workers > 1:
        pool = multiprocessing.Pool(
            args.workers,
            initializer=_worker_init,
            initargs=(args.input, args.source_tokenizer, args.target_tokenizer, args.multimodal),
        )
        doc_iter = pool.imap(_retokenize_document, range(num_docs))
    else:
        _worker_init(args.input, args.source_tokenizer, args.target_tokenizer, args.multimodal)
        doc_iter = map(_retokenize_document, range(num_docs))

    docs_written = 0
    seqs_written = 0

    for doc_idx, new_tokens_list, modes_list in doc_iter:
        for tokens, mode in zip(new_tokens_list, modes_list):
            builder.add_item(torch.tensor(tokens), mode=mode)
            seqs_written += 1
        builder.end_document()
        docs_written += 1

        if docs_written % args.log_interval == 0:
            print(f"  Processed {docs_written}/{num_docs} documents...")

    if args.workers > 1:
        pool.close()
        pool.join()

    builder.finalize(get_idx_path(args.output_prefix))

    print(f"\nRetokenization complete -> {args.output_prefix}")
    print(f"  Documents written: {docs_written}")
    print(f"  Sequences written: {seqs_written}")

    # ── Verification ──
    print("\nVerification:")
    output_dataset = IndexedDataset(args.output_prefix, multimodal=args.multimodal)
    out_num_docs = len(output_dataset.document_indices) - 1
    out_num_seqs = len(output_dataset.index)
    del output_dataset

    doc_match = out_num_docs == num_docs
    seq_match = out_num_seqs == num_seqs
    print(f"  Documents: {out_num_docs} (expected {num_docs}) — {'PASS' if doc_match else 'FAIL'}")
    print(f"  Sequences: {out_num_seqs} (expected {num_seqs}) — {'PASS' if seq_match else 'FAIL'}")

    if not (doc_match and seq_match):
        sys.exit(1)


if __name__ == "__main__":
    main()
