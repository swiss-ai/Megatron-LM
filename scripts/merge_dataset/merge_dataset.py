"""Merge multiple Megatron .bin/.idx datasets into one, in explicit order.

Usage:
    python scripts/merge_dataset/merge_dataset.py \
        --inputs dataset_a dataset_b dataset_c \
        --output-prefix merged_output \
        --multimodal  # optional
"""

import argparse
import os
import sys

from megatron.core.datasets.indexed_dataset import (
    IndexedDataset,
    IndexedDatasetBuilder,
    get_bin_path,
    get_idx_path,
)


def get_args():
    parser = argparse.ArgumentParser(
        description="Merge multiple Megatron indexed datasets in explicit order."
    )

    parser.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        required=True,
        help="Ordered list of input dataset prefixes (no .bin/.idx extension)",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        required=True,
        help="Output dataset prefix (no extension)",
    )
    parser.add_argument(
        "--multimodal",
        action="store_true",
        help="Whether the datasets are multimodal",
    )

    args = parser.parse_args()

    for prefix in args.inputs:
        assert os.path.exists(get_idx_path(prefix)), f"Missing index file: {get_idx_path(prefix)}"
        assert os.path.exists(get_bin_path(prefix)), f"Missing data file: {get_bin_path(prefix)}"

    output_dir = os.path.dirname(args.output_prefix)
    if output_dir:
        assert os.path.isdir(output_dir), f"Output directory does not exist: {output_dir}"

    return args


def main():
    args = get_args()

    # Open the first dataset to determine dtype
    first_dataset = IndexedDataset(args.inputs[0], multimodal=args.multimodal)
    dtype = first_dataset.index.dtype
    del first_dataset

    builder = IndexedDatasetBuilder(
        get_bin_path(args.output_prefix), dtype=dtype, multimodal=args.multimodal
    )

    total_sequences = 0
    total_documents = 0

    for i, prefix in enumerate(args.inputs):
        dataset = IndexedDataset(prefix, multimodal=args.multimodal)
        num_seq = len(dataset.index)
        num_doc = len(dataset.document_indices)
        print(f"  [{i}] {prefix}: {num_seq} sequences, {num_doc} documents")
        assert dataset.index.dtype == dtype, (
            f"dtype mismatch: expected {dtype}, got {dataset.index.dtype} for {prefix}"
        )
        total_sequences += num_seq
        total_documents += num_doc
        del dataset

        builder.add_index(prefix)

    builder.finalize(get_idx_path(args.output_prefix))

    print(f"\nMerged {len(args.inputs)} datasets -> {args.output_prefix}")
    print(f"  Total sequences: {total_sequences}")
    # Each part's document_indices includes a sentinel; the merged dataset has only one sentinel,
    # so subtract the extra (N-1) sentinels from the sum.
    print(f"  Total documents: {total_documents - (len(args.inputs) - 1)}")


if __name__ == "__main__":
    main()
