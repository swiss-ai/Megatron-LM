"""Split a Megatron .bin/.idx dataset into parts by ratio at document boundaries.

Usage:
    python scripts/split_dataset/split_dataset.py \
        --input my_dataset \
        --output-prefix my_dataset_split \
        --ratios 0.8 0.2 \
        --multimodal  # optional
"""

import argparse
import os

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
        description="Split a Megatron indexed dataset into parts by ratio at document boundaries."
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
        help="Base output prefix; produces PREFIX_part0, PREFIX_part1, ...",
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        required=True,
        help="Split ratios (must sum to 1.0), e.g. 0.8 0.2",
    )
    parser.add_argument(
        "--multimodal",
        action="store_true",
        help="Whether the dataset is multimodal",
    )

    args = parser.parse_args()

    assert os.path.exists(get_idx_path(args.input)), f"Missing index file: {get_idx_path(args.input)}"
    assert os.path.exists(get_bin_path(args.input)), f"Missing data file: {get_bin_path(args.input)}"

    ratio_sum = sum(args.ratios)
    assert abs(ratio_sum - 1.0) < 1e-6, f"Ratios must sum to 1.0, got {ratio_sum}"
    assert all(r > 0 for r in args.ratios), "All ratios must be positive"

    output_dir = os.path.dirname(args.output_prefix)
    if output_dir:
        assert os.path.isdir(output_dir), f"Output directory does not exist: {output_dir}"

    return args


def compute_split_points(num_docs, ratios):
    """Compute document split points from ratios using cumulative rounding."""
    cumulative = numpy.cumsum(ratios)
    split_points = [0]
    for c in cumulative[:-1]:
        split_points.append(int(round(c * num_docs)))
    split_points.append(num_docs)
    return split_points


def main():
    args = get_args()

    dataset = IndexedDataset(args.input, multimodal=args.multimodal)
    doc_indices = dataset.document_indices
    num_docs = len(doc_indices) - 1
    dtype = dataset.index.dtype

    print(f"Input: {args.input}")
    print(f"  Sequences: {len(dataset.index)}")
    print(f"  Documents: {num_docs}")

    split_points = compute_split_points(num_docs, args.ratios)
    print(f"  Document split points: {split_points}")

    for part_idx in range(len(args.ratios)):
        start_doc = split_points[part_idx]
        end_doc = split_points[part_idx + 1]
        num_part_docs = end_doc - start_doc

        if num_part_docs == 0:
            print(f"\n  Part {part_idx}: 0 documents (skipped)")
            continue

        seq_start = int(doc_indices[start_doc])
        seq_end = int(doc_indices[end_doc])

        part_prefix = f"{args.output_prefix}_part{part_idx}"
        builder = IndexedDatasetBuilder(
            get_bin_path(part_prefix), dtype=dtype, multimodal=args.multimodal
        )

        # Iterate document by document
        for doc_idx in range(start_doc, end_doc):
            doc_seq_start = int(doc_indices[doc_idx])
            doc_seq_end = int(doc_indices[doc_idx + 1])
            for seq_idx in range(doc_seq_start, doc_seq_end):
                if args.multimodal:
                    tokens, mode = dataset[seq_idx]
                else:
                    tokens = dataset[seq_idx]
                    mode = 0
                builder.add_item(torch.tensor(tokens), mode=mode)
            builder.end_document()

        builder.finalize(get_idx_path(part_prefix))

        print(f"\n  Part {part_idx}: {part_prefix}")
        print(f"    Documents: {num_part_docs}")
        print(f"    Sequences: {seq_end - seq_start}")

    del dataset
    print("\nDone.")


if __name__ == "__main__":
    main()
