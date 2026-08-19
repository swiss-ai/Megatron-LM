"""Fast split of a Megatron .bin/.idx dataset into parts by ratio at document boundaries.

Instead of reading/writing sequences one by one, this script:
  1. Slices the index arrays (sequence_lengths, sequence_pointers, document_indices,
     sequence_modes) using numpy operations.
  2. Copies the relevant byte range from the source .bin file in a single bulk transfer.
  3. Writes a new .idx file directly from the sliced arrays.

This makes it orders of magnitude faster than the item-by-item approach on large datasets.

Usage:
    python scripts/split_dataset/split_dataset_fast.py \
        --input my_dataset \
        --output-prefix my_dataset_split \
        --ratios 0.8 0.2 \
        --multimodal  # optional
"""

import argparse
import os
import struct
import shutil

import numpy

from megatron.core.datasets.indexed_dataset import (
    DType,
    IndexedDataset,
    _INDEX_HEADER,
    get_bin_path,
    get_idx_path,
)

COPY_BUFFER_SIZE = 64 * 1024 * 1024  # 64 MiB


def get_args():
    parser = argparse.ArgumentParser(
        description="Fast split of a Megatron indexed dataset into parts by ratio at document boundaries."
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


def write_idx_file(
    idx_path,
    dtype,
    sequence_lengths,
    sequence_pointers,
    document_indices,
    sequence_modes,
):
    """Write a .idx file from pre-computed arrays."""
    with open(idx_path, "wb") as f:
        # Header
        f.write(_INDEX_HEADER)
        # Version
        f.write(struct.pack("<Q", 1))
        # Dtype code
        f.write(struct.pack("<B", DType.code_from_dtype(dtype)))
        # Sequence count
        f.write(struct.pack("<Q", len(sequence_lengths)))
        # Document count
        f.write(struct.pack("<Q", len(document_indices)))
        # Sequence lengths (int32)
        f.write(numpy.ascontiguousarray(sequence_lengths, dtype=numpy.int32).tobytes())
        # Sequence pointers (int64)
        f.write(numpy.ascontiguousarray(sequence_pointers, dtype=numpy.int64).tobytes())
        # Document indices (int64)
        f.write(numpy.ascontiguousarray(document_indices, dtype=numpy.int64).tobytes())
        # Sequence modes (int8), only if multimodal
        if sequence_modes is not None:
            f.write(numpy.ascontiguousarray(sequence_modes, dtype=numpy.int8).tobytes())


def copy_bin_range(src_bin_path, dst_bin_path, byte_start, byte_end):
    """Copy a byte range [byte_start, byte_end) from src to dst .bin file."""
    num_bytes = byte_end - byte_start
    if num_bytes == 0:
        # Create empty file
        open(dst_bin_path, "wb").close()
        return

    with open(src_bin_path, "rb") as src, open(dst_bin_path, "wb") as dst:
        src.seek(byte_start)
        remaining = num_bytes
        while remaining > 0:
            chunk = min(COPY_BUFFER_SIZE, remaining)
            data = src.read(chunk)
            if not data:
                break
            dst.write(data)
            remaining -= len(data)


def main():
    args = get_args()

    dataset = IndexedDataset(args.input, multimodal=args.multimodal)
    index = dataset.index

    doc_indices = index.document_indices    # int64, length = document_count
    seq_lengths = index.sequence_lengths    # int32, length = sequence_count
    seq_pointers = index.sequence_pointers  # int64, length = sequence_count
    seq_modes = index.sequence_modes        # int8 or None
    dtype = index.dtype
    num_docs = index.document_count

    print(f"Input: {args.input}")
    print(f"  Sequences: {index.sequence_count}")
    print(f"  Documents: {num_docs}")

    split_points = compute_split_points(num_docs, args.ratios)
    print(f"  Document split points: {split_points}")

    src_bin_path = get_bin_path(args.input)

    for part_idx in range(len(args.ratios)):
        start_doc = split_points[part_idx]
        end_doc = split_points[part_idx + 1]
        num_part_docs = end_doc - start_doc

        if num_part_docs == 0:
            print(f"\n  Part {part_idx}: 0 documents (skipped)")
            continue

        # Sequence range for this part
        seq_start = int(doc_indices[start_doc])
        seq_end = int(doc_indices[end_doc]) if end_doc < num_docs else index.sequence_count
        num_seqs = seq_end - seq_start

        # Slice index arrays for this part
        part_seq_lengths = seq_lengths[seq_start:seq_end]

        # Recompute pointers relative to byte offset 0 for the new .bin file
        itemsize = DType.size(dtype)
        part_seq_pointers = numpy.zeros(num_seqs, dtype=numpy.int64)
        if num_seqs > 0:
            numpy.cumsum(part_seq_lengths.astype(numpy.int64) * itemsize, out=part_seq_pointers)
            # Shift right: pointers[i] = sum of lengths[0..i-1] * itemsize
            part_seq_pointers = numpy.roll(part_seq_pointers, 1)
            part_seq_pointers[0] = 0

        # Adjust document indices relative to seq_start
        # Include end_doc+1 to capture the sentinel entry (doc_indices always has num_docs+1 entries)
        part_doc_indices = doc_indices[start_doc:end_doc + 1] - seq_start

        # Slice sequence modes if multimodal
        part_seq_modes = None
        if seq_modes is not None:
            part_seq_modes = seq_modes[seq_start:seq_end]

        # Compute byte range in source .bin
        byte_start = int(seq_pointers[seq_start])
        if seq_end < index.sequence_count:
            byte_end = int(seq_pointers[seq_end])
        else:
            # Last sequence: byte_end = pointer + length * itemsize
            byte_end = int(seq_pointers[seq_end - 1]) + int(seq_lengths[seq_end - 1]) * itemsize

        # Write output files
        part_prefix = f"{args.output_prefix}_part{part_idx}"

        copy_bin_range(src_bin_path, get_bin_path(part_prefix), byte_start, byte_end)

        write_idx_file(
            get_idx_path(part_prefix),
            dtype,
            part_seq_lengths,
            part_seq_pointers,
            part_doc_indices,
            part_seq_modes,
        )

        print(f"\n  Part {part_idx}: {part_prefix}")
        print(f"    Documents: {len(part_doc_indices)}")
        print(f"    Sequences: {num_seqs}")
        print(f"    Bytes copied: {byte_end - byte_start:,}")

    del dataset
    print("\nDone.")


if __name__ == "__main__":
    main()