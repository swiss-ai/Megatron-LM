#!/usr/bin/env python3
"""Inspect a Megatron indexed dataset (.idx) file.

Prints number of sequences, number of documents, document indices array length,
and the first/last 4 entries of the document indices array.

Usage:
    python scripts/inspect_idx.py <path_to_dataset_prefix_or_idx_file>
"""

import argparse
import struct
import numpy as np


def inspect_idx(path: str) -> None:
    if not path.endswith(".idx"):
        path = path + ".idx"

    with open(path, "rb") as f:
        magic = f.read(9)
        assert magic == b"MMIDIDX\x00\x00", f"Invalid magic header: {magic!r}"

        version = struct.unpack("<Q", f.read(8))[0]
        dtype_code = struct.unpack("<B", f.read(1))[0]
        seq_count = struct.unpack("<Q", f.read(8))[0]
        doc_count = struct.unpack("<Q", f.read(8))[0]

        # Skip sequence_lengths (int32) and sequence_pointers (int64)
        f.seek(seq_count * 4 + seq_count * 8, 1)

        doc_indices = np.frombuffer(f.read(doc_count * 8), dtype=np.int64).copy()

    print(f"File:               {path}")
    print(f"Version:            {version}")
    print(f"Dtype code:         {dtype_code}")
    print(f"Num sequences:      {seq_count}")
    print(f"Num documents:      {doc_count - 1}")
    print(f"Doc indices length: {doc_count}")
    print()

    n = min(4, len(doc_indices))
    print(f"First {n} doc indices: {doc_indices[:n].tolist()}")
    print(f"Last  {n} doc indices: {doc_indices[-n:].tolist()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect a Megatron .idx file")
    parser.add_argument("path", help="Path to dataset prefix or .idx file")
    args = parser.parse_args()
    inspect_idx(args.path)
