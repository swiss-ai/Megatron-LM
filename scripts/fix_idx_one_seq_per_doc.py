#!/usr/bin/env python3
"""Fix a Megatron .idx file so that every sequence becomes its own document.

The .bin file is untouched — only the .idx file is rewritten with new
document boundaries (one document per sequence).

Usage:
    python fix_idx_one_seq_per_doc.py <prefix> [--dry-run]

where <prefix> is the path without .bin/.idx extension, e.g.:
    python fix_idx_one_seq_per_doc.py /path/to/dataset

Options:
    --dry-run   Print what would change without writing anything.
"""

import argparse
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

_INDEX_HEADER = b"MMIDIDX\x00\x00"

DTYPE_CODES = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float64,
    7: np.float32,
    8: np.uint16,
}


def read_idx(path):
    """Read a Megatron .idx file and return all its components."""
    with open(path, "rb") as f:
        magic = f.read(9)
        assert magic == _INDEX_HEADER, f"Bad magic: {magic}"

        version = struct.unpack("<Q", f.read(8))[0]
        assert version == 1, f"Unsupported version: {version}"

        dtype_code = struct.unpack("<B", f.read(1))[0]
        dtype = DTYPE_CODES[dtype_code]

        seq_count = struct.unpack("<Q", f.read(8))[0]
        doc_count = struct.unpack("<Q", f.read(8))[0]

        seq_lengths = np.frombuffer(f.read(seq_count * 4), dtype=np.int32).copy()
        seq_pointers = np.frombuffer(f.read(seq_count * 8), dtype=np.int64).copy()
        doc_indices = np.frombuffer(f.read(doc_count * 8), dtype=np.int64).copy()

        remaining = f.read()
        if len(remaining) == seq_count:
            seq_modes = np.frombuffer(remaining, dtype=np.int8).copy()
        else:
            seq_modes = None

    return {
        "dtype_code": dtype_code,
        "dtype": dtype,
        "seq_count": seq_count,
        "doc_count": doc_count,
        "seq_lengths": seq_lengths,
        "seq_pointers": seq_pointers,
        "doc_indices": doc_indices,
        "seq_modes": seq_modes,
    }


def write_idx(path, dtype_code, seq_lengths, seq_pointers, doc_indices, seq_modes=None):
    """Write a Megatron .idx file."""
    with open(path, "wb") as f:
        f.write(_INDEX_HEADER)
        f.write(struct.pack("<Q", 1))  # version
        f.write(struct.pack("<B", dtype_code))
        f.write(struct.pack("<Q", len(seq_lengths)))  # sequence count
        f.write(struct.pack("<Q", len(doc_indices)))  # document count
        f.write(np.array(seq_lengths, dtype=np.int32).tobytes(order="C"))
        f.write(np.array(seq_pointers, dtype=np.int64).tobytes(order="C"))
        f.write(np.array(doc_indices, dtype=np.int64).tobytes(order="C"))
        if seq_modes is not None:
            f.write(np.array(seq_modes, dtype=np.int8).tobytes(order="C"))


def main():
    parser = argparse.ArgumentParser(
        description="Fix a Megatron .idx file so every sequence is its own document."
    )
    parser.add_argument("prefix", help="Dataset prefix (path without .bin/.idx extension)")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print what would change without writing."
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create a .idx.bak backup (backup is created by default).",
    )
    args = parser.parse_args()

    idx_path = Path(args.prefix + ".idx")
    bin_path = Path(args.prefix + ".bin")

    if not idx_path.exists():
        print(f"ERROR: {idx_path} not found", file=sys.stderr)
        sys.exit(1)
    if not bin_path.exists():
        print(f"WARNING: {bin_path} not found (only the .idx is modified, but this is unusual)")

    data = read_idx(idx_path)

    seq_count = data["seq_count"]
    old_doc_count = data["doc_count"]

    # New document indices: [0, 1, 2, ..., seq_count]
    # This means doc_count = seq_count + 1, following the Megatron convention
    # where doc_indices[0] = 0 is a sentinel and each subsequent entry is the
    # exclusive end-of-document sequence index.
    new_doc_indices = np.arange(seq_count + 1, dtype=np.int64)
    new_doc_count = len(new_doc_indices)

    print(f"File:            {idx_path}")
    print(f"Dtype code:      {data['dtype_code']} ({data['dtype'].__name__})")
    print(f"Sequence count:  {seq_count}")
    print(f"Old doc count:   {old_doc_count}")
    print(f"New doc count:   {new_doc_count}")
    print(f"Has seq modes:   {data['seq_modes'] is not None}")

    if old_doc_count == new_doc_count and np.array_equal(data["doc_indices"], new_doc_indices):
        print("\nAlready correct — nothing to do.")
        return

    if args.dry_run:
        print("\n[DRY RUN] Would rewrite the .idx file. No changes made.")
        return

    # Backup
    if not args.no_backup:
        backup_path = str(idx_path) + ".bak"
        shutil.copy2(idx_path, backup_path)
        print(f"\nBackup saved to: {backup_path}")

    write_idx(
        idx_path,
        dtype_code=data["dtype_code"],
        seq_lengths=data["seq_lengths"],
        seq_pointers=data["seq_pointers"],
        doc_indices=new_doc_indices,
        seq_modes=data["seq_modes"],
    )
    print(f"Wrote fixed idx: {idx_path}")

    # Verify by reading back
    verify = read_idx(idx_path)
    assert verify["seq_count"] == seq_count
    assert verify["doc_count"] == new_doc_count
    assert np.array_equal(verify["seq_lengths"], data["seq_lengths"])
    assert np.array_equal(verify["seq_pointers"], data["seq_pointers"])
    assert np.array_equal(verify["doc_indices"], new_doc_indices)
    print("Verification passed.")


if __name__ == "__main__":
    main()
