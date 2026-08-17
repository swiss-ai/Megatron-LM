import argparse
import os
import struct
import sys
from pathlib import Path
from typing import List, Optional

_INDEX_HEADER = b"MMIDIDX\x00\x00"


def idx_total_tokens(idx_path: str) -> int:
    """Sum of sequence lengths recorded in a Megatron .idx file (total token count)."""
    with open(idx_path, "rb") as stream:
        header = stream.read(9)
        assert header == _INDEX_HEADER, f"bad header, cannot read: {idx_path}"
        stream.read(8)  # version
        stream.read(1)  # dtype code
        sequence_count = struct.unpack("<Q", stream.read(8))[0]
        stream.read(8)  # document_count
        sequence_lengths = struct.unpack(f"<{sequence_count}i", stream.read(4 * sequence_count))
        return sum(sequence_lengths)


def create_data_prefix(list_of_paths: List[str], min_tokens: Optional[int] = None):
    list_of_bin_files = []
    # Select all .bin files
    for path in list_of_paths:
        path_to_files = [
            os.path.join(dp, f)
            for dp, _, fn in os.walk(os.path.expanduser(path))
            for f in fn
        ]
        list_of_bin_files.extend(
            [
                raw_file
                for raw_file in path_to_files
                if Path(raw_file).suffix.lower().endswith(".bin")
            ]
        )

    list_of_bin_files = [
        bin_file[:-4] for bin_file in list_of_bin_files
    ]  # NOTE(tj.solergibert) Delete .bin extension to have file prefixes

    if min_tokens is not None:
        # NOTE Files with fewer than `min_tokens` total tokens cannot yield a single
        # training sample. Megatron derives per-file blend weights from the number of
        # samples each file produces, so a file that produces zero samples gets weight 0,
        # which trips `assert all(w > 0 for w in weights)` in BlendedDataset and crashes
        # the whole job. Drop them here instead.
        kept, dropped = [], []
        for prefix in list_of_bin_files:
            (kept if idx_total_tokens(prefix + ".idx") >= min_tokens else dropped).append(prefix)
        if dropped:
            print(
                f"[create_data_config] Skipping {len(dropped)} file(s) with < {min_tokens} "
                f"total tokens (cannot yield a full sample):",
                file=sys.stderr,
            )
            for prefix in dropped:
                print(f"  - {prefix}", file=sys.stderr)
        list_of_bin_files = kept

    return list_of_bin_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--paths",
        type=str,
        required=True,
        help="Comma separated list of paths to generate the config from. e.g. -p /path/to/dataset/A,/path/to/dataset/B,/path/to/dataset/C",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=None,
        help="If set, drop any file whose total token count is below seq-length + 1, since "
        "such a file cannot yield a single training sample and would otherwise crash "
        "Megatron's blended dataset builder with a zero-weight assertion.",
    )
    args = parser.parse_args()

    paths = [x.strip() for x in args.paths.split(",")]
    min_tokens = args.seq_length + 1 if args.seq_length is not None else None
    data_prefix = create_data_prefix(paths, min_tokens=min_tokens)
    print(*data_prefix, sep=" ")
