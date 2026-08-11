"""
This script is a modified version of create_data_config.py.

It will find all bin/idx pairs in the given paths and assign overall weights according to the number of tokens in each dataset. All weights are normalized to sum to the given weight.

ex:

python $MEGATRON_LM_DIR/scripts/tools/create_weighted_data_config_by_tokens.py \
      --paths /iopsstor/scratch/cscs/jpcoles/a06/phase-5 \
      --weight 0.1)
"""

import argparse
import os
import struct
import sys
from pathlib import Path
from typing import List, Tuple


def get_dataset_size(prefix: str) -> int:
    """Get the number of tokens from a .bin file.

    Args:
        prefix: Dataset prefix (without .bin or .idx extension)

    Returns:
        Number of tokens in the dataset
    """
    bin_file = f"{prefix}.bin"
    if not os.path.isfile(bin_file):
        raise FileNotFoundError(f"Binary file not found: {bin_file}")

    return os.path.getsize(bin_file) // 4


def create_data_prefix(list_of_paths: List[str]) -> List[str]:
    """Find all .bin files in the given paths and return their prefixes.

    Args:
        list_of_paths: List of directory paths to search

    Returns:
        List of dataset prefixes (without .bin extension)
    """
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
    ]  # Delete .bin extension to have file prefixes

    return list_of_bin_files


def calculate_proportional_weights(prefixes: List[str], total_weight: float) -> List[Tuple[float, str]]:
    """Calculate proportional weights for datasets based on their sizes.

    Args:
        prefixes: List of dataset prefixes
        total_weight: Total weight to distribute (e.g., 0.1)

    Returns:
        List of (weight, prefix) tuples
    """
    # Get sizes for all datasets
    sizes = []
    for prefix in prefixes:
        size = get_dataset_size(prefix)
        sizes.append(size)

    total_size = sum(sizes)

    if total_size == 0:
        raise ValueError("Total dataset size is zero")

    # Calculate proportional weights
    weighted_prefixes = []
    for size, prefix in zip(sizes, prefixes):
        weight = total_weight * size / total_size
        weighted_prefixes.append((weight, prefix))

    return weighted_prefixes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create weighted data config with proportional weights based on dataset sizes"
    )
    parser.add_argument(
        "-p",
        "--paths",
        type=str,
        required=True,
        help="Comma separated list of paths to generate the config from. e.g. -p /path/to/dataset/A,/path/to/dataset/B,/path/to/dataset/C",
    )
    parser.add_argument(
        "-w",
        "--weight",
        type=float,
        required=True,
        help="Total weight to distribute among datasets (e.g., 0.1). Weights will be proportional to dataset sizes.",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["megatron", "verbose"],
        default="megatron",
        help="Output format: 'megatron' outputs space-separated weights and prefixes, 'verbose' shows breakdown",
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        choices=["sft", "pretrain", "none"],
        default="none",
        help="Prepend an explicit dataset-type marker ('sft:' or 'pretrain:') to each emitted "
             "prefix. 'none' (default) emits bare prefixes for backward compatibility.",
    )
    args = parser.parse_args()

    paths = [x.strip() for x in args.paths.split(",")]
    prefixes = create_data_prefix(paths)

    if not prefixes:
        print("Error: No .bin files found in the specified paths", file=sys.stderr)
        sys.exit(1)

    weighted_prefixes = calculate_proportional_weights(prefixes, args.weight)

    marker = f"{args.dataset_type}:" if args.dataset_type != "none" else ""

    if args.format == "verbose":
        print(f"Found {len(prefixes)} datasets with total weight {args.weight}")
        print(f"{'Weight':<12} {'Percentage':<12} {'Size':<15} {'Prefix'}")
        print("-" * 80)
        for weight, prefix in weighted_prefixes:
            size = get_dataset_size(prefix)
            percentage = (weight / args.weight) * 100
            print(f"{weight:<12.6f} {percentage:<12.2f} {size:<15,} {marker}{prefix}")
        print("-" * 80)
        total = sum(w for w, _ in weighted_prefixes)
        print(f"Total weight: {total:.6f}")
        print()
        print("Megatron format:")

    # Output in Megatron format: weight1 prefix1 weight2 prefix2 ...
    output_parts = []
    for weight, prefix in weighted_prefixes:
        output_parts.extend([f"{weight:.6f}", f"{marker}{prefix}"])

    print(*output_parts, sep=" ")
