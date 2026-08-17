"""
python3 scripts/tokenization/prepare_dumps.py --dataset-folder /capstor/store/cscs/swissai/a06/datasets_raw/fineweb-2 --filter-in snapshot/data train --filter-out _removed und_ --preprocessing-metadata-folder datasets/fineweb-2 --n-dumps 10
"""

import argparse
import os
from pathlib import Path
from typing import List


def get_files(path_to_folder: List, fmt: str) -> List:
    files = [
        os.path.join(dp, f)
        for dp, _, fn in os.walk(os.path.expanduser(path_to_folder), followlinks=True)
        for f in fn
    ]

    if len(files) == 0:
        raise ValueError(f"No .{fmt} files found in {path_to_folder}")

    filtered_files = [
        raw_file
        for raw_file in files
        if Path(raw_file).name.lower().endswith(f".{fmt}")
        or Path(raw_file).name.lower().endswith(f".{fmt}.gz")
    ]

    return filtered_files


def filter_in(list_of_files: List, list_of_folders: List) -> List:
    return_list_of_files = []
    for folder in list_of_folders:
        return_list_of_files.extend([file for file in list_of_files if (folder in file and file not in return_list_of_files)])
    return return_list_of_files


def filter_out(list_of_files: List, list_of_folders: List) -> List:
    for folder in list_of_folders:
        list_of_files = [file for file in list_of_files if folder not in file]
    return list_of_files


def listOfTuples(l1: List, l2: List) -> List:
    assert len(l1) == len(l2)
    return list(map(lambda x, y: (x, y), l1, l2))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-folder",
        type=str,
        required=True,
        help="Path to a folder containing recursively dataset files",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["parquet", "jsonl"],
        default="parquet",
        help="File format to look for (default: parquet)",
    )
    parser.add_argument(
        "--filter-in",
        nargs="+",
        help="Name of the paths to filter in. e.g. --filter-in esp en fr",
        default=None,
    )
    parser.add_argument(
        "--filter-out",
        nargs="+",
        help="Name of the paths to filter out. e.g. --filter out test valid",
        default=None,
    )
    parser.add_argument(
        "--preprocessing-metadata-folder",
        type=str,
        required=True,
        help="Path to a folder to store the generated metadata files",
    )
    parser.add_argument(
        "--n-dumps",
        type=int,
        default=None,
        help="Total number of dumps to split the files into. If None it will automatically compute this value based on the amount and size of parquet files",
    )
    args = parser.parse_args()

    return args


def main(args):
    print(f"Scanning .{args.format} files in {args.dataset_folder}...")
    dataset_files = get_files(args.dataset_folder, args.format)
    print(f"Found a total of {len(dataset_files)} in {args.dataset_folder}")
    dataset_files = (
        filter_in(dataset_files, args.filter_in) if args.filter_in else dataset_files
    )
    dataset_files = (
        filter_out(dataset_files, args.filter_out) if args.filter_out else dataset_files
    )
    size_of_dataset_files = [
        os.path.getsize(dataset_file) for dataset_file in dataset_files
    ]
    print(
        f"Total number of files filtered to tokenize: {len(dataset_files)} ({sum(size_of_dataset_files) / 1e9:.2f} GB)"
    )

    number_of_dumps = args.n_dumps if args.n_dumps else int(sum(size_of_dataset_files) / 150e9) + 1
    if number_of_dumps > len(dataset_files):
        number_of_dumps = len(dataset_files)

    print(f"Splitting {len(dataset_files)} into {number_of_dumps} dumps...")
    files_sizes_tuples = listOfTuples(dataset_files, size_of_dataset_files)
    files_sizes_tuples = sorted(
        files_sizes_tuples, key=lambda x: x[1], reverse=True
    )

    dump_folder_files = [[] for _ in range(number_of_dumps)]
    dump_folder_size = [0] * number_of_dumps

    for dataset_file, dataset_file_size in files_sizes_tuples:
        min_ind = dump_folder_size.index(min(dump_folder_size))
        dump_folder_files[min_ind].append(dataset_file)
        dump_folder_size[min_ind] += dataset_file_size

    PATH_TO_DATASET_SYMLINK = os.path.join(
        args.preprocessing_metadata_folder, "raw-dataset-link"
    )
    PATH_TO_DUMP_FOLDER = os.path.join(args.preprocessing_metadata_folder, "dumps")

    # Create folder to store paths files
    Path(PATH_TO_DUMP_FOLDER).mkdir(parents=True, exist_ok=True)

    # Create symlink to original dataset
    if not os.path.islink(PATH_TO_DATASET_SYMLINK):
        os.symlink(args.dataset_folder, PATH_TO_DATASET_SYMLINK)

    for i, (dump_files, dump_size) in enumerate(
        zip(dump_folder_files, dump_folder_size)
    ):
        print(
            f"[ Dump {i} | {dump_size / 1e9:.2f} GB | {len(dump_files)} Files | ~{20 * dump_size / 3600e9:.2f} hours to tokenize (@20 s per GB)]"
        )

        relative_paths = [
            os.path.relpath(path, args.dataset_folder) for path in dump_files
        ]
        with open(os.path.join(PATH_TO_DUMP_FOLDER, f"paths_file_{i}.txt"), "w") as f:
            for relative_path in relative_paths:
                f.write(f"{relative_path}\n")

    print("Finished preparing the dumps!")


if __name__ == "__main__":
    _args = get_args()
    main(_args)
