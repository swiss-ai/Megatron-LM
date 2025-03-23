"""Initializes evaluation metadata."""

import json
import os

from check_evaluations import EVAL_METADATA_PATH


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Initializes evaluation metadata.")
    parser.add_argument(
        "--checkpoints_root",
        default="/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/Meg-Runs/main-runs-v1",
    )
    parser.add_argument(
        "--model_dirs",
        nargs="+",
        default=[
            "apertus3-1b-21-nodes",
            "apertus3-3b-64-nodes",
            "apertus3-8b-128-nodes",
        ],
    )
    parser.add_argument(
        "--model_names",
        nargs="+",
        default=["Apertus3-1.5B", "Apertus3-3B", "Apertus3-8B"],
    )
    parser.add_argument("--checkpoints_dir", default="checkpoints")
    return parser.parse_args()


def main(args):
    assert not os.path.exists(EVAL_METADATA_PATH), "Metadata file already exists"
    metadata = {}
    for model_name, model_dir in zip(args.model_names, args.model_dirs):
        metadata[model_name] = {
            "checkpoints": os.path.join(
                args.checkpoints_root, model_dir, args.checkpoints_dir
            ),
            "iterations": {},
        }
    with open(EVAL_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)


if __name__ == "__main__":
    main(parse_args())
