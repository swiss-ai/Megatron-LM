"""Excludes earlier iterations form being evaluated by setting their state to finished."""

import os
import re

from check_evaluations import State, EvalMetadata


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Excludes some iterations from being evaluated."
    )
    parser.add_argument("--eval_metadata_path", default="$SCRATCH/eval_metadata.json")
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
    parser.add_argument("--checkpoints_dir", default="checkpoints")
    parser.add_argument(
        "--model_names",
        nargs="+",
        default=["Apertus3-1.5B", "Apertus3-3B", "Apertus3-8B"],
    )
    parser.add_argument(
        "--max_iteration_already_evaluated",
        nargs="+",
        type=int,
        default=[1_500_000, 990_000, 866_000],
    )
    return parser.parse_args()


def main(args):
    for model_name, model_dir, max_iter in zip(
        args.model_names, args.model_dirs, args.max_iteration_already_evaluated
    ):
        eval_metadata = EvalMetadata()
        checkpoints_dir = os.path.join(
            args.checkpoints_root, model_dir, args.checkpoints_dir
        )
        for d in os.listdir(checkpoints_dir):
            if d.startswith("iter_"):
                match = re.search(r"^iter_(\d+)$", d)
                if not match:
                    print(f"Warning: Unknown checkpoint dir naming: {d}")
                    continue
                iteration = int(match.group(1))
                if iteration <= max_iter:
                    eval_metadata.update_iteration_metadata(
                        model_name,
                        iteration,
                        State.FINISHED,
                        checkpoint_dir=os.path.join(checkpoints_dir, d),
                    )
        print(
            f"Set the evaluations for {model_name} to *finished* until iteration {max_iter}"
        )


if __name__ == "__main__":
    main(parse_args())
