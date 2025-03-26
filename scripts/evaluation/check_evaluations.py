"""Regularly checks Imanol's checkpoints to submit evaluation runs."""

import json
import os
import re
import shutil
import time
from collections import defaultdict
from enum import Enum

State = Enum(
    "State",
    [
        ("NOT_EVALUATED", 0),
        ("SUBMITTED", 1),
        ("RUNNING", 2),
        ("FINISHED", 3),
        ("FAILED", 4),
    ],
)

EVAL_METADATA_PATH = "/users/amarfurt/eval_metadata.json"


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Checks for evaluations to run.")

    # personal paths
    parser.add_argument(
        "--logs_root", default="/iopsstor/scratch/cscs/amarfurt/eval-logs"
    )
    parser.add_argument(
        "--hf_temp_dir", default="/iopsstor/scratch/cscs/amarfurt/hf_checkpoints"
    )
    parser.add_argument(
        "--hf_storage_dir", default="/capstor/store/cscs/swissai/a06/hf_checkpoints"
    )
    parser.add_argument(
        "--container_path",
        default="/iopsstor/scratch/cscs/amarfurt/envs/ngc_pt_jan.toml",
    )

    # model checkpoints
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
    parser.add_argument(
        "--model_tokens_per_iter",
        nargs="+",
        type=int,
        default=[2_064_384, 2_097_152, 4_194_304],
    )
    parser.add_argument("--checkpoints_dir", default="checkpoints")

    # W&B experiment tracking
    parser.add_argument("--wandb_entity", default="epflmlo-epfl")
    parser.add_argument("--wandb-project", default="swissai-eval-main-v1.1")
    parser.add_argument("--wandb_api_key")

    # evaluation settings
    parser.add_argument("--bs", default="auto")
    parser.add_argument("--tasks", default="swissai_eval")
    parser.add_argument(
        "--evaluate_every",
        type=int,
        default=100_000,
        help="Evaluate every N iterations",
    )

    # other settings
    parser.add_argument("--num_hf_checkpoints_to_keep", type=int, default=5)
    return parser.parse_args()


class EvalMetadata:
    def __init__(self, path=EVAL_METADATA_PATH):
        self.path = path
        with open(self.path) as f:
            self.metadata = json.load(f)

    def get_model_metadata(self, model_name):
        return self.metadata[model_name]

    def get_iteration_metadata(self, model_name, iteration):
        iterations_metadata = self.metadata[model_name]["iterations"]
        iteration = str(iteration)  # JSON keys are strings
        if iteration in iterations_metadata:
            return iterations_metadata[iteration]
        return {
            "state": State.NOT_EVALUATED.name,
            "timestamp": time.time(),
        }

    def update_iteration_metadata(self, model_name, iteration, new_state, **kwargs):
        iteration = str(iteration)  # JSON keys are strings
        new_state_name = new_state.name if type(new_state) is State else new_state
        if iteration not in self.metadata[model_name]["iterations"]:
            self.metadata[model_name]["iterations"][iteration] = {}
        self.metadata[model_name]["iterations"][iteration].update(kwargs)
        self.metadata[model_name]["iterations"][iteration]["state"] = new_state_name
        self.metadata[model_name]["iterations"][iteration]["timestamp"] = time.time()
        with open(self.path, "w") as f:
            json.dump(self.metadata, f, indent=4)


def submit_new_evaluations(args):
    assert os.environ["WANDB_API_KEY"] or args.wandb_api_key, "No W&B API key provided"
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    eval_metadata = EvalMetadata()

    for model_name, model_dir, model_tokens in zip(
        args.model_names, args.model_dirs, args.model_tokens_per_iter
    ):
        print(f"Checking {model_name} checkpoints")

        # prepare submit script arguments
        match = re.match(r"apertus3-(\d)b-.*", model_dir)
        assert match, "Unknown naming pattern for model dir: {model_dir}"
        size = int(match.group(1))
        assert size in [1, 3, 8], "Unknown model size"

        # find checkpoints from iterations at the given intervals
        checkpoints_dir = os.path.join(
            args.checkpoints_root, model_dir, args.checkpoints_dir
        )
        iters_to_evaluate = []
        for d in os.listdir(checkpoints_dir):
            if d.startswith("iter_"):
                match = re.match(r"iter_(\d+)$", d)
                if not match:
                    print(f"Warning: Unknown checkpoint dir naming: {d}")
                    continue
                iteration = int(match.group(1))
                if iteration % args.evaluate_every != 0:
                    continue
                iters_to_evaluate.append((iteration, os.path.join(checkpoints_dir, d)))

        # submit evaluations for checkpoints that have not been evaluated yet
        for iteration, iteration_dir in iters_to_evaluate:
            if (
                eval_metadata.get_iteration_metadata(model_name, iteration)["state"]
                == State.NOT_EVALUATED.name
            ):
                print(f"Submitting evaluation for iteration {iteration}")
                jobname = f"{model_name}_iter_{iteration}"
                arguments = f"""
--container-path {args.container_path}
--logs-root {args.logs_root}
--convert-to-hf
--hf-temp-dir {args.hf_temp_dir}
--size {size}
--wandb-entity {args.wandb_entity}
--wandb-project {args.wandb_project}
--wandb-id {model_name}
--name {jobname}
--bs {args.bs}
--tokens-per-iter {model_tokens}
--tasks {args.tasks}
""".strip().replace("\n", " ")
                command = f"bash {cur_dir}/submit_evaluation.sh {checkpoints_dir} {arguments} --iterations {iteration}"
                print(f"Running command: {command}")
                res = os.system(command)
                if res:
                    raise RuntimeError(
                        f"Submitting evaluation for iteration {iteration} returned with return code: {res}"
                    )
                eval_metadata.update_iteration_metadata(
                    model_name,
                    iteration,
                    State.SUBMITTED,
                    checkpoint_dir=iteration_dir,
                    eval_out=os.path.join(args.logs_root, "slurm", f"{jobname}.out"),
                    eval_err=os.path.join(args.logs_root, "slurm", f"{jobname}.err"),
                )


def check_running(args):
    """Checks if evaluations in the *submitted* state are now running."""
    eval_metadata = EvalMetadata()
    for model_name in args.model_names:
        model_metadata = eval_metadata.get_model_metadata(model_name)
        for iteration, iteration_metadata in model_metadata["iterations"].items():
            if iteration_metadata["state"] == State.SUBMITTED.name:
                if os.path.exists(iteration_metadata["eval_out"]):
                    eval_metadata.update_iteration_metadata(
                        model_name, iteration, State.RUNNING
                    )


def check_finished(args):
    """Checks if evaluations in the *running* state are finished or have failed."""
    eval_metadata = EvalMetadata()
    for model_name in args.model_names:
        model_metadata = eval_metadata.get_model_metadata(model_name)
        for iteration, iteration_metadata in model_metadata["iterations"].items():
            if iteration_metadata["state"] == State.RUNNING.name:
                with open(iteration_metadata["eval_out"]) as f:
                    lines = list(map(str.strip, f.readlines()))
                if "Evaluation finished." in lines[-1]:
                    eval_metadata.update_iteration_metadata(
                        model_name, iteration, State.FINISHED
                    )
                elif "Evaluation failed." in lines[-1]:
                    eval_metadata.update_iteration_metadata(
                        model_name, iteration, State.FAILED
                    )
                    print(
                        f"Evaluation of model {model_name} at iteration {iteration} failed."
                    )
                    print("1. Check the error logs:")
                    print(f"less {iteration_metadata['eval_err']}")
                    print("2. Reset the state of the evaluation by running:")
                    cur_dir = os.path.dirname(os.path.realpath(__file__))
                    script_path = os.path.join(cur_dir, "change_eval_state.py")
                    print(
                        f"python {script_path} --model_name {model_name} --iteration {iteration} --state not_evaluated"
                    )


def cleanup_hf_checkpoints(args):
    """Moves converted checkpoints from temp to storage dir, only keeps the latest N checkpoints."""
    eval_metadata = EvalMetadata()

    # move checkpoints from temp to storage dir (need to wait for evaluation to finish)
    for checkpoint_dir in os.listdir(args.hf_temp_dir):
        match = re.match(r"(Apertus3-\d\.?\d?B)_iter_(\d+)$", checkpoint_dir)
        if not match:
            continue
        model_name = match.group(1)
        iteration = int(match.group(2))
        iteration_metadata = eval_metadata.get_iteration_metadata(model_name, iteration)
        if iteration_metadata["state"] == State.FINISHED.name:
            os.rename(
                os.path.join(args.hf_temp_dir, checkpoint_dir),
                os.path.join(args.hf_storage_dir, checkpoint_dir),
            )

    # for each model size, find all checkpoint dirs with their iterations
    iters_with_dirs = defaultdict(list)
    for checkpoint_dir in os.listdir(args.hf_storage_dir):
        match = re.match(r"Apertus3-(\d\.?\d?)B_iter_(\d+)$", checkpoint_dir)
        if match:
            iters_with_dirs[match.group(1)].append(
                (int(match.group(2)), os.path.join(args.hf_storage_dir, checkpoint_dir))
            )

    # sort by iteration and keep the N latest checkpoints
    for model_size, dirs in iters_with_dirs.items():
        for iteration, d in sorted(dirs)[: -args.num_hf_checkpoints_to_keep]:
            shutil.rmtree(d)


def main(args):
    # submit evaluations for checkpoints not evaluated yet
    submit_new_evaluations(args)

    # check if evaluations that have been submitted are now running
    check_running(args)

    # check if evaluations that have been running are now finished
    check_finished(args)

    # cleanup tasks
    # remove HF checkpoints per model size exceeding `num_hf_checkpoints_to_keep`
    cleanup_hf_checkpoints(args)

    # TODO check if finished evaluations have been synced with W&B


if __name__ == "__main__":
    main(parse_args())
