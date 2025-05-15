"""Regularly checks model checkpoints to submit evaluation runs."""

import os
import re
import json
import shutil
from collections import defaultdict

from evaluations_metadata import EvalMetadata, State


def find_evaluations(config):
    """Finds checkpoints that need to be evaluated and sets their state to *waiting*."""
    eval_metadata = EvalMetadata()
    for model_name, model_config in config["models"].items():
        # find checkpoints from iterations at the given intervals
        for model_dir in model_config["model_dirs"]:
            for d in os.listdir(model_dir):
                if d.startswith("iter_"):
                    match = re.match(r"iter_(\d+)$", d)
                    if not match:
                        print(f"Warning: Unknown checkpoint dir naming: {d}")
                        continue
                    iteration = int(match.group(1))

                    # start evaluating at a specific iteration
                    if iteration < model_config["start_eval_at_iter"]:
                        continue

                    # only evaluate every N iterations
                    if iteration % model_config["evaluate_every"] != 0:
                        continue

                    # only evaluate iterations not evaluated yet
                    if eval_metadata.get_state(model_name, iteration) is None:
                        eval_metadata.update_iteration_metadata(
                            model_name,
                            iteration,
                            State.WAITING,
                            checkpoint_dir=os.path.join(model_dir, d),
                        )


def submit_waiting(config):
    """Submits evaluations for checkpoints that are in the *waiting* state, not more than 1 per model."""
    assert os.environ["WANDB_API_KEY"] or config["wandb_api_key"], "No W&B API key provided"
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    eval_metadata = EvalMetadata()

    for model_name, model_config in config["models"].items():
        model_metadata = eval_metadata.get_model_metadata(model_name)

        # find the checkpoint in the *waiting* state at the lowest iteration
        iteration_to_evaluate = None
        for iteration in sorted(model_metadata["iterations"]):
            if eval_metadata.get_state(model_name, iteration) == State.WAITING:
                iteration_to_evaluate = iteration
                break
        if iteration_to_evaluate is None:
            # no checkpoints in *waiting* state
            continue

        # check if there are already submitted/running evaluations for this model
        for iteration in model_metadata["iterations"]:
            if eval_metadata.get_state(model_name, iteration) in [
                State.SUBMITTED,
                State.RUNNING,
            ]:
                # skip this model, as we can only submit one evaluation at a time
                iteration_to_evaluate = None
                break
        if iteration_to_evaluate is None:
            continue

        # submit evaluations for checkpoints that have not been evaluated yet
        jobname = f"{model_name}_iter_{iteration_to_evaluate}"
        checkpoint_dir = model_metadata["iterations"][iteration_to_evaluate]["checkpoint_dir"]
        arguments = f"""
--container-path {config["container_path"]}
--logs-root {config["logs_root"]}
--convert-to-hf
--hf-temp-dir {config["hf_temp_dir"]}
--size {model_config["model_size"]}
--wandb-entity {config["wandb_entity"]}
--wandb-project {config["wandb_project"]}
--wandb-id {model_name}
--name {jobname}
--bs {config["bs"]}
--tokens-per-iter {model_config["tokens_per_iter"]}
--tasks {config["tasks"]}
--limit {model_config["limit"] if model_config["limit"] > 0 else "null"}
""".strip().replace("\n", " ")
        command = f"bash {cur_dir}/submit_evaluation.sh {checkpoint_dir} {arguments} --iterations {iteration_to_evaluate}"
        res = os.system(command)
        if res:
            raise RuntimeError(
                f"Submitting evaluation for iteration {iteration_to_evaluate} returned with return code: {res}.\nCommand: {command}"
            )

        # update eval metadata, setting the iteration to submitted
        eval_metadata.update_iteration_metadata(
            model_name,
            iteration_to_evaluate,
            State.SUBMITTED,
            eval_out=os.path.join(config["logs_root"], "slurm", f"{jobname}.out"),
            eval_err=os.path.join(config["logs_root"], "slurm", f"{jobname}.err"),
        )


def check_running(config):
    """Checks if evaluations in the *submitted* state are now running."""
    eval_metadata = EvalMetadata()
    for model_name in config["models"]:
        model_metadata = eval_metadata.get_model_metadata(model_name)
        for iteration, iteration_metadata in model_metadata["iterations"].items():
            if eval_metadata.get_state(model_name, iteration) == State.SUBMITTED:
                if os.path.exists(iteration_metadata["eval_out"]):
                    eval_metadata.update_iteration_metadata(
                        model_name, iteration, State.RUNNING
                    )


def check_finished(config):
    """Checks if evaluations in the *running* state are finished or have failed."""
    eval_metadata = EvalMetadata()
    for model_name in config["models"]:
        model_metadata = eval_metadata.get_model_metadata(model_name)
        for iteration, iteration_metadata in model_metadata["iterations"].items():
            if eval_metadata.get_state(model_name, iteration) == State.RUNNING:
                with open(iteration_metadata["eval_out"], encoding="latin-1") as f:
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
                        f"python {script_path} --model_name {model_name} --iteration {iteration} --state waiting"
                    )


def cleanup_hf_checkpoints(config):
    """Moves converted checkpoints from temp to storage dir, only keeps the latest N checkpoints."""
    eval_metadata = EvalMetadata()

    # move checkpoints from temp to storage dir (need to wait for evaluation to finish)
    for checkpoint_dir in os.listdir(config["hf_temp_dir"]):
        match = re.match(r"(Apertus3-\d\.?\d?B)_iter_(\d+)$", checkpoint_dir)
        if not match:
            continue
        model_name = match.group(1)
        iteration = int(match.group(2))
        if eval_metadata.get_state(model_name, iteration) == State.FINISHED:
            shutil.move(
                os.path.join(config["hf_temp_dir"], checkpoint_dir),
                os.path.join(config["hf_storage_dir"], checkpoint_dir),
            )

    # for each model size, find all checkpoint dirs with their iterations
    iters_with_dirs = defaultdict(list)
    for checkpoint_dir in os.listdir(config["hf_storage_dir"]):
        match = re.match(r"Apertus3-(\d\.?\d?)B_iter_(\d+)$", checkpoint_dir)
        if match:
            iters_with_dirs[match.group(1)].append(
                (int(match.group(2)), os.path.join(config["hf_storage_dir"], checkpoint_dir))
            )

    # sort by iteration and keep the N latest checkpoints
    for model_size, dirs in iters_with_dirs.items():
        for iteration, d in sorted(dirs)[: -config["num_hf_checkpoints_to_keep"]]:
            try:
                shutil.rmtree(d)
            except PermissionError:
                # someone else saved their checkpoint in the same directory, without write access to others
                # ignore these checkpoints
                pass


def main(config):
    # find checkpoints that need to be evaluated and put them into the waiting state
    find_evaluations(config)

    # submit waiting evaluations, at most one per model
    submit_waiting(config)

    # check if evaluations that have been submitted are now running
    check_running(config)

    # check if evaluations that have been running are now finished
    check_finished(config)

    # cleanup tasks
    # remove HF checkpoints per model size exceeding `num_hf_checkpoints_to_keep`
    cleanup_hf_checkpoints(config)

    # TODO check if finished evaluations have been synced with W&B


if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    config_file = os.path.join(cur_dir, "evaluations_config.json")
    with open(config_file, 'r') as f:
        config = json.load(f)
    main(config)
