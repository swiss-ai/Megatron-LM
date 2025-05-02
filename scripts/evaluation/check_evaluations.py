"""Regularly checks Imanol's checkpoints to submit evaluation runs."""

import os
import re
import shutil
from collections import defaultdict

from evaluations_metadata import EvalMetadata, State


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
        "--checkpoints_roots",
        nargs="+",
        default=[
            "/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/Meg-Runs/main-runs-v1",
            "/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/Meg-Runs/main-runs-v1",
            "/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/Meg-Runs/main-runs-v1",
            "/iopsstor/scratch/cscs/schlag/main_run_70B_megatron/Megatron-LM/logs/Meg-Runs/main-runs-v1",
        ],
    )
    parser.add_argument(
        "--model_dirs",
        nargs="+",
        default=[
            "apertus3-1b-21-nodes",
            "apertus3-3b-64-nodes",
            "apertus3-8b-128-nodes",
            "apertus3-70b-512-nodes-1e-5lr",
        ],
    )
    parser.add_argument(
        "--model_names",
        nargs="+",
        default=["Apertus3-1.5B", "Apertus3-3B", "Apertus3-8B", "Apertus3-70B"],
    )
    parser.add_argument(
        "--model_tokens_per_iter",
        nargs="+",
        type=int,
        default=[2_064_384, 2_097_152, 4_194_304, 8_388_608],
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
        nargs="+",
        type=int,
        default=[100_000, 100_000, 100_000, 30_000],
        help="Evaluate every N iterations",
    )
    parser.add_argument(
        "--limits",
        nargs="+",
        type=int,
        default=[0, 0, 0, 1000],
        help="Limits samples per lm-eval-harness task (0 = no limit)",
    )

    # other settings
    parser.add_argument("--num_hf_checkpoints_to_keep", type=int, default=5)
    return parser.parse_args()


def submit_new_evaluations(args):
    assert os.environ["WANDB_API_KEY"] or args.wandb_api_key, "No W&B API key provided"
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    eval_metadata = EvalMetadata()

    for checkpoints_root, model_name, model_dir, model_tokens, evaluate_every, limit in zip(
        args.checkpoints_roots,
        args.model_names,
        args.model_dirs,
        args.model_tokens_per_iter,
        args.evaluate_every,
        args.limits,
    ):
        # prepare submit script arguments
        match = re.match(r"apertus3-(\d\d?)b-.*", model_dir)
        assert match, f"Unknown naming pattern for model dir: {model_dir}"
        size = int(match.group(1))
        assert size in [1, 3, 8, 70], "Unknown model size"

        # find checkpoints from iterations at the given intervals
        checkpoints_dir = os.path.join(
            checkpoints_root, model_dir, args.checkpoints_dir
        )
        iters_to_evaluate = []
        for d in os.listdir(checkpoints_dir):
            if d.startswith("iter_"):
                match = re.match(r"iter_(\d+)$", d)
                if not match:
                    print(f"Warning: Unknown checkpoint dir naming: {d}")
                    continue
                iteration = int(match.group(1))

                # only evaluate every N iterations
                if iteration % evaluate_every != 0:
                    continue

                # only evaluate iterations not evaluated yet
                eval_state = eval_metadata.get_state(model_name, iteration)
                if eval_state is None or eval_state == State.NOT_EVALUATED:
                    iters_to_evaluate.append(
                        (iteration, os.path.join(checkpoints_dir, d))
                    )

        # skip if no iterations need evaluation
        if not iters_to_evaluate:
            continue

        # submit evaluations for checkpoints that have not been evaluated yet
        all_iterations = ",".join([str(it) for it, _ in sorted(iters_to_evaluate)])
        start_to_end = str(sorted(iters_to_evaluate)[0][0])
        if len(iters_to_evaluate) > 1:
            start_to_end += "-" + str(sorted(iters_to_evaluate)[-1][0])
        jobname = f"{model_name}_iter_{start_to_end}"
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
--limit {limit if limit > 0 else "null"}
""".strip().replace("\n", " ")
        command = f"bash {cur_dir}/submit_evaluation.sh {checkpoints_dir} {arguments} --iterations {all_iterations}"
        res = os.system(command)
        if res:
            raise RuntimeError(
                f"Submitting evaluation for iteration(s) {start_to_end} returned with return code: {res}.\nCommand: {command}"
            )

        # update eval metadata, setting each evaluated iteration to submitted
        for iteration, iteration_dir in iters_to_evaluate:
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
            if eval_metadata.get_state(model_name, iteration) == State.SUBMITTED:
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
        if eval_metadata.get_state(model_name, iteration) == State.FINISHED:
            shutil.move(
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
