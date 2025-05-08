import math
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import pandas as pd
import wandb
import numpy as np
from tqdm import tqdm

MULTILINGUAL_TASKS = [
    "include_base_44",
    "global_mmlu",
    "xcopa",
    "xnli",
    "xwinograd",
    "pawsx",
    "m_hellaswag",
    "m_arc",
]
ENGLISH_TASKS = [
    "hellaswag",
    "arc_easy",
    "arc_challenge",
    "lambada_openai",
    "lambada_standard",
    "winogrande",
    "piqa",
    "openbookqa",
    "commonsense_qa",
    "mmlu",
    "mmlu_continuation",
    "gsm8k",
    "wikitext",
    "lambada",
    "hellaswag",
    "squadv2",
]
ENGLISH_METRICS = [
    "acc",
    "acc_norm",
    "f1",
    "perplexity",
    "acc_stderr",
    "acc_norm_stderr",
    "perplexity_stderr",
]
MULTILINGUAL_METRICS = ["acc", "acc_norm", "acc_stderr", "acc_norm_stderr"]

SWISSAI_EVAL_METRICS = [
    "acc",
    "acc_norm",
    "acc_stderr",
    "acc_norm_stderr",
    "perplexity",
    "perplexity_stderr",
    "f1",
]

SWISSAI_EVAL_TASKS = ["english_macro", "multilingual_macro"]

ALL_GROUPS = SWISSAI_EVAL_TASKS + MULTILINGUAL_TASKS + ENGLISH_TASKS + ["swissai_eval_macro"]


def get_all_runs(entity: str, project: str):
    """Retrieve all runs from the WandB project."""
    api = wandb.Api()
    return list(api.runs(f"{entity}/{project}"))


def mean(arr):
    """Calculate the mean of an array."""
    return sum(arr) / len(arr)


# NOTE: this is taken from lm_eval, which uses the actual sizes of the subtasks
# but we don't have that here, so we just use 2 for all subtasks (needs to be larger than 1)
def pooled_sample_stderr(stderrs, sizes=None):
    """
    Aggregate bootstrapped stderrs across subtasks in a group.
    """
    if sizes is None:
        # this is a hack, we should use the actual sizes of the subtasks, and only use 2 because for 1 there is a divison by zero
        sizes = [2] * len(stderrs)

    assert len(stderrs) == len(sizes)

    pooled_sample_var = (
        sum([(size - 1) * stderr**2 * size for size, stderr in zip(sizes, stderrs)])
    ) / (sum(sizes) - len(sizes))

    return np.sqrt(pooled_sample_var / sum(sizes)).item()


def extract_metrics_by_prefix(metrics, prefix, extract_type=None, ignore_names=None):
    """
    Extract metrics that match a given prefix.

    Args:
        metrics: Dictionary of metrics
        prefix: Prefix string to match
        extract_type: Whether to extract specific metric types
        ignore_names: List of names to ignore

    Returns:
        Dictionary of metric types with lists of values
    """
    extracted = {}

    for metric_name, value in metrics.items():
        if metric_name.startswith(prefix):
            if ignore_names:
                if any(name in metric_name for name in ignore_names):
                    continue
            if extract_type:
                metric_type = metric_name.split("/")[-1]  # e.g., "acc", "acc_norm"
                if metric_type not in extracted:
                    extracted[metric_type] = []
                extracted[metric_type].append(value)
            else:
                extracted[metric_name] = value

    return extracted


def calculate_aggregates(values_dict, group_name):
    """
    Calculate the mean and stderr for a given prefix.

    Args:
        values_dict: Dictionary of lists of values
        group_name: Name of the group to calculate aggregates for

    Returns:
        Dictionary of metric types with lists of values
    """
    group_metrics = {}
    for metric_type, values in values_dict.items():
        if values:
            if metric_type.endswith("stderr"):
                group_metrics[f"{group_name}/{metric_type}"] = pooled_sample_stderr(values)
            else:
                group_metrics[f"{group_name}/{metric_type}"] = mean(values)

    return group_metrics


def calculate_aggregates_for_group(metrics, group_name, group_metric_names, group_tasks):
    """
    Calculate the mean and stderr for a given group which is defined by a list of tasks and a list of metrics.

    """
    group_metrics = {metric: [] for metric in group_metric_names}
    for metric_name, value in metrics.items():
        if "/" in metric_name:
            task, metric_type = metric_name.split("/")
            if task in group_tasks:
                if metric_type in group_metrics:
                    group_metrics[metric_type].append(value)

    group_agg = calculate_aggregates(group_metrics, group_name)
    return group_agg


def check_if_metrics_available(step_metrics, list_of_tasks):
    for task in list_of_tasks:
        if not any(k.startswith(task) for k in step_metrics.keys()):
            # not all tasks are in step_metrics, skip this run step
            return False
        for metric in MULTILINGUAL_METRICS + ENGLISH_METRICS:
            # check if there is a None value for any task+metric combination
            metric_key = f"{task}/{metric}"
            if metric_key in step_metrics:
                if step_metrics[metric_key] is None:
                    return False
    return True



def process_metrics_for_step(step_metrics):
    new_metrics = {}

    # first runs did not include these metrics, compute again
    if ("m_hellaswag/acc" not in step_metrics or "m_arc/acc" not in step_metrics) or step_metrics[
        "m_arc/acc"
    ] is None:
        # the 'de' tasks are just exemplary, but we need to check if hellaswa_{lang} and arc_{lang} are available
        if not check_if_metrics_available(step_metrics, ["hellaswag_de", "arc_de"]):
            return None
        
        # Extract benchmark-specific metrics
        hellaswag_metrics = extract_metrics_by_prefix(step_metrics, "hellaswag_", extract_type=True)
        arc_metrics = extract_metrics_by_prefix(
            step_metrics, "arc_", extract_type=True, ignore_names=["arc_easy", "arc_challenge"]
        )

        # Calculate benchmark aggregates
        new_metrics.update(calculate_aggregates(hellaswag_metrics, "m_hellaswag"))
        new_metrics.update(calculate_aggregates(arc_metrics, "m_arc"))
        step_metrics.update(new_metrics)

    if not check_if_metrics_available(step_metrics, MULTILINGUAL_TASKS + ENGLISH_TASKS):
        return None
    # Calculate multilingual aggregates
    multilingual_agg = calculate_aggregates_for_group(
        step_metrics, "multilingual_macro", MULTILINGUAL_METRICS, MULTILINGUAL_TASKS
    )
    new_metrics.update(multilingual_agg)

    # Calculate english aggregates
    english_agg = calculate_aggregates_for_group(
        step_metrics, "english_macro", ENGLISH_METRICS, ENGLISH_TASKS
    )
    new_metrics.update(english_agg)

    # Calculate swissai_eval aggregates with the new metrics
    swissai_agg = calculate_aggregates_for_group(
        new_metrics, "swissai_eval_macro", SWISSAI_EVAL_METRICS, SWISSAI_EVAL_TASKS
    )
    new_metrics.update(swissai_agg)
    new_metrics["ConsumedTokens"] = step_metrics["ConsumedTokens"]
    new_metrics["OptimizerStep"] = step_metrics["OptimizerStep"]

    step_metrics.update(new_metrics)
    return step_metrics


def update_aggregate_metrics(entity: str, project: str, run: wandb.run, iterations: list[int] = []):
    print(f"Processing run: {run.name} to update aggregates")

    history = run.scan_history()
    history_list = list(history)
    if not history_list:
        print(f"No history found for run {run.name}")
        return

    with wandb.init(
        project=project, entity=entity, id=run.id, config=run.config, resume="must"
    ) as update_run:
        # Process each step in the history
        for step in tqdm(history_list):
            if "ConsumedTokens" not in step or "OptimizerStep" not in step:
                continue
            if iterations and step["OptimizerStep"] not in iterations:
                continue
            updated_metrics = process_metrics_for_step(step)
            if updated_metrics:
                wandb.log(updated_metrics)

    print(f"Finished processing run: {run.name}")


def log_eval_table(entity: str, project: str, run: wandb.run):
    # Download history.
    ignore_cols = ["evaluation/group_eval_results", "evaluation/eval_results", "eval_table"]
    history = defaultdict(dict)

    for row in run.scan_history():
        if "ConsumedTokens" not in row:
            continue
        optstep = row["ConsumedTokens"]
        for name, value in row.items():
            if (
                name == "ConsumedTokens"
                or name == "OptimizerStep"
                or (
                    not name.startswith("_")
                    and name.split("/")[0] in ALL_GROUPS
                    and name not in ignore_cols
                    and value is not None
                    and not math.isnan(value)
                )
            ):
                history[optstep][name] = value
    history = pd.DataFrame(list(history.values()))
    last_step = np.max(history["ConsumedTokens"])
    last_row = history[history["ConsumedTokens"] == last_step]
    assert last_row.shape[0] == 1
    last_row = last_row.iloc[0, :]
    last_row = last_row.dropna()

    # Get task -> list of metrics dict.
    names = list(filter(lambda col: col != "ConsumedTokens", last_row.index))
    last_row = last_row[names]
    tasks = defaultdict(list)
    tasks_maybe_stranded = defaultdict(lambda: defaultdict(list))
    for col in names:
        if col == "OptimizerStep":
            continue
        task, metric = col.split("/")
        belongs_here = "stderr" not in metric
        if belongs_here:
            tasks[task].append(metric)

    # Build metrics row.
    data = {"ConsumedTokens": last_step, "OptimizerStep": last_row["OptimizerStep"]}
    for task, metrics in tasks.items():
        for metric in metrics:
            data[f"{task}/{metric}"] = last_row[f"{task}/{metric}"]

    # Now connect to the run in question and attempt to find the table.
    with wandb.init(entity=entity, project=project, id=run.id) as update_run:
        df = pd.DataFrame([data])
        table = wandb.Table(dataframe=df)
        wandb.log({"eval_table": table, "ConsumedTokens": last_step})
        wandb.finish()


def main(
    entity: str,
    project: str,
    runid: str = None,
    update_aggregates: bool = True,
    create_table: bool = False,
    iterations: list[int] = [],
):
    if runid is None:
        runs = get_all_runs(entity, project)
    else:
        api = wandb.Api()
        runs = [api.run(f"{entity}/{project}/{runid}")]
    for run in runs:
        if update_aggregates:
            update_aggregate_metrics(entity, project, run, iterations)
        if create_table:
            log_eval_table(entity, project, run)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--runid", required=True)
    parser.add_argument("--iterations", type=int, nargs="+", default=[])
    parser.add_argument("--update-aggregates", action="store_true")
    parser.add_argument("--create-table", action="store_true")
    args = parser.parse_args()
    main(**vars(args))
