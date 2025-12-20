"""
Evaluate Megatron models using lm-evaluation-harness with EvalHarnessAdaptor

This script uses the EvalHarnessAdaptor from lm_evaluate.py to evaluate
Megatron checkpoints with the lm-eval harness.

Usage:
    torchrun --nproc_per_node=1 ckpt_convert_scripts/eval.py \
        --load /path/to/checkpoint \
        --bf16 \
        --tasks hellaswag,arc_easy \
        --batch-size 1 \
        --output-path eval_results.json
"""
import argparse
import json
import os
import sys
from functools import partial

import numpy as np
import torch

try:
    from lm_eval import evaluator
except ImportError:
    raise ImportError("Please install the lm-eval-harness package")

from megatron.training import initialize_megatron, get_args
from lm_evaluate import EvalHarnessAdaptor
from model_provider import model_provider
from gpt_builders import gpt_builder


def parse_eval_args():
    """Parse eval-specific args and leave the rest for Megatron."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', default='hellaswag,arc_easy,winogrande')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--output-path', default='eval_results.json')

    eval_args, remaining_argv = parser.parse_known_args()

    # Replace sys.argv with remaining args for Megatron
    sys.argv = [sys.argv[0]] + remaining_argv

    return vars(eval_args)


def main():
    # Parse eval args before Megatron initialization
    eval_args = parse_eval_args()

    print("Loading Megatron model for evaluation")

    initialize_megatron(
        args_defaults={
            'use_checkpoint_args': True,      # Load model config & tokenizer from checkpoint
            'no_load_rng': True,              # Skip RNG state (not needed for inference)
            'no_load_optim': True,            # Skip optimizer state (not needed for inference)
            'bf16': True,                      # Required for MoE with grouped GEMM
            "micro_batch_size": eval_args['batch_size'], # I think this can be put to 1 as I don't think this value is used in inference (this is only for training to load the bin tokenized files)
        },
    )

    torch.distributed.barrier()

    # Create EvalHarnessAdaptor which handles Megatron initialization and model loading
    lm_obj = EvalHarnessAdaptor(
        pretrained="gpt2",
        batch_size=eval_args['batch_size'],
        model_provider=partial(model_provider, gpt_builder)
    )

    args = get_args()
    tasks = eval_args['tasks'].split(",")
    model_path = args.load

    # Evaluate tasks
    all_results = {"results": {}, "configs": {}}

    for task in tasks:
        results = evaluator.simple_evaluate(
            model=lm_obj,
            tasks=[task],
            batch_size=eval_args['batch_size'],
        )

        if results:
            all_results["results"].update(results.get("results", {}))
            all_results["configs"].update(results.get("configs", {}))

    if lm_obj.is_main:
        if not all_results["results"]:
            print("Warning: No results returned from evaluation")
            return

        # Save and print results
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        for task in tasks:
            if task in all_results["results"]:
                print(f"\n{task}:")
                for metric, value in all_results["results"][task].items():
                    if not metric.startswith("alias"):
                        print(f"  {metric}: {value}")

        def handle_non_serializable(o):
            if isinstance(o, (np.int64, np.int32)):
                return int(o)
            elif isinstance(o, set):
                return list(o)
            else:
                return str(o)

        if eval_args['output_path']:
            os.makedirs(os.path.dirname(eval_args['output_path']) if os.path.dirname(eval_args['output_path']) else ".", exist_ok=True)

        all_results["model_info"] = {"model_path": model_path}

        with open(eval_args['output_path'], "w") as f:
            json.dump(
                all_results, f, indent=2, default=handle_non_serializable, ensure_ascii=False
            )
        print(f"\nResults saved to: {eval_args['output_path']}")

if __name__ == "__main__":
    main()
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
