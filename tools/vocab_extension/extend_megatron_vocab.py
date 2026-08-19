#!/usr/bin/env python3
# Copyright (c) 2026, Swiss AI. All rights reserved.

"""Extend vocab tables in a Megatron torch checkpoint.

This is a thin wrapper around ``tools/checkpoint/convert.py`` using the core
loader and saver. The loader reconstructs full vocab tables from existing TP
shards, the saver appends deterministic rows before Megatron padding and TP/PP
sharding, then writes a new Megatron torch checkpoint.
"""

import argparse
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-dir", required=True, help="Source Megatron torch checkpoint directory.")
    parser.add_argument("--save-dir", required=True, help="Output Megatron torch checkpoint directory.")
    parser.add_argument("--model-type", default="GPT", choices=["GPT", "BERT"])
    parser.add_argument(
        "--source-vocab-size",
        type=int,
        required=True,
        help="Original non-padded vocab size to preserve bitwise.",
    )
    parser.add_argument(
        "--target-vocab-size",
        type=int,
        default=None,
        help="Target real vocab size after adding tokenizer-addressable tokens.",
    )
    parser.add_argument(
        "--target-tokenizer-dir",
        default=None,
        help="Optional tokenizer directory. If --target-vocab-size is omitted, len(tokenizer) is used.",
    )
    parser.add_argument("--target-tensor-parallel-size", type=int, default=None)
    parser.add_argument("--target-pipeline-parallel-size", type=int, default=None)
    parser.add_argument("--target-expert-parallel-size", type=int, default=1)
    parser.add_argument(
        "--target-num-layers-per-virtual-pipeline-stage",
        type=int,
        default=None,
        help="If set, dispatch to the VPP-aware saver so the output checkpoint "
             "matches a training run with --num-layers-per-virtual-pipeline-stage. "
             "Source must be non-VPP. Topology is validated by the saver against "
             "num_layers from the source checkpoint metadata.",
    )
    parser.add_argument("--make-vocab-size-divisible-by", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--init-method",
        default="mean-std",
        choices=["mean-std", "megatron-original"],
        help="How to initialize appended vocab rows.",
    )
    parser.add_argument("--megatron-path", default=None)
    parser.add_argument(
        "--loader-transformer-impl",
        default="transformer_engine",
        choices=["local", "transformer_engine"],
    )
    parser.add_argument(
        "--saver-transformer-impl",
        default="transformer_engine",
        choices=["local", "transformer_engine"],
    )
    parser.add_argument(
        "--loader-saver-mode",
        default="process",
        choices=["serial", "process"],
        help="Passed through to convert.py. Defaults to process because core loader and saver both initialize Megatron globals.",
    )
    parser.add_argument("--max-queue-size", type=int, default=2)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the underlying convert.py command and exit.",
    )
    return parser.parse_args()


def tokenizer_vocab_size(tokenizer_dir: str) -> int:
    try:
        import transformers
    except ImportError as exc:
        raise SystemExit(
            "--target-tokenizer-dir requires transformers to be installed"
        ) from exc

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_dir,
        trust_remote_code=True,
    )
    return len(tokenizer)


def main():
    args = parse_args()

    tokenizer_size = None
    if args.target_tokenizer_dir is not None:
        tokenizer_size = tokenizer_vocab_size(args.target_tokenizer_dir)

    target_vocab_size = args.target_vocab_size
    if target_vocab_size is None:
        if tokenizer_size is None:
            raise SystemExit("Provide --target-vocab-size or --target-tokenizer-dir.")
        target_vocab_size = tokenizer_size

    if tokenizer_size is not None and target_vocab_size < tokenizer_size:
        raise SystemExit(
            f"--target-vocab-size ({target_vocab_size}) cannot be smaller than "
            f"len(target tokenizer) ({tokenizer_size})."
        )
    if target_vocab_size < args.source_vocab_size:
        raise SystemExit(
            f"--target-vocab-size ({target_vocab_size}) cannot be smaller than "
            f"--source-vocab-size ({args.source_vocab_size})."
        )
    if args.make_vocab_size_divisible_by <= 0:
        raise SystemExit("--make-vocab-size-divisible-by must be positive.")
    if args.target_tensor_parallel_size is not None and args.target_tensor_parallel_size <= 0:
        raise SystemExit("--target-tensor-parallel-size must be positive when provided.")
    if (
        args.target_pipeline_parallel_size is not None
        and args.target_pipeline_parallel_size <= 0
    ):
        raise SystemExit("--target-pipeline-parallel-size must be positive when provided.")

    # This script lives in tools/vocab_extension/; vocab_extension.py /
    # topology.py / convert.py / saver_*.py live in tools/checkpoint/.
    checkpoint_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, "checkpoint")
    )
    sys.path.insert(0, checkpoint_dir)
    from vocab_extension import vpp_saver_dispatch

    saver_name, vpp_argv = vpp_saver_dispatch(
        args.target_num_layers_per_virtual_pipeline_stage
    )

    padded_vocab_size = None
    if args.target_tensor_parallel_size is not None:
        multiple = args.make_vocab_size_divisible_by * args.target_tensor_parallel_size
        padded_vocab_size = ((target_vocab_size + multiple - 1) // multiple) * multiple
        if args.target_tensor_parallel_size > 1:
            os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")

    convert_argv = [
        "convert.py",
        "--model-type", args.model_type,
        "--loader", "core",
        "--saver", saver_name,
        "--load-dir", args.load_dir,
        "--save-dir", args.save_dir,
        "--true-vocab-size", str(args.source_vocab_size),
        "--loader-transformer-impl", args.loader_transformer_impl,
        "--saver-transformer-impl", args.saver_transformer_impl,
        "--target-expert-parallel-size", str(args.target_expert_parallel_size),
        "--vocab-extension-source-size", str(args.source_vocab_size),
        "--vocab-extension-target-size", str(target_vocab_size),
        "--vocab-extension-seed", str(args.seed),
        "--vocab-extension-init-method", args.init_method,
        "--vocab-extension-make-vocab-size-divisible-by",
        str(args.make_vocab_size_divisible_by),
        "--loader-saver-mode", args.loader_saver_mode,
        "--max-queue-size", str(args.max_queue_size),
    ]
    if args.target_tensor_parallel_size is not None:
        convert_argv.extend([
            "--target-tensor-parallel-size",
            str(args.target_tensor_parallel_size),
        ])
    if args.target_pipeline_parallel_size is not None:
        convert_argv.extend([
            "--target-pipeline-parallel-size",
            str(args.target_pipeline_parallel_size),
        ])
    convert_argv.extend(vpp_argv)
    if args.megatron_path is not None:
        convert_argv.extend(["--megatron-path", args.megatron_path])

    print("Running vocab extension via:")
    print(" ".join(convert_argv))
    print(f"source_vocab_size={args.source_vocab_size}")
    if tokenizer_size is not None:
        print(f"target_tokenizer_size={tokenizer_size}")
    print(f"target_vocab_size={target_vocab_size}")
    print(f"make_vocab_size_divisible_by={args.make_vocab_size_divisible_by}")
    if args.target_num_layers_per_virtual_pipeline_stage is not None:
        print(
            "target_num_layers_per_virtual_pipeline_stage="
            f"{args.target_num_layers_per_virtual_pipeline_stage}"
        )
    print(f"init_method={args.init_method}")
    if padded_vocab_size is not None:
        print(f"padded_vocab_size={padded_vocab_size}")
        print(f"padding_rows={padded_vocab_size - target_vocab_size}")

    if args.dry_run:
        return

    sys.argv = convert_argv
    from convert import main as convert_main

    convert_main()


if __name__ == "__main__":
    main()
