#!/usr/bin/env python3
# Copyright (c) 2026, Swiss AI. All rights reserved.

"""Convert an Apertus HuggingFace checkpoint to Megatron with extended vocab.

The Apertus HF loader reads the original HF tensors. The core saver then
extends the full vocab tables deterministically, applies Megatron vocab
padding, shards across TP/PP, and writes a Megatron torch checkpoint.
"""

import argparse
import json
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-dir", required=True, help="Source Apertus HuggingFace checkpoint.")
    parser.add_argument("--save-dir", required=True, help="Output Megatron torch checkpoint directory.")
    parser.add_argument(
        "--tokenizer-model",
        default=None,
        help="HF tokenizer path. Defaults to --load-dir.",
    )
    parser.add_argument(
        "--source-vocab-size",
        type=int,
        default=None,
        help="Original real vocab size to preserve. Defaults to len(--tokenizer-model).",
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
        help="Optional extended tokenizer dir. If target size is omitted, len(tokenizer) is used.",
    )
    parser.add_argument("--target-tensor-parallel-size", type=int, required=True)
    parser.add_argument("--target-pipeline-parallel-size", type=int, required=True)
    parser.add_argument("--target-expert-parallel-size", type=int, default=1)
    parser.add_argument(
        "--target-num-layers-per-virtual-pipeline-stage",
        type=int,
        default=None,
        help="If set, dispatch to the VPP-aware saver. Topology is validated by "
             "the saver against num_layers from the source checkpoint metadata.",
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
        "--saver-transformer-impl",
        default="transformer_engine",
        choices=["local", "transformer_engine"],
    )
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Parameter dtype used for conversion.",
    )
    parser.add_argument(
        "--loader-saver-mode",
        default="process",
        choices=["serial", "process"],
        help="Use process mode for large checkpoints so loader/saver overlap with bounded queue memory.",
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
        raise SystemExit("transformers is required to read tokenizer size") from exc

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_dir,
        trust_remote_code=True,
    )
    return len(tokenizer)


def config_vocab_size(model_dir: str) -> int | None:
    config_path = os.path.join(model_dir, "config.json")
    try:
        with open(config_path) as f:
            config = json.load(f)
    except FileNotFoundError:
        return None
    return config.get("vocab_size")


def main():
    args = parse_args()

    tokenizer_model = args.tokenizer_model or args.load_dir
    source_vocab_size = args.source_vocab_size
    if source_vocab_size is None:
        source_vocab_size = tokenizer_vocab_size(tokenizer_model)

    hf_config_vocab_size = config_vocab_size(args.load_dir)
    if hf_config_vocab_size is not None and hf_config_vocab_size < source_vocab_size:
        raise SystemExit(
            f"config vocab_size ({hf_config_vocab_size}) is smaller than source vocab size "
            f"({source_vocab_size})."
        )

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
    if target_vocab_size < source_vocab_size:
        raise SystemExit(
            f"--target-vocab-size ({target_vocab_size}) cannot be smaller than "
            f"source_vocab_size ({source_vocab_size})."
        )
    if args.make_vocab_size_divisible_by <= 0:
        raise SystemExit("--make-vocab-size-divisible-by must be positive.")
    if args.target_tensor_parallel_size <= 0:
        raise SystemExit("--target-tensor-parallel-size must be positive.")
    if args.target_pipeline_parallel_size <= 0:
        raise SystemExit("--target-pipeline-parallel-size must be positive.")

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

    multiple = args.make_vocab_size_divisible_by * args.target_tensor_parallel_size
    padded_vocab_size = ((target_vocab_size + multiple - 1) // multiple) * multiple
    if args.target_tensor_parallel_size > 1:
        os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")

    convert_argv = [
        "convert.py",
        "--model-type", "GPT",
        "--loader", "apertus",
        "--saver", saver_name,
        "--load-dir", args.load_dir,
        "--save-dir", args.save_dir,
        "--tokenizer-model", tokenizer_model,
        "--true-vocab-size", str(source_vocab_size),
        "--make-vocab-size-divisible-by", str(args.make_vocab_size_divisible_by),
        "--saver-transformer-impl", args.saver_transformer_impl,
        "--target-tensor-parallel-size", str(args.target_tensor_parallel_size),
        "--target-pipeline-parallel-size", str(args.target_pipeline_parallel_size),
        "--target-expert-parallel-size", str(args.target_expert_parallel_size),
        "--vocab-extension-source-size", str(source_vocab_size),
        "--vocab-extension-target-size", str(target_vocab_size),
        "--vocab-extension-seed", str(args.seed),
        "--vocab-extension-init-method", args.init_method,
        "--vocab-extension-make-vocab-size-divisible-by",
        str(args.make_vocab_size_divisible_by),
        "--loader-saver-mode", args.loader_saver_mode,
        "--max-queue-size", str(args.max_queue_size),
    ]
    if args.dtype == "bf16":
        convert_argv.append("--bf16")
    elif args.dtype == "fp16":
        convert_argv.append("--fp16")
    convert_argv.extend(vpp_argv)
    if args.megatron_path is not None:
        convert_argv.extend(["--megatron-path", args.megatron_path])

    print("Running Apertus HF vocab extension via:")
    print(" ".join(convert_argv))
    print(f"source_vocab_size={source_vocab_size}")
    if hf_config_vocab_size is not None:
        print(f"hf_config_vocab_size={hf_config_vocab_size}")
    if tokenizer_size is not None:
        print(f"target_tokenizer_size={tokenizer_size}")
    print(f"target_vocab_size={target_vocab_size}")
    print(f"make_vocab_size_divisible_by={args.make_vocab_size_divisible_by}")
    print(f"target_tensor_parallel_size={args.target_tensor_parallel_size}")
    print(f"target_pipeline_parallel_size={args.target_pipeline_parallel_size}")
    if args.target_num_layers_per_virtual_pipeline_stage is not None:
        print(
            "target_num_layers_per_virtual_pipeline_stage="
            f"{args.target_num_layers_per_virtual_pipeline_stage}"
        )
    print(f"init_method={args.init_method}")
    print(f"padded_vocab_size={padded_vocab_size}")
    print(f"padding_rows={padded_vocab_size - target_vocab_size}")

    if args.dry_run:
        return

    sys.argv = convert_argv
    from convert import main as convert_main

    convert_main()


if __name__ == "__main__":
    main()
