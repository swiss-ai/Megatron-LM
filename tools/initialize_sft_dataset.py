#!/usr/bin/env python
"""Initialize SFT dataset with packing to determine sample counts.

This script initializes the SFT dataset in the same way as pretrain_gpt.py
but exits immediately after dataset initialization. This is useful for the
first phase of packed SFT training where you need to determine the actual
number of packed samples without loading the full model.

IMPORTANT REQUIREMENTS:
    These arguments MUST match your intended training run (they affect the
    cache hash that allows the training script to reuse the precomputed index):
        --seed, --split, --seq-length, --data-path, --tokenizer-*,
        --ap-sft-packing-strategy, --ap-sft-load-loss-mask,
        --max-docs-per-bin-sft

    The requested TRAIN SAMPLE COUNT is also part of the cache hash. It is
    --train-samples if given, otherwise --train-iters * --global-batch-size.
    Because the whole point of this script is to discover the packed sample
    count you will pass as --train-samples, cache reuse needs two init runs:
        1. Run once (any --train-iters) to learn the packed samples per epoch N.
        2. Re-run with --train-samples <N> (or your chosen multiple) so the
           index is cached under the same hash your training run will compute.
    A training run whose sample count differs from the init run still works,
    it just silently rebuilds the packing index at launch.

    These arguments do NOT affect the packing index or its cache hash:
        --tensor-model-parallel-size, --pipeline-model-parallel-size,
        --expert-model-parallel-size, --num-layers, --hidden-size,
        --num-attention-heads, --max-position-embeddings

    Model architecture args (num-layers, hidden-size, num-attention-heads,
    max-position-embeddings) are required by Megatron's argument validation
    but have no effect on dataset construction. Use any valid dummy values.

Usage:
    python tools/initialize_sft_dataset.py <same data/training arguments as pretrain_gpt.py>

    Must include: --ap-sft --ap-sft-pack-samples

    Data paths may carry the explicit dataset-type markers understood by the
    training run (e.g. "sft:/data/dolly" / "pretrain:/data/fineweb") for mixed
    blends. --ap-sft is required either way: it enables run-level SFT mode,
    while markers only control per-entry dataset dispatch.

Example (single-GPU initialization for any training topology):
    torchrun --nproc_per_node=1 tools/initialize_sft_dataset.py \\
        --tensor-model-parallel-size 1 \\
        --pipeline-model-parallel-size 1 \\
        --num-layers 1 --hidden-size 128 --num-attention-heads 1 \\
        --max-position-embeddings 8192 \\
        --seq-length 8192 \\
        --data-path /path/to/data \\
        --tokenizer-type HuggingFaceTokenizer \\
        --tokenizer-model /path/to/tokenizer \\
        --ap-sft --ap-sft-pack-samples \\
        --train-iters 1000 \\
        --global-batch-size 8 \\
        --micro-batch-size 1
"""

import sys
from pathlib import Path

MEGATRON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MEGATRON_DIR))

from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.training.initialize import initialize_megatron
from megatron.training.training import get_train_valid_test_num_samples, update_train_iters

# Use the exact same config construction and rank predicate as training so the
# dataset cache hash is identical for the same data/training arguments.
from pretrain_gpt import core_gpt_dataset_config_from_args, is_dataset_built_on_rank


def build_train_valid_test_datasets(train_val_test_num_samples):
    """Build the train, test, and validation datasets.

    Args:
        train_val_test_num_samples: A list containing the number of samples in train, test, and validation.
    """
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        train_val_test_num_samples,
        is_dataset_built_on_rank,
        config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")


def get_train_val_test_num_samples():
    """Train/valid/test sample counts aligned with megatron.training.training.

    Must produce the same train_samples as the training script for the cache
    hash to match, so this calls training's own get_train_valid_test_num_samples
    after replicating the two preconditions training establishes before calling
    it: train_iters derived from train_samples, and args.iteration set from the
    checkpoint (0 for a fresh run — a run resumed past a
    --phase-transition-iterations boundary computes a smaller train sample
    count and will rebuild the packing index at launch).
    """
    args = get_args()

    if args.train_samples:
        update_train_iters(args)
    if not hasattr(args, "iteration"):
        args.iteration = 0

    return get_train_valid_test_num_samples()


def main():
    """Main function to initialize dataset and exit."""

    # Initialize Megatron (this handles argument parsing and distributed setup)
    initialize_megatron(
        extra_args_provider=None,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
    )

    args = get_args()

    # Validate required arguments
    if not args.ap_sft or not args.ap_sft_pack_samples:
        print_rank_0("=" * 80)
        print_rank_0("ERROR: This script requires both --ap-sft and --ap-sft-pack-samples flags")
        print_rank_0("=" * 80)
        print_rank_0("This script is specifically for determining packed sample counts")
        print_rank_0("before running a full SFT training job with sample packing.")
        print_rank_0("")
        print_rank_0("Usage:")
        print_rank_0("  python tools/initialize_sft_dataset.py <args> --ap-sft --ap-sft-pack-samples")
        print_rank_0("")
        print_rank_0("For normal training without packing, use pretrain_gpt.py")
        print_rank_0("=" * 80)
        sys.exit(1)

    print_rank_0("=" * 80)
    print_rank_0("SFT Dataset Initialization Script")
    print_rank_0("Builds the packing index and reports ONE-EPOCH packing statistics.")
    print_rank_0("=" * 80)
    print_rank_0("")
    print_rank_0("For cache reuse, these args MUST match your training run:")
    print_rank_0("  seed, seq-length, split, data-path, tokenizer, packing settings")
    print_rank_0("  (strategy, load-loss-mask, max-docs-per-bin-sft), and the")
    print_rank_0("  requested train sample count (--train-samples, or")
    print_rank_0("  --train-iters * --global-batch-size).")
    print_rank_0("  For cache reuse, re-run this script with the final --train-samples")
    print_rank_0("  once you know the packed sample count (see module docstring).")
    print_rank_0("")
    print_rank_0("These do NOT affect the cache hash (any value works here):")
    print_rank_0("  TP/PP/EP, num-layers, hidden-size, num-attention-heads")
    print_rank_0("")
    print_rank_0(f"  Seed:              {args.seed}")
    print_rank_0(f"  Seq length:        {args.seq_length}")
    print_rank_0(f"  Global batch size: {args.global_batch_size}")
    train_samples = args.train_samples if args.train_samples else args.train_iters * args.global_batch_size
    print_rank_0(f"  Train samples:     {train_samples}")
    print_rank_0(f"  Packing strategy:  {args.ap_sft_packing_strategy}")
    print_rank_0("=" * 80)
    print_rank_0("")

    # Calculate train/val/test sample counts
    train_val_test_num_samples = get_train_val_test_num_samples()

    # Build datasets (this will trigger the packing process)
    build_train_valid_test_datasets(train_val_test_num_samples)


if __name__ == "__main__":
    main()