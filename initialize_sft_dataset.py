#!/usr/bin/env python
"""Initialize SFT dataset with packing to determine sample counts.

This script initializes the SFT dataset in the same way as pretrain_gpt.py
but exits immediately after dataset initialization. This is useful for the
first phase of packed SFT training where you need to determine the actual
number of packed samples without loading the full model.

IMPORTANT REQUIREMENTS:
    - SEED MUST MATCH your intended training run (determines packing)
    - Parallelism settings (TP/PP/EP) MUST match your actual training run
    - World size can be smaller (minimal DP) but TP/PP/EP must be identical
    - Model architecture params are only needed for validation (not used in packing)

Usage:
    python initialize_sft_dataset.py <same arguments as pretrain_gpt.py>

    Must include: --ap-sft --ap-sft-pack-samples

Example (for a training run with TP=8, PP=4):
    torchrun --nproc_per_node=32 initialize_sft_dataset.py \\
        --tensor-model-parallel-size 8 \\
        --pipeline-model-parallel-size 4 \\
        --num-layers 32 \\
        --hidden-size 4096 \\
        --seq-length 2048 \\
        --data-path /path/to/data \\
        --tokenizer-type GPT2BPETokenizer \\
        --vocab-file /path/to/vocab.json \\
        --merge-file /path/to/merges.txt \\
        --sft \\
        --ap-sft-pack-samples \\
        --train-iters 1000 \\
        --global-batch-size 8

    # Note: nproc_per_node = TP * PP = 32 (world size can be = TP*PP*minimal_DP)
"""

import sys
import json
from typing import List, Optional, Tuple

from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.training import get_tokenizer
from megatron.core import mpu
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.gpt_dataset import MockGPTDataset, GPTDataset
from megatron.core.tokenizers.text.utils.build_tokenizer import build_tokenizer
from megatron.training.initialize import initialize_megatron
from megatron.training.tokenizer.tokenizer_omni_metadata import populate_omni_metadata_from_tokenizer
from megatron.training.tokenizer.tokenizer_sft_metadata import populate_sft_information_from_tokenizer
from megatron.training.utils import get_blend_and_blend_per_split


def is_dataset_built_on_rank():
    """Determine if dataset should be built on this rank."""
    return (
        mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()
    ) and mpu.get_tensor_model_parallel_rank() == 0


def core_gpt_dataset_config_from_args(args):
    """Create GPTDatasetConfig from command line arguments."""
    if args.legacy_tokenizer:
        tokenizer = get_tokenizer()
    else:
        tokenizer = build_tokenizer(args)

    # Keep tokenizer-derived metadata behavior aligned with pretrain_gpt.py.
    populate_omni_metadata_from_tokenizer(args, tokenizer)
    populate_sft_information_from_tokenizer(args, tokenizer)

    # Sometimes --data-path is too long, instead we parse it from a file.
    blend: Optional[Tuple[List[str], Optional[List[float]]]]
    blend_per_split: Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]]
    blend, blend_per_split = get_blend_and_blend_per_split(args)

    sequences_per_dataset = None
    if args.per_dataset_sequences_path is not None:
        with open(args.per_dataset_sequences_path, "r") as f:
            sequences_per_dataset = json.load(f)

    modality_weights = {}
    omnimodal_config = getattr(args, "omnimodal_config", None)
    if omnimodal_config is not None:
        for modality in omnimodal_config.get("modalities", []):
            name = modality.get("name")
            if not name:
                continue
            weight = getattr(args, f"{name}_weight", None)
            if weight is not None:
                modality_weights[name] = weight

    # Backward-compatible fallback for legacy modality names.
    for name in ("vision", "audio"):
        weight = getattr(args, f"{name}_weight", None)
        if weight is not None:
            modality_weights.setdefault(name, weight)

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=blend,
        blend_per_split=blend_per_split,
        split=args.split,
        multiple_validation_sets=args.multiple_validation_sets,
        full_validation=args.full_validation,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
        object_storage_cache_path=args.object_storage_cache_path,
        mid_level_dataset_surplus=args.mid_level_dataset_surplus,
        allow_ambiguous_pad_tokens=args.allow_ambiguous_pad_tokens,
        fast_cache_load=args.dataloader_fast_cache_load,
        defer_npy_index_mmap=args.dataloader_defer_npy_index_mmap,
        sequences_per_dataset=sequences_per_dataset,
        goldfish_loss=args.goldfish_loss,
        goldfish_k=args.goldfish_k,
        goldfish_h=args.goldfish_h,
        modality_weights=modality_weights,
        vision_weight=args.vision_weight,
        audio_weight=args.audio_weight,
        sft_mask_special_tokens=args.ap_sft_mask_special_tokens,
        sft_plw=args.ap_sft_plw,
        sft_pack_samples=args.ap_sft_pack_samples,
        sft_equalize_sample_loss=args.ap_sft_equalize_sample_loss,
        sft_load_loss_mask=args.ap_sft_load_loss_mask,
        sft_truncate_right=args.ap_sft_truncate_right,
    )


def build_train_valid_test_datasets(train_val_test_num_samples):
    """Build the train, test, and validation datasets.

    Args:
        train_val_test_num_samples: A list containing the number of samples in train, test, and validation.

    Returns:
        train_ds, valid_ds, test_ds: The constructed datasets
    """
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)

    if args.ap_sft:
        from megatron.training.datasets.apertus_sft_dataset import ApertusSFTDataset
        dataset_type = ApertusSFTDataset
    elif args.sft:
        from megatron.training.datasets.sft_dataset import SFTDataset
        dataset_type = SFTDataset
    elif args.mock_data:
        dataset_type = MockGPTDataset
    else:
        dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type,
        train_val_test_num_samples,
        is_dataset_built_on_rank,
        config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")


def get_train_val_test_num_samples():
    """Train/valid/test sample counts aligned with megatron.training.training."""
    args = get_args()

    if args.train_samples:
        train_samples = args.train_samples
    else:
        train_samples = args.train_iters * args.global_batch_size

    if args.full_validation:
        eval_samples = None
    else:
        if args.skip_train:
            eval_iters = args.eval_iters
        else:
            assert args.train_iters is not None
            eval_iters = (args.train_iters // args.eval_interval + 1) * args.eval_iters
        eval_samples = eval_iters * args.global_batch_size
    test_samples = args.eval_iters * args.global_batch_size

    return [train_samples, eval_samples, test_samples]


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
        print_rank_0("  python initialize_sft_dataset.py <args> --ap-sft --ap-sft-pack-samples")
        print_rank_0("")
        print_rank_0("For normal training without packing, use pretrain_gpt.py")
        print_rank_0("=" * 80)
        sys.exit(1)

    print_rank_0("=" * 80)
    print_rank_0("SFT Dataset Initialization Script")
    print_rank_0("This script will build the dataset index and report ONE-EPOCH packing statistics")
    print_rank_0("=" * 80)
    print_rank_0("")
    print_rank_0("IMPORTANT: SEED must match your intended training run! It determines packing and thus num of samples.")
    print_rank_0(f"  Seed: {args.seed}")
    print_rank_0("=" * 80)
    print_rank_0("")
    print_rank_0("IMPORTANT: Parallelism settings (TP/PP/EP) should match your training run!")
    print_rank_0(f"  Tensor Parallel: {args.tensor_model_parallel_size}")
    print_rank_0(f"  Pipeline Parallel: {args.pipeline_model_parallel_size}")
    if args.expert_model_parallel_size > 1:
        print_rank_0(f"  Expert Parallel: {args.expert_model_parallel_size}")
    print_rank_0(f"  World Size: {args.world_size}")
    print_rank_0("")
    print_rank_0("Note: Model architecture parameters (--num-layers, --hidden-size, etc.)")
    print_rank_0("      are only needed to pass Megatron's validation. They don't affect")
    print_rank_0("      the dataset packing calculation, which only depends on:")
    print_rank_0("      - Data paths and tokenizer")
    print_rank_0("      - Sequence length (--seq-length)")
    print_rank_0("      - Global batch size (--global-batch-size)")
    print_rank_0("      - Parallelism settings (TP/PP/EP)")
    print_rank_0("=" * 80)
    print_rank_0("")

    # Calculate train/val/test sample counts
    train_val_test_num_samples = get_train_val_test_num_samples()

    # Build datasets (this will trigger the packing process)
    build_train_valid_test_datasets(train_val_test_num_samples)


if __name__ == "__main__":
    main()
