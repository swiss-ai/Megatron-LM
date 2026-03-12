# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain and SFT GPT."""

import json
import math
from functools import partial
from typing import List, Optional, Tuple

import torch

from gpt_builders import gpt_builder
from megatron.core import parallel_state
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, MockGPTDataset
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.tokenizers.text.utils.build_tokenizer import build_tokenizer
from megatron.core.utils import StragglerDetector, get_attr_wrapped_model, unwrap_model
from megatron.training import get_args, get_timers, get_tokenizer, inprocess_restart, pretrain, print_rank_0
from megatron.training.datasets.apertus_sft_dataset import ApertusSFTDataset
from megatron.training.datasets.sft_dataset import SFTDataset
from megatron.training.datasets.fim_dataset import GPTFIMDataset, GPTFIMDatasetConfig
from megatron.training.tokenizer.tokenizer_omni_metadata import populate_omni_metadata_from_tokenizer
from megatron.training.tokenizer.tokenizer_sft_metadata import populate_sft_information_from_tokenizer
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    get_blend_and_blend_per_split,
    is_first_or_last_pipeline_stage,
)
from model_provider import model_provider

try:
    from megatron.post_training.arguments import add_modelopt_args
    from megatron.post_training.loss_func import loss_func as loss_func_modelopt

    has_nvidia_modelopt = True
except ImportError:
    has_nvidia_modelopt = False

stimer = StragglerDetector()


def tokens_to_packed_seq_params(input_ids, eod_token, orig_seq_len, qkv_format='thd', cu_seqlens_padded=None):
    """
    Compute PackedSeqParams from input tokens using EOD token boundaries.

    Args:
        input_ids: Input token IDs, shape assumed flattened (1, tokens) or (tokens)
        eod_token: End-of-Document token ID (from tokenizer.eod)
        orig_seq_len: Original sequence length for fixed boundaries
        qkv_format: QKV format - 'sbhd' (default) or 'thd' (for CP with padding)
        cu_seqlens_padded: Optional padded cumulative lengths for context parallelism

    Returns:
        PackedSeqParams with cu_seqlens respecting both EOD and orig_seq_len boundaries
    """
    from megatron.core.packed_seq_params import PackedSeqParams

    # Create boundaries at fixed intervals (based on orig_seq_len)
    # Find EOD token positions (+1 to mark position AFTER eod)
    # Concatenate and sort to get all boundaries (fixed + EOD)
    cu_seq, _ = torch.sort(torch.cat((
        torch.arange(0, input_ids.size(-1) + orig_seq_len, orig_seq_len, device=input_ids.device, dtype=torch.int32),
        (input_ids.flatten() == eod_token).nonzero()[:, 0].int() + 1,
    )))

    # Compute max sequence length between boundaries
    max_len = (cu_seq[1:] - cu_seq[:-1]).max()

    return PackedSeqParams(
        cu_seqlens_q=cu_seq,
        cu_seqlens_kv=cu_seq,
        max_seqlen_q=max_len,
        max_seqlen_kv=max_len,
        qkv_format=qkv_format,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
    )



def get_batch(data_iterator, vp_stage=None):
    """Generate a batch with optional packed-sequence + context-parallelism support."""

    if not is_first_or_last_pipeline_stage(vp_stage):
        return (None, None, None, None, None, None), None

    args = get_args()
    batch = get_batch_on_this_tp_rank(data_iterator)

    packed_seq_params = None

    if args.use_packed_seq_params:
        tokenizer = get_tokenizer()
        qkv_format = 'thd'
        cp_size = parallel_state.get_context_parallel_world_size()

        tokens_full_flat = batch["tokens"].view(1, -1)
        total_tokens = tokens_full_flat.size(-1)
        device = tokens_full_flat.device

        # Step 1a: compute cu_seqlens from EOD boundaries
        eod_positions = (tokens_full_flat.flatten() == tokenizer.eod).nonzero()[:, 0].int() + 1
        fixed_boundaries = torch.arange(
            0, total_tokens + args.seq_length, args.seq_length,
            device=device, dtype=torch.int32,
        )
        cu_seq, _ = torch.sort(torch.unique(torch.cat((fixed_boundaries, eod_positions))))
        cu_seq = cu_seq[cu_seq <= total_tokens]
        if cu_seq[-1] != total_tokens:
            cu_seq = torch.cat([
                cu_seq,
                torch.tensor([total_tokens], device=device, dtype=cu_seq.dtype),
            ])

        # Step 1b: merge sequences that are too short for CP 
        _divisibility = 2 * cp_size
        _seq_lens = cu_seq[1:] - cu_seq[:-1]
        _keep = torch.cat([
            torch.tensor([True], device=device),
            _seq_lens >= _divisibility,
        ])
        cu_seq = cu_seq[_keep]

        if cp_size > 1:

            from megatron.core.packed_seq_params import PackedSeqParams
            from transformer_engine.pytorch.attention.dot_product_attention.context_parallel import (
                pad_thd_sequences_for_cp,
                generate_positional_ids_for_cp,
                get_batch_on_this_cp_rank as te_get_batch_on_this_cp_rank,
            )

            divisibility = 2 * cp_size
            cp_group = parallel_state.get_context_parallel_group()
            cp_rank = parallel_state.get_context_parallel_rank()

            # Step 2a. Pad every sub-sequence so its length is divisible by 2*cp_size
            input_ids_padded, labels_padded, cu_seqlens_padded = pad_thd_sequences_for_cp(
                batch["tokens"].view(-1).cpu(),
                batch["labels"].view(-1).cpu(),
                cu_seq.cpu(),
                divisibility_factor=divisibility,
                padding_token_id=tokenizer.eod,
                padding_label_id=-100,
            )
            input_ids_padded = input_ids_padded.to(device)
            labels_padded = labels_padded.to(device)
            cu_seqlens_padded = cu_seqlens_padded.to(device=device, dtype=torch.int32)

            assert (cu_seqlens_padded[1:] - cu_seqlens_padded[:-1]).min() % divisibility == 0, \
                "Some padded sequence lengths are not divisible by 2*cp_size after merging."

            # Step 2b. Pad loss_mask -> Unfortunately, there wasn't any function applying padding to the loss function in transformers_engine, 
            # but the current line of code follow the same algorithm that the other implementations have.
            loss_mask_flat = batch["loss_mask"].view(-1)
            seqlens = cu_seq[1:] - cu_seq[:-1]
            padding_amounts = [
                ((l.item() + divisibility - 1) // divisibility) * divisibility - l.item()
                for l in seqlens
            ]
            loss_mask_seqs = [
                loss_mask_flat[cu_seq[i]:cu_seq[i + 1]]
                for i in range(len(cu_seq) - 1)
            ]
            loss_mask_padded = torch.cat([
                torch.cat([seq, torch.zeros(pad, dtype=seq.dtype, device=seq.device)])
                if pad > 0 else seq
                for seq, pad in zip(loss_mask_seqs, padding_amounts)
            ])

            # Step 2b-bis. Pad assistant_mask (same algorithm as loss_mask)
            if batch.get("assistant_mask") is not None:
                am_flat = batch["assistant_mask"].view(-1)
                am_seqs = [
                    am_flat[cu_seq[i]:cu_seq[i + 1]]
                    for i in range(len(cu_seq) - 1)
                ]
                am_padded = torch.cat([
                    torch.cat([seq, torch.zeros(pad, dtype=seq.dtype, device=seq.device)])
                    if pad > 0 else seq
                    for seq, pad in zip(am_seqs, padding_amounts)
                ])
            else:
                am_padded = None

            # Step 2c. Generate position_ids for padded sequences
            position_ids_padded = generate_positional_ids_for_cp(
                cu_seq.cpu(), divisibility, dtype=batch["position_ids"].dtype
            ).to(device)

            # Step 3a: DualChunkSwap slicing
            input_ids_padded, labels_padded, position_ids_padded = (
                te_get_batch_on_this_cp_rank(
                    cu_seqlens_padded,
                    input_ids_padded,
                    labels_padded,
                    position_ids_padded,
                    cp_group=cp_group,
                    qvk_format='thd',
                )
            )

            # Steo 3b: Again, similar to the transformer_engine logic, we select the loss_mask tensor related to the specific CP rank of this GPU.
            total_slices = 2 * cp_size
            slice_sizes = (
                (cu_seqlens_padded[1:] - cu_seqlens_padded[:-1]) // total_slices
            )
            cp_rank_indices = []
            
            # On this particular rank, for each sequence, get two slices, one from the beginning
            # and one from the end. 
            for slice_size, seq_start in zip(slice_sizes, cu_seqlens_padded[:-1]):
                cp_rank_indices.append(torch.arange(
                    seq_start + cp_rank * slice_size,
                    seq_start + (cp_rank + 1) * slice_size,
                    device=device,
                ))
                cp_rank_indices.append(torch.arange(
                    seq_start + (total_slices - cp_rank - 1) * slice_size,
                    seq_start + (total_slices - cp_rank) * slice_size,
                    device=device,
                ))
            loss_mask_padded = loss_mask_padded.index_select(
                0, torch.cat(cp_rank_indices)
            )
            if am_padded is not None:
                am_padded = am_padded.index_select(0, torch.cat(cp_rank_indices))

            batch["tokens"] = input_ids_padded.unsqueeze(0)
            batch["labels"] = labels_padded.unsqueeze(0)
            batch["loss_mask"] = loss_mask_padded.unsqueeze(0)
            batch["position_ids"] = position_ids_padded.unsqueeze(0)
            batch["attention_mask"] = None
            batch["assistant_mask"] = am_padded.unsqueeze(0) if am_padded is not None else None

            #  Step 4: construct PackedSeqParams 
            max_padded_len = (cu_seqlens_padded[1:] - cu_seqlens_padded[:-1]).max()

            assert max_padded_len % divisibility == 0, \
                f"max_padded_len {max_padded_len} not divisible by {divisibility}"

            packed_seq_params = PackedSeqParams(
                cu_seqlens_q=cu_seqlens_padded,  # We are returning input_ids_padded as "tokens", cu_seqlens is no longer necessary. This is a bit hacky.
                cu_seqlens_kv=cu_seqlens_padded, # We are returning input_ids_padded as "tokens", cu_seqlens is no longer necessary. This is a bit hacky.
                max_seqlen_q=max_padded_len,
                max_seqlen_kv=max_padded_len,
                qkv_format=qkv_format,
                cu_seqlens_q_padded=cu_seqlens_padded,
                cu_seqlens_kv_padded=cu_seqlens_padded,
            )

        else:
            # cp_size == 1: No need to cut the sequence length
            packed_seq_params = tokens_to_packed_seq_params(
                tokens_full_flat,
                eod_token=tokenizer.eod,
                orig_seq_len=args.seq_length,
                qkv_format=qkv_format,
            )
            for key in ["tokens", "labels", "loss_mask", "position_ids", "assistant_mask"]:
                if batch.get(key) is not None:
                    batch[key] = batch[key].view(1, -1)

    else:
        batch = get_batch_on_this_cp_rank(batch)

    return batch.values(), packed_seq_params




# define spiky loss as a loss that's 10x the max loss observed
SPIKY_LOSS_FACTOR = 10
MODALITY_WEIGHT_NAMES = ("vision", "audio")


def get_current_modality_weight(args, modality: str) -> float:
    """Compute current modality loss weight using static or decay config."""
    if not getattr(args, f"{modality}_weight_decay"):
        return getattr(args, f"{modality}_weight")

    step = getattr(args, "curr_iteration", 0)
    start_step = getattr(args, f"{modality}_weight_decay_start_step")
    end_step = getattr(args, f"{modality}_weight_decay_end_step")
    w_max = getattr(args, f"{modality}_weight_max")
    w_min = getattr(args, f"{modality}_weight_min")

    if step <= start_step:
        return w_max
    if step >= end_step:
        return w_min

    progress = (step - start_step) / (end_step - start_step)
    schedule = getattr(args, f"{modality}_weight_decay_schedule")
    if schedule == "cosine":
        return w_min + 0.5 * (w_max - w_min) * (1.0 + math.cos(math.pi * progress))
    return w_max + (w_min - w_max) * progress


def apply_decay_modality_weights(loss_mask: torch.Tensor, labels: torch.Tensor, args, current_weights: dict) -> torch.Tensor:
    """Apply decayed modality weights only on currently active (non-zero) loss entries."""
    if labels is None:
        return loss_mask

    loss_mask_modified = False
    for modality, current_weight in current_weights.items():
        if not getattr(args, f"{modality}_weight_decay", False):
            continue

        offset = getattr(args, f"{modality}_token_offset", None)
        vocab_size = getattr(args, f"{modality}_vocab_size", None)
        if offset is None or vocab_size is None:
            raise ValueError(
                f"{modality}_weight_decay is enabled but token metadata is missing. "
                f"Expected args.{modality}_token_offset and args.{modality}_vocab_size from tokenizer omnimodal_config."
            )

        modality_mask = (labels >= offset) & (labels < offset + vocab_size)
        active_modality_mask = modality_mask & (loss_mask > 0)
        if not loss_mask_modified:
            loss_mask = loss_mask.clone()
            loss_mask_modified = True
        loss_mask[active_modality_mask] = current_weight

    return loss_mask


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor, model: Optional[GPTModel] = None, labels: torch.Tensor = None, assistant_mask: torch.Tensor = None, current_modality_weights: dict = None):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
        model (GPTModel, optional): The model (can be wrapped)
        labels: tensor with labels to allow reporting of special losses based on label ids

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()

    if has_nvidia_modelopt and getattr(args, 'modelopt_enabled', False):  # [ModelOpt]
        loss, num_tokens, report = loss_func_modelopt(loss_mask, output_tensor, model=model)
    else:
        losses = output_tensor.view(-1).float()
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses * loss_mask)

        num_tokens = loss_mask.sum().clone().detach().to(torch.int)
        report = {'lm loss': torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])}

        # Assistant-only loss tracking for Apertus SFT
        if args.ap_sft and assistant_mask is not None:
            assistant_mask_flat = assistant_mask.view(-1).float()
            assistant_loss_sum = torch.sum(losses * assistant_mask_flat)
            assistant_count = assistant_mask_flat.sum().clone().detach().to(torch.int)
            report['assistant loss'] = torch.cat([
                assistant_loss_sum.clone().detach().view(1), assistant_count.view(1)
            ])

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
    # Check for spiky loss
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=SPIKY_LOSS_FACTOR,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=False,
        )

    # --- Per-modality token losses ---
    # Skip when using modelopt since `losses` is not defined in that code path
    is_modelopt = has_nvidia_modelopt and getattr(args, 'modelopt_enabled', False)
    if labels is not None and getattr(args, 'base_vocab_size', None) is not None and not is_modelopt:
        losses_flat = losses.detach().reshape(-1)
        labels_flat = labels.reshape(-1)
        loss_mask_flat = loss_mask.reshape(-1)

        modalities = [('text', 0, args.base_vocab_size)]
        omnimodal_config = getattr(args, 'omnimodal_config', None)
        if omnimodal_config is not None:
            for modality in omnimodal_config.get('modalities', []):
                modalities.append((modality['name'], modality['offset'], modality['vocab_size']))

        sum_list = []
        weighted_count_list = []
        raw_count_list = []
        raw_error_list = []
        for name, offset, vocab_size in modalities:
            in_range = (labels_flat >= offset) & (labels_flat < offset + vocab_size)
            in_range_f = in_range.float()
            weights = loss_mask_flat * in_range_f
            sum_list.append(torch.sum(losses_flat * weights))
            weighted_count_list.append(torch.sum(weights))
            raw_count_list.append(torch.sum(in_range_f))
            raw_error_list.append(torch.sum(losses_flat * in_range_f))

        # Stack as [num_modalities, 4] where dim=1 is [loss_sum, loss_weight_sum, token_count, raw_error]
        # Raw error is not affected by loss masking or weighting - enables track of error for modality tokens even if not supervising them in loss.
        # Don't reduce here - train_step() handles the DP/CP reduction for all report entries
        modality_stats = torch.stack((torch.stack(sum_list), torch.stack(weighted_count_list), torch.stack(raw_count_list), torch.stack(raw_error_list)), dim=1)

        for i, (name, _, _) in enumerate(modalities):
            report[f'{name}_token_loss'] = modality_stats[i][[0,1]]
            report[f'{name}_weighted_loss'] = modality_stats[i][[0,2]]
            report[f'{name}_error'] = modality_stats[i][[3, 2]]

    for modality, weight in (current_modality_weights or {}).items():
        if getattr(args, f"log_{modality}_weight", False):
            report[f"{modality}-weight"] = torch.tensor(weight, device=loss_mask.device)

    # Pre-final-layernorm activation norm (if captured by hook).
    if model is not None and args.log_pre_final_ln_norm:
        unwrapped_model = unwrap_model(model)
        decoder = getattr(unwrapped_model, "decoder", None)
        final_ln = getattr(decoder, "final_layernorm", None) if decoder is not None else None
        pre_ln_norm = getattr(final_ln, "_last_pre_ln_norm", None) if final_ln is not None else None
        if pre_ln_norm is not None:
            pre_ln_norm = pre_ln_norm.clone().detach()
            report["pre_final_ln_norm"] = torch.stack(
                (pre_ln_norm, torch.tensor(1.0, device=pre_ln_norm.device))
            )
    return loss, num_tokens, report


def forward_step(data_iterator, model: GPTModel, return_schedule_plan: bool = False):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
        return_schedule_plan (bool): Whether to return the schedule plan instead of the output tensor
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        vp_stage = get_attr_wrapped_model(model, "vp_stage")
        batch_values, packed_seq_params = get_batch(data_iterator, vp_stage)
        tokens, labels, loss_mask, attention_mask, position_ids, assistant_mask = batch_values

    timers('batch-generator').stop()

    # Dynamic modality weight decay: update active modality tokens only.
    current_modality_weights = {
        modality: get_current_modality_weight(args, modality) for modality in MODALITY_WEIGHT_NAMES
    }
    loss_mask = apply_decay_modality_weights(loss_mask, labels, args, current_modality_weights)

    with stimer:
        if args.use_legacy_models:
            output_tensor = model(tokens, position_ids, attention_mask, labels=labels,
                                  packed_seq_params=packed_seq_params)
        else:
            if return_schedule_plan:
                assert args.overlap_moe_expert_parallel_comm, \
                    "overlap_moe_expert_parallel_comm must be enabled to return the schedule plan"
                schedule_plan = model.build_schedule_plan(
                    tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask,
                    packed_seq_params=packed_seq_params
                )
                return schedule_plan, partial(
                    loss_func,
                    loss_mask,
                    model=model,
                    labels=labels,
                    assistant_mask=assistant_mask,
                    current_modality_weights=current_modality_weights,
                )
            else:
                output_tensor = model(
                    tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask,
                    packed_seq_params=packed_seq_params
                )

    # [ModelOpt]: model is needed to access ModelOpt distillation losses
    return output_tensor, partial(
        loss_func,
        loss_mask,
        model=model,
        labels=labels,
        assistant_mask=assistant_mask,
        current_modality_weights=current_modality_weights,
    )


def is_dataset_built_on_rank(vp_stage=None):
    return is_first_or_last_pipeline_stage(vp_stage) and parallel_state.get_tensor_model_parallel_rank() == 0


def core_gpt_dataset_config_from_args(args):
    if args.legacy_tokenizer:
        tokenizer = get_tokenizer()
    else:
        tokenizer = build_tokenizer(args)

    # Populate metadata for both tokenizer paths. This keeps goldfish exemption and
    # modality offsets/vocab available to GPT dataset construction and reporting.
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

    data_args = {
        "random_seed": args.seed,
        "sequence_length": args.seq_length,
        "blend": blend,
        "blend_per_split": blend_per_split,
        "split": args.split,
        "multiple_validation_sets": args.multiple_validation_sets,
        "full_validation": args.full_validation,
        "num_dataset_builder_threads": args.num_dataset_builder_threads,
        "path_to_cache": args.data_cache_path,
        "mmap_bin_files": args.mmap_bin_files,
        "tokenizer": tokenizer,
        "reset_position_ids": args.reset_position_ids,
        "reset_attention_mask": args.reset_attention_mask,
        "eod_mask_loss": args.eod_mask_loss,
        "create_attention_mask": args.create_attention_mask_in_dataloader,
        "object_storage_cache_path": args.object_storage_cache_path,
        "mid_level_dataset_surplus": args.mid_level_dataset_surplus,
        "allow_ambiguous_pad_tokens": args.allow_ambiguous_pad_tokens,
        "fast_cache_load": args.dataloader_fast_cache_load,
        "sequences_per_dataset": sequences_per_dataset,
        "defer_npy_index_mmap": args.dataloader_defer_npy_index_mmap,
        "goldfish_loss": args.goldfish_loss,
        "goldfish_k": args.goldfish_k,
        "goldfish_h": args.goldfish_h,
        "modality_weights": modality_weights,
        "vision_weight": args.vision_weight,
        "audio_weight": args.audio_weight,
        "sft_plw": args.ap_sft_plw,
        "sft_load_loss_mask": args.ap_sft_load_loss_mask,
        "sft_mask_special_tokens": args.ap_sft_mask_special_tokens,
        "sft_pack_samples": args.ap_sft_pack_samples,
        "sft_equalize_sample_loss": args.ap_sft_equalize_sample_loss,
    }

    # add FIM args to the config
    if args.fim_data:
        extra_tokens = {
            "prefix": args.fim_prefix_token,
            "middle": args.fim_middle_token,
            "suffix": args.fim_suffix_token,
            "pad": args.fim_pad_token,
            "eod": args.fim_eod_token,
        }
        data_args.update(
            {
                "fim_rate": args.fim_rate,
                "fim_spm_rate": args.fim_spm_rate,
                "fim_extra_tokens": extra_tokens,
                "fim_split_sample": args.fim_split_sample,
                "fim_fragment_rate": args.fim_fragment_rate,
                "fim_no_prefix": args.fim_no_prefix,
            }
        )
        return GPTFIMDatasetConfig(**data_args)

    return GPTDatasetConfig(**data_args)


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)

    if args.ap_sft:
        dataset_type = ApertusSFTDataset
    elif args.sft:
        dataset_type = SFTDataset
    else:
        if args.mock_data:
            dataset_type = MockGPTDataset
        elif args.fim_data:
            dataset_type = GPTFIMDataset
        else:
            dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type, train_val_test_num_samples, partial(is_dataset_built_on_rank, vp_stage=vp_stage), config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    # Optionally enable inprocess restart on pretrain
    pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

    pretrain(
        train_valid_test_datasets_provider,
        partial(model_provider, gpt_builder),
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
        extra_args_provider=add_modelopt_args if has_nvidia_modelopt else None,
        store=store,
    )
