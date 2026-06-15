"""Pretrain and SFT GPT."""

import json
import math
from functools import partial
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist

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


def _ensure_complete_batch_on_all_tp_ranks(batch, device):
    """
    Broadcast all batch fields from TP-rank-0 to sibling TP ranks.

    When PP > 1, get_batch_on_this_tp_rank will leave some fields as None on
    non-zero TP ranks (e.g. tokens on the last PP stage). This function
    ensures every TP rank has the full batch so they can all compute
    packed_seq_params independently.

    Collective: all TP ranks must call. No-op when TP = 1.
    """
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    if tp_size <= 1:
        return

    tp_group = parallel_state.get_tensor_model_parallel_group()
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    tp_src = dist.get_global_rank(tp_group, 0)

    fields_and_dtypes = [
        ("tokens", torch.long), ("labels", torch.long),
        ("loss_mask", torch.float), ("position_ids", torch.long),
        ("assistant_mask", torch.float),  
    ]


    # Step 1: broadcast shape from TP-rank-0 (which always has all fields).
    shape_buf = torch.zeros(2, device=device, dtype=torch.int64)
    if tp_rank == 0:
        ref = next(v for v in [batch.get(k) for k, _ in fields_and_dtypes] if v is not None)
        shape_buf[0] = ref.shape[0]
        shape_buf[1] = ref.shape[1] if ref.dim() > 1 else 1
    dist.broadcast(shape_buf, src=tp_src, group=tp_group)
    shape = (int(shape_buf[0].item()), int(shape_buf[1].item()))

    # Step 2: broadcast required fields unconditionally.  All TP ranks must
    # participate in every broadcast (it's a collective).  Ranks that
    # already have a field get it overwritten with the same data; ranks
    # that had None receive it for the first time.
    for key, dtype in fields_and_dtypes:
        val = batch.get(key)
        if val is None:
            val = torch.empty(shape, device=device, dtype=dtype)
        if not val.is_contiguous():
            val = val.contiguous()
        dist.broadcast(val, src=tp_src, group=tp_group)
        batch[key] = val


def _packed_seq_params_from_batch(batch, args, tokenizer, cp_size, device):
    """Compute PackedSeqParams from a local batch, applying CP padding if needed.

    When CP > 1, this function modifies the batch in-place (padding tokens,
    labels, loss_mask, assistant_mask, and position_ids to lengths divisible
    by 2*CP, then slicing for DualChunkSwap).  This changes tensor sizes,
    which is why every TP rank that participates in the forward pass must call
    this function independently.

    Returns (packed_seq_params, modified_batch).
    """
    from megatron.core.packed_seq_params import PackedSeqParams

    qkv_format = 'thd'
    tokens_full_flat = batch["tokens"].view(1, -1)
    total_tokens = tokens_full_flat.size(-1)

    # Step 1a: compute cu_seqlens from EOD boundaries + fixed seq_length intervals
    cu_seq, _ = torch.sort(torch.unique(torch.cat((
        torch.arange(0, total_tokens + args.seq_length, args.seq_length, device=device, dtype=torch.int32),
        (tokens_full_flat.flatten() == tokenizer.eod).nonzero()[:, 0].int() + 1)))
        )

    # Step 1b: merge sub-sequences that are too short for CP
    if cp_size > 1:
        from transformer_engine.pytorch.attention.dot_product_attention.context_parallel import (
            pad_thd_sequences_for_cp,
            generate_positional_ids_for_cp,
            get_batch_on_this_cp_rank as te_get_batch_on_this_cp_rank,
        )

        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        sp_enabled = getattr(args, 'sequence_parallel', False)

        _divisibility = 2 * cp_size * (tp_size if sp_enabled else 1)
        _seq_lens = cu_seq[1:] - cu_seq[:-1]
        _keep = torch.cat([
            torch.tensor([True], device=device),
            _seq_lens >= _divisibility,
        ])
        cu_seq = cu_seq[_keep]

        divisibility = 2 * cp_size * (tp_size if sp_enabled else 1)

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

        # Step 2b. Pad loss_mask (no TE utility exists for this)
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

        # Step 3b: select loss_mask (and assistant_mask) slices for this CP rank
        total_slices = 2 * cp_size
        slice_sizes = (cu_seqlens_padded[1:] - cu_seqlens_padded[:-1]) // total_slices
        cp_rank_indices = []
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
        loss_mask_padded = loss_mask_padded.index_select(0, torch.cat(cp_rank_indices))
        if am_padded is not None:
            am_padded = am_padded.index_select(0, torch.cat(cp_rank_indices))

        batch["tokens"] = input_ids_padded.unsqueeze(0)
        batch["labels"] = labels_padded.unsqueeze(0)
        batch["loss_mask"] = loss_mask_padded.unsqueeze(0)
        batch["position_ids"] = position_ids_padded.unsqueeze(0)
        batch["attention_mask"] = None
        batch["assistant_mask"] = am_padded.unsqueeze(0) if am_padded is not None else None

        max_padded_len = int((cu_seqlens_padded[1:] - cu_seqlens_padded[:-1]).max().item())
        assert max_padded_len % divisibility == 0, \
            f"max_padded_len {max_padded_len} not divisible by {divisibility}"

        packed_seq_params = PackedSeqParams(
            cu_seqlens_q=cu_seqlens_padded,
            cu_seqlens_kv=cu_seqlens_padded,
            max_seqlen_q=max_padded_len,
            max_seqlen_kv=max_padded_len,
            qkv_format=qkv_format,
            cu_seqlens_q_padded=cu_seqlens_padded,
            cu_seqlens_kv_padded=cu_seqlens_padded,
        )

    else:
        # CP = 1: use the already-computed cu_seq, no padding needed.
        max_len = (cu_seq[1:] - cu_seq[:-1]).max()
        packed_seq_params = PackedSeqParams(
            cu_seqlens_q=cu_seq,
            cu_seqlens_kv=cu_seq,
            max_seqlen_q=max_len,
            max_seqlen_kv=max_len,
            qkv_format=qkv_format,
            cu_seqlens_q_padded=None,
            cu_seqlens_kv_padded=None,
        )
        for key in ["tokens", "labels", "loss_mask", "position_ids", "assistant_mask"]:
            if batch.get(key) is not None:
                batch[key] = batch[key].view(1, -1)

    return packed_seq_params, batch


def _compute_and_broadcast_packed_seq_params(batch, args, device):
    """Compute packed_seq_params on TP-rank-0 and broadcast to sibling TP ranks.

    Used for middle PP stages where only TP-rank-0 has meaningful batch data
    and the batch itself is discarded (activations arrive via PP P2P).

    Returns packed_seq_params on all TP ranks.
    """
    from megatron.core.packed_seq_params import PackedSeqParams

    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    cp_size = parallel_state.get_context_parallel_world_size()
    tokenizer = get_tokenizer()

    # TP-rank-0 computes; others will receive via broadcast.
    packed_seq_params = None
    if tp_rank == 0:
        packed_seq_params, _ = _packed_seq_params_from_batch(
            batch, args, tokenizer, cp_size, device
        )

    # Broadcast packed_seq_params to other TP ranks.
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    if tp_size <= 1:
        return packed_seq_params

    tp_group = parallel_state.get_tensor_model_parallel_group()
    tp_src_global = dist.get_global_rank(tp_group, 0)

    # Fixed-size buffer: [n_elements, max_seqlen, cu_seqlens...]
    buf_size = args.seq_length + 2
    buf = torch.zeros(buf_size, dtype=torch.int32, device=device)

    if tp_rank == 0:
        cu_sq = packed_seq_params.cu_seqlens_q
        msv = packed_seq_params.max_seqlen_q
        n = cu_sq.shape[0]
        buf[0] = n
        buf[1] = int(msv.item() if hasattr(msv, "item") else msv)
        buf[2:2 + n] = cu_sq.to(dtype=torch.int32, device=device)

    dist.broadcast(buf, src=tp_src_global, group=tp_group)

    if tp_rank != 0:
        n = int(buf[0].item())
        max_seqlen = int(buf[1].item())
        cu_seqlens = buf[2:2 + n].clone()
        packed_seq_params = PackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            qkv_format='thd',
            cu_seqlens_q_padded=cu_seqlens,
            cu_seqlens_kv_padded=cu_seqlens,
        )

    return packed_seq_params


def get_batch(data_iterator, vp_stage=None):
    """Generate a batch with packed-sequence + CP/PP support.

    Overview of the approach for each PP stage type:

    - First/last PP stages: all TP ranks compute packed_seq_params independently
    from identical data (broadcasted via get_batch_on_this_tp_rank). When PP > 1
    and TP > 1, missing batch fields on non-zero TP ranks are filled in first.

    - Middle PP stages: TP-rank-0 computes packed_seq_params from its local
    dataloader and broadcasts to sibling TP ranks. Batch values != packed_seq_params
    are set to None since activations arrive via PP P2P, not from the dataloader.

    Returns a 6-tuple (tokens, labels, loss_mask, attention_mask, position_ids,
    assistant_mask) and packed_seq_params. assistant_mask is None for pretraining.
    """
    args = get_args()

    if not is_first_or_last_pipeline_stage(vp_stage):
        if not args.use_packed_seq_params:
            return (None, None, None, None, None, None), None

        # All TP ranks must call get_batch_on_this_tp_rank (it's a collective).
        batch = get_batch_on_this_tp_rank(data_iterator)
        device = torch.cuda.current_device()

        packed_seq_params = _compute_and_broadcast_packed_seq_params(batch, args, device)
        return (None, None, None, None, None, None), packed_seq_params

    batch = get_batch_on_this_tp_rank(data_iterator)

    if not args.use_packed_seq_params:
        batch = get_batch_on_this_cp_rank(batch)
        return (
            batch.get("tokens"),
            batch.get("labels"),
            batch.get("loss_mask"),
            batch.get("attention_mask"),
            batch.get("position_ids"),
            batch.get("assistant_mask"),
        ), None

    cp_size = parallel_state.get_context_parallel_world_size()
    device = torch.cuda.current_device()
    tokenizer = get_tokenizer()

    # When PP > 1, the last stage's non-zero TP ranks have tokens=None
    # (get_batch_on_this_tp_rank only broadcasts what that stage needs for
    # loss computation).  If CP > 1, all TP ranks must run the CP padding
    # logic (which reads tokens), so we fill in the missing fields first.
    pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    tp_size = parallel_state.get_tensor_model_parallel_world_size()

    if pp_size > 1 and tp_size > 1:
        _ensure_complete_batch_on_all_tp_ranks(batch, device)

    packed_seq_params, batch = _packed_seq_params_from_batch(
        batch, args, tokenizer, cp_size, device
    )

    return (
        batch.get("tokens"),
        batch.get("labels"),
        batch.get("loss_mask"),
        batch.get("attention_mask"),
        batch.get("position_ids"),
        batch.get("assistant_mask"),
    ), packed_seq_params


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


def loss_func(
    loss_mask: torch.Tensor,
    output_tensor: torch.Tensor,
    model: Optional[GPTModel] = None,
    labels: torch.Tensor = None,
    assistant_mask: torch.Tensor = None,
    current_modality_weights: dict = None,
):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
        model (GPTModel, optional): The model (can be wrapped)
        labels: tensor with labels to allow reporting of special losses based on label ids
        assistant_mask: optional mask selecting only assistant tokens (Apertus SFT)

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
        if getattr(args, 'ap_sft', False) and assistant_mask is not None:
            assistant_mask_flat = assistant_mask.view(-1).float()
            assistant_loss_sum = torch.sum(losses * assistant_mask_flat)
            assistant_count = assistant_mask_flat.sum().clone().detach().to(torch.int)
            report['assistant_loss'] = torch.cat([
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
    args = get_args()
    # When use_packed_seq_params is enabled, ALL PP stages build a dataloader
    # on TP-rank-0. Middle stages use it only to compute cu_seqlens locally,
    # avoiding cross-stage P2P for packed sequence metadata.
    if args.use_packed_seq_params:
        return parallel_state.get_tensor_model_parallel_rank() == 0
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
        "loss_mask_token_ids": getattr(args, "loss_mask_token_ids", None),
        # Apertus SFT packing / masking options
        "sft_plw": args.ap_sft_plw,
        "sft_load_loss_mask": args.ap_sft_load_loss_mask,
        "sft_mask_special_tokens": args.ap_sft_mask_special_tokens,
        "sft_pack_samples": args.ap_sft_pack_samples,
        "sft_packing_strategy": args.ap_sft_packing_strategy,
        "sft_equalize_sample_loss": args.ap_sft_equalize_sample_loss,
        "sft_truncate_right": args.ap_sft_truncate_right,
        "pretraining_packing_strategy": args.pretraining_packing_strategy,
        "max_docs_per_bin": args.max_docs_per_bin,
        "max_docs_per_bin_sft": args.max_docs_per_bin_sft,
        "ap_sft_auto_tag": args.ap_sft,
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

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        train_val_test_num_samples, partial(is_dataset_built_on_rank, vp_stage=vp_stage), config
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