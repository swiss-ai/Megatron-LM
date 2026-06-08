# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from typing import Optional, Tuple

import psutil
import torch


def chunk_bias(bias, parallel_mode, tp_size=1, ep_size=1):
    assert parallel_mode in ["row", "column"]
    if bias.dim() == 2:
        num_experts, hidden_size = bias.shape
        if parallel_mode == 'column':
            bias = bias.reshape(ep_size, num_experts // ep_size, tp_size, hidden_size // tp_size)
            bias = bias.permute(0, 2, 1, 3) # (ep_size, tp_size, local_eps, hidden_size)
        else:
            bias = bias.reshape(ep_size, num_experts // ep_size, hidden_size) # (ep_size, local_eps, hidden_size)
        return bias
    else:
        hidden_size = bias.shape
        if parallel_mode == "column":
            bias = bias.reshape(tp_size, hidden_size[0] // tp_size) # (tp_size, hidden_size)
        return bias


def chunk_weight(weight, parallel_mode, tp_size=1, ep_size=1):
    assert parallel_mode in ["row", "column"]
    if weight.dim() == 3:
        num_experts, out_features, in_features = weight.shape
        if parallel_mode == "column":
            weight = weight.reshape(ep_size, num_experts // ep_size, tp_size, out_features // tp_size, in_features)
            weight = weight.permute(0, 2, 1, 3, 4)
        else:
            weight = weight.reshape(ep_size, num_experts // ep_size, out_features, tp_size, in_features // tp_size)
            weight = weight.permute(0, 3, 1, 2, 4)
        return weight # (ep_size, tp_size, local_eps, output_features, in_features)
    else:
        out_features, in_features = weight.shape
        if parallel_mode == "column":
            weight = weight.reshape(tp_size, out_features // tp_size, in_features)
        else:
            weight = weight.reshape(out_features, tp_size, in_features // tp_size).permute(1, 0, 2)
        return weight # (tp_size, output_features, in_features)


def pop_xielu_params(msg: dict) -> Tuple[
    torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]
]:
    """Decode XIELU params from a loader queue message; tolerate both key schemes.

    ``loader_base``    emits ``"mlp xielu alpha p"`` / ``"mlp xielu alpha n"`` and
    optionally ``"mlp xielu beta"`` / ``"mlp xielu eps"``.
    ``loader_apertus`` emits ``"mlp alpha_p"`` / ``"mlp alpha_n"``.

    Centralizes the cross-loader key disagreement so savers don't fork on it.
    Returns ``(alpha_p, alpha_n, beta, eps)`` where ``beta`` and ``eps`` are
    ``None`` if absent.
    """
    if "mlp xielu alpha p" in msg:
        return (
            msg.pop("mlp xielu alpha p"),
            msg.pop("mlp xielu alpha n"),
            msg.pop("mlp xielu beta", None),
            msg.pop("mlp xielu eps", None),
        )
    return msg.pop("mlp alpha_p"), msg.pop("mlp alpha_n"), None, None


def print_memory_usage(key, rank, num_ranks):
    '''Print memory usage.'''
    process = psutil.Process()
    mem_info = process.memory_info()
    print("> memory usage: '%s', rank %d / %d, mem %.1f/%.1f gb." % (
        key,
        rank,
        num_ranks,
        mem_info.rss / 1024**3,
        100 * mem_info.rss / process.memory_percent() / 1024**3,
    ))

class _ConverterFakeProcessGroup:
    def __init__(self, rank=0, size=1):
        self._rank = rank
        self._size = size

    def rank(self):
        return self._rank

    def size(self):
        return self._size

    def set_rank(self, rank):
        self._rank = rank

    def set_size(self, size):
        self._size = size


def init_converter_fake_groups(mpu, *, tp_size, pp_size, ep_size, cp_size=1):
    """Wire fake process groups onto the Megatron mpu module for converter-only runs.

    Returns the dict of created groups so callers can later flip ranks via
    ``set_converter_fake_group_ranks``. Both loader and saver converters share
    this; drift here silently breaks TP/PP/EP correctness.
    """
    cp_size = max(1, cp_size or 1)

    groups = {
        "tp": _ConverterFakeProcessGroup(size=tp_size),
        "pp": _ConverterFakeProcessGroup(size=pp_size),
        "mp": _ConverterFakeProcessGroup(size=tp_size * pp_size),
        "embd": _ConverterFakeProcessGroup(size=min(pp_size, 2)),
        "pos_embd": _ConverterFakeProcessGroup(size=min(pp_size, 2)),
        "dp": _ConverterFakeProcessGroup(size=1),
        "dp_cp": _ConverterFakeProcessGroup(size=cp_size),
        "intra_dp_cp": _ConverterFakeProcessGroup(size=cp_size),
        "tp_dp": _ConverterFakeProcessGroup(size=tp_size),
        "tp_dp_cp": _ConverterFakeProcessGroup(size=tp_size * cp_size),
        "cp": _ConverterFakeProcessGroup(size=cp_size),
        "tp_cp": _ConverterFakeProcessGroup(size=tp_size * cp_size),
        "ep": _ConverterFakeProcessGroup(size=ep_size),
        "expt_tp": _ConverterFakeProcessGroup(size=tp_size),
        "tp_ep": _ConverterFakeProcessGroup(size=tp_size * ep_size),
        "tp_ep_pp": _ConverterFakeProcessGroup(size=tp_size * ep_size * pp_size),
        "expt_dp": _ConverterFakeProcessGroup(size=1),
        "intra_expt_dp": _ConverterFakeProcessGroup(size=1),
        "inter_dist_opt": _ConverterFakeProcessGroup(size=1),
        "intra_dist_opt": _ConverterFakeProcessGroup(size=1),
    }

    mpu._TENSOR_MODEL_PARALLEL_GROUP = groups["tp"]
    mpu._PIPELINE_MODEL_PARALLEL_GROUP = groups["pp"]
    mpu._MODEL_PARALLEL_GROUP = groups["mp"]
    mpu._EMBEDDING_GROUP = groups["embd"]
    mpu._POSITION_EMBEDDING_GROUP = groups["pos_embd"]
    mpu._CONTEXT_PARALLEL_GROUP = groups["cp"]
    mpu._TENSOR_AND_CONTEXT_PARALLEL_GROUP = groups["tp_cp"]

    mpu._DATA_PARALLEL_GROUP = groups["dp"]
    mpu._DATA_PARALLEL_GROUP_GLOO = groups["dp"]
    mpu._DATA_PARALLEL_GROUP_WITH_CP = groups["dp_cp"]
    mpu._DATA_PARALLEL_GROUP_WITH_CP_GLOO = groups["dp_cp"]
    mpu._INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP = groups["intra_dp_cp"]
    mpu._INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP_GLOO = groups["intra_dp_cp"]
    mpu._TENSOR_AND_DATA_PARALLEL_GROUP = groups["tp_dp"]
    mpu._TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = groups["tp_dp_cp"]

    mpu._EXPERT_MODEL_PARALLEL_GROUP = groups["ep"]
    mpu._EXPERT_TENSOR_PARALLEL_GROUP = groups["expt_tp"]
    mpu._EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP = groups["tp_ep"]
    mpu._EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP = groups["tp_ep_pp"]
    mpu._EXPERT_DATA_PARALLEL_GROUP = groups["expt_dp"]
    mpu._EXPERT_DATA_PARALLEL_GROUP_GLOO = groups["expt_dp"]
    mpu._INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP = groups["intra_expt_dp"]
    mpu._INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP_GLOO = groups["intra_expt_dp"]
    mpu._INTER_PARTIAL_EXPERT_DATA_PARALLEL_GROUP = groups["inter_dist_opt"]
    mpu._INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP = groups["intra_dist_opt"]

    mpu._HIERARCHICAL_CONTEXT_PARALLEL_GROUPS = [groups["cp"]] if cp_size > 1 else []
    mpu._EMBEDDING_GLOBAL_RANKS = [0] if pp_size == 1 else [0, pp_size - 1]
    mpu._POSITION_EMBEDDING_GLOBAL_RANKS = list(mpu._EMBEDDING_GLOBAL_RANKS)
    mpu._PIPELINE_GLOBAL_RANKS = list(range(pp_size))
    mpu._TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = list(range(tp_size))
    mpu._MODEL_PARALLEL_GLOBAL_RANKS = list(range(tp_size * pp_size))
    mpu._DATA_PARALLEL_GLOBAL_RANKS = [0]
    mpu._DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = [0]
    mpu._CONTEXT_PARALLEL_GLOBAL_RANKS = [0]

    return groups


def set_converter_fake_group_ranks(
    mpu, groups, *, tp_rank, pp_rank, ep_rank=0, tp_size, ep_size,
):
    """Update fake-group ranks for the (tp, pp, ep) shard currently being processed."""
    mp_rank = pp_rank * tp_size + tp_rank
    tp_ep_rank = ep_rank * tp_size + tp_rank
    tp_ep_pp_rank = pp_rank * (tp_size * ep_size) + tp_ep_rank

    groups["tp"].set_rank(tp_rank)
    groups["pp"].set_rank(pp_rank)
    groups["mp"].set_rank(mp_rank)
    groups["tp_dp"].set_rank(tp_rank)
    groups["tp_dp_cp"].set_rank(tp_rank)
    groups["cp"].set_rank(0)
    groups["tp_cp"].set_rank(tp_rank)
    groups["ep"].set_rank(ep_rank)
    groups["expt_tp"].set_rank(tp_rank)
    groups["tp_ep"].set_rank(tp_ep_rank)
    groups["tp_ep_pp"].set_rank(tp_ep_pp_rank)
    groups["embd"].set_rank(0)
    groups["pos_embd"].set_rank(0)
    groups["dp"].set_rank(0)
    groups["dp_cp"].set_rank(0)
    groups["intra_dp_cp"].set_rank(0)
    groups["expt_dp"].set_rank(0)
    groups["intra_expt_dp"].set_rank(0)
    groups["inter_dist_opt"].set_rank(0)
    groups["intra_dist_opt"].set_rank(0)

    mpu.set_tensor_model_parallel_rank(tp_rank)
    mpu.set_pipeline_model_parallel_rank(pp_rank)
    mpu.set_expert_tensor_parallel_rank(tp_rank)
    mpu.set_expert_model_parallel_rank(ep_rank)
