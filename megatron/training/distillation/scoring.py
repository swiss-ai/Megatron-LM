"""Teacher scoring: output-layer logits in bounded chunks, then top-k or sampling."""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region

from .cache_format import sample_distinct


def teacher_rows(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    vocab_limit: int,
    mode: str,
    slots: int,
    chunk: int,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Score ``hidden`` rows and return ``(ids, probs)``, both ``[rows, slots]``.

    Args:
        hidden: ``[rows, hidden]`` final-norm hidden states of the positions to store.
        weight, bias: this rank's shard of the output layer (vocab-parallel over TP).
        vocab_limit: only ids below it are scored; the softmax renormalizes over them.
        mode: ``topk`` keeps the most likely ids, ``sample`` stores distinct sampled
            ids with count / draws.
        chunk: rows per output-layer call, which bounds the logits memory to
            ``chunk x vocab`` values.
    """
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    ids_parts, prob_parts = [], []
    for start in range(0, hidden.shape[0], chunk):
        h = hidden[start:start + chunk]
        if tp_size == 1:
            logits = F.linear(h, weight[:vocab_limit], None if bias is None else bias[:vocab_limit])
        else:
            local = F.linear(h, weight, bias)
            logits = gather_from_tensor_model_parallel_region(local)[:, :vocab_limit]
        probs = torch.softmax(logits.float(), dim=-1)
        del logits
        if mode == 'topk':
            values, ids = torch.topk(probs, slots, dim=-1, sorted=True)
        else:
            ids, values = sample_distinct(probs, slots, generator)
        ids_parts.append(ids.to(torch.int32))
        prob_parts.append(values)
        del probs
    if not ids_parts:
        empty = hidden.new_zeros((0, slots))
        return empty.to(torch.int32), empty.float()
    return torch.cat(ids_parts), torch.cat(prob_parts)
