"""Teacher scoring: output-layer logits in bounded chunks, then top-k or sampling."""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region

from .cache_format import sample_distinct


def teacher_rows(
    hidden: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    vocab_limit: int,
    mode: str,
    slots: int,
    chunk: int,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Score ``hidden`` rows and return ``(ids, probs, entropy, label_logprob)``.

    ``ids`` and ``probs`` are ``[rows, slots]``; ``entropy`` and ``label_logprob``
    are ``[rows]`` float32, computed from the full distribution over the ids below
    ``vocab_limit``: its entropy, and the log-probability of the row's label
    (-inf for labels at or above ``vocab_limit``, which it cannot produce).

    Args:
        hidden: ``[rows, hidden]`` final-norm hidden states of the positions to store.
        labels: ``[rows]`` target token of each row.
        weight, bias: this rank's shard of the output layer (vocab-parallel over TP).
        vocab_limit: only ids below it are scored; the softmax renormalizes over them.
        mode: ``topk`` keeps the most likely ids, ``sample`` stores distinct sampled
            ids with count / draws.
        chunk: rows per output-layer call, which bounds the logits memory to
            ``chunk x vocab`` values.
    """
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    parts = []
    for start in range(0, hidden.shape[0], chunk):
        h = hidden[start:start + chunk]
        if tp_size == 1:
            logits = F.linear(h, weight[:vocab_limit], None if bias is None else bias[:vocab_limit])
        else:
            local = F.linear(h, weight, bias)
            logits = gather_from_tensor_model_parallel_region(local)[:, :vocab_limit]
        logits = logits.float()
        log_norm = torch.logsumexp(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        entropy = (log_norm - (probs * logits).sum(-1)).clamp_(min=0)
        target = labels[start:start + chunk].long()
        in_vocab = (target >= 0) & (target < vocab_limit)
        label_logits = logits.gather(1, target.clamp(0, vocab_limit - 1).unsqueeze(1)).squeeze(1)
        label_logprob = torch.where(in_vocab, label_logits - log_norm, float('-inf'))
        del logits
        if mode == 'topk':
            values, ids = torch.topk(probs, slots, dim=-1, sorted=True)
        else:
            ids, values = sample_distinct(probs, slots, generator)
        parts.append((ids.to(torch.int32), values, entropy, label_logprob))
        del probs
    if not parts:
        empty = hidden.new_zeros((0, slots), dtype=torch.float32)
        none = hidden.new_zeros((0,), dtype=torch.float32)
        return empty.to(torch.int32), empty, none, none.clone()
    return tuple(torch.cat(column) for column in zip(*parts))
