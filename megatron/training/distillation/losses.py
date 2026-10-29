"""Student losses against a sparse teacher distribution from a teacher cache.

For each position with loss, with student logits ``z``, teacher ids ``K`` and their
stored probabilities ``p`` (top-k probabilities, or sampled frequencies):

* ``ce``: cross-entropy on the label over the student's whole vocabulary.
* ``kd``: cross-entropy against the teacher over the ids below ``vocab_limit``, with
  the student's softmax ``q`` renormalized over them:

  - plain: ``-sum_K p log q``. For sampled frequencies, which sum to 1, this is an
    unbiased estimate of the cross-entropy against the teacher's full distribution.
    For top-k probabilities it is the Apertus-v1.1 loss: the teacher mass outside
    ``K`` is dropped, and the gradient pushes the student's mass there to zero.
  - tail: adds ``-t log(1 - sum_K q)`` with ``t = 1 - sum_K p``, the teacher's mass
    outside ``K`` as one bucket, so the student keeps that mass instead.

Both are computed from fp32 logits in chunks of rows and recomputed in the
backward pass, so at most ``chunk x vocab`` fp32 values exist at a time.
"""

from typing import Dict, Optional

import torch


def _chunks(n: int, chunk: int):
    for start in range(0, n, chunk):
        yield start, min(n, start + chunk)


def _row_terms(z, labels, ids, probs, vocab_limit: int, tail: bool) -> Dict[str, torch.Tensor]:
    """Per-row loss terms of one chunk of fp32 logits ``z`` (``[rows, vocab]``)."""
    terms = {'lse': torch.logsumexp(z, dim=-1)}
    terms['ce'] = terms['lse'] - z.gather(1, labels.unsqueeze(1)).squeeze(1)
    z_text = z[:, :vocab_limit]
    terms['lse_text'] = torch.logsumexp(z_text, dim=-1)
    mass = probs.sum(-1)
    # -sum_K p log q = mass * logsumexp(z_text) - sum_K p z
    terms['kd'] = mass * terms['lse_text'] - (probs * z_text.gather(1, ids)).sum(-1)
    if tail:
        lse_tail = torch.logsumexp(z_text.scatter(1, ids, float('-inf')), dim=-1)
        # -t log(1 - sum_K q) = t * (logsumexp(z_text) - logsumexp(z_text outside K))
        terms['kd'] = terms['kd'] + (1 - mass).clamp(min=0) * (terms['lse_text'] - lse_tail)
        terms['lse_tail'] = lse_tail
    return terms


class _SparseDistillation(torch.autograd.Function):
    """``(kd, ce)`` per row; see the module docstring."""

    @staticmethod
    def forward(ctx, logits, labels, ids, probs, vocab_limit, tail, chunk):
        n = logits.shape[0]
        out = {key: logits.new_zeros(n, dtype=torch.float32)
               for key in ('kd', 'ce', 'lse', 'lse_text') + (('lse_tail',) if tail else ())}
        for start, end in _chunks(n, chunk):
            terms = _row_terms(logits[start:end].float(), labels[start:end], ids[start:end],
                               probs[start:end], vocab_limit, tail)
            for key, value in terms.items():
                out[key][start:end] = value
        ctx.save_for_backward(logits, labels, ids, probs, out['lse'], out['lse_text'],
                              out.get('lse_tail'))
        ctx.vocab_limit, ctx.tail, ctx.chunk = vocab_limit, tail, chunk
        return out['kd'], out['ce']

    @staticmethod
    def backward(ctx, grad_kd, grad_ce):
        logits, labels, ids, probs, lse, lse_text, lse_tail = ctx.saved_tensors
        vocab_limit = ctx.vocab_limit
        grad = torch.empty_like(logits)
        for start, end in _chunks(logits.shape[0], ctx.chunk):
            z = logits[start:end].float()
            k, p = ids[start:end], probs[start:end]
            # Cross-entropy: softmax minus the one-hot label.
            g = torch.exp(z - lse[start:end, None])
            g.scatter_add_(1, labels[start:end, None], torch.full_like(g[:, :1], -1.0))
            g *= grad_ce[start:end, None]
            q = torch.exp(z[:, :vocab_limit] - lse_text[start:end, None])
            if ctx.tail:
                # Target: p on K, and the teacher's mass outside K spread over the other
                # ids in proportion to the student's own distribution there.
                t = (1 - p.sum(-1)).clamp(min=0)
                target = torch.exp(z[:, :vocab_limit] - lse_tail[start:end, None])
                target *= t[:, None]
                target.scatter_(1, k, 0.0)
                target.scatter_add_(1, k, p)
                q -= target
            else:
                # The stored mass scales the student's whole distribution (the top-k
                # case without a tail), minus p on K.
                q *= p.sum(-1, keepdim=True)
                q.scatter_add_(1, k, -p)
            q *= grad_kd[start:end, None]
            g[:, :vocab_limit] += q
            grad[start:end] = g.to(grad.dtype)
        return grad, None, None, None, None, None, None


def sparse_distillation(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ids: torch.Tensor,
    probs: torch.Tensor,
    vocab_limit: int,
    tail: bool,
    chunk: int = 2048,
):
    """Return ``(kd, ce)``, both ``[rows]`` float32.

    Args:
        logits: ``[rows, vocab]`` student logits over its whole (padded) vocabulary,
            only for the positions with loss.
        labels: ``[rows]`` target ids.
        ids, probs: ``[rows, slots]`` teacher ids (below ``vocab_limit``) and their
            probabilities or sampled frequencies.
        tail: add the tail-bucket term (for top-k caches).
        chunk: rows per fp32 block.
    """
    if tail:
        assert ids.shape[1] < vocab_limit, 'the tail bucket needs ids outside the stored ones'
    return _SparseDistillation.apply(logits, labels.long(), ids.long(), probs.float(), vocab_limit,
                                     tail, chunk)


@torch.no_grad()
def validation_terms(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ids: torch.Tensor,
    probs: torch.Tensor,
    vocab_limit: int,
    topk: bool,
    entropy: Optional[torch.Tensor] = None,
    chunk: int = 2048,
) -> Dict[str, torch.Tensor]:
    """Per-row diagnostics, the same for every student, from one cache.

    ``kl``: KL divergence from the teacher, ``teacher_ce`` minus the teacher's
    entropy. With sampled frequencies (and the cached ``entropy``) it is an unbiased
    estimate of the KL over the full distribution. With top-k probabilities the mass
    outside ``K`` counts as one bucket on both sides, which gives a lower bound.
    ``confidence`` / ``correct``: the student's top-1 probability over its whole
    vocabulary and whether that token is the label.
    """
    n = logits.shape[0]
    out = {key: logits.new_zeros(n, dtype=torch.float32)
           for key in ('teacher_ce', 'confidence', 'correct')}
    labels, ids, probs = labels.long(), ids.long(), probs.float()
    for start, end in _chunks(n, chunk):
        z = logits[start:end].float()
        terms = _row_terms(z, labels[start:end], ids[start:end], probs[start:end], vocab_limit,
                           tail=topk)
        out['teacher_ce'][start:end] = terms['kd']
        top, top_ids = z.max(dim=-1)
        out['confidence'][start:end] = torch.exp(top - terms['lse'])
        out['correct'][start:end] = (top_ids == labels[start:end]).float()
    if topk:
        # The entropy of the bucketed teacher: the stored ids plus one tail bucket.
        t = (1 - probs.sum(-1)).clamp(min=0)
        bucketed = -torch.special.xlogy(probs, probs).sum(-1) - torch.special.xlogy(t, t)
        out['kl'] = out['teacher_ce'] - bucketed
    elif entropy is not None:
        out['kl'] = out['teacher_ce'] - entropy.float()
    return out
