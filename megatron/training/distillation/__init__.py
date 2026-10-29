"""Teacher caches for offline knowledge distillation (Apertus 1.5).

``get_teacher_logits.py`` runs a frozen teacher over the regular Megatron data
pipeline and writes a sparse cache of its next-token distribution: the top-k
tokens, or sampled tokens with their frequencies, at every position with loss.

The cache is self-contained: each file also stores the tokens, labels, loss mask
and position ids exactly as the teacher's data pipeline produced them, so a
student trained from it sees the same data and the same masks (assistant-only SFT
loss, modality weights, task-marker masks, Goldfish) as the teacher's own run.

``pretrain_gpt_distill.py`` trains a student from a cache: ``cache_reader`` gives
each data-parallel rank its micro-batches, ``losses`` the distillation and label
losses.
"""

from .arguments import add_distillation_args
