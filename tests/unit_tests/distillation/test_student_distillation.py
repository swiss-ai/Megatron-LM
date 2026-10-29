"""CPU tests for training students from a teacher cache: losses and cache reader.

Run with ``pytest --noconftest tests/unit_tests/distillation`` (no GPU or test
data needed).
"""

import pytest
import torch

from megatron.training.distillation.cache_format import (
    FORMAT_NAME,
    FORMAT_VERSION,
    ShardWriter,
    compress_chunk,
    encode_chunk,
    file_path,
    sample_distinct,
    write_manifest,
    write_progress,
)
from megatron.training.distillation.cache_reader import (
    CacheSequences,
    RankSpans,
    build_cache_iterator,
)
from megatron.training.distillation.losses import sparse_distillation, validation_terms

VOCAB, TEXT = 40, 30  # Student vocabulary; the teacher's ids are below TEXT.


def _reference(logits, labels, ids, probs, tail):
    """The losses with plain autograd, in float64."""
    z = logits.double()
    ce = torch.logsumexp(z, -1) - z.gather(1, labels[:, None]).squeeze(1)
    logq = torch.log_softmax(z[:, :TEXT], -1).gather(1, ids)
    kd = -(probs.double() * logq).sum(-1)
    if tail:
        t = (1 - probs.double().sum(-1)).clamp(min=0)
        kd = kd - t * torch.log1p(-logq.exp().sum(-1))
    return kd, ce


def _teacher(rows, slots, mode, seed=0):
    g = torch.Generator().manual_seed(seed)
    p = torch.softmax(torch.randn(rows, TEXT, generator=g) * 2, -1)
    if mode == 'topk':
        probs, ids = torch.topk(p, slots, dim=-1)
    else:
        ids, probs = sample_distinct(p, slots, torch.Generator().manual_seed(seed + 1))
    return p, ids, probs


@pytest.mark.parametrize('mode,tail', [('topk', False), ('topk', True), ('sample', False)])
@pytest.mark.parametrize('chunk', [3, 64])
def test_loss_matches_autograd(mode, tail, chunk):
    rows = 11
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(rows, VOCAB, generator=g, dtype=torch.float64) * 3
    labels = torch.randint(0, VOCAB, (rows,), generator=g)
    _, ids, probs = _teacher(rows, 16 if mode == 'sample' else 8, mode)
    if mode == 'sample':
        # Token 0 drawn for real in a row that also has padding (id 0, frequency 0).
        ids[0, 0], probs[0, 0] = 0, probs[0, 0]
        assert (probs[0] == 0).any()
    w_kd, w_ce = torch.rand(rows, generator=g), torch.rand(rows, generator=g)

    x = logits.clone().requires_grad_()
    kd, ce = sparse_distillation(x, labels, ids, probs, TEXT, tail, chunk)
    (w_kd * kd + w_ce * ce).sum().backward()
    y = logits.clone().requires_grad_()
    ref_kd, ref_ce = _reference(y, labels, ids, probs, tail)
    (w_kd * ref_kd + w_ce * ref_ce).sum().backward()

    torch.testing.assert_close(kd.double(), ref_kd, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(ce.double(), ref_ce, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(x.grad, y.grad, rtol=1e-5, atol=1e-6)
    assert x.grad[:, TEXT:].abs().max() > 0  # The label loss also trains the other ids.


def test_tail_bucket_is_minimized_by_the_teacher():
    """At q = p only the tail loss has zero gradient; v1.1's pushes the tail mass to zero."""
    p, ids, probs = _teacher(6, 4, 'topk')
    logits = torch.full((6, VOCAB), -30.0)
    logits[:, :TEXT] = p.log()
    labels = torch.zeros(6, dtype=torch.long)
    grads = {}
    for tail in (False, True):
        x = logits.clone().requires_grad_()
        kd, _ = sparse_distillation(x, labels, ids, probs, TEXT, tail)
        kd.sum().backward()
        grads[tail] = x.grad[:, :TEXT]
    assert grads[True].abs().max() < 1e-6
    outside = torch.ones(6, TEXT, dtype=torch.bool).scatter_(1, ids, False)
    assert (grads[False][outside] > 0).all()  # Gradient descent lowers every tail logit.


def test_sampled_loss_is_unbiased():
    """Averaged over draws, the sampled loss is the cross-entropy against the teacher."""
    p, _, _ = _teacher(1, 4, 'topk')
    rows = p.expand(4000, -1).contiguous()
    ids, freqs = sample_distinct(rows, 16, torch.Generator().manual_seed(5))
    logits = torch.randn(1, VOCAB, generator=torch.Generator().manual_seed(6)).expand(4000, -1)
    kd, _ = sparse_distillation(logits.contiguous(), torch.zeros(4000, dtype=torch.long), ids,
                                freqs, TEXT, tail=False)
    exact = -(p[0] * torch.log_softmax(logits[0, :TEXT], -1)).sum()
    torch.testing.assert_close(kd.mean(), exact, atol=0.02, rtol=0)


@pytest.mark.parametrize('mode', ['topk', 'sample'])
def test_validation_terms(mode):
    p, ids, probs = _teacher(8, 256 if mode == 'sample' else 6, mode)
    entropy = torch.special.entr(p).sum(-1)
    labels = p.argmax(-1)
    logits = torch.full((8, VOCAB), -30.0)
    logits[:, :TEXT] = p.log()
    terms = validation_terms(logits, labels, ids, probs, TEXT, mode == 'topk', entropy)
    if mode == 'topk':
        torch.testing.assert_close(terms['kl'], torch.zeros(8), atol=1e-5, rtol=0)
    else:
        # Unbiased, not exact: one set of draws per row.
        assert terms['kl'].abs().mean() < 0.2
    torch.testing.assert_close(terms['confidence'], p.max(-1).values, atol=1e-5, rtol=0)
    assert (terms['correct'] == 1).all()

    worse = logits.clone()
    worse[:, :TEXT] += torch.randn(8, TEXT, generator=torch.Generator().manual_seed(3))
    worse_terms = validation_terms(worse, labels, ids, probs, TEXT, mode == 'topk', entropy)
    assert worse_terms['kl'].mean() > terms['kl'].mean()


def _write_cache(path, num_files=3, seqs_per_file=8, seqs_per_chunk=4, seq_length=6, slots=5):
    """A small top-k cache whose tokens encode the sequence number."""
    write_manifest(str(path), {
        'format': FORMAT_NAME, 'version': FORMAT_VERSION, 'mode': 'topk', 'slots': slots,
        'seq_length': seq_length, 'seqs_per_file': seqs_per_file,
        'seqs_per_chunk': seqs_per_chunk, 'vocab_limit': TEXT, 'num_files': num_files,
    })
    g = torch.Generator().manual_seed(0)
    for file_id in range(num_files):
        writer = ShardWriter(file_path(str(path), file_id), {'file_id': file_id})
        for chunk in range(seqs_per_file // seqs_per_chunk):
            first = file_id * seqs_per_file + chunk * seqs_per_chunk
            seq = torch.arange(first, first + seqs_per_chunk)
            tokens = seq[:, None].repeat(1, seq_length)
            loss_mask = (torch.rand(seqs_per_chunk, seq_length, generator=g) > 0.3).float()
            rows = int(loss_mask.sum())
            # Every teacher row records its sequence in its first id.
            ids = torch.randint(0, TEXT, (rows, slots), generator=g)
            ids[:, 0] = tokens[loss_mask > 0] % TEXT
            payload = encode_chunk(tokens, tokens, loss_mask, torch.arange(seq_length).repeat(
                seqs_per_chunk, 1), None, ids, torch.rand(rows, slots, generator=g),
                torch.rand(rows, generator=g), -torch.rand(rows, generator=g), 'fp32')
            writer.add(compress_chunk(payload, 1), seqs_per_chunk, rows)
        writer.close()
    write_progress(str(path), num_files, 1)


@pytest.mark.parametrize('dp_size', [1, 2, 4])
@pytest.mark.parametrize('global_batch', [4, 8, 16])
def test_rank_spans(dp_size, global_batch):
    if global_batch % dp_size:
        return
    steps = range(1, 4)
    pieces = {r: list(RankSpans(steps, global_batch, r, dp_size, seqs_per_chunk=4))
              for r in range(dp_size)}
    for step in steps:
        # Each step covers its global batch, whatever the number of ranks, rank r taking
        # the contiguous block r; every piece lies in one chunk.
        per_rank = global_batch // dp_size
        for r in range(dp_size):
            first = step * global_batch + r * per_rank
            mine = [s for s in pieces[r] if first <= s[0] < first + per_rank]
            assert sum((list(range(*s)) for s in mine), []) == list(range(first, first + per_rank))
            assert all((s[1] - 1) // 4 == s[0] // 4 for s in mine)


@pytest.mark.parametrize('num_workers', [0, 2])
@pytest.mark.parametrize('dp_size,micro_batch_size', [(1, 2), (2, 1), (2, 2), (4, 1)])
def test_cache_iterator(tmp_path, num_workers, dp_size, micro_batch_size):
    _write_cache(tmp_path)
    cache = CacheSequences(str(tmp_path))
    assert cache.num_sequences == 24
    global_batch = 8
    seen = []
    for rank in range(dp_size):
        it = build_cache_iterator(cache, range(1, 3), global_batch, micro_batch_size, rank,
                                  dp_size, num_workers)
        for step in range(1, 3):
            per_rank = global_batch // dp_size
            first = step * global_batch + rank * per_rank
            for micro in range(per_rank // micro_batch_size):
                batch = next(it)
                expect = torch.arange(first + micro * micro_batch_size,
                                      first + (micro + 1) * micro_batch_size)
                assert torch.equal(batch['tokens'][:, 0], expect)
                # Teacher rows follow the loss positions of exactly these sequences.
                rows = batch['tokens'][batch['loss_mask'] > 0]
                assert batch['teacher_ids'].shape[0] == rows.numel()
                assert torch.equal(batch['teacher_ids'][:, 0].long(), rows % TEXT)
                assert batch['teacher_entropy'].shape == (rows.numel(),)
                seen += expect.tolist()
        with pytest.raises(StopIteration):
            next(it)
    assert sorted(seen) == list(range(8, 24))


def test_cache_iterator_cycles_and_checks_size(tmp_path):
    _write_cache(tmp_path)
    cache = CacheSequences(str(tmp_path))
    valid = build_cache_iterator(cache, range(2), 4, 2, 0, 1, 0, base=16, cycle=True)
    firsts = [int(next(valid)['tokens'][0, 0]) for _ in range(8)]
    assert firsts == [16, 18, 20, 22] * 2  # The same sequences at every evaluation.
    with pytest.raises(AssertionError, match='completed sequences'):
        build_cache_iterator(cache, range(4), 8, 2, 0, 1, 0)
