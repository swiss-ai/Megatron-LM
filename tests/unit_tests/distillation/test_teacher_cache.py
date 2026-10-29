"""CPU tests for the teacher-cache format, writer and scoring.

Run with ``pytest --noconftest tests/unit_tests/distillation`` (no GPU or test
data needed).
"""

import os

import pytest
import torch

from megatron.core import parallel_state
from megatron.training.distillation.cache_format import (
    FORMAT_NAME,
    FORMAT_VERSION,
    FileView,
    check_manifest_compatible,
    encode_file,
    file_path,
    load_file,
    load_manifest,
    load_progress,
    sample_distinct,
    save_file,
    write_manifest,
    write_progress,
)
from megatron.training.distillation.cache_writer import AsyncFileWriter
from megatron.training.distillation.scoring import teacher_rows


def _batch(seqs=3, seq_length=16, slots=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, 1000, (seqs, seq_length), generator=g)
    labels = torch.randint(0, 1000, (seqs, seq_length), generator=g)
    loss_mask = (torch.rand(seqs, seq_length, generator=g) > 0.4).float()
    loss_mask[0, :5] = 0.5  # Fractional modality weights are stored as they are.
    position_ids = torch.arange(seq_length).repeat(seqs, 1)
    assistant_mask = loss_mask > 0
    rows = int((loss_mask > 0).sum())
    ids = torch.randint(0, 1000, (rows, slots), generator=g)
    probs = torch.rand(rows, slots, generator=g)
    return tokens, labels, loss_mask, position_ids, assistant_mask, ids, probs


@pytest.mark.parametrize('compress_level', [0, 1])
@pytest.mark.parametrize('prob_dtype', ['fp16', 'fp32'])
def test_file_roundtrip(tmp_path, compress_level, prob_dtype):
    tokens, labels, loss_mask, position_ids, assistant_mask, ids, probs = _batch()
    payload = encode_file(tokens, labels, loss_mask, position_ids, assistant_mask, ids, probs,
                          prob_dtype)
    path = file_path(str(tmp_path), 12345)
    save_file(path, payload, compress_level)
    assert path.endswith(os.path.join('files', '000001', '0000012345.pt.gz'))
    assert not [name for name in os.listdir(os.path.dirname(path)) if name.endswith('.tmp')]

    batch = FileView(load_file(path)).sequences(1, 3)
    assert torch.equal(batch['tokens'], tokens[1:3])
    assert torch.equal(batch['labels'], labels[1:3])
    assert torch.equal(batch['loss_mask'], loss_mask[1:3])
    assert torch.equal(batch['position_ids'], position_ids[1:3])
    assert torch.equal(batch['assistant_mask'], assistant_mask[1:3].float())

    # Teacher rows land on the positions with loss and are zero elsewhere.
    slots, seq_length = ids.shape[1], tokens.shape[1]
    has_loss = (loss_mask > 0).reshape(-1)
    dense_ids = torch.zeros(has_loss.numel(), slots, dtype=torch.long)
    dense_probs = torch.zeros(has_loss.numel(), slots)
    dense_ids[has_loss] = ids
    stored = probs.to(torch.float16 if prob_dtype == 'fp16' else torch.float32)
    dense_probs[has_loss] = stored.float()
    window = slice(seq_length, 3 * seq_length)
    assert torch.equal(batch['teacher_ids'].long().reshape(-1, slots), dense_ids[window])
    assert torch.equal(batch['teacher_probs'].reshape(-1, slots), dense_probs[window])


def test_encode_rejects_row_mismatch():
    tokens, labels, loss_mask, position_ids, _, ids, probs = _batch()
    with pytest.raises(ValueError):
        encode_file(tokens, labels, loss_mask, position_ids, None, ids[1:], probs[1:], 'fp32')


def test_manifest_and_progress(tmp_path):
    manifest = {'format': FORMAT_NAME, 'version': FORMAT_VERSION, 'mode': 'topk', 'slots': 256,
                'writer': {'data_parallel_size': 4}}
    write_manifest(str(tmp_path), manifest)
    loaded = load_manifest(str(tmp_path))
    assert loaded == manifest
    # The writer's own topology may change between runs; the cache settings may not.
    check_manifest_compatible(loaded, dict(manifest, writer={'data_parallel_size': 8}))
    with pytest.raises(ValueError, match='slots'):
        check_manifest_compatible(loaded, dict(manifest, slots=64))
    assert load_manifest(str(tmp_path / 'missing')) is None

    assert load_progress(str(tmp_path)) == 0
    write_progress(str(tmp_path), 40, 4)
    assert load_progress(str(tmp_path)) == 40


def test_sample_distinct():
    probs = torch.softmax(torch.randn(64, 50, generator=torch.Generator().manual_seed(0)) * 2, -1)
    ids, freqs = sample_distinct(probs, 32, torch.Generator().manual_seed(1))
    assert ids.shape == freqs.shape == (64, 32)
    torch.testing.assert_close(freqs.sum(-1), torch.ones(64))
    for row_ids, row_freqs in zip(ids, freqs):
        used = row_freqs > 0
        n = int(used.sum())
        assert used[:n].all() and not used[n:].any()  # Distinct ids first, then padding.
        assert len(set(row_ids[:n].tolist())) == n
        assert (row_ids[n:] == 0).all()
        counts = row_freqs[:n] * 32
        torch.testing.assert_close(counts, counts.round())

    # The same seed gives the same draws.
    ids_again, freqs_again = sample_distinct(probs, 32, torch.Generator().manual_seed(1))
    assert torch.equal(ids, ids_again) and torch.equal(freqs, freqs_again)

    # Frequencies are an unbiased estimate of the probabilities.
    rows = probs[:1].expand(4000, -1).contiguous()
    many_ids, many_freqs = sample_distinct(rows, 32, torch.Generator().manual_seed(2))
    estimate = torch.zeros(50).index_add_(0, many_ids.reshape(-1), many_freqs.reshape(-1)) / 4000
    torch.testing.assert_close(estimate, probs[0], atol=0.01, rtol=0)


def test_async_writer(tmp_path):
    tokens, labels, loss_mask, position_ids, _, ids, probs = _batch()
    payload = encode_file(tokens, labels, loss_mask, position_ids, None, ids, probs, 'fp16')
    writer = AsyncFileWriter(str(tmp_path), num_threads=2, compress_level=1)
    for step in range(5):
        writer.submit(step, 4 * step + 1, payload)
    assert writer.close() == 5
    for step in range(5):
        assert os.path.exists(file_path(str(tmp_path), 4 * step + 1))


def test_async_writer_raises_on_failure(tmp_path):
    (tmp_path / 'files').write_text('a file where the writer needs a directory')
    tokens, labels, loss_mask, position_ids, _, ids, probs = _batch()
    payload = encode_file(tokens, labels, loss_mask, position_ids, None, ids, probs, 'fp16')
    writer = AsyncFileWriter(str(tmp_path), num_threads=1, compress_level=0)
    writer.submit(0, 0, payload)
    with pytest.raises(OSError):
        writer.close()


@pytest.fixture
def single_tp_rank(monkeypatch):
    monkeypatch.setattr(parallel_state, 'get_tensor_model_parallel_world_size', lambda: 1)


@pytest.mark.parametrize('chunk', [3, 7, 64])
def test_teacher_rows_topk(single_tp_rank, chunk):
    g = torch.Generator().manual_seed(0)
    hidden = torch.randn(10, 8, generator=g)
    weight = torch.randn(40, 8, generator=g)  # 40 padded rows, of which 30 are scored.
    ids, probs = teacher_rows(hidden, weight, None, vocab_limit=30, mode='topk', slots=5,
                              chunk=chunk)
    ref_probs, ref_ids = torch.topk(torch.softmax(hidden @ weight[:30].T, dim=-1), 5, dim=-1)
    assert ids.dtype == torch.int32 and ids.shape == (10, 5)
    assert torch.equal(ids.long(), ref_ids)
    torch.testing.assert_close(probs, ref_probs)


def test_teacher_rows_sample(single_tp_rank):
    g = torch.Generator().manual_seed(0)
    hidden = torch.randn(6, 8, generator=g)
    weight = torch.randn(40, 8, generator=g)
    bias = torch.randn(40, generator=g)
    ids, freqs = teacher_rows(hidden, weight, bias, vocab_limit=30, mode='sample', slots=16,
                              chunk=4, generator=torch.Generator().manual_seed(3))
    assert ids.shape == freqs.shape == (6, 16)
    torch.testing.assert_close(freqs.sum(-1), torch.ones(6))
    assert (ids < 30).all()


def test_teacher_rows_empty(single_tp_rank):
    ids, probs = teacher_rows(torch.zeros(0, 8), torch.randn(40, 8), None, 30, 'topk', 5, 4)
    assert ids.shape == (0, 5) and probs.shape == (0, 5)
