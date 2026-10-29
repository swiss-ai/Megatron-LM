"""CPU tests for the teacher-cache format, writer, scoring and calibration blend.

Run with ``pytest --noconftest tests/unit_tests/distillation`` (no GPU or test
data needed).
"""

import importlib.util
import os
import struct

import pytest
import torch

from megatron.core import parallel_state
from megatron.training.distillation.cache_format import (
    FORMAT_NAME,
    FORMAT_VERSION,
    FileView,
    ShardReader,
    ShardWriter,
    check_manifest_compatible,
    compress_chunk,
    encode_chunk,
    file_path,
    load_manifest,
    load_progress,
    sample_distinct,
    write_manifest,
    write_progress,
)
from megatron.training.distillation.cache_writer import AsyncShardWriter, FileBatchSampler
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
    entropy = torch.rand(rows, generator=g) * 3
    label_logprob = -torch.rand(rows, generator=g) * 5
    label_logprob[0] = float('-inf')  # A label outside the teacher's vocabulary.
    return (tokens, labels, loss_mask, position_ids, assistant_mask, ids, probs, entropy,
            label_logprob)


def _write_shard(path, chunks, compress_level=1, prob_dtype='fp32'):
    writer = ShardWriter(path, {'file_id': 7, 'first_sequence': 70})
    for chunk in chunks:
        payload = encode_chunk(*chunk, prob_dtype)
        ids = chunk[5]
        writer.add(compress_chunk(payload, compress_level), chunk[0].shape[0],
                   0 if ids is None else ids.shape[0])
    writer.close()


@pytest.mark.parametrize('compress_level', [0, 1])
@pytest.mark.parametrize('prob_dtype', ['fp16', 'fp32'])
def test_shard_roundtrip(tmp_path, compress_level, prob_dtype):
    chunks = [_batch(seed=0), _batch(seed=1)]
    path = file_path(str(tmp_path), 12345)
    _write_shard(path, chunks, compress_level, prob_dtype)
    assert path.endswith(os.path.join('files', '00012', '00012345.zip'))
    assert os.listdir(os.path.dirname(path)) == ['00012345.zip']  # No temporary files left.

    with ShardReader(path) as shard:
        assert shard.num_chunks == 2
        assert shard.index['file_id'] == 7 and shard.index['first_sequence'] == 70
        assert [c['rows'] for c in shard.index['chunks']] == [c[5].shape[0] for c in chunks]
        view = FileView(shard.chunk(1))  # Random access to the second chunk.
    batch = view.sequences(1, 3)
    tokens, labels, loss_mask, position_ids, assistant_mask, ids, probs, entropy, label_logprob = (
        chunks[1]
    )
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
    for key, values in (('teacher_entropy', entropy), ('teacher_label_logprob', label_logprob)):
        dense = torch.zeros(has_loss.numel())
        dense[has_loss] = values
        assert torch.equal(batch[key].reshape(-1), dense[window])

    # Compact rows: the same rows, in the order of the loss positions.
    compact = view.sequences(1, 3, dense=False)
    window_rows = has_loss[window]
    assert torch.equal(compact['teacher_ids'].long(), dense_ids[window][window_rows])
    assert torch.equal(compact['teacher_probs'], dense_probs[window][window_rows])
    first = int(has_loss[:seq_length].sum())
    rows = slice(first, first + int(window_rows.sum()))
    assert torch.equal(compact['teacher_entropy'], entropy[rows])
    assert torch.equal(compact['teacher_label_logprob'], label_logprob[rows])
    assert torch.equal(compact['tokens'], batch['tokens'])


def test_shard_writer_removes_leftovers(tmp_path):
    path = file_path(str(tmp_path), 3)
    other = file_path(str(tmp_path), 4)
    os.makedirs(os.path.dirname(path))
    for leftover in (f'{path}.111.222.tmp', f'{other}.111.222.tmp'):
        open(leftover, 'wb').close()
    _write_shard(path, [_batch()])
    # Only the leftover of the shard being written goes; other shards' files stay.
    assert sorted(os.listdir(os.path.dirname(path))) == ['00000003.zip', '00000004.zip.111.222.tmp']


def test_tokens_only_chunk(tmp_path):
    tokens, labels, loss_mask, position_ids, _, ids, probs, entropy, label_logprob = _batch()
    path = file_path(str(tmp_path), 0)
    _write_shard(path, [(tokens, labels, loss_mask, position_ids, None, None, None, None, None)])
    with ShardReader(path) as shard:
        payload = shard.chunk(0)
    assert not any(key.startswith('teacher_') for key in payload)
    batch = FileView(payload).sequences(0, 3)
    assert torch.equal(batch['tokens'], tokens) and 'teacher_ids' not in batch
    with pytest.raises(ValueError):
        encode_chunk(tokens, labels, loss_mask, position_ids, None, ids, probs, None, None, 'fp16')


def test_encode_rejects_row_mismatch():
    tokens, labels, loss_mask, position_ids, _, ids, probs, entropy, label_logprob = _batch()
    with pytest.raises(ValueError):
        encode_chunk(tokens, labels, loss_mask, position_ids, None, ids[1:], probs[1:],
                     entropy[1:], label_logprob[1:], 'fp32')
    with pytest.raises(ValueError):
        encode_chunk(tokens, labels, loss_mask, position_ids, None, ids, probs, entropy[1:],
                     label_logprob, 'fp32')


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


@pytest.mark.parametrize('num_files', [7, 8])
@pytest.mark.parametrize('dp_size', [1, 2, 4])
def test_file_batch_sampler(dp_size, num_files):
    seqs_per_file, micro_batch_size = 6, 2
    seen = {}
    for rank in range(dp_size):
        sampler = FileBatchSampler(0, num_files, seqs_per_file, micro_batch_size, rank, dp_size)
        batches = list(sampler)
        assert len(batches) == len(sampler) == len(range(rank, num_files, dp_size)) * 3
        # Each file arrives as consecutive micro-batches of contiguous samples.
        for i in range(0, len(batches), 3):
            samples = sum(batches[i:i + 3], [])
            file_id = samples[0] // seqs_per_file
            assert samples == list(range(file_id * 6, file_id * 6 + 6))
            assert file_id % dp_size == rank and file_id not in seen
            seen[file_id] = rank
    assert sorted(seen) == list(range(num_files))

    # Resuming skips whole files, whatever the number of ranks.
    resumed = FileBatchSampler(4, num_files, seqs_per_file, micro_batch_size, 0, 2)
    assert next(iter(resumed)) == [24, 25]
    with pytest.raises(AssertionError):
        FileBatchSampler(0, num_files, seqs_per_file, 4, 0, 1)
    # A rank past the last shard has nothing to do.
    assert list(FileBatchSampler(4, 5, seqs_per_file, micro_batch_size, 3, 4)) == []


def test_async_shard_writer(tmp_path):
    batch = _batch()
    tokens, ids = batch[0], batch[5]
    payload = encode_chunk(*batch, 'fp32')
    writer = AsyncShardWriter(str(tmp_path), num_threads=2, compress_level=1, max_pending=2)
    for step in range(3):
        writer.begin(step, 4 * step + 1, {'file_id': 4 * step + 1})
        for _ in range(5):
            writer.add(payload, tokens.shape[0], ids.shape[0])
        writer.end()
    writer.skip(3)  # A rank without a shard in the last round.
    assert writer.close() == 4
    for step in range(3):
        with ShardReader(file_path(str(tmp_path), 4 * step + 1)) as shard:
            assert shard.num_chunks == 5
            assert torch.equal(shard.chunk(4)['teacher_probs'], payload['teacher_probs'])


def test_async_shard_writer_raises_on_failure(tmp_path):
    (tmp_path / 'files').write_text('a file where the writer needs a directory')
    batch = _batch()
    tokens, ids = batch[0], batch[5]
    payload = encode_chunk(*batch, 'fp32')
    writer = AsyncShardWriter(str(tmp_path), num_threads=1, compress_level=0, max_pending=1)
    writer.begin(0, 0, {})
    for _ in range(4):  # More chunks than the queue holds: the failed shard must not block.
        writer.add(payload, tokens.shape[0], ids.shape[0])
    writer.end()
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
    labels = torch.randint(0, 30, (10,), generator=g)
    labels[3] = 35  # Outside the scored vocabulary.
    ids, probs, entropy, label_logprob = teacher_rows(
        hidden, labels, weight, None, vocab_limit=30, mode='topk', slots=5, chunk=chunk
    )
    ref = torch.softmax(hidden @ weight[:30].T, dim=-1)
    ref_probs, ref_ids = torch.topk(ref, 5, dim=-1)
    assert ids.dtype == torch.int32 and ids.shape == (10, 5)
    assert torch.equal(ids.long(), ref_ids)
    torch.testing.assert_close(probs, ref_probs)
    torch.testing.assert_close(entropy, torch.special.entr(ref).sum(-1))
    ref_label = ref.gather(1, labels.clamp(max=29).unsqueeze(1)).squeeze(1).log()
    keep = labels < 30
    torch.testing.assert_close(label_logprob[keep], ref_label[keep])
    assert torch.isneginf(label_logprob[3])


def test_teacher_rows_sample(single_tp_rank):
    g = torch.Generator().manual_seed(0)
    hidden = torch.randn(6, 8, generator=g)
    weight = torch.randn(40, 8, generator=g)
    bias = torch.randn(40, generator=g)
    labels = torch.randint(0, 30, (6,), generator=g)
    ids, freqs, entropy, label_logprob = teacher_rows(
        hidden, labels, weight, bias, vocab_limit=30, mode='sample', slots=16, chunk=4,
        generator=torch.Generator().manual_seed(3),
    )
    assert ids.shape == freqs.shape == (6, 16)
    torch.testing.assert_close(freqs.sum(-1), torch.ones(6))
    assert (ids < 30).all()
    ref = torch.softmax(hidden @ weight[:30].T + bias[:30], dim=-1)
    torch.testing.assert_close(entropy, torch.special.entr(ref).sum(-1))
    torch.testing.assert_close(label_logprob, ref.gather(1, labels.unsqueeze(1)).squeeze(1).log())


def test_teacher_rows_empty(single_tp_rank):
    ids, probs, entropy, label_logprob = teacher_rows(
        torch.zeros(0, 8), torch.zeros(0, dtype=torch.long), torch.randn(40, 8), None, 30, 'topk',
        5, 4
    )
    assert ids.shape == (0, 5) and probs.shape == (0, 5)
    assert entropy.shape == label_logprob.shape == (0,)


def _load_calibration_blend():
    path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'examples',
                        'apertus_distillation', 'calibration_blend.py')
    spec = importlib.util.spec_from_file_location('calibration_blend', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_dataset(prefix, tokens):
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    with open(prefix + '.idx', 'wb') as f:
        f.write(b'MMIDIDX\x00\x00' + struct.pack('<Q', 1) + struct.pack('<B', 4))
    with open(prefix + '.bin', 'wb') as f:
        f.write(b'\x00' * 4 * tokens)


def test_calibration_blend(tmp_path):
    module = _load_calibration_blend()
    _fake_dataset(str(tmp_path / 'text' / 'web'), 300)
    _fake_dataset(str(tmp_path / 'text' / 'apertus_sft_filtered' / 'chat'), 100)
    _fake_dataset(str(tmp_path / 'text' / 'bad' / 'broken'), 50)
    _fake_dataset(str(tmp_path / 'image' / 'captions'), 10)
    groups = [('text', 0.6, [str(tmp_path / 'text')]), ('image', 0.4, [str(tmp_path / 'image')])]
    entries = dict((prefix, weight) for weight, prefix in
                   module.blend(groups, exclude=['*/bad/*']))
    assert entries == pytest.approx({
        'pretrain:' + str(tmp_path / 'text' / 'web'): 0.45,
        'sft:' + str(tmp_path / 'text' / 'apertus_sft_filtered' / 'chat'): 0.15,
        'pretrain:' + str(tmp_path / 'image' / 'captions'): 0.4,
    })
