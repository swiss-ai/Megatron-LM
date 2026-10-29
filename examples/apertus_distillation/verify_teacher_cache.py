"""Check a teacher cache written by get_teacher_logits.py.

Reads the shards back and checks their index, and in every chunk the shapes,
dtypes, row counts, token ids, probabilities, entropies and label log-probabilities,
then prints summary statistics. With ``--compare`` it also checks that a second
cache holds the same data and close teacher distributions, e.g. a resumed run or a
run with a different parallel layout. A top-k and a sample cache of the same data
order are compared on their data, entropies and label log-probabilities.

Usage:
    python examples/apertus_distillation/verify_teacher_cache.py CACHE_DIR \
        [--files N] [--compare OTHER_CACHE_DIR]
"""

import argparse
import os
import random
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from megatron.training.distillation.cache_format import (  # noqa: E402
    PROB_DTYPES,
    TEACHER_KEYS,
    ShardReader,
    file_path,
    load_manifest,
    load_progress,
)

DATA_KEYS = ('tokens', 'labels', 'loss_mask', 'position_ids')
# Manifest keys that must match for two caches to hold the same data, chunk by chunk.
SAME_DATA_KEYS = ('seq_length', 'seqs_per_file', 'seqs_per_chunk', 'vocab_limit', 'data', 'seed')


def check_chunk(manifest, payload):
    """Validate one chunk and return its statistics."""
    seqs, seq_length = manifest['seqs_per_chunk'], manifest['seq_length']
    slots, vocab_limit = manifest['slots'], manifest['vocab_limit']
    for key, dtype in (('tokens', torch.int32), ('labels', torch.int32),
                       ('position_ids', torch.int32), ('loss_mask', torch.float32)):
        assert payload[key].dtype == dtype, f'{key} has dtype {payload[key].dtype}'
        assert payload[key].shape == (seqs, seq_length), f'{key} has shape {payload[key].shape}'
    rows = int((payload['loss_mask'] > 0).sum())
    has_loss = payload['loss_mask'] > 0
    labels = payload['labels'].long()
    stats = {
        'positions': seqs * seq_length,
        'rows': rows,
        'nontext_targets': int((has_loss & (labels >= vocab_limit)).sum()),
        'fractional_weights': int((has_loss & (payload['loss_mask'] < 1)).sum()),
        'assistant': int(payload['assistant_mask'].sum()) if 'assistant_mask' in payload else 0,
    }
    if manifest['mode'] == 'tokens':
        assert not any(key in payload for key in TEACHER_KEYS)
        return stats
    assert all(key in payload for key in TEACHER_KEYS), 'missing teacher fields'
    ids, probs = payload['teacher_ids'], payload['teacher_probs']
    entropy, label_logprob = payload['teacher_entropy'], payload['teacher_label_logprob']
    assert entropy.dtype == label_logprob.dtype == torch.float32
    assert entropy.shape == label_logprob.shape == (rows,), 'entropy / label log-prob shape'
    assert torch.isfinite(entropy).all() and (entropy >= -1e-4).all(), 'invalid entropy'
    row_labels = labels[has_loss]
    in_vocab = row_labels < vocab_limit
    assert torch.isfinite(label_logprob[in_vocab]).all() and (label_logprob <= 1e-4).all(), (
        'invalid label log-probability'
    )
    assert torch.isneginf(label_logprob[~in_vocab]).all(), 'non-text labels must have -inf'
    stats['entropy'] = entropy.sum().item()
    stats['label_nll'] = -label_logprob[in_vocab].sum().item()
    stats['text_rows'] = int(in_vocab.sum())
    assert ids.dtype == torch.int32 and probs.dtype == PROB_DTYPES[manifest['prob_dtype']]
    assert ids.shape == probs.shape == (rows, slots), (
        f'teacher shape {tuple(ids.shape)} for {rows} loss positions and {slots} slots'
    )
    p = probs.float()
    assert (ids >= 0).all() and (ids < vocab_limit).all(), 'teacher id outside [0, vocab_limit)'
    assert (p >= 0).all() and torch.isfinite(p).all(), 'negative or non-finite probability'
    sums = p.sum(-1)
    if rows:
        if manifest['mode'] == 'topk':
            assert (p[:, 1:] - p[:, :-1] <= 1e-3).all(), 'top-k probabilities are not sorted'
            assert (sums <= 1 + 1e-2).all(), f'top-k mass above 1: {sums.max().item()}'
            stats['mass'] = sums.sum().item()
            # The label log-prob is exact where the label is a stored id (above fp16's
            # subnormals); the stored ids alone have less entropy than the distribution.
            stored = ids.long() == row_labels[:, None]
            stats['label_stored'] = int(stored.any(-1).sum())
            check = stored & (p >= 1e-4)
            if check.any():
                error = (p[check].log() - label_logprob[check.any(-1)]).abs().max().item()
                assert error < 2e-3, f'label log-prob differs from its stored probability by {error:.1e}'
            partial_entropy = -torch.special.xlogy(p, p).sum(-1)
            assert (entropy >= partial_entropy - 1e-2).all(), 'entropy below that of the stored ids'
        else:
            counts = p * slots
            assert torch.allclose(counts, counts.round(), atol=1e-2), 'frequencies are not count / draws'
            assert torch.allclose(sums, torch.ones_like(sums), atol=1e-2), 'frequencies do not sum to 1'
            used = p > 0
            stats['distinct'] = used.sum().item()
            for row_ids, row_used in zip(ids[:200], used[:200]):
                kept = row_ids[row_used].tolist()
                assert len(set(kept)) == len(kept), 'repeated id among sampled tokens'
        stats['top1'] = p[:, 0].sum().item() if manifest['mode'] == 'topk' else p.max(-1).values.sum().item()
    return stats


def _top_ids(ids, probs):
    return ids.gather(1, probs.argmax(-1, keepdim=True)).squeeze(1)


def compare_chunks(manifest, a, b, same_mode=True, block=128):
    """Data must be identical; teacher distributions are compared by total variation,
    entropies and label log-probabilities by their largest difference."""
    for key in DATA_KEYS:
        assert torch.equal(a[key], b[key]), f'{key} differs'
    rows = int((a['loss_mask'] > 0).sum())
    result = {'rows': rows, 'top1_agree': rows, 'tv_sum': 0.0, 'tv_max': 0.0, 'exact': rows,
              'entropy_diff': 0.0, 'label_diff': 0.0}
    if manifest['mode'] == 'tokens' or rows == 0:
        return result
    result['entropy_diff'] = (a['teacher_entropy'] - b['teacher_entropy']).abs().max().item()
    finite = torch.isfinite(a['teacher_label_logprob'])
    assert torch.equal(finite, torch.isfinite(b['teacher_label_logprob'])), 'non-text rows differ'
    result['label_diff'] = (a['teacher_label_logprob'][finite]
                            - b['teacher_label_logprob'][finite]).abs().max().item() if finite.any() else 0.0
    if not same_mode:
        return result
    ids_a, ids_b = a['teacher_ids'].long(), b['teacher_ids'].long()
    p_a, p_b = a['teacher_probs'].float(), b['teacher_probs'].float()
    result.update(top1_agree=0, exact=0)
    result['exact'] = int(((ids_a == ids_b).all(-1) & (p_a == p_b).all(-1)).sum())
    result['top1_agree'] = int((_top_ids(ids_a, p_a) == _top_ids(ids_b, p_b)).sum())
    dense = torch.zeros(block, manifest['vocab_limit'])
    for start in range(0, rows, block):
        end = min(rows, start + block)
        diff = dense[:end - start].zero_()
        diff.scatter_add_(1, ids_a[start:end], p_a[start:end])
        diff.scatter_add_(1, ids_b[start:end], -p_b[start:end])
        tv = diff.abs().sum(-1) / 2
        result['tv_sum'] += tv.sum().item()
        result['tv_max'] = max(result['tv_max'], tv.max().item())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('cache_dir')
    parser.add_argument('--files', type=int, default=0, help='Shards to check (0: all completed shards).')
    parser.add_argument('--compare', default=None, help='Second cache to compare shard by shard.')
    parser.add_argument('--min-top1-agreement', type=float, default=0.95)
    parser.add_argument('--max-mean-tv', type=float, default=0.02,
                        help='Largest allowed mean total-variation distance per position.')
    parser.add_argument('--max-field-diff', type=float, default=0.1,
                        help='Largest allowed difference of an entropy or label log-probability '
                        '(nats).')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    manifest = load_manifest(args.cache_dir)
    assert manifest is not None, f'no manifest in {args.cache_dir}'
    completed = load_progress(args.cache_dir)
    assert 0 < completed <= manifest['num_files'], f'{completed} of {manifest["num_files"]} shards complete'
    file_ids = list(range(completed))
    if args.files and args.files < completed:
        file_ids = sorted({0, completed - 1, *random.Random(args.seed).sample(file_ids, args.files)})

    chunks_per_file = manifest['seqs_per_file'] // manifest['seqs_per_chunk']
    totals, size = {}, 0
    for file_id in file_ids:
        path = file_path(args.cache_dir, file_id)
        try:
            with ShardReader(path) as shard:
                index = shard.index
                assert index['file_id'] == file_id, f'index says shard {index["file_id"]}'
                assert index['first_sequence'] == file_id * manifest['seqs_per_file']
                assert shard.num_chunks == chunks_per_file, f'{shard.num_chunks} chunks'
                for chunk, payload in enumerate(shard.chunks()):
                    stats = check_chunk(manifest, payload)
                    assert stats['rows'] == index['chunks'][chunk]['rows'] or manifest['mode'] == 'tokens'
                    for key, value in stats.items():
                        totals[key] = totals.get(key, 0) + value
        except AssertionError as error:
            sys.exit(f'FAIL shard {file_id}: {error}')
        size += os.path.getsize(path)

    rows = max(totals['rows'], 1)
    print(f'{args.cache_dir}: mode={manifest["mode"]} slots={manifest["slots"]} '
          f'vocab_limit={manifest["vocab_limit"]} prob_dtype={manifest["prob_dtype"]}')
    print(f'  shards: {completed}/{manifest["num_files"]} complete, {len(file_ids)} checked, '
          f'{size / len(file_ids) / 2**30:.2f} GiB each on average')
    print(f'  positions with loss: {totals["rows"]:,} of {totals["positions"]:,} '
          f'({totals["rows"] / totals["positions"]:.1%}), {size / rows:,.0f} file bytes per such position')
    if 'mass' in totals:
        print(f'  mean top-{manifest["slots"]} mass {totals["mass"] / rows:.4f}, '
              f'mean top-1 prob {totals["top1"] / rows:.4f}')
    if 'distinct' in totals:
        print(f'  mean distinct ids {totals["distinct"] / rows:.1f} of {manifest["slots"]} draws, '
              f'mean max frequency {totals["top1"] / rows:.4f}')
    if 'entropy' in totals:
        print(f'  mean teacher entropy {totals["entropy"] / rows:.4f} nats, teacher loss on text '
              f'labels {totals["label_nll"] / max(totals["text_rows"], 1):.4f} nats'
              + (f', label among the stored ids at {totals["label_stored"] / rows:.1%} of positions'
                 if 'label_stored' in totals else ''))
    print(f'  loss positions with non-text targets: {totals["nontext_targets"]:,}, '
          f'with fractional weights: {totals["fractional_weights"]:,}, '
          f'assistant positions: {totals["assistant"]:,}')
    if manifest['mode'] == 'tokens':
        tokens_total = totals['positions']
        print(f'  {tokens_total:,} tokens in {len(file_ids)} shards, {size / tokens_total:.1f} bytes per token')

    if args.compare:
        other = load_manifest(args.compare)
        for key in SAME_DATA_KEYS:
            if manifest.get(key) != other.get(key):
                sys.exit(f'FAIL compare: {key} differs between the caches')
        same_mode = all(manifest.get(k) == other.get(k) for k in ('mode', 'slots'))
        if not same_mode and 'tokens' in (manifest['mode'], other['mode']):
            sys.exit('FAIL compare: a tokens cache can only be compared with another one')
        shared = [f for f in file_ids if f < load_progress(args.compare)]
        assert shared, 'the caches share no completed files'
        agg = {'rows': 0, 'top1_agree': 0, 'tv_sum': 0.0, 'tv_max': 0.0, 'exact': 0,
               'entropy_diff': 0.0, 'label_diff': 0.0}
        for file_id in shared:
            with ShardReader(file_path(args.cache_dir, file_id)) as a, \
                    ShardReader(file_path(args.compare, file_id)) as b:
                for chunk in range(a.num_chunks):
                    try:
                        result = compare_chunks(manifest, a.chunk(chunk), b.chunk(chunk), same_mode)
                    except AssertionError as error:
                        sys.exit(f'FAIL compare shard {file_id} chunk {chunk}: {error}')
                    for key in agg:
                        agg[key] = (max(agg[key], result[key]) if key.endswith(('_max', '_diff'))
                                    else agg[key] + result[key])
        n = max(agg['rows'], 1)
        top1, mean_tv = agg['top1_agree'] / n, agg['tv_sum'] / n
        fields = (f'max entropy difference {agg["entropy_diff"]:.2e}, max label log-prob '
                  f'difference {agg["label_diff"]:.2e}')
        if same_mode:
            print(f'  compare with {args.compare}: {len(shared)} shards, data identical, '
                  f'{agg["exact"] / n:.1%} rows bit-identical, top-1 agreement {top1:.2%}, '
                  f'mean TV distance {mean_tv:.2e}, max {agg["tv_max"]:.3f}, {fields}')
        else:
            print(f'  compare with {args.compare} ({other["mode"]}): {len(shared)} shards, data '
                  f'identical, {fields}')
        if top1 < args.min_top1_agreement or mean_tv > args.max_mean_tv:
            sys.exit('FAIL compare: teacher distributions differ more than allowed')
        if max(agg['entropy_diff'], agg['label_diff']) > args.max_field_diff:
            sys.exit('FAIL compare: entropies or label log-probabilities differ more than allowed')
    print('OK')


if __name__ == '__main__':
    main()
