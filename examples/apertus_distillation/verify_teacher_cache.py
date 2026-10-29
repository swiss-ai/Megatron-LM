"""Check a teacher cache written by get_teacher_logits.py.

Reads the files back and checks shapes, dtypes, row counts, token ids and
probabilities, then prints summary statistics. With ``--compare`` it also checks
that a second cache holds the same data and close teacher distributions, e.g. a
resumed run or a run with a different parallel layout.

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
    file_path,
    load_file,
    load_manifest,
    load_progress,
)

DATA_KEYS = ('tokens', 'labels', 'loss_mask', 'position_ids')


def check_file(manifest, payload):
    """Validate one payload and return its statistics."""
    seqs, seq_length = manifest['seqs_per_file'], manifest['seq_length']
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
        assert 'teacher_ids' not in payload and 'teacher_probs' not in payload
        return stats
    ids, probs = payload['teacher_ids'], payload['teacher_probs']
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


def compare_files(manifest, a, b, block=128):
    """Data must be identical; teacher distributions are compared by total variation."""
    for key in DATA_KEYS:
        assert torch.equal(a[key], b[key]), f'{key} differs'
    if manifest['mode'] == 'tokens':
        rows = int((a['loss_mask'] > 0).sum())
        return {'rows': rows, 'top1_agree': rows, 'tv_sum': 0.0, 'tv_max': 0.0, 'exact': rows}
    ids_a, ids_b = a['teacher_ids'].long(), b['teacher_ids'].long()
    p_a, p_b = a['teacher_probs'].float(), b['teacher_probs'].float()
    rows = ids_a.shape[0]
    result = {'rows': rows, 'top1_agree': 0, 'tv_sum': 0.0, 'tv_max': 0.0, 'exact': 0}
    if rows == 0:
        return result
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
    parser.add_argument('--files', type=int, default=0, help='Files to check (0: all completed files).')
    parser.add_argument('--compare', default=None, help='Second cache to compare file by file.')
    parser.add_argument('--min-top1-agreement', type=float, default=0.95)
    parser.add_argument('--max-mean-tv', type=float, default=0.02,
                        help='Largest allowed mean total-variation distance per position.')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    manifest = load_manifest(args.cache_dir)
    assert manifest is not None, f'no manifest in {args.cache_dir}'
    completed = load_progress(args.cache_dir)
    assert 0 < completed <= manifest['num_files'], f'{completed} of {manifest["num_files"]} files complete'
    file_ids = list(range(completed))
    if args.files and args.files < completed:
        file_ids = sorted({0, completed - 1, *random.Random(args.seed).sample(file_ids, args.files)})

    totals, size = {}, 0
    for file_id in file_ids:
        path = file_path(args.cache_dir, file_id)
        try:
            stats = check_file(manifest, load_file(path))
        except AssertionError as error:
            sys.exit(f'FAIL file {file_id}: {error}')
        size += os.path.getsize(path)
        for key, value in stats.items():
            totals[key] = totals.get(key, 0) + value

    rows = max(totals['rows'], 1)
    print(f'{args.cache_dir}: mode={manifest["mode"]} slots={manifest["slots"]} '
          f'vocab_limit={manifest["vocab_limit"]} prob_dtype={manifest["prob_dtype"]}')
    print(f'  files: {completed}/{manifest["num_files"]} complete, {len(file_ids)} checked')
    print(f'  positions with loss: {totals["rows"]:,} of {totals["positions"]:,} '
          f'({totals["rows"] / totals["positions"]:.1%}), {size / rows:,.0f} file bytes per such position')
    if 'mass' in totals:
        print(f'  mean top-{manifest["slots"]} mass {totals["mass"] / rows:.4f}, '
              f'mean top-1 prob {totals["top1"] / rows:.4f}')
    if 'distinct' in totals:
        print(f'  mean distinct ids {totals["distinct"] / rows:.1f} of {manifest["slots"]} draws, '
              f'mean max frequency {totals["top1"] / rows:.4f}')
    print(f'  loss positions with non-text targets: {totals["nontext_targets"]:,}, '
          f'with fractional weights: {totals["fractional_weights"]:,}, '
          f'assistant positions: {totals["assistant"]:,}')
    if manifest['mode'] == 'tokens':
        tokens_total = totals['positions']
        print(f'  {tokens_total:,} tokens in {len(file_ids)} files, {size / tokens_total:.1f} bytes per token')

    if args.compare:
        other = load_manifest(args.compare)
        for key in ('mode', 'slots', 'seq_length', 'seqs_per_file', 'vocab_limit', 'data', 'seed'):
            if manifest.get(key) != other.get(key):
                sys.exit(f'FAIL compare: {key} differs between the caches')
        shared = [f for f in file_ids if f < load_progress(args.compare)]
        assert shared, 'the caches share no completed files'
        agg = {'rows': 0, 'top1_agree': 0, 'tv_sum': 0.0, 'tv_max': 0.0, 'exact': 0}
        for file_id in shared:
            try:
                result = compare_files(manifest, load_file(file_path(args.cache_dir, file_id)),
                                       load_file(file_path(args.compare, file_id)))
            except AssertionError as error:
                sys.exit(f'FAIL compare file {file_id}: {error}')
            for key in agg:
                agg[key] = max(agg[key], result[key]) if key == 'tv_max' else agg[key] + result[key]
        n = max(agg['rows'], 1)
        top1, mean_tv = agg['top1_agree'] / n, agg['tv_sum'] / n
        print(f'  compare with {args.compare}: {len(shared)} files, data identical, '
              f'{agg["exact"] / n:.1%} rows bit-identical, top-1 agreement {top1:.2%}, '
              f'mean TV distance {mean_tv:.2e}, max {agg["tv_max"]:.3f}')
        if top1 < args.min_top1_agreement or mean_tv > args.max_mean_tv:
            sys.exit('FAIL compare: teacher distributions differ more than allowed')
    print('OK')


if __name__ == '__main__':
    main()
