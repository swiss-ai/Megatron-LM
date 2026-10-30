#!/usr/bin/env python3
"""Checks of the Stage P data and caches, before and during the run (data_mix.md).

    blends   before any job, no GPU or torch needed: the frozen blends against the plan
             and the inventory, the input files, the teacher, the repository, the disk.
             launch_stage_p.sh runs it before every submission, so a rerun also finds
             input files that changed or vanished.
    cache    after a writer run: settings, finished shards, data mix, every dataset
             present, no repeated sequences.
    pruning  after the calibration set: the same, and no sequence shared with a cache.
    same     after the first shards of the real run: same data and teacher targets as the
             preflight cache, whatever the number of GPUs.

Every check prints PASS or FAIL, and the script exits with an error if any failed.

    python3 examples/apertus_distillation/check_stage_p.py blends --root CACHE_ROOT
"""

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import struct
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)

DEFAULT_ROOT = '/capstor/store/cscs/swissai/infra01/apertus_1p5_distillation_cache'
TEACHER = ('/capstor/store/cscs/swissai/infra01/apertus_checkpoints/v1p5/megatron/8B/pretraining/'
           'stage3_iter_0478000')
TARGET_TOKENS = 197031624704
SEQ_LEN = 8192
SEQUENCES = TARGET_TOKENS // SEQ_LEN
PRUNING_SEQUENCES = 1024
SPACE_NEEDED = 130e12  # bytes, ~110 TB of top-256 and a margin

# data_mix.md, sections 3 and 4: tokens (or shares) by teacher stage and domain.
MIX = {
    'distill': {('stage 3', 'text'): 112.50, ('stage 3', 'vision'): 34.34,
                ('stage 2', 'vision'): 3.82, ('stage 3', 'audio'): 46.02},
    'pruning': {('stage 3', 'text'): 50.0, ('stage 3', 'vision'): 31.5,
                ('stage 2', 'vision'): 3.5, ('stage 3', 'audio'): 15.0},
}
NUM_DATASETS = 483
STAGE2_TOKENS = {'img2txt_blip3_grounding_50m': 47.52e9, 'img2txt_recap_datacomp_1b': 243.35e9,
                 'img2txt_llava85m_midtrain': 153.11e9}
EXCLUDED = 'swiss-caselaw-preprocessed/dump-1/00019_tokens'
DTYPE_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 8, 7: 4, 8: 2}
IMAGE_IDS, SPEECH_IDS = (131272, 262344), (262344, 266440)
# finemath holds documents of a million tokens, some of them tables of zeros. Their 8k
# windows repeat without any fault in the blend, so sequences of so few distinct tokens
# are counted apart from real repeats.
PATTERN_TOKENS = 8

failures = []


def check(ok, what):
    print(f'  {"PASS" if ok else "FAIL"}  {what}', flush=True)
    if not ok:
        failures.append(what)


def read_blend(path):
    with open(path) as f:
        words = f.read().split()
    return [(float(weight), prefix) for weight, prefix in zip(words[0::2], words[1::2])]


def plain(prefix):
    return prefix.split(':', 1)[1]


def place(prefix):
    """(teacher stage, domain) of a blend entry, from its path."""
    if 'text_stage_3' in prefix:
        return 'stage 3', 'text'
    if 'audio-datasets' in prefix:
        return 'stage 3', 'audio'
    return ('stage 2' if 'stage2_tokenized' in prefix else 'stage 3'), 'vision'


def shares(entries):
    totals = {}
    for weight, prefix in entries:
        totals[place(prefix)] = totals.get(place(prefix), 0.0) + weight
    return totals


def read_index(prefix):
    """Tokens, sequences and documents of a dataset, from its files."""
    with open(prefix + '.idx', 'rb') as f:
        header = f.read(34)
    assert header[:9] == b'MMIDIDX\x00\x00', f'{prefix}.idx is not an index file'
    sequences, documents = struct.unpack('<QQ', header[18:34])  # An index of N + 1 boundaries.
    return os.path.getsize(prefix + '.bin') // DTYPE_SIZES[header[17]], sequences, documents - 1


def check_blend(name, entries):
    print(f'{name} blend')
    weights = sum(weight for weight, _ in entries)
    check(abs(weights - 1) < 1e-6, f'weights sum to 1 ({weights:.8f})')
    check(len(entries) == NUM_DATASETS, f'{len(entries)} datasets, expected {NUM_DATASETS}')
    files = [os.path.realpath(plain(prefix) + '.bin') for _, prefix in entries]
    check(len(set(files)) == len(files), 'no file appears twice')
    check(not any(EXCLUDED in file or EXCLUDED in prefix
                  for file, (_, prefix) in zip(files, entries)), 'the corrupt caselaw file is out')
    wrong = [p for _, p in entries if p.startswith('sft:') != ('apertus_sft' in p.lower())]
    check(not wrong, 'only the SFT data has the SFT loader')
    planned = MIX[name]
    total = sum(planned.values())
    measured = shares(entries)
    for key in sorted(planned):
        share, target = measured.get(key, 0.0) * 100, planned[key] / total * 100
        check(abs(share - target) < 0.02, f'{key[0]} {key[1]}: {share:.2f}% (plan {target:.2f}%)')
    check(set(measured) == set(planned), 'no other stage or domain')


def check_inputs(root, entries, inventory):
    print('input files')
    rows = {}
    with open(inventory) as f:
        for row in csv.DictReader(f):
            rows[os.path.realpath(row['resolved_path'] or row['path'])] = row
    missing, wrong = [], []
    used = set()
    for _, prefix in entries:
        path = plain(prefix)
        try:
            tokens, sequences, documents = read_index(path)
        except (OSError, AssertionError) as error:
            missing.append(f'{path}: {error}')
            continue
        name = os.path.basename(path)
        real = os.path.realpath(path + '.bin')
        if name in STAGE2_TOKENS:
            if abs(tokens - STAGE2_TOKENS[name]) > 0.01e9:
                wrong.append(path)
        elif real not in rows:
            wrong.append(f'{path}: not in the inventory')
        else:
            row, used = rows[real], used | {real}
            if (tokens, sequences, documents) != (int(row['tokens']), int(row['sequences']),
                                                  int(row['documents'])):
                wrong.append(path)
    check(not missing, f'all {len(entries)} datasets are readable' + (f': {missing[:3]}' if missing else ''))
    check(not wrong, 'tokens, sequences and documents match the inventory' + (f': {wrong[:3]}' if wrong else ''))
    check(not set(rows) - used, f'every inventory file is used ({len(used)} of {len(rows)})')

    snapshot = os.path.join(root, 'inputs.tsv')
    current = {}
    for _, prefix in entries:
        for suffix in ('.bin', '.idx'):
            real = os.path.realpath(plain(prefix) + suffix)
            if os.path.exists(real):
                stat = os.stat(real)
                current[real] = f'{stat.st_size}\t{int(stat.st_mtime)}'
    if not os.path.exists(snapshot):
        with open(snapshot, 'w') as f:
            f.writelines(f'{path}\t{value}\n' for path, value in sorted(current.items()))
        check(True, f'recorded {len(current)} input files in {snapshot}')
    else:
        with open(snapshot) as f:
            frozen = dict(line.rstrip('\n').split('\t', 1) for line in f)
        changed = [path for path, value in frozen.items() if current.get(path) != value]
        check(not changed, 'no input file changed since the first check'
              + (f': {changed[:3]}' if changed else ''))
    scratch = sum('/scratch/' in os.path.realpath(plain(p) + '.bin') for _, p in entries)
    print(f'  NOTE  {scratch} of {len(entries)} datasets live in personal scratch folders')


def blends(args):
    root = args.root
    paths = {name: os.path.join(root, name, 'blend.txt') for name in MIX}
    for name, path in paths.items():
        if not os.path.exists(path):
            sys.exit(f'{path} is missing: run stage_p_blends.sh')
    distill, pruning = read_blend(paths['distill']), read_blend(paths['pruning'])
    check_blend('distill', distill)
    check_blend('pruning', pruning)
    check_inputs(root, distill, args.inventory)

    print('separation')
    weights = dict((prefix, weight) for weight, prefix in distill)
    check(set(p for _, p in pruning) <= set(weights), 'the pruning blend only uses datasets of the distill blend')
    check(os.path.realpath(paths['distill']) != os.path.realpath(paths['pruning']), 'two separate blend files and folders')
    tight = 0.0
    for weight, prefix in pruning:
        tokens = read_index(plain(prefix))[0]
        needed = math.ceil(weights[prefix] * SEQUENCES) + math.ceil(weight * PRUNING_SEQUENCES)
        needed *= SEQ_LEN
        tight = max(tight, needed / tokens)
    check(tight <= 1, f'every dataset has sequences left for the pruning set (fullest {tight:.1%})')

    print('teacher, repository, disk')
    parts = os.listdir(TEACHER)
    check(any(n.endswith('.distcp') for n in parts) and '.metadata' in parts, f'teacher checkpoint: {len(parts)} files')
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=REPO).decode()
    commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=REPO).decode().strip()
    check(not dirty, f'no uncommitted changes to tracked files (commit {commit})')
    stat = os.statvfs(root)
    free = stat.f_bavail * stat.f_frsize
    check(free > SPACE_NEEDED, f'{free / 1e15:.2f} PB free on the cache file system')


def tokens_of(cache_dir, limit=None):
    """Tokens of the first finished sequences of a cache, chunk by chunk."""
    import torch
    from megatron.training.distillation.cache_format import ShardReader, file_path, load_progress
    seen = 0
    for file_id in range(load_progress(cache_dir)):
        with ShardReader(file_path(cache_dir, file_id)) as shard:
            for chunk in shard.chunks():
                yield chunk['tokens'].to(torch.int64)
                seen += chunk['tokens'].shape[0]
                if limit is not None and seen >= limit:
                    return


def sequence_counts(cache_dir, limit=None):
    """Sequences per domain, hashes of the sequences, and the number of mere patterns."""
    import torch
    counts, hashes, patterns = {'text': 0, 'vision': 0, 'audio': 0}, [], 0
    for tokens in tokens_of(cache_dir, limit):
        image = ((tokens >= IMAGE_IDS[0]) & (tokens < IMAGE_IDS[1])).any(1)
        speech = ((tokens >= SPEECH_IDS[0]) & (tokens < SPEECH_IDS[1])).any(1)
        for seq, is_image, is_speech in zip(tokens, image.tolist(), speech.tolist()):
            counts['vision' if is_image else 'audio' if is_speech else 'text'] += 1
            if len(torch.unique(seq)) <= PATTERN_TOKENS:
                patterns += 1
            else:
                hashes.append(hashlib.sha1(seq.numpy().tobytes()).digest())
    return counts, hashes, patterns


def check_settings(cache_dir, manifest, blend):
    print(f'settings of {cache_dir}')
    teacher = manifest.get('teacher') or {}
    for key, want in (('mode', 'topk'), ('slots', 256), ('prob_dtype', 'fp32'), ('seq_length', SEQ_LEN),
                      ('vocab_limit', 131072), ('seed', 41)):
        check(manifest.get(key) == want, f'{key} = {manifest.get(key)}')
    loaded = os.path.realpath(os.path.join(teacher.get('load', ''), 'iter_0478000'))
    check(teacher.get('ckpt_step') == 478000 and loaded == os.path.realpath(TEACHER),
          f'teacher {loaded}, step {teacher.get("ckpt_step")}')
    with open(blend, 'rb') as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    check(manifest['data'].get('data_args_path_sha256') == digest, 'the manifest holds the frozen blend')
    leftovers = glob.glob(os.path.join(cache_dir, 'files', '*', '*.tmp'))
    check(not leftovers, f'no half-written shards left ({len(leftovers)})')


def check_domains(counts, planned, tolerance):
    total = sum(counts.values())
    plan = {}
    for (_, domain), value in planned.items():
        plan[domain] = plan.get(domain, 0.0) + value
    for domain in sorted(plan):
        share, target = counts[domain] / total * 100, plan[domain] / sum(plan.values()) * 100
        check(abs(share - target) < tolerance, f'{domain}: {share:.1f}% of {total} sequences (plan {target:.1f}%)')


def check_datasets(args, entries):
    """The blend's own index: sequences per dataset over the whole cache."""
    import numpy
    expected = sum(math.ceil(SEQUENCES * weight) for weight, _ in entries)
    found = None
    for path in glob.glob(os.path.join(args.data_cache, '*dataset_index.npy')):
        index = numpy.load(path, mmap_mode='r')
        if index.shape[0] == expected:
            found = index
    if found is None:
        return check(False, f'no blend index of {expected} sequences in {args.data_cache}')
    counts = numpy.bincount(found[:SEQUENCES], minlength=len(entries))
    planned = numpy.array([weight * SEQUENCES for weight, _ in entries])
    off = numpy.abs(counts - planned)
    check((counts > 0).all(), f'all {len(entries)} datasets are in the cache, the smallest with {counts.min()} sequences')
    check((off <= numpy.maximum(1, 0.01 * planned)).all(),
          f'sequences per dataset within 1% of the plan, or one sequence (worst {off.max():.1f} sequences)')


def cache(args):
    from megatron.training.distillation.cache_format import load_manifest
    manifest = load_manifest(args.cache_dir)
    blend = os.path.join(args.root, 'distill', 'blend.txt')
    check_settings(args.cache_dir, manifest, blend)
    print('data')
    check(manifest['data'].get('distill_heldout_blend') is None, 'not a held-out set')
    counts, hashes, patterns = sequence_counts(args.cache_dir)
    check_domains(counts, MIX['distill'], 3.0)
    check(len(set(hashes)) == len(hashes), f'no repeated sequence in {len(hashes)}')
    print(f'  NOTE  {patterns} sequences of at most {PATTERN_TOKENS} distinct tokens left out of that check')
    check_datasets(args, read_blend(blend))


def pruning(args):
    from megatron.training.distillation.cache_format import load_manifest
    manifest = load_manifest(args.cache_dir)
    check_settings(args.cache_dir, manifest, os.path.join(args.root, 'distill', 'blend.txt'))
    print('data')
    data = manifest['data']
    with open(os.path.join(args.root, 'pruning', 'blend.txt'), 'rb') as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    check(data.get('distill_heldout_blend_sha256') == digest, 'the manifest holds the frozen pruning blend')
    check(data.get('distill_heldout_after') == SEQUENCES, f'held out after {data.get("distill_heldout_after")} sequences')
    counts, hashes, _ = sequence_counts(args.cache_dir)
    check(sum(counts.values()) == PRUNING_SEQUENCES, f'{sum(counts.values())} sequences')
    check_domains(counts, MIX['pruning'], 5.0)
    check(len(set(hashes)) == len(hashes), 'no repeated sequence')
    print('separation')
    other = set(sequence_counts(args.against)[1])
    check(not other & set(hashes), f'no sequence shared with the {len(other)} of {args.against}')


def same(args):
    import torch
    from megatron.training.distillation.cache_reader import CacheSequences
    a, b = CacheSequences(args.cache_dir), CacheSequences(args.against)
    total = min(a.num_sequences, b.num_sequences)
    step = a.seqs_per_chunk
    check(a.seqs_per_chunk == b.seqs_per_chunk and total > 0, f'comparing {total} sequences')
    exact = rows = 0
    for start in range(0, total, step):
        x, y = a[(start, start + step)], b[(start, start + step)]
        for key in ('tokens', 'labels', 'loss_mask', 'position_ids'):
            if not torch.equal(x[key], y[key]):
                check(False, f'{key} differ from sequence {start} on')
                return
        exact += int(((x['teacher_ids'] == y['teacher_ids']).all(1)
                      & (x['teacher_probs'] == y['teacher_probs']).all(1)).sum())
        rows += x['teacher_ids'].shape[0]
    check(True, 'identical data')
    check(exact / rows > 0.999, f'{exact / rows:.2%} of {rows} teacher rows are bit-identical')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('step', choices=['blends', 'cache', 'pruning', 'same'])
    parser.add_argument('cache_dir', nargs='?', help='The cache to check.')
    parser.add_argument('--root', default=DEFAULT_ROOT)
    parser.add_argument('--against', help='pruning, same: the other cache.')
    parser.add_argument('--inventory', default=os.path.join(REPO, '..', 'stage3', 'dataset-inventory-2026-10-28.csv'))
    parser.add_argument('--data-cache', default=os.path.join(REPO, '..', 'datasets', 'cache'))
    args = parser.parse_args()
    {'blends': blends, 'cache': cache, 'pruning': pruning, 'same': same}[args.step](args)
    print(f'{len(failures)} checks failed: {failures}' if failures else 'ALL CHECKS PASSED')
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
