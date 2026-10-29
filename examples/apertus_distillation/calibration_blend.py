"""Write a weighted data blend for a pruning calibration set.

Each group (e.g. text, vision, audio) gets a fixed share of the blend; inside a
group, datasets are weighted by their token counts, as Megatron does for an
unweighted blend. SFT prefixes (paths containing ``apertus_sft``) are tagged
``sft:`` and the rest ``pretrain:``, as in the Apertus 1.5 stage-3 launcher.
The output file is meant for ``--data-args-path``.

Usage:
    python calibration_blend.py --output blend.txt \
        --group text 0.50 /path/text_stage_3 \
        --group vision 0.35 /path/vision/8B \
        --group audio 0.15 /path/audio \
        [--exclude '*/swiss-caselaw-preprocessed/dump-1/00019_tokens']
"""

import argparse
import fnmatch
import os
import struct
import sys

# Megatron indexed-dataset dtype codes and their sizes in bytes.
DTYPE_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 8, 7: 4, 8: 2}


def token_count(prefix):
    """Tokens in a Megatron .bin/.idx dataset, from the file size and the index dtype."""
    with open(prefix + '.idx', 'rb') as f:
        header = f.read(18)
    if header[:9] != b'MMIDIDX\x00\x00':
        raise ValueError(f'{prefix}.idx is not a Megatron index file')
    dtype_code = struct.unpack('<B', header[17:18])[0]
    return os.path.getsize(prefix + '.bin') // DTYPE_SIZES[dtype_code]


def prefixes(paths, exclude):
    """Dataset prefixes under the given directories or prefixes, sorted."""
    found = set()
    for path in paths:
        if os.path.isdir(path):
            for root, _, files in os.walk(path, followlinks=True):
                found.update(os.path.join(root, f[:-4]) for f in files if f.endswith('.bin'))
        else:
            found.add(path[:-4] if path.endswith('.bin') else path)
    return sorted(
        p for p in found
        if not any(fnmatch.fnmatch(os.path.realpath(p + '.bin')[:-4], pattern)
                   or fnmatch.fnmatch(p, pattern) for pattern in exclude)
    )


def blend(groups, exclude):
    """Return ``[(weight, tagged_prefix), ...]`` with group shares summing to 1."""
    total_share = sum(share for _, share, _ in groups)
    entries = []
    for name, share, paths in groups:
        found = prefixes(paths, exclude)
        if not found:
            raise ValueError(f'group {name}: no datasets under {paths}')
        counts = [token_count(p) for p in found]
        group_tokens = sum(counts)
        for prefix, count in zip(found, counts):
            tag = 'sft:' if 'apertus_sft' in prefix.lower() else 'pretrain:'
            entries.append((share / total_share * count / group_tokens, tag + prefix))
        print(f'{name}: {len(found)} datasets, {group_tokens / 1e9:.2f}B tokens, '
              f'share {share / total_share:.1%}', file=sys.stderr)
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--group', nargs='+', action='append', required=True,
                        metavar=('NAME SHARE', 'PATH'),
                        help='Group name, share of the blend, then directories or prefixes.')
    parser.add_argument('--exclude', action='append', default=[],
                        help='Glob of prefixes to leave out (matched on the path and its target).')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    groups = []
    for group in args.group:
        if len(group) < 3:
            parser.error('--group needs a name, a share and at least one path')
        groups.append((group[0], float(group[1]), group[2:]))
    entries = blend(groups, args.exclude)
    with open(args.output, 'w') as f:
        f.write(' '.join(f'{weight:.10g} {prefix}' for weight, prefix in entries) + '\n')
    print(f'wrote {len(entries)} weighted prefixes to {args.output}', file=sys.stderr)


if __name__ == '__main__':
    main()
