#!/usr/bin/env python3
"""Freeze distill-rest/blend.txt: what Stage P and the pruning set leave of Stage 3.

Each line gives a dataset and, as its weight, the sequences left in it. The writer takes
them through --distill-heldout-blend, reading each dataset on after both sets. The Stage P
blend reads every dataset for under one pass, so these are its sequences it never reached.
A rerun rebuilds the blend and stops if it differs from the frozen file. Needs numpy and
torch, and the dataset index of the Stage P blend.

    python3 examples/apertus_distillation/stage3_rest_blend.py --root CACHE_ROOT
"""

import argparse
import os
import sys

import check_stage_p


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', default=check_stage_p.DEFAULT_ROOT)
    args = parser.parse_args()
    left = check_stage_p.rest_of_stage3(args.root, os.path.join(args.root, 'dataset-index'))
    text = ''.join(f'{sequences} {prefix}\n' for prefix, sequences in left.items())
    output = os.path.join(args.root, 'distill-rest', 'blend.txt')
    if os.path.exists(output):
        with open(output) as f:
            if f.read() != text:
                sys.exit(f'{output} differs from the data on disk')
        return
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, 'w') as f:
        f.write(text)
    os.chmod(output, 0o444)
    print(f'{output}: {len(left)} datasets, {sum(left.values())} sequences')


if __name__ == '__main__':
    main()
