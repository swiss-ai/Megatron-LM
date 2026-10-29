"""Command-line arguments of the teacher-cache writer."""


def add_distillation_args(parser):
    """Add distillation arguments to a Megatron argument parser."""
    group = parser.add_argument_group(title='distillation')

    group.add_argument('--distill-cache-dir', type=str, default=None,
                       help='Directory of the teacher cache written by get_teacher_logits.py.')
    group.add_argument('--distill-cache-mode', type=str, default='topk',
                       choices=['topk', 'sample', 'tokens'],
                       help='topk: store the teacher\'s k most likely tokens and their '
                       'probabilities. sample: draw --distill-samples tokens per position at '
                       'T=1 and store each distinct token with its count / draws, which is an '
                       'unbiased estimate of the full distribution. tokens: store the data '
                       'only (tokens, labels, masks), without loading a teacher, e.g. for a '
                       'pruning calibration set.')
    group.add_argument('--distill-topk', type=int, default=256,
                       help='Tokens stored per position in topk mode.')
    group.add_argument('--distill-samples', type=int, default=64,
                       help='Draws per position in sample mode.')
    group.add_argument('--distill-vocab-size', type=int, default=None,
                       help='Restrict teacher and student distributions to token ids below this '
                       'value, renormalizing over them (131072 keeps the Apertus 1.5 text '
                       'vocabulary). Defaults to the tokenizer vocabulary size.')
    group.add_argument('--distill-prob-dtype', type=str, default='fp32',
                       choices=['fp32', 'fp16'],
                       help='Storage dtype of cached teacher probabilities. fp32 matches the '
                       'Megatron-LM-Distill format; fp16 makes top-256 files about 17%% '
                       'smaller with at most 0.05%% relative error.')

    # Writer.
    group.add_argument('--distill-cache-num-samples', type=int, default=None,
                       help='Number of sequences to score, rounded up to whole shards. It '
                       'fixes the data '
                       'order, so set it to the largest cache you may want: a run can stop '
                       'early and resume later, but cannot grow a finished cache.')
    group.add_argument('--distill-seqs-per-file', type=int, default=1024,
                       help='Sequences per shard file, a multiple of --distill-seqs-per-chunk. '
                       'Shards are large on purpose, to keep the number of files low.')
    group.add_argument('--distill-seqs-per-chunk', type=int, default=32,
                       help='Sequences per independently compressed chunk inside a shard, a '
                       'multiple of the micro batch size. A reader loads one chunk at a time.')
    group.add_argument('--distill-logits-chunk', type=int, default=4096,
                       help='Positions per output-layer chunk when scoring, which bounds the '
                       'memory used for logits.')
    group.add_argument('--distill-writer-threads', type=int, default=8,
                       help='Background threads compressing chunks per rank.')
    group.add_argument('--distill-compress-level', type=int, default=1,
                       help='gzip compression level of chunks (0 stores them uncompressed).')
    group.add_argument('--distill-sample-seed', type=int, default=1234,
                       help='Base seed for sample mode. Each file uses seed + file id, so the '
                       'draws do not depend on the number of GPUs.')
    group.add_argument('--distill-progress-interval', type=int, default=1,
                       help='Shards per rank between progress records and log lines.')
    group.add_argument('--distill-exit-duration-in-mins', type=float, default=None,
                       help='Stop at a shard boundary when the next shard would end after this '
                       'many minutes, so a job ends before its time limit without losing a '
                       'half-written shard. A rerun resumes after it.')
    group.add_argument('--distill-self-check', action='store_true',
                       help='On the first step, recompute a few rows with the full output '
                       'layer and check them against the chunked scoring.')
    return parser


def distill_slots(args):
    """Number of (token, probability) slots stored per position."""
    return {'topk': args.distill_topk, 'sample': args.distill_samples, 'tokens': 0}[
        args.distill_cache_mode
    ]
