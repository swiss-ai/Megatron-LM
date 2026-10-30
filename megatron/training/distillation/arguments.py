"""Command-line arguments of the teacher-cache writer and of students trained from a cache."""


def add_distillation_args(parser):
    """Add distillation arguments to a Megatron argument parser."""
    group = parser.add_argument_group(title='distillation')

    group.add_argument('--distill-cache-dir', type=str, default=None,
                       help='Directory of the teacher cache: written by get_teacher_logits.py, '
                       'read for training by pretrain_gpt_distill.py.')
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
    group.add_argument('--distill-cache-stop-files', type=int, default=None,
                       help='Write only shards 0 .. N-1 of the cache in this run. The data order '
                       'still covers --distill-cache-num-samples, so a later run without this '
                       'flag continues where this one stopped.')
    group.add_argument('--distill-heldout-blend', type=str, default=None,
                       help='Blend file with the weights of a held-out set, e.g. the pruning '
                       'calibration set. The writer builds the datasets of --data-args-path and '
                       'reads the sequences after the first --distill-heldout-after ones of '
                       'that blend, so the set shares no tokens with a cache of that blend.')
    group.add_argument('--distill-heldout-after', type=int, default=None,
                       help='Sequences of the --data-args-path blend to stay clear of: the '
                       'sequences of the training cache.')
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

    # Student.
    group.add_argument('--distill-kd-weight', type=float, default=0.9,
                       help='Weight of the distillation term; the cross-entropy on the labels '
                       'gets 1 - weight. 0 trains on the labels alone, from the same data.')
    group.add_argument('--distill-tail', action='store_true',
                       help='Top-k caches: add the teacher\'s mass outside the stored ids as one '
                       'tail bucket, -(1 - sum p) log(1 - sum q). Without it the loss is the '
                       'Apertus-v1.1 one, which pushes the student\'s mass there to zero. '
                       'No effect on sample caches, whose frequencies sum to 1.')
    group.add_argument('--distill-valid-cache-dir', type=str, default=None,
                       help='Cache to validate on (default: --distill-cache-dir), e.g. the sample '
                       'cache of the same data order, so every student is measured the same way.')
    group.add_argument('--distill-valid-start', type=int, default=None,
                       help='First validation sequence. Every evaluation reads the same '
                       'eval-iters x global-batch-size sequences from there; with the training '
                       'data order they must come after the last training sequence.')
    group.add_argument('--distill-loss-chunk', type=int, default=1024,
                       help='Positions per fp32 block in the distillation loss, which bounds its '
                       'memory to chunk x vocab values.')
    return parser


def distill_slots(args):
    """Number of (token, probability) slots stored per position."""
    return {'topk': args.distill_topk, 'sample': args.distill_samples, 'tokens': 0}[
        args.distill_cache_mode
    ]
