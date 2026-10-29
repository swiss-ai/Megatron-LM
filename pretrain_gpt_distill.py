"""Train a student from a teacher cache (Apertus 1.5 offline distillation).

Sequences and teacher distributions come from a cache written by get_teacher_logits.py,
which holds the teacher pipeline's own tokens and masks, instead of the GPT datasets
(data order: megatron/training/distillation/cache_reader.py). At every position with
loss, weighted by the cache's loss mask, the student minimizes

    kd_weight * KD + (1 - kd_weight) * CE

with CE the cross-entropy on the label over the student's whole vocabulary and KD the
cross-entropy against the teacher over the cache's ``vocab_limit`` ids
(megatron/training/distillation/losses.py):

* sample cache: an unbiased estimate of the cross-entropy against the teacher's full
  distribution;
* top-k cache: the Apertus-v1.1 loss, or with ``--distill-tail`` the tail-bucket loss.

Validation reads the same held-out sequences at every evaluation
(``--distill-valid-start``, optionally from ``--distill-valid-cache-dir``) and reports,
besides both losses, the KL from the teacher (unbiased on a sample cache, a lower bound
on a top-k one), the teacher's own loss on the labels, and the student's top-1
confidence and accuracy.

Supports TP = PP = CP = 1, which the students fit, and a fixed global batch.
"""

import os
from functools import partial

import torch

import pretrain_gpt
from gpt_builders import gpt_builder
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.training import get_args, get_timers, get_tokenizer, pretrain, print_rank_0
from megatron.training.distillation import add_distillation_args
from megatron.training.distillation.cache_reader import CacheSequences, build_cache_iterator
from megatron.training.distillation.losses import sparse_distillation, validation_terms
from model_provider import model_provider

# Manifest keys that must match for two caches' sequence numbers to mean the same data.
DATA_ORDER_KEYS = ('data', 'seed', 'seq_length', 'num_files', 'seqs_per_file', 'tokenizer')

# Manifests of the training and validation caches, set by the dataset provider.
_MANIFESTS = {}


def _check_args(args, manifest):
    assert args.dataloader_type == 'external', '--dataloader-type must be external'
    assert (args.tensor_model_parallel_size == args.pipeline_model_parallel_size
            == args.context_parallel_size == 1), 'only TP = PP = CP = 1 is supported'
    assert args.rampup_batch_size is None, 'batch size ramp-up is not supported'
    assert args.train_iters, '--train-iters is required'
    assert args.use_packed_seq_params, (
        '--use-packed-seq-params is required so documents do not attend to each other, as '
        'in the teacher\'s runs'
    )
    assert args.mtp_num_layers is None, 'multi-token prediction is not supported'
    assert manifest['mode'] in ('topk', 'sample'), f'{manifest["mode"]} caches have no teacher'
    assert args.seq_length == manifest['seq_length'], (
        f'--seq-length {args.seq_length} differs from the cache\'s {manifest["seq_length"]}'
    )
    assert args.distill_vocab_size in (None, manifest['vocab_limit']), (
        f'--distill-vocab-size differs from the cache\'s {manifest["vocab_limit"]}'
    )
    assert 0 <= args.distill_kd_weight <= 1, '--distill-kd-weight must be in [0, 1]'
    assert args.consumed_train_samples % args.global_batch_size == 0
    if manifest.get('tokenizer') and os.path.realpath(manifest['tokenizer']) != os.path.realpath(
            args.tokenizer_model):
        print_rank_0(f'WARNING: the cache was written with tokenizer {manifest["tokenizer"]}, '
                     f'this run uses {args.tokenizer_model}')


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    """Iterators over this rank's micro-batches of the training and validation caches."""
    args = get_args()
    train_cache = CacheSequences(args.distill_cache_dir)
    _check_args(args, train_cache.manifest)
    _MANIFESTS['train'] = train_cache.manifest
    batch = args.global_batch_size
    common = dict(global_batch=batch, micro_batch_size=args.micro_batch_size,
                  dp_rank=mpu.get_data_parallel_rank(), dp_size=mpu.get_data_parallel_world_size(),
                  num_workers=args.num_workers)
    first_step = args.consumed_train_samples // batch
    train = build_cache_iterator(train_cache, range(first_step, args.train_iters), **common)
    print_rank_0(f'> training on sequences {first_step * batch}..{args.train_iters * batch - 1} '
                 f'of {args.distill_cache_dir} ({train_cache.manifest["mode"]}, '
                 f'{train_cache.manifest["slots"]} slots)')

    valid = None
    if args.eval_iters > 0:
        valid_dir = args.distill_valid_cache_dir or args.distill_cache_dir
        valid_cache = CacheSequences(valid_dir)
        manifest = valid_cache.manifest
        assert args.distill_valid_start is not None, 'validation needs --distill-valid-start'
        assert manifest['mode'] in ('topk', 'sample'), f'{valid_dir} has no teacher'
        assert manifest['vocab_limit'] == train_cache.manifest['vocab_limit']
        if all(manifest.get(k) == train_cache.manifest.get(k) for k in DATA_ORDER_KEYS):
            assert args.distill_valid_start >= args.train_iters * batch, (
                'validation sequences overlap the training sequences'
            )
        _MANIFESTS['valid'] = manifest
        valid = build_cache_iterator(valid_cache, range(args.eval_iters),
                                     base=args.distill_valid_start, cycle=True, **common)
        print_rank_0(f'> validating on sequences {args.distill_valid_start}..'
                     f'{args.distill_valid_start + args.eval_iters * batch - 1} of {valid_dir} '
                     f'({manifest["mode"]}, {manifest["slots"]} slots)')
    return train, valid, None


def get_batch(data_iterator):
    """Move a micro-batch to the GPU and pack its documents as in the teacher's run."""
    args = get_args()
    data = next(data_iterator)
    batch = {key: value.cuda(non_blocking=True) for key, value in data.items()}
    packed_seq_params, batch = pretrain_gpt._packed_seq_params_from_batch(
        batch, args, get_tokenizer(), 1, torch.cuda.current_device()
    )
    return batch, packed_seq_params


def _sum_and_weight(values, weights):
    return torch.stack([torch.sum(weights * values), weights.sum()]).detach()


def loss_func(batch, output_tensor):
    """Distillation and label losses of one micro-batch; see the module docstring."""
    args = get_args()
    training = torch.is_grad_enabled()
    manifest = _MANIFESTS['train' if training else 'valid']
    topk = manifest['mode'] == 'topk'

    weights = batch['loss_mask'].reshape(-1)
    rows = weights > 0
    weights = weights[rows]
    labels = batch['labels'].reshape(-1)[rows]
    logits = output_tensor.reshape(-1, output_tensor.shape[-1])[rows]
    ids, probs = batch['teacher_ids'], batch['teacher_probs']
    kd, ce = sparse_distillation(logits, labels, ids, probs, manifest['vocab_limit'],
                                 tail=args.distill_tail and topk, chunk=args.distill_loss_chunk)
    kd_weight = args.distill_kd_weight
    loss = torch.sum(weights * (kd_weight * kd + (1 - kd_weight) * ce))
    num_tokens = weights.sum().clone().detach().to(torch.int)
    report = {'lm loss': _sum_and_weight(ce, weights), 'kd loss': _sum_and_weight(kd, weights)}

    if not training:
        terms = validation_terms(logits, labels, ids, probs, manifest['vocab_limit'], topk,
                                 batch['teacher_entropy'], args.distill_loss_chunk)
        report['kl to teacher'] = _sum_and_weight(terms['kl'], weights)
        report['top-1 confidence'] = _sum_and_weight(terms['confidence'], weights)
        report['top-1 accuracy'] = _sum_and_weight(terms['correct'], weights)
        label_logprob = batch['teacher_label_logprob']
        in_vocab = torch.isfinite(label_logprob)
        report['teacher lm loss'] = _sum_and_weight(
            torch.where(in_vocab, -label_logprob, 0.0), weights * in_vocab
        )

    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        for rejection_func, what in ((torch.isnan, 'NaN'), (torch.isinf, 'Inf')):
            rerun_state_machine.validate_result(
                result=loss,
                rejection_func=rejection_func,
                message=f'found {what} in local forward loss calculation',
                tolerance=0.0,  # forward pass calculations are determinisic
                fatal=True,
            )
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=pretrain_gpt.SPIKY_LOSS_FACTOR,
                context='loss',
            ),
            message='Spiky loss',
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=False,
        )
    return loss, num_tokens, report


def forward_step(data_iterator, model):
    """Forward step: the student's logits, scored by ``loss_func``."""
    timers = get_timers()
    timers('batch-generator', log_level=2).start()
    batch, packed_seq_params = get_batch(data_iterator)
    timers('batch-generator').stop()
    output_tensor = model(batch['tokens'], batch['position_ids'], None,
                          packed_seq_params=packed_seq_params)
    return output_tensor, partial(loss_func, batch)


if __name__ == '__main__':
    train_valid_test_datasets_provider.is_distributed = True
    pretrain(
        train_valid_test_datasets_provider,
        partial(model_provider, gpt_builder),
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'dataloader_type': 'external'},
        extra_args_provider=add_distillation_args,
    )
