"""Write a teacher cache for offline distillation (Apertus 1.5).

Runs a frozen teacher over the regular Megatron data pipeline, with the same data
flags as the teacher's training run (blend, SFT packing, masks), and stores at
every position with loss either the teacher's top-k tokens or a set of sampled
tokens. See megatron/training/distillation/cache_format.py for the layout.

Each data-parallel rank writes one file per step: file ``f`` holds sequences
``f * micro_batch_size ...`` of the data order, independent of the number of GPUs.
``progress.json`` records the completed files, and a rerun with the same
arguments resumes after them.
"""

import math
import os
import subprocess
import time
from functools import partial

import torch

import pretrain_gpt
from gpt_builders import gpt_builder
from megatron.core import mpu
from megatron.core.utils import unwrap_model
from megatron.training import get_args, get_model, get_tokenizer, print_rank_0
from megatron.training.checkpointing import load_checkpoint
from megatron.training.distillation import add_distillation_args
from megatron.training.distillation.arguments import distill_slots
from megatron.training.distillation.cache_format import (
    FORMAT_NAME,
    FORMAT_VERSION,
    check_manifest_compatible,
    encode_file,
    load_manifest,
    load_progress,
    write_manifest,
    write_progress,
)
from megatron.training.distillation.cache_writer import AsyncFileWriter
from megatron.training.distillation.scoring import teacher_rows
from megatron.training.initialize import initialize_megatron
from megatron.training.training import build_train_valid_test_data_iterators, update_train_iters
from model_provider import model_provider

# Data arguments that change which tokens, masks or order the pipeline produces.
DATA_IDENTITY_ARGS = (
    'data_path', 'data_args_path', 'per_split_data_args_path', 'train_data_path', 'split',
    'seq_length', 'reset_position_ids', 'reset_attention_mask', 'eod_mask_loss',
    'goldfish_loss', 'goldfish_k', 'goldfish_h', 'vision_weight', 'audio_weight',
    'loss_mask_token_ids', 'ap_sft', 'ap_sft_plw', 'ap_sft_mask_special_tokens',
    'ap_sft_pack_samples', 'ap_sft_packing_strategy', 'ap_sft_equalize_sample_loss',
    'ap_sft_long_ctx_loss', 'ap_sft_truncate_right', 'ap_sft_load_loss_mask',
    'pretraining_packing_strategy', 'max_docs_per_bin', 'max_docs_per_bin_sft',
)


def _check_args(args):
    assert args.distill_cache_dir, '--distill-cache-dir is required'
    assert args.distill_cache_num_samples, '--distill-cache-num-samples is required'
    assert args.load, '--load must point to the teacher checkpoint'
    assert args.pipeline_model_parallel_size == 1, 'the writer supports PP=1 only'
    assert args.context_parallel_size == 1, 'the writer supports CP=1 only'
    assert not args.sequence_parallel, 'the writer does not support sequence parallelism'
    assert args.use_packed_seq_params, (
        '--use-packed-seq-params is required so documents do not attend to each other, '
        'as in the Apertus 1.5 training runs'
    )
    assert args.mtp_num_layers is None, 'multi-token prediction is not supported'
    assert args.global_batch_size == args.micro_batch_size * args.data_parallel_size, (
        'the writer scores one micro-batch per rank per step: leave --global-batch-size '
        'unset or set it to micro-batch-size x data-parallel-size'
    )


def _git_commit():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return None


def _manifest(args, vocab_limit, slots, num_files):
    return {
        'format': FORMAT_NAME,
        'version': FORMAT_VERSION,
        'mode': args.distill_cache_mode,
        'slots': slots,
        'seq_length': args.seq_length,
        'seqs_per_file': args.micro_batch_size,
        'vocab_limit': vocab_limit,
        'prob_dtype': args.distill_prob_dtype,
        'seed': args.seed,
        'sample_seed': args.distill_sample_seed if args.distill_cache_mode == 'sample' else None,
        'data': {key: getattr(args, key, None) for key in DATA_IDENTITY_ARGS},
        'tokenizer': args.tokenizer_model,
        'teacher': {'load': os.path.realpath(args.load), 'ckpt_step': args.ckpt_step},
        'num_files': num_files,
        'num_samples': num_files * args.micro_batch_size,
        'writer': {
            'commit': _git_commit(),
            'tensor_model_parallel_size': args.tensor_model_parallel_size,
            'data_parallel_size': args.data_parallel_size,
            'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
        },
    }


def _prepare_cache_dir(cache_dir, manifest):
    existing = load_manifest(cache_dir)
    if existing is None:
        write_manifest(cache_dir, manifest)
        print_rank_0(f'> new teacher cache in {cache_dir}')
    else:
        check_manifest_compatible(existing, manifest)
        print_rank_0(f'> resuming teacher cache in {cache_dir}')


def _sync_progress(args, writer, start_file, dp_size, num_files, final=False):
    """Record the files every rank has finished, as a contiguous prefix."""
    if writer is not None:
        local = writer.close() if final else writer.completed_steps()
    else:
        local = 2**62  # TP ranks > 0 write nothing and never hold progress back.
    done = torch.tensor([local], dtype=torch.long, device='cuda')
    torch.distributed.all_reduce(done, op=torch.distributed.ReduceOp.MIN)
    completed = min(num_files, start_file + int(done.item()) * dp_size)
    if torch.distributed.get_rank() == 0:
        write_progress(args.distill_cache_dir, completed, dp_size)
    return completed


def _self_check(gpt, hidden_rows, ids, probs, vocab_limit, mode):
    """Compare chunked scoring with the model's own output layer on a few rows.

    Collective over the tensor-parallel group: every TP rank must call it (they
    all hold the same rows, so they all return early together).
    """
    n = min(8, hidden_rows.shape[0])
    if n == 0:
        return
    weight = gpt.shared_embedding_or_output_weight() if gpt.share_embeddings_and_output_weights else None
    logits, _ = gpt.output_layer(hidden_rows[:n].unsqueeze(1), weight=weight, runtime_gather_output=True)
    ref = torch.softmax(logits.squeeze(1)[:, :vocab_limit].float(), dim=-1)
    if mode == 'topk':
        ref_probs, ref_ids = torch.topk(ref, ids.shape[1], dim=-1)
        prob_err = (ref_probs - probs[:n].float()).abs().max().item()
        overlap = min(
            len(set(a.tolist()) & set(b.tolist())) for a, b in zip(ref_ids, ids[:n].long())
        )
        ok = prob_err < 5e-3 and overlap >= ids.shape[1] - 4
        detail = f'max prob error {prob_err:.2e}, min top-k overlap {overlap}/{ids.shape[1]}'
    else:
        sums = probs[:n].float().sum(-1)
        drawn = torch.gather(ref, 1, ids[:n].long())
        ok = bool(torch.allclose(sums, torch.ones_like(sums))) and bool(
            (drawn[probs[:n] > 0] > 0).all()
        )
        detail = f'frequency sums {sums.min().item():.4f}..{sums.max().item():.4f}'
    if mpu.get_tensor_model_parallel_rank() == 0:
        print(f'[rank {torch.distributed.get_rank()}] self-check {"passed" if ok else "FAILED"} '
              f'on {n} rows: {detail}', flush=True)
    assert ok, 'chunked teacher scoring disagrees with the output layer'


def main():
    initialize_megatron(
        extra_args_provider=add_distillation_args,
        args_defaults={'no_load_rng': True, 'no_load_optim': True, 'exit_on_missing_checkpoint': True},
    )
    args = get_args()
    _check_args(args)

    dp_size = mpu.get_data_parallel_world_size()
    dp_rank = mpu.get_data_parallel_rank()
    is_writer = mpu.get_tensor_model_parallel_rank() == 0
    tokenizer = get_tokenizer()
    vocab_limit = args.distill_vocab_size or tokenizer.vocab_size
    assert vocab_limit <= args.padded_vocab_size, (
        f'--distill-vocab-size {vocab_limit} exceeds the model vocabulary {args.padded_vocab_size}'
    )
    slots = distill_slots(args)
    mode = args.distill_cache_mode
    seqs_per_file = args.micro_batch_size
    num_files = math.ceil(args.distill_cache_num_samples / (seqs_per_file * dp_size)) * dp_size

    # Manifest and resume point.
    manifest = _manifest(args, vocab_limit, slots, num_files)
    if torch.distributed.get_rank() == 0:
        _prepare_cache_dir(args.distill_cache_dir, manifest)
    torch.distributed.barrier()
    start_file = (load_progress(args.distill_cache_dir) // dp_size) * dp_size
    if start_file >= num_files:
        print_rank_0(f'> cache already complete: {num_files} files')
        return

    # Teacher. Keep the decoder's final norm, but run the output layer in chunks
    # over the positions with loss only.
    model = get_model(partial(model_provider, gpt_builder), wrap_with_ddp=False)
    load_checkpoint(model, None, None, strict=True)
    gpt = unwrap_model(model)[0]
    gpt.eval()
    gpt.requires_grad_(False)
    gpt.post_process = False
    assert gpt.decoder.post_process, 'the decoder must apply the final norm'
    weight = (gpt.shared_embedding_or_output_weight() if gpt.share_embeddings_and_output_weights
              else gpt.output_layer.weight)
    bias = getattr(gpt.output_layer, 'bias', None)

    # Data: the same pipeline as training, skipping the sequences already cached.
    args.train_iters = None
    args.train_samples = num_files * seqs_per_file
    update_train_iters(args)
    args.iteration = 0
    args.consumed_train_samples = start_file * seqs_per_file
    args.eval_iters = 0
    pretrain_gpt.train_valid_test_datasets_provider.is_distributed = True
    train_iterator, _, _ = build_train_valid_test_data_iterators(
        pretrain_gpt.train_valid_test_datasets_provider
    )

    writer = (AsyncFileWriter(args.distill_cache_dir, args.distill_writer_threads,
                              args.distill_compress_level) if is_writer else None)
    generator = (torch.Generator(device=torch.device('cuda', torch.cuda.current_device()))
                 if mode == 'sample' else None)
    total_steps = (num_files - start_file) // dp_size
    print_rank_0(
        f'> scoring files {start_file}..{num_files - 1} in {total_steps} steps: mode={mode}, '
        f'slots={slots}, vocab_limit={vocab_limit}, {seqs_per_file} x {args.seq_length} tokens per file'
    )

    started = time.time()
    loss_rows = 0
    with torch.inference_mode():
        for step in range(total_steps):
            file_id = start_file + step * dp_size + dp_rank
            (tokens, labels, loss_mask, _, position_ids, assistant_mask), packed_seq_params = (
                pretrain_gpt.get_batch(train_iterator)
            )
            hidden = gpt(tokens, position_ids, None, packed_seq_params=packed_seq_params)
            rows = loss_mask.reshape(-1) > 0
            hidden_rows = hidden.reshape(-1, hidden.shape[-1])[rows]
            if generator is not None:
                generator.manual_seed(args.distill_sample_seed + file_id)
            ids, probs = teacher_rows(hidden_rows, weight, bias, vocab_limit, mode, slots,
                                      args.distill_logits_chunk, generator)
            if args.distill_self_check and step == 0:
                _self_check(gpt, hidden_rows, ids, probs, vocab_limit, mode)
            loss_rows += int(rows.sum())
            if writer is not None:
                shape = (seqs_per_file, args.seq_length)
                payload = encode_file(
                    tokens.view(shape), labels.view(shape), loss_mask.view(shape),
                    position_ids.view(shape),
                    None if assistant_mask is None else assistant_mask.view(shape),
                    ids, probs, args.distill_prob_dtype,
                )
                writer.submit(step, file_id, payload)
            del hidden, hidden_rows, ids, probs

            if (step + 1) % args.distill_progress_interval == 0 and step + 1 < total_steps:
                completed = _sync_progress(args, writer, start_file, dp_size, num_files)
                elapsed = time.time() - started
                tokens_done = (step + 1) * dp_size * seqs_per_file * args.seq_length
                print_rank_0(
                    f'> step {step + 1}/{total_steps}: {completed}/{num_files} files complete, '
                    f'{tokens_done / elapsed:,.0f} tokens/s, '
                    f'{loss_rows / ((step + 1) * seqs_per_file * args.seq_length):.1%} of positions '
                    f'stored on rank 0, ETA {elapsed / (step + 1) * (total_steps - step - 1) / 60:.1f} min'
                )

    completed = _sync_progress(args, writer, start_file, dp_size, num_files, final=True)
    print_rank_0(f'> done: {completed}/{num_files} files complete in {(time.time() - started) / 60:.1f} min')
    torch.distributed.barrier()


if __name__ == '__main__':
    main()
