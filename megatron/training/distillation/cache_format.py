"""On-disk layout of the teacher cache.

A cache directory holds:

* ``manifest.json``: format, layout and the settings that produced the cache.
* ``progress.json``: number of leading files that are completely written.
* ``files/<file_id // 10000>/<file_id>.pt.gz``: one file per writer micro-batch.

File ``f`` holds the sequences ``f * seqs_per_file ... (f + 1) * seqs_per_file - 1``
of the data pipeline's order. It is a gzipped ``torch.save`` of a dict with:

* ``tokens``, ``labels``, ``position_ids``: ``[seqs_per_file, seq_length]`` int32
* ``loss_mask``: ``[seqs_per_file, seq_length]`` float32, exactly as the data
  pipeline produced it
* ``assistant_mask``: optional ``[seqs_per_file, seq_length]`` bool
* ``teacher_ids``: ``[rows, slots]`` int32
* ``teacher_probs``: ``[rows, slots]`` float16 or float32

Teacher rows are stored only where ``loss_mask > 0``, in row-major order of
``(sequence, position)``; the reader re-expands them. Positions that carry no
loss never reach the loss, so storing them would only cost space.

In topk mode a row holds the teacher's most likely tokens and their
probabilities. In sample mode it holds the distinct tokens drawn and their
count / draws, padded with zero probability.
"""

import gzip
import json
import os
import time
from typing import Dict, Optional

import torch

FORMAT_NAME = 'apertus-distill-cache'
FORMAT_VERSION = 1
MANIFEST_NAME = 'manifest.json'
PROGRESS_NAME = 'progress.json'
FILES_PER_DIR = 10000

# Manifest keys that must match when a writer resumes into an existing cache.
IDENTITY_KEYS = (
    'format', 'version', 'mode', 'slots', 'seq_length', 'seqs_per_file', 'vocab_limit',
    'prob_dtype', 'seed', 'sample_seed', 'data', 'tokenizer', 'teacher', 'num_files',
)

PROB_DTYPES = {'fp16': torch.float16, 'fp32': torch.float32}


def file_path(cache_dir: str, file_id: int) -> str:
    """Path of cache file ``file_id``; files are spread over subdirectories."""
    return os.path.join(
        cache_dir, 'files', f'{file_id // FILES_PER_DIR:06d}', f'{file_id:010d}.pt.gz'
    )


def _write_json_atomic(path: str, payload: dict) -> None:
    tmp = f'{path}.{os.getpid()}.tmp'
    with open(tmp, 'w') as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_manifest(cache_dir: str, manifest: dict) -> None:
    """Write ``manifest.json`` atomically."""
    os.makedirs(cache_dir, exist_ok=True)
    _write_json_atomic(os.path.join(cache_dir, MANIFEST_NAME), manifest)


def load_manifest(cache_dir: str) -> Optional[dict]:
    """Return the manifest, or None if the directory has none."""
    path = os.path.join(cache_dir, MANIFEST_NAME)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        manifest = json.load(f)
    if manifest.get('format') != FORMAT_NAME:
        raise ValueError(f'{path} is not an {FORMAT_NAME} manifest')
    if manifest.get('version') != FORMAT_VERSION:
        raise ValueError(
            f'{path} has format version {manifest.get("version")}, expected {FORMAT_VERSION}'
        )
    return manifest


def check_manifest_compatible(existing: dict, new: dict) -> None:
    """Raise if a resumed writer would mix settings within one cache."""
    diffs = [
        f'{key}: cache has {existing.get(key)!r}, this run has {new.get(key)!r}'
        for key in IDENTITY_KEYS
        if existing.get(key) != new.get(key)
    ]
    if diffs:
        raise ValueError('Cache settings differ from this run:\n  ' + '\n  '.join(diffs))


def write_progress(cache_dir: str, completed_files: int, data_parallel_size: int) -> None:
    """Record that files ``0 .. completed_files - 1`` are complete."""
    _write_json_atomic(
        os.path.join(cache_dir, PROGRESS_NAME),
        {
            'completed_files': int(completed_files),
            'data_parallel_size': int(data_parallel_size),
            'updated': time.strftime('%Y-%m-%dT%H:%M:%S'),
        },
    )


def load_progress(cache_dir: str) -> int:
    """Number of leading files that are completely written (0 for a new cache)."""
    path = os.path.join(cache_dir, PROGRESS_NAME)
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return int(json.load(f)['completed_files'])


def encode_file(
    tokens: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    position_ids: torch.Tensor,
    assistant_mask: Optional[torch.Tensor],
    teacher_ids: torch.Tensor,
    teacher_probs: torch.Tensor,
    prob_dtype: str,
) -> Dict[str, torch.Tensor]:
    """Build the payload of one cache file from CPU or GPU tensors.

    ``tokens``, ``labels``, ``loss_mask``, ``position_ids`` and ``assistant_mask``
    are ``[seqs, seq_length]``; ``teacher_ids`` / ``teacher_probs`` are
    ``[rows, slots]`` with one row per position where ``loss_mask > 0``.
    """
    rows = int((loss_mask > 0).sum())
    if teacher_ids.shape[0] != rows or teacher_probs.shape != teacher_ids.shape:
        raise ValueError(
            f'teacher rows {tuple(teacher_ids.shape)} / {tuple(teacher_probs.shape)} '
            f'do not match {rows} loss positions'
        )
    payload = {
        'tokens': tokens.to('cpu', torch.int32),
        'labels': labels.to('cpu', torch.int32),
        'loss_mask': loss_mask.to('cpu', torch.float32),
        'position_ids': position_ids.to('cpu', torch.int32),
        'teacher_ids': teacher_ids.to('cpu', torch.int32),
        'teacher_probs': teacher_probs.to('cpu', PROB_DTYPES[prob_dtype]),
    }
    if assistant_mask is not None:
        payload['assistant_mask'] = assistant_mask.to('cpu', torch.bool)
    return payload


def save_file(path: str, payload: Dict[str, torch.Tensor], compress_level: int) -> None:
    """Write a payload atomically: a temporary file, then a rename."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f'{path}.{os.getpid()}.{time.monotonic_ns()}.tmp'
    try:
        if compress_level > 0:
            with gzip.open(tmp, 'wb', compresslevel=compress_level) as f:
                torch.save(payload, f)
        else:
            with open(tmp, 'wb') as f:
                torch.save(payload, f)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def load_file(path: str) -> Dict[str, torch.Tensor]:
    """Read a cache file (compressed or not)."""
    with open(path, 'rb') as f:
        magic = f.read(2)
    opener = gzip.open if magic == b'\x1f\x8b' else open
    with opener(path, 'rb') as f:
        return torch.load(f, map_location='cpu', weights_only=True)


class FileView:
    """Expands the compact teacher rows of one cache file, a few sequences at a time."""

    def __init__(self, payload: Dict[str, torch.Tensor]):
        self.payload = payload
        self.loss_rows = payload['loss_mask'] > 0
        counts = self.loss_rows.sum(dim=1)
        self.row_offsets = torch.zeros(counts.numel() + 1, dtype=torch.long)
        torch.cumsum(counts, dim=0, out=self.row_offsets[1:])
        if int(self.row_offsets[-1]) != payload['teacher_ids'].shape[0]:
            raise ValueError(
                f'cache file has {payload["teacher_ids"].shape[0]} teacher rows for '
                f'{int(self.row_offsets[-1])} loss positions'
            )

    @property
    def num_seqs(self) -> int:
        return self.payload['tokens'].shape[0]

    def sequences(self, start: int, end: int) -> Dict[str, torch.Tensor]:
        """Dense tensors for sequences ``start .. end - 1``.

        Teacher tensors are ``[n, seq_length, slots]``; positions without loss get
        zero probability, so they contribute nothing to the distillation term.
        """
        p = self.payload
        seq_length = p['tokens'].shape[1]
        slots = p['teacher_ids'].shape[1]
        n = end - start
        lo, hi = int(self.row_offsets[start]), int(self.row_offsets[end])
        rows = self.loss_rows[start:end].reshape(-1)
        teacher_ids = torch.zeros((n * seq_length, slots), dtype=torch.int32)
        teacher_probs = torch.zeros((n * seq_length, slots), dtype=torch.float32)
        teacher_ids[rows] = p['teacher_ids'][lo:hi]
        teacher_probs[rows] = p['teacher_probs'][lo:hi].float()
        batch = {
            'tokens': p['tokens'][start:end].long(),
            'labels': p['labels'][start:end].long(),
            'loss_mask': p['loss_mask'][start:end].float(),
            'position_ids': p['position_ids'][start:end].long(),
            'teacher_ids': teacher_ids.view(n, seq_length, slots),
            'teacher_probs': teacher_probs.view(n, seq_length, slots),
        }
        if 'assistant_mask' in p:
            batch['assistant_mask'] = p['assistant_mask'][start:end].float()
        else:
            batch['assistant_mask'] = torch.zeros((n, seq_length), dtype=torch.float32)
        return batch


def sample_distinct(
    probs: torch.Tensor, num_samples: int, generator: Optional[torch.Generator] = None
):
    """Draw ``num_samples`` tokens per row and return distinct ids with count / draws.

    Args:
        probs: ``[rows, vocab]`` probabilities (rows need not be normalized).

    Returns:
        ``(ids, freqs)``, both ``[rows, num_samples]``. Each row lists its distinct
        tokens first; the remaining slots have id 0 and frequency 0.
    """
    draws = torch.multinomial(probs, num_samples, replacement=True, generator=generator)
    draws, _ = torch.sort(draws, dim=-1)
    first = torch.ones_like(draws, dtype=torch.bool)
    first[:, 1:] = draws[:, 1:] != draws[:, :-1]
    group = torch.cumsum(first.long(), dim=-1) - 1
    counts = torch.zeros(draws.shape, dtype=torch.float32, device=draws.device)
    counts.scatter_add_(-1, group, torch.ones_like(counts))
    ids = torch.zeros_like(draws)
    ids.scatter_(-1, group, draws)
    return ids, counts / num_samples
