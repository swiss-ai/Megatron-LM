"""On-disk layout of the teacher cache.

A cache directory holds:

* ``manifest.json``: format, layout and the settings that produced the cache.
* ``progress.json``: number of leading shard files that are completely written.
* ``files/<file_id // 1000>/<file_id>.zip``: large shard files.

Shard ``f`` holds the sequences ``f * seqs_per_file ... (f + 1) * seqs_per_file - 1``
of the data pipeline's order. It is an uncompressed zip archive (readable with
``unzip -l``) of independently gzipped chunks, so a reader can load one chunk
without decompressing the whole shard:

* ``00000.pt.gz``, ``00001.pt.gz``, ...: ``seqs_per_chunk`` sequences each.
* ``index.json``: shard id, first sequence, and per chunk its sequences and rows.

A chunk is a gzipped ``torch.save`` of a dict with:

* ``tokens``, ``labels``, ``position_ids``: ``[seqs_per_chunk, seq_length]`` int32
* ``loss_mask``: ``[seqs_per_chunk, seq_length]`` float32, exactly as the data
  pipeline produced it
* ``assistant_mask``: optional ``[seqs_per_chunk, seq_length]`` bool
* ``teacher_ids``: ``[rows, slots]`` int32 (absent in tokens mode)
* ``teacher_probs``: ``[rows, slots]`` float32 or float16 (absent in tokens mode)
* ``teacher_entropy``: ``[rows]`` float32, entropy of the teacher's full distribution
  (absent in tokens mode)
* ``teacher_label_logprob``: ``[rows]`` float32, the teacher's log-probability of the
  label, -inf for labels at or above ``vocab_limit`` (absent in tokens mode)

Teacher rows are stored only where ``loss_mask > 0``, in row-major order of
``(sequence, position)``; ``FileView`` re-expands them. Positions that carry no
loss never reach the loss, so storing them would only cost space.

In topk mode a row holds the teacher's most likely tokens and their
probabilities. In sample mode it holds the distinct tokens drawn and their
count / draws, padded with zero probability. Tokens mode stores the data only.

Entropy and label log-probability are exact in both modes. With them a reader can
turn a cross-entropy against the teacher into a KL divergence, and compare a
student's loss on the labels with the teacher's own.
"""

import contextlib
import glob
import gzip
import io
import json
import os
import time
import zipfile
from typing import Dict, Iterator, List, Optional

import torch

FORMAT_NAME = 'apertus-distill-cache'
FORMAT_VERSION = 3
MANIFEST_NAME = 'manifest.json'
PROGRESS_NAME = 'progress.json'
INDEX_NAME = 'index.json'
FILES_PER_DIR = 1000

# Manifest keys that must match when a writer resumes into an existing cache.
IDENTITY_KEYS = (
    'format', 'version', 'mode', 'slots', 'seq_length', 'seqs_per_file', 'seqs_per_chunk',
    'vocab_limit', 'prob_dtype', 'seed', 'sample_seed', 'data', 'tokenizer', 'teacher',
    'num_files',
)

PROB_DTYPES = {'fp16': torch.float16, 'fp32': torch.float32}

# Per-row teacher fields: (ids, probs) per slot, then one float32 value per row.
TEACHER_KEYS = ('teacher_ids', 'teacher_probs', 'teacher_entropy', 'teacher_label_logprob')


def file_path(cache_dir: str, file_id: int) -> str:
    """Path of shard ``file_id``; shards are spread over subdirectories."""
    return os.path.join(
        cache_dir, 'files', f'{file_id // FILES_PER_DIR:05d}', f'{file_id:08d}.zip'
    )


def chunk_name(chunk: int) -> str:
    return f'{chunk:05d}.pt.gz'


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
    """Record that shards ``0 .. completed_files - 1`` are complete."""
    _write_json_atomic(
        os.path.join(cache_dir, PROGRESS_NAME),
        {
            'completed_files': int(completed_files),
            'data_parallel_size': int(data_parallel_size),
            'updated': time.strftime('%Y-%m-%dT%H:%M:%S'),
        },
    )


def load_progress(cache_dir: str) -> int:
    """Number of leading shards that are completely written (0 for a new cache)."""
    path = os.path.join(cache_dir, PROGRESS_NAME)
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return int(json.load(f)['completed_files'])


def encode_chunk(
    tokens: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    position_ids: torch.Tensor,
    assistant_mask: Optional[torch.Tensor],
    teacher_ids: Optional[torch.Tensor],
    teacher_probs: Optional[torch.Tensor],
    teacher_entropy: Optional[torch.Tensor],
    teacher_label_logprob: Optional[torch.Tensor],
    prob_dtype: str,
) -> Dict[str, torch.Tensor]:
    """Build the payload of one chunk from CPU or GPU tensors.

    ``tokens``, ``labels``, ``loss_mask``, ``position_ids`` and ``assistant_mask``
    are ``[seqs, seq_length]``. The teacher fields have one row per position where
    ``loss_mask > 0``: ``teacher_ids`` / ``teacher_probs`` are ``[rows, slots]``,
    ``teacher_entropy`` / ``teacher_label_logprob`` are ``[rows]``. In tokens mode
    all four are None.
    """
    payload = {
        'tokens': tokens.to('cpu', torch.int32),
        'labels': labels.to('cpu', torch.int32),
        'loss_mask': loss_mask.to('cpu', torch.float32),
        'position_ids': position_ids.to('cpu', torch.int32),
    }
    teacher = (teacher_ids, teacher_probs, teacher_entropy, teacher_label_logprob)
    if any(t is None for t in teacher) and any(t is not None for t in teacher):
        raise ValueError('pass all teacher fields (ids, probs, entropy, label log-prob), or none')
    if teacher_ids is not None:
        rows = int((loss_mask > 0).sum())
        if (teacher_ids.shape[0] != rows or teacher_probs.shape != teacher_ids.shape
                or teacher_entropy.shape != (rows,) or teacher_label_logprob.shape != (rows,)):
            raise ValueError(
                f'teacher fields {[tuple(t.shape) for t in teacher]} do not match {rows} '
                f'loss positions'
            )
        payload['teacher_ids'] = teacher_ids.to('cpu', torch.int32)
        payload['teacher_probs'] = teacher_probs.to('cpu', PROB_DTYPES[prob_dtype])
        payload['teacher_entropy'] = teacher_entropy.to('cpu', torch.float32)
        payload['teacher_label_logprob'] = teacher_label_logprob.to('cpu', torch.float32)
    if assistant_mask is not None:
        payload['assistant_mask'] = assistant_mask.to('cpu', torch.bool)
    return payload


def compress_chunk(payload: Dict[str, torch.Tensor], compress_level: int) -> bytes:
    """Serialize and gzip one chunk payload."""
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return gzip.compress(buffer.getvalue(), compresslevel=compress_level)


def decompress_chunk(data: bytes) -> Dict[str, torch.Tensor]:
    return torch.load(io.BytesIO(gzip.decompress(data)), map_location='cpu', weights_only=True)


class ShardWriter:
    """Writes one shard: chunks in order, then the index, then an atomic rename."""

    def __init__(self, path: str, index: dict):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # One process writes a shard at a time, so temporary files of this shard are
        # left over from a killed attempt; each can be as large as a whole shard.
        for stale in glob.glob(glob.escape(path) + '.*.tmp'):
            with contextlib.suppress(FileNotFoundError):
                os.remove(stale)
        self.path = path
        self.tmp = f'{path}.{os.getpid()}.{time.monotonic_ns()}.tmp'
        self.index = dict(index, chunks=[])
        self.zip = zipfile.ZipFile(self.tmp, 'w', compression=zipfile.ZIP_STORED, allowZip64=True)

    def add(self, data: bytes, num_seqs: int, rows: int) -> None:
        chunk = len(self.index['chunks'])
        self.zip.writestr(chunk_name(chunk), data)
        self.index['chunks'].append({'num_seqs': num_seqs, 'rows': rows, 'bytes': len(data)})

    def close(self) -> None:
        self.zip.writestr(INDEX_NAME, json.dumps(self.index))
        self.zip.close()
        with open(self.tmp, 'rb+') as f:
            os.fsync(f.fileno())
        os.replace(self.tmp, self.path)

    def abort(self) -> None:
        self.zip.close()
        if os.path.exists(self.tmp):
            os.remove(self.tmp)


class ShardReader:
    """Random access to the chunks of one shard."""

    def __init__(self, path: str):
        self.path = path
        self.zip = zipfile.ZipFile(path, 'r')
        self.index = json.loads(self.zip.read(INDEX_NAME))

    @property
    def num_chunks(self) -> int:
        return len(self.index['chunks'])

    def chunk(self, chunk: int) -> Dict[str, torch.Tensor]:
        return decompress_chunk(self.zip.read(chunk_name(chunk)))

    def chunks(self) -> Iterator[Dict[str, torch.Tensor]]:
        for chunk in range(self.num_chunks):
            yield self.chunk(chunk)

    def close(self) -> None:
        self.zip.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class FileView:
    """Expands the compact teacher rows of one chunk, a few sequences at a time."""

    def __init__(self, payload: Dict[str, torch.Tensor]):
        self.payload = payload
        self.has_teacher = 'teacher_ids' in payload
        self.loss_rows = payload['loss_mask'] > 0
        counts = self.loss_rows.sum(dim=1)
        self.row_offsets = torch.zeros(counts.numel() + 1, dtype=torch.long)
        torch.cumsum(counts, dim=0, out=self.row_offsets[1:])
        if self.has_teacher and int(self.row_offsets[-1]) != payload['teacher_ids'].shape[0]:
            raise ValueError(
                f'chunk has {payload["teacher_ids"].shape[0]} teacher rows for '
                f'{int(self.row_offsets[-1])} loss positions'
            )

    @property
    def num_seqs(self) -> int:
        return self.payload['tokens'].shape[0]

    def sequences(self, start: int, end: int, dense: bool = True) -> Dict[str, torch.Tensor]:
        """Tensors for sequences ``start .. end - 1`` of the chunk.

        Teacher tensors, when the chunk has them, are ``[n, seq_length, slots]`` for
        ids and probabilities and ``[n, seq_length]`` for entropy and label
        log-probability. Positions without loss get zeros, so they contribute
        nothing to a distillation term. With ``dense=False`` they keep one row per
        position with loss instead, in row-major order of ``(sequence, position)``.
        """
        p = self.payload
        seq_length = p['tokens'].shape[1]
        n = end - start
        batch = {
            'tokens': p['tokens'][start:end].long(),
            'labels': p['labels'][start:end].long(),
            'loss_mask': p['loss_mask'][start:end].float(),
            'position_ids': p['position_ids'][start:end].long(),
        }
        if self.has_teacher and not dense:
            lo, hi = int(self.row_offsets[start]), int(self.row_offsets[end])
            batch['teacher_ids'] = p['teacher_ids'][lo:hi]
            batch['teacher_probs'] = p['teacher_probs'][lo:hi].float()
            batch['teacher_entropy'] = p['teacher_entropy'][lo:hi]
            batch['teacher_label_logprob'] = p['teacher_label_logprob'][lo:hi]
        elif self.has_teacher:
            slots = p['teacher_ids'].shape[1]
            lo, hi = int(self.row_offsets[start]), int(self.row_offsets[end])
            rows = self.loss_rows[start:end].reshape(-1)
            teacher_ids = torch.zeros((n * seq_length, slots), dtype=torch.int32)
            teacher_probs = torch.zeros((n * seq_length, slots), dtype=torch.float32)
            teacher_ids[rows] = p['teacher_ids'][lo:hi]
            teacher_probs[rows] = p['teacher_probs'][lo:hi].float()
            batch['teacher_ids'] = teacher_ids.view(n, seq_length, slots)
            batch['teacher_probs'] = teacher_probs.view(n, seq_length, slots)
            for key in ('teacher_entropy', 'teacher_label_logprob'):
                values = torch.zeros(n * seq_length, dtype=torch.float32)
                values[rows] = p[key][lo:hi]
                batch[key] = values.view(n, seq_length)
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
