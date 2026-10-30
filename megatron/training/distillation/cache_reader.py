"""Reading a teacher cache for student training.

Training step ``s`` uses sequences ``s * G ... (s + 1) * G - 1`` of the cache, the data
order of the teacher's pipeline, for a global batch of ``G`` sequences. Data-parallel
rank ``r`` reads the contiguous block ``r`` of ``G / dp_size`` of them, so a chunk is
decompressed by one rank only when blocks hold whole chunks, and the order does not
depend on the number of GPUs: a run resumes at ``consumed_samples`` on any layout.

DataLoader workers read pieces of blocks that lie in one chunk; ``MicroBatches``
splits them into micro-batches in order. Teacher fields stay compact: one row per
position with loss, in row-major order of ``(sequence, position)``, the order of
``loss_mask > 0`` over the micro-batch.
"""

import itertools
import multiprocessing.util
import os
import tempfile
from typing import Dict, Iterator, Optional, Tuple

import torch

from .cache_format import FileView, ShardReader, file_path, load_manifest, load_progress


class CacheSequences(torch.utils.data.Dataset):
    """Sequences of the completed shards of a cache, indexed by ``(start, end)``.

    A range must lie within one chunk. Each process, forked workers included, keeps
    its own last shard open and its last chunk decompressed.
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.manifest = load_manifest(cache_dir)
        assert self.manifest is not None, f'no teacher cache in {cache_dir}'
        self.seqs_per_file = self.manifest['seqs_per_file']
        self.seqs_per_chunk = self.manifest['seqs_per_chunk']
        self.num_sequences = load_progress(cache_dir) * self.seqs_per_file
        self._shard: Optional[ShardReader] = None
        self._view: Optional[Tuple[Tuple[int, int], FileView]] = None
        self._pid = os.getpid()

    def __getstate__(self):
        # Workers open their own files.
        return dict(self.__dict__, _shard=None, _view=None)

    def __getitem__(self, span: Tuple[int, int]) -> Dict[str, torch.Tensor]:
        if self._pid != os.getpid():
            # A forked worker: the inherited shard shares its read position with the
            # parent and the other workers, so their reads would corrupt each other.
            self._shard, self._view, self._pid = None, None, os.getpid()
        start, end = span
        assert 0 <= start < end <= self.num_sequences, (
            f'sequences {start}..{end - 1} are not in the {self.num_sequences} completed '
            f'sequences of {self.cache_dir}'
        )
        file_id, offset = divmod(start, self.seqs_per_file)
        chunk, first = divmod(offset, self.seqs_per_chunk)
        assert first + end - start <= self.seqs_per_chunk, 'a range must lie within one chunk'
        if self._view is None or self._view[0] != (file_id, chunk):
            if self._shard is None or self._shard.index['file_id'] != file_id:
                if self._shard is not None:
                    self._shard.close()
                self._shard = ShardReader(file_path(self.cache_dir, file_id))
            self._view = ((file_id, chunk), FileView(self._shard.chunk(chunk)))
        return self._view[1].sequences(first, first + end - start, dense=False)


class RankSpans:
    """The ``(start, end)`` pieces one data-parallel rank reads, step after step.

    Args:
        steps: training steps to cover, e.g. ``range(first, last)``; ``cycle`` repeats
            them forever (validation reads the same sequences at every evaluation).
        base: first sequence of step 0.
    """

    def __init__(self, steps: range, global_batch: int, dp_rank: int, dp_size: int,
                 seqs_per_chunk: int, base: int = 0, cycle: bool = False):
        assert global_batch % dp_size == 0, (
            f'global batch {global_batch} is not a multiple of the data-parallel size {dp_size}'
        )
        self.steps, self.cycle = steps, cycle
        self.global_batch, self.per_rank = global_batch, global_batch // dp_size
        self.dp_rank, self.seqs_per_chunk, self.base = dp_rank, seqs_per_chunk, base

    def __iter__(self) -> Iterator[Tuple[int, int]]:
        steps = itertools.cycle(self.steps) if self.cycle else self.steps
        for step in steps:
            start = self.base + step * self.global_batch + self.dp_rank * self.per_rank
            end = start + self.per_rank
            while start < end:
                stop = min(end, (start // self.seqs_per_chunk + 1) * self.seqs_per_chunk)
                yield start, stop
                start = stop


class MicroBatches:
    """Splits the pieces a DataLoader yields into micro-batches, in order."""

    def __init__(self, pieces, micro_batch_size: int):
        self.pieces = iter(pieces)
        self.micro_batch_size = micro_batch_size
        self.piece: Optional[Dict[str, torch.Tensor]] = None
        self.offset = 0

    def __iter__(self):
        return self

    def __next__(self) -> Dict[str, torch.Tensor]:
        if self.piece is None or self.offset == self.piece['tokens'].shape[0]:
            self.piece = next(self.pieces)
            counts = (self.piece['loss_mask'] > 0).sum(dim=1)
            self.row_offsets = torch.cat([counts.new_zeros(1), torch.cumsum(counts, 0)])
            self.offset = 0
        start, end = self.offset, self.offset + self.micro_batch_size
        assert end <= self.piece['tokens'].shape[0], (
            'pieces must hold whole micro-batches: make the micro batch size divide the '
            'sequences per chunk and the per-rank batch'
        )
        self.offset = end
        lo, hi = int(self.row_offsets[start]), int(self.row_offsets[end])
        return {key: value[lo:hi] if key.startswith('teacher_') else value[start:end]
                for key, value in self.piece.items()}


def check_worker_tmpdir(num_workers: int):
    """Fail at once where DataLoader workers would hang.

    Workers hand tensors to the main process through Unix sockets in the multiprocessing
    temp dir, under TMPDIR. Python 3.12 does not fall back to a shorter directory when a
    socket path is too long (108 bytes): the workers' queue threads fail and the main
    process waits for the first batch forever.
    """
    if num_workers > 0:
        socket = os.path.join(multiprocessing.util.get_temp_dir(), 'listener-12345678')
        assert len(os.fsencode(socket)) < 108, (
            f'TMPDIR={tempfile.gettempdir()} is too long for the sockets of DataLoader '
            'workers: set a shorter TMPDIR, such as /tmp'
        )


def build_cache_iterator(
    cache: CacheSequences,
    steps: range,
    global_batch: int,
    micro_batch_size: int,
    dp_rank: int,
    dp_size: int,
    num_workers: int,
    base: int = 0,
    cycle: bool = False,
) -> MicroBatches:
    """Micro-batches of this rank for ``steps``; see the module docstring."""
    needed = base + steps.stop * global_batch
    assert not steps or needed <= cache.num_sequences, (
        f'{cache.cache_dir} has {cache.num_sequences} completed sequences, {needed} are needed'
    )
    check_worker_tmpdir(num_workers)
    spans = RankSpans(steps, global_batch, dp_rank, dp_size, cache.seqs_per_chunk, base, cycle)
    loader = torch.utils.data.DataLoader(
        cache, sampler=spans, batch_size=None, num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return MicroBatches(loader, micro_batch_size)
