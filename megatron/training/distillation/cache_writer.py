"""Data order and background writing of cache shards."""

import functools
import queue
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, Iterator, List, Sequence

import numpy
import torch

from .cache_format import ShardWriter, compress_chunk, file_path


class FileBatchSampler:
    """Micro-batches that give each data-parallel rank whole shards.

    Shard ``f`` holds samples ``f * seqs_per_file ... (f + 1) * seqs_per_file - 1`` of
    the dataset order, so shard contents do not depend on the number of GPUs. Rank
    ``r`` scores shards ``start_file + r, start_file + r + dp_size, ...``, each in
    ``seqs_per_file / micro_batch_size`` micro-batches.
    """

    def __init__(self, start_file: int, num_files: int, seqs_per_file: int,
                 micro_batch_size: int, dp_rank: int, dp_size: int):
        assert seqs_per_file % micro_batch_size == 0, (
            f'seqs per file ({seqs_per_file}) must be a multiple of the micro batch size '
            f'({micro_batch_size})'
        )
        assert start_file % dp_size == 0
        self.files = range(start_file + dp_rank, num_files, dp_size)
        self.seqs_per_file = seqs_per_file
        self.micro_batch_size = micro_batch_size

    def __iter__(self) -> Iterator[List[int]]:
        for file_id in self.files:
            first = file_id * self.seqs_per_file
            for offset in range(0, self.seqs_per_file, self.micro_batch_size):
                yield list(range(first + offset, first + offset + self.micro_batch_size))

    def __len__(self) -> int:
        return len(self.files) * (self.seqs_per_file // self.micro_batch_size)


class HeldOutSequences(torch.utils.data.Dataset):
    """Sequences of a blend's datasets that the blend's first ``used`` sequences don't reach.

    A blend reads each dataset in that dataset's own shuffled order, and a dataset's
    sequences are disjoint windows of its tokens. So its first ``used`` sequences are the
    first sequences of every dataset, and reading on from there gives a set that shares
    no tokens with them. ``weights`` share out ``size`` sequences over the datasets, in
    the order of ``blended.datasets``; ``seed`` shuffles them.
    """

    def __init__(self, blended, used: int, weights: Sequence[float], size: int, seed: int):
        self.datasets = blended.datasets
        assert len(weights) == len(self.datasets)
        blended[0]  # Maps the blend's index if another rank built it.
        assert used <= len(blended), f'the blend has {len(blended)} sequences, not {used}'
        first = numpy.bincount(blended.dataset_index[:used], minlength=len(self.datasets))
        quota = size * numpy.asarray(weights, dtype=numpy.float64) / sum(weights)
        counts = numpy.floor(quota).astype(numpy.int64)
        counts[numpy.argsort(counts - quota, kind='stable')[:size - counts.sum()]] += 1
        pairs = []
        for dataset_id in numpy.flatnonzero(counts).tolist():
            begin, end = int(first[dataset_id]), int(first[dataset_id] + counts[dataset_id])
            assert end <= len(self.datasets[dataset_id]), (
                f'dataset {dataset_id} has {len(self.datasets[dataset_id])} sequences: too few '
                f'for {counts[dataset_id]} held out after {begin} used'
            )
            pairs.extend((dataset_id, sample_id) for sample_id in range(begin, end))
        self.index = numpy.array(pairs)[numpy.random.RandomState(seed).permutation(size)]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        dataset_id, sample_id = int(self.index[idx][0]), int(self.index[idx][1])
        return {'dataset_id': dataset_id, **self.datasets[dataset_id][sample_id]}


def collate_samples(samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """``default_collate`` after casting each field to one dtype across the micro-batch.

    The datasets of a blend can disagree: ``assistant_mask`` is int64 from ``GPTDataset``
    and bool from the SFT dataset, and a worker stacks into the first sample's dtype.
    """
    for key, value in samples[0].items():
        if isinstance(value, torch.Tensor):
            dtype = functools.reduce(torch.promote_types, (s[key].dtype for s in samples))
            for sample in samples:
                sample[key] = sample[key].to(dtype)
    return torch.utils.data.default_collate(samples)


class AsyncShardWriter:
    """Writes shards in the background.

    Chunks are compressed on a thread pool and appended to their shard, in order,
    by a single assembler thread; a shard becomes visible only once complete.
    ``max_pending`` bounds the chunks held in memory. Errors are raised on the
    next call after the shard that failed.

    Usage per shard: ``begin(step, file_id, index)``, ``add(...)`` per chunk, ``end()``.
    ``abort()`` after an error drops the open shard, so that no thread is left waiting
    for its chunks and the process can exit.
    """

    def __init__(self, cache_dir: str, num_threads: int, compress_level: int, max_pending: int = 0):
        self.cache_dir = cache_dir
        self.compress_level = compress_level
        self.max_pending = max_pending or 2 * num_threads
        self.compressors = ThreadPoolExecutor(max_workers=num_threads)
        self.assembler = ThreadPoolExecutor(max_workers=1)
        self.compressing = threading.BoundedSemaphore(num_threads)
        self.chunks = None
        self.pending: Dict[int, Future] = {}
        self.done_steps = set()
        self.contiguous_steps = 0

    def begin(self, step: int, file_id: int, index: dict) -> None:
        """Start shard ``file_id``; ``step`` is this rank's local shard counter."""
        assert self.chunks is None, 'end() the previous shard first'
        self._collect()
        self.chunks = queue.Queue(maxsize=self.max_pending)
        self.pending[step] = self.assembler.submit(
            self._assemble, file_path(self.cache_dir, file_id), index, self.chunks
        )

    def add(self, payload: Dict[str, torch.Tensor], num_seqs: int, rows: int) -> None:
        """Queue the next chunk of the current shard."""
        self.compressing.acquire()
        try:
            future = self.compressors.submit(compress_chunk, payload, self.compress_level)
        except BaseException:
            self.compressing.release()
            raise
        future.add_done_callback(lambda _: self.compressing.release())
        self.chunks.put((future, num_seqs, rows))

    def end(self) -> None:
        """Close the current shard; it is written once its chunks are compressed."""
        self.chunks.put(None)
        self.chunks = None

    def abort(self) -> None:
        """Drop the open shard after an error; see the class docstring."""
        self.compressors.shutdown(wait=False, cancel_futures=True)
        if self.chunks is not None:
            failed = Future()
            failed.set_exception(RuntimeError('shard aborted'))
            self.chunks.put((failed, 0, 0))
            self.chunks.put(None)
            self.chunks = None
        self.assembler.shutdown(wait=False)

    def skip(self, step: int) -> None:
        """Count a step without a shard (the short last round) as done."""
        self.done_steps.add(step)
        self._advance()

    def completed_steps(self) -> int:
        """Number of leading local shards that are on disk."""
        self._collect()
        return self.contiguous_steps

    def close(self) -> int:
        """Wait for every shard and return the number of completed local shards."""
        for step, future in list(self.pending.items()):
            future.result()
            self.done_steps.add(step)
            del self.pending[step]
        self._advance()
        self.compressors.shutdown(wait=True)
        self.assembler.shutdown(wait=True)
        return self.contiguous_steps

    @staticmethod
    def _assemble(path: str, index: dict, chunks: queue.Queue) -> None:
        error, shard = None, None
        try:
            shard = ShardWriter(path, index)
        except BaseException as e:
            error = e
        while True:
            item = chunks.get()
            if item is None:
                break
            if error is not None:
                continue  # Keep draining so the producer never blocks on a full queue.
            future, num_seqs, rows = item
            try:
                shard.add(future.result(), num_seqs, rows)
            except BaseException as e:
                error = e
        if error is not None:
            if shard is not None:
                shard.abort()
            raise error
        shard.close()

    def _collect(self) -> None:
        finished: List[int] = [step for step, future in self.pending.items() if future.done()]
        for step in finished:
            self.pending.pop(step).result()  # Re-raises write errors.
            self.done_steps.add(step)
        self._advance()

    def _advance(self) -> None:
        while self.contiguous_steps in self.done_steps:
            self.done_steps.remove(self.contiguous_steps)
            self.contiguous_steps += 1
