"""Data order and background writing of cache files."""

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, Iterator, List

import torch

from .cache_format import file_path, save_file


class FileBatchSampler:
    """Micro-batches that give each data-parallel rank whole files.

    File ``f`` holds samples ``f * seqs_per_file ... (f + 1) * seqs_per_file - 1`` of
    the dataset order, so file contents do not depend on the number of GPUs. Rank
    ``r`` scores files ``start_file + r, start_file + r + dp_size, ...``, each in
    ``seqs_per_file / micro_batch_size`` micro-batches.
    """

    def __init__(self, start_file: int, num_files: int, seqs_per_file: int,
                 micro_batch_size: int, dp_rank: int, dp_size: int):
        assert seqs_per_file % micro_batch_size == 0, (
            f'seqs per file ({seqs_per_file}) must be a multiple of the micro batch size '
            f'({micro_batch_size})'
        )
        assert start_file % dp_size == 0 and num_files % dp_size == 0
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


class AsyncFileWriter:
    """Compresses and writes cache files on a thread pool.

    ``submit`` blocks once ``max_pending`` files are queued, which bounds host
    memory. Write errors are raised on the next ``submit`` or ``completed_steps``.
    """

    def __init__(self, cache_dir: str, num_threads: int, compress_level: int, max_pending: int = 0):
        self.cache_dir = cache_dir
        self.compress_level = compress_level
        self.pool = ThreadPoolExecutor(max_workers=num_threads)
        self.slots = threading.BoundedSemaphore(max_pending or 2 * num_threads)
        self.pending: Dict[int, Future] = {}
        self.done_steps = set()
        self.contiguous_steps = 0

    def submit(self, step: int, file_id: int, payload: Dict[str, torch.Tensor]) -> None:
        """Queue ``payload`` as file ``file_id``; ``step`` is this rank's local step index."""
        self._collect()
        self.slots.acquire()
        try:
            future = self.pool.submit(
                save_file, file_path(self.cache_dir, file_id), payload, self.compress_level
            )
        except BaseException:
            self.slots.release()
            raise
        future.add_done_callback(lambda _: self.slots.release())
        self.pending[step] = future

    def completed_steps(self) -> int:
        """Number of leading local steps whose files are on disk."""
        self._collect()
        return self.contiguous_steps

    def close(self) -> int:
        """Wait for every queued file and return the number of completed steps."""
        for step, future in list(self.pending.items()):
            future.result()
            self.done_steps.add(step)
            del self.pending[step]
        self._advance()
        self.pool.shutdown(wait=True)
        return self.contiguous_steps

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
