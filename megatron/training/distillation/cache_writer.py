"""Background writing of cache files and cross-rank progress tracking."""

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, List

import torch

from .cache_format import file_path, save_file


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
