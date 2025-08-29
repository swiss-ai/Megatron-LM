# Prefetching LogitsLoader with rank-aware prefetch and deadlock-safe loading,
# optimized for strictly sequential access.
#
# Key properties:
# - Worker-only I/O. The main thread never touches disk.
# - Blocks only when the CURRENT file is not yet cached (cold start / catch-up).
# - STRICT cache cap: {current file + next N files along THIS RANK's path}.
# - Rank-aware prefetch + eviction (handles iteration boundaries cleanly).
# - Pin CPU tensors on cache admission so H2D copies can be truly async.
#
# Spawn one instance per DP rank (recommended).

import os
import json
import multiprocessing as mp
from typing import Dict, Tuple, Any, Optional, Set

import torch

from logging import getLogger
logger = getLogger(__name__)


# ---------------------- USER-DEFINED CONSTANTS ----------------------
TENSORS_DIR = "/capstor/store/cscs/swissai/infra01/distillation/70B_processed_logits"
TOPK = 256
SEQS_PER_FILE = 32          # 32 sequences per dp file
FILES_PER_ITER = 128        # 128 dp files per iteration
SEQS_PER_ITER = SEQS_PER_FILE * FILES_PER_ITER  # 4096 sequences per iteration

# CONSUMED_IDS = set()
# -------------------------------------------------------------------

# ---------------------- Helpers (importable at module top) ----------------------
def _filepath_for_seq(seq: int) -> str:
    return os.path.join(TENSORS_DIR, f"{seq // 32:010d}.pt")


def _load_one_file(seq: int) -> Dict[str, torch.Tensor]:
    """Load a single DP file from disk (CPU tensors) and return the three buffers."""
    file_path = _filepath_for_seq(seq)
    tensor_sd = torch.load(file_path, weights_only=False)
    
    # {
    #     "input_ids": input_ids_buffer,
    #     "labels": labels_buffer,
    #     "exp_logits": exp_logits_buffer,
    #     "index": index_buffer,
    #     "loss_mask": loss_mask,
    # }

    return tensor_sd


def _prefetch_worker(task_q: mp.Queue, result_q: mp.Queue, *, n_threads: int = 8):
    from concurrent.futures import ThreadPoolExecutor

    def _run_one(key):
        try:
            payload = _load_one_file(key)
            result_q.put((key, payload))
        except Exception as e:
            result_q.put((key, e))

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        while True:
            msg = task_q.get()
            if msg is None:  # sentinel
                break
            pool.submit(_run_one, msg)

# ---------------------- Optimized Main class ----------------------
class LogitsLoader:
    """
    Prefetching logits loader optimized for sequential loads:

    - Blocks only when the current file is not yet cached.
    - Worker-only disk I/O; main thread never does synchronous reads.
    - Rank-aware prefetch + eviction (handles iteration boundaries).
    - CPU tensors are pinned upon cache admission for async H2D.
    """

    def __init__(self, prefetch_ahead_files: int = 32, start_method: str = "spawn"):
        """
        prefetch_ahead_files:
            number of FUTURE files to keep ready (not counting current).
            total resident files <= (1 + prefetch_ahead_files).
            e.g., 3 => keep current + next 3 = 4 total.
        start_method:
            "spawn" (safe around CUDA; works cross-platform) or "fork" (Linux/macOS only; testing).
        """
        # Active buffers (CPU) for the currently mounted file
        self.input_ids_buffer: Optional[torch.Tensor] = None
        self.labels_buffer: Optional[torch.Tensor] = None
        self.exp_logits_buffer: Optional[torch.Tensor] = None
        self.index_buffer: Optional[torch.Tensor] = None
        self.loss_mask_buffer: Optional[torch.Tensor] = None

        # Prefetch config
        self.prefetch_ahead: int = max(0, int(prefetch_ahead_files))

        # IPC (avoid inheriting CUDA state; do not touch CUDA in worker)
        ctx = mp.get_context(start_method)
        self._task_q: mp.Queue = ctx.Queue(maxsize=prefetch_ahead_files)
        self._result_q: mp.Queue = ctx.Queue(maxsize=prefetch_ahead_files)
        self._worker: mp.Process = ctx.Process(
            target=_prefetch_worker, args=(self._task_q, self._result_q), daemon=True
        )
        self._worker.start()
        self.expected_seqs = {}

        # Cache positions
        self._cached_seq: int =  -1_000_000

        # Sticky runtime info (filled on first get_seq)
        self._device: Optional[torch.device] = None

    # -------------------- Public API --------------------
    def get_seq(
        self,
        common_seq_counter: int,
        seqs_to_consume_per_dp: int,
        dp_rank: int,
        dp_world_size: int,
    ) -> Dict[str, torch.Tensor]:
        # Invariants
        assert seqs_to_consume_per_dp <= SEQS_PER_FILE
        assert SEQS_PER_FILE % seqs_to_consume_per_dp == 0
        assert FILES_PER_ITER % dp_world_size == 0

        # Sticky runtime info
        if self._device is None:
            self._device = torch.device('cuda', torch.cuda.current_device())

        # Compute this rank's absolute local sequence position
        block_start = (common_seq_counter // SEQS_PER_ITER) * SEQS_PER_ITER
        files_per_rank = FILES_PER_ITER // dp_world_size
        block_section_offset = dp_rank * files_per_rank * SEQS_PER_FILE
        section_pos = (common_seq_counter % SEQS_PER_ITER) // dp_world_size
        local_seq_counter = block_start + block_section_offset + section_pos

        # Ensure the file containing local_seq_counter is ready
        file_start_seq = (local_seq_counter // SEQS_PER_FILE) * SEQS_PER_FILE
        
        # Inital order of self.prefetch_ahead files
        if self._cached_seq == -1_000_000:
            for i in range(0, self.prefetch_ahead):
                future_seq = self._kth_future_file_start_seq(
                    file_start_seq, i, dp_rank, dp_world_size
                )
                self._task_q.put_nowait(future_seq)
        
        # Get the file from the result queue if it is not already cached
        if file_start_seq == self._cached_seq:
            pass
        elif file_start_seq > self._cached_seq:
            self._queue_to_cache(file_start_seq, dp_rank, dp_world_size)
            self._cached_seq = file_start_seq        

        # Slice within file
        diff = local_seq_counter - self._cached_seq

        input_ids = self.input_ids_buffer[diff:diff + seqs_to_consume_per_dp]
        labels = self.labels_buffer[diff:diff + seqs_to_consume_per_dp]
        exp_logits = self.exp_logits_buffer[:, diff:diff + seqs_to_consume_per_dp]
        index = self.index_buffer[:, diff:diff + seqs_to_consume_per_dp]
        loss_mask = self.loss_mask_buffer[diff:diff + seqs_to_consume_per_dp]
                
        # for i in range(local_seq_counter, local_seq_counter + seqs_to_consume_per_dp):
        #     CONSUMED_IDS.add(i)

        return {
            'input_ids': input_ids.to(self._device, non_blocking=True),
            'labels': labels.to(self._device, non_blocking=True),
            'exp_logits': exp_logits.to(self._device, non_blocking=True),
            'index': index.to(self._device, non_blocking=True),
            'loss_mask': loss_mask.to(self._device, non_blocking=True),
            'attention_mask': None,
            'position_ids': torch.arange(input_ids.shape[1], dtype=torch.long, device=self._device),
        }
        
    def _queue_to_cache(self, file_start_seq: int, dp_rank: int, dp_world_size: int) -> None:
        # Queue to cache
        # logger.warning(f"Awaiting file {file_start_seq}")
        while file_start_seq not in self.expected_seqs:
            seq, payload = self._result_q.get()
            self.expected_seqs[seq] = payload
        
        payload = self.expected_seqs[file_start_seq]
        del self.expected_seqs[file_start_seq]
        
        if isinstance(payload, Exception):
            raise payload
        
        for v in payload.values():
            if isinstance(v, torch.Tensor):
                try:
                    v.pin_memory()
                except Exception:
                    pass
                
        self.input_ids_buffer = payload["input_ids"]
        self.labels_buffer = payload["labels"]
        self.exp_logits_buffer = payload["exp_logits"]
        self.index_buffer = payload["index"]
        self.loss_mask_buffer = payload["loss_mask"]
        
        # Add next file to load to queue
        next_file_start_seq = self._kth_future_file_start_seq(
            file_start_seq, self.prefetch_ahead, dp_rank, dp_world_size
        )
        self._task_q.put_nowait(next_file_start_seq)

    def close(self):
        """Gracefully stop the worker process."""
        try:
            self._task_q.put_nowait(None)  # sentinel
        except Exception:
            pass
        if hasattr(self, "_worker") and self._worker.is_alive():
            self._worker.join(timeout=5)

    def __del__(self):
        self.close()    

    # -------- Rank-aware future computation --------
    def _kth_future_file_start_seq(self, current_file_start_seq: int, k: int, dp_rank: int, dp_world_size: int) -> int:
        """
        Return the file_start_seq (multiple of 32) for the kth file ahead along THIS RANK's path,
        correctly jumping across iteration boundaries to the start of the rank's shard in the next iter.
        """
        files_per_rank = FILES_PER_ITER // dp_world_size
        rank_dp0 = dp_rank * files_per_rank

        curr_iter = current_file_start_seq // SEQS_PER_ITER
        curr_dp   = (current_file_start_seq % SEQS_PER_ITER) // SEQS_PER_FILE
        rel_in_rank = curr_dp - rank_dp0                  # 0 .. files_per_rank-1
        total = rel_in_rank + k
        iter_advance, pos_in_rank = divmod(total, files_per_rank)

        target_iter = curr_iter + iter_advance
        target_dp   = rank_dp0 + pos_in_rank
        return target_iter * SEQS_PER_ITER + target_dp * SEQS_PER_FILE
