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
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict
from torch.distributed.checkpoint.format_utils import FileSystemReader, _EmptyStateDictLoadPlanner


# ---------------------- USER-DEFINED CONSTANTS ----------------------
TOPK = 256
TENSORS_DIR = "/capstor/scratch/cscs/asolergi/main_run_70B_megatron/Megatron-LM/logs/Meg-Runs/main-runs-v1/apertus3-70b-512-nodes-1e-5lr/70b-probs-tensors"
ITERATION_TO_JOBID_PATH = "/iopsstor/scratch/cscs/blacksamorez/Megatron-LM-QAT/iteration_to_jobid.json"
ITERATION_TO_JOBID = json.load(open(ITERATION_TO_JOBID_PATH))
# -------------------------------------------------------------------

SEQS_PER_FILE = 32          # 32 sequences per dp file
FILES_PER_ITER = 128        # 128 dp files per iteration
SEQS_PER_ITER = SEQS_PER_FILE * FILES_PER_ITER  # 4096 sequences per iteration


# ---------------------- Helpers (importable at module top) ----------------------
def _file_key_from_dp_seq(dp_seq_counter: int) -> Tuple[int, int]:
    """Map an absolute dp-sequence index to (orig_iter, orig_dp file index)."""
    orig_iter = dp_seq_counter // SEQS_PER_ITER
    orig_iter_seq = dp_seq_counter % SEQS_PER_ITER
    orig_dp = orig_iter_seq // SEQS_PER_FILE
    return orig_iter, orig_dp


def _filepath_for_key(orig_iter: int, orig_dp: int) -> str:
    jobid = ITERATION_TO_JOBID[str(orig_iter)]
    return os.path.join(TENSORS_DIR, f"{jobid}-iter-{orig_iter}-dp-{orig_dp}")


def _load_one_file(orig_iter: int, orig_dp: int) -> Dict[str, torch.Tensor]:
    """Load a single DP file from disk (CPU tensors) and return the three buffers."""
    file_path = _filepath_for_key(orig_iter, orig_dp)
    tensor_sd: Dict[str, Any] = {}
    _load_state_dict(
        tensor_sd,
        storage_reader=FileSystemReader(file_path),
        planner=_EmptyStateDictLoadPlanner(),
        no_dist=True,
    )

    # Expected shapes after processing:
    # - input_ids_buffer: [32, T]  (seq, tokens)
    # - exp_logits_buffer: [T, 32, K]
    # - index_buffer: [T, 32, 4*TOPK] (after offsets applied)
    input_ids_buffer = tensor_sd["labels"].transpose(0, 1).contiguous()  # [32, T]
    exp_logits_buffer = tensor_sd["exp_logits"].contiguous()             # [T, 32, K]
    index_buffer = tensor_sd["index"].contiguous()                       # [T, 32, 4*TOPK]

    # Apply banked offsets in-place (CPU)
    index_buffer[:, :, :TOPK] += 0
    index_buffer[:, :, TOPK:2 * TOPK] += 32768
    index_buffer[:, :, 2 * TOPK:3 * TOPK] += 32768 * 2
    index_buffer[:, :, 3 * TOPK:4 * TOPK] += 32768 * 3

    return {
        "input_ids": input_ids_buffer,
        "exp_logits": exp_logits_buffer,
        "index": index_buffer,
    }


def _prefetch_worker(task_q: mp.Queue, result_q: mp.Queue):
    """
    Worker process:
      - Receives (orig_iter, orig_dp) keys.
      - Loads CPU tensors and sends (key, payload) back.
      - Sends (key, Exception) on failure.
    """
    seen: Set[Tuple[int, int]] = set()
    while True:
        msg = task_q.get()
        if msg is None:  # sentinel
            break
        key = tuple(msg)
        if key in seen:
            continue
        seen.add(key)
        try:
            payload = _load_one_file(key[0], key[1])
            result_q.put((key, payload))
        except Exception as e:
            result_q.put((key, e))


# ---------------------- Optimized Main class ----------------------
class LogitsLoader:
    """
    Prefetching logits loader optimized for sequential loads:

    - Blocks only when the current file is not yet cached.
    - Worker-only disk I/O; main thread never does synchronous reads.
    - Rank-aware prefetch + eviction (handles iteration boundaries).
    - CPU tensors are pinned upon cache admission for async H2D.
    """

    def __init__(self, prefetch_ahead_files: int = 3, start_method: str = "spawn"):
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
        self.exp_logits_buffer: Optional[torch.Tensor] = None
        self.index_buffer: Optional[torch.Tensor] = None

        # Sequence index at the start of the currently mounted file (multiple of 32)
        self.cached_seq_counter: int = -SEQS_PER_FILE

        # Prefetch config
        self.prefetch_ahead: int = max(0, int(prefetch_ahead_files))

        # IPC (avoid inheriting CUDA state; do not touch CUDA in worker)
        ctx = mp.get_context(start_method)
        self._task_q: mp.Queue = ctx.Queue(maxsize=256)
        self._result_q: mp.Queue = ctx.Queue(maxsize=256)
        self._worker: mp.Process = ctx.Process(
            target=_prefetch_worker, args=(self._task_q, self._result_q), daemon=True
        )
        self._worker.start()

        # In-memory cache (CPU tensors) of loaded files
        self._cache: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}
        self._pending: Set[Tuple[int, int]] = set()

        # Anchor file (current) for eviction policy
        self._anchor_file_start_seq: Optional[int] = None

        # Sticky runtime info (filled on first get_seq)
        self._last_dp_rank: Optional[int] = None
        self._last_dp_world_size: Optional[int] = None
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
        if self._last_dp_rank is None:
            self._last_dp_rank = dp_rank
        if self._last_dp_world_size is None:
            self._last_dp_world_size = dp_world_size
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
        self._ensure_file_ready_async_then_wait(file_start_seq)  # worker-only I/O + wait only for current
        self._activate_file(file_start_seq)

        # Slice within file
        diff = local_seq_counter - self.cached_seq_counter
        if diff < 0 or diff + seqs_to_consume_per_dp > SEQS_PER_FILE:
            # Unexpected jump (e.g., caller advanced across file boundary). Mount correct file.
            file_start_seq = (local_seq_counter // SEQS_PER_FILE) * SEQS_PER_FILE
            self._ensure_file_ready_async_then_wait(file_start_seq)
            self._activate_file(file_start_seq)
            diff = local_seq_counter - self.cached_seq_counter
            assert 0 <= diff <= SEQS_PER_FILE - seqs_to_consume_per_dp, (diff, seqs_to_consume_per_dp)

        input_ids = self.input_ids_buffer[diff:diff + seqs_to_consume_per_dp]
        exp_logits = self.exp_logits_buffer[:, diff:diff + seqs_to_consume_per_dp]
        index = self.index_buffer[:, diff:diff + seqs_to_consume_per_dp]

        # Rank-aware prefetch for the next N files (jump correctly across iter boundaries)
        for k in range(1, self.prefetch_ahead + 1):
            future_seq = self._kth_future_file_start_seq(
                file_start_seq, k, dp_rank, dp_world_size
            )
            self._maybe_enqueue_file(future_seq)

        # Opportunistically drain completed loads & enforce strict eviction
        self._drain_results(non_blocking=True)
        self._evict_strict()

        return {
            'input_ids': input_ids.to(self._device, non_blocking=True),
            'exp_logits': exp_logits.to(self._device, non_blocking=True),
            'index': index.to(self._device, non_blocking=True),
            'loss_mask': torch.ones(seqs_to_consume_per_dp, input_ids.shape[1], device=self._device),
            'attention_mask': None,
            'position_ids': torch.arange(input_ids.shape[1], dtype=torch.long, device=self._device),
        }

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

    # -------------------- Internal helpers --------------------
    def _activate_file(self, file_start_seq: int) -> None:
        """Mount cached file tensors into active buffers; set anchor for eviction."""
        key = _file_key_from_dp_seq(file_start_seq)
        slot = self._cache.get(key)
        if slot is None:
            # Should be rare (only if prefetch fell behind): wait for it now.
            self._wait_for_key(key)
            slot = self._cache[key]

        self.input_ids_buffer = slot["input_ids"]
        self.exp_logits_buffer = slot["exp_logits"]
        self.index_buffer = slot["index"]
        self.cached_seq_counter = file_start_seq
        self._anchor_file_start_seq = file_start_seq

    def _ensure_file_ready_async_then_wait(self, file_start_seq: int) -> None:
        """
        Ensure 'file' is (or will be) in cache using the worker only.
        Enqueue the current file (guaranteed), best-effort prefetch futures,
        then wait for the current file if still missing.
        """
        key = _file_key_from_dp_seq(file_start_seq)
        if key in self._cache:
            return

        # 1) Guarantee the current file is enqueued (may block on put if queue is full)
        self._enqueue_key_blocking(key)

        # 2) Immediately schedule next files along the rank path (best effort)
        if self._last_dp_rank is not None and self._last_dp_world_size is not None:
            for k in range(1, self.prefetch_ahead + 1):
                future_seq = self._kth_future_file_start_seq(
                    file_start_seq, k, self._last_dp_rank, self._last_dp_world_size
                )
                self._maybe_enqueue_file(future_seq)

        # 3) Drain any completed loads and then wait for current if still missing
        self._drain_results(non_blocking=True)
        if key not in self._cache:
            self._wait_for_key(key)  # blocks only for the current file

    def _wait_for_key(self, key: Tuple[int, int]) -> None:
        """Block until 'key' arrives; admit other arrivals into cache while waiting."""
        while key not in self._cache:
            k, payload = self._result_q.get()  # blocking
            if isinstance(payload, Exception):
                self._pending.discard(k)
                if k == key:
                    raise payload
                continue
            # Pin tensors upon admission so H2D .to(..., non_blocking=True) is effective
            for v in payload.values():
                if isinstance(v, torch.Tensor):
                    try:
                        v.pin_memory()
                    except Exception:
                        pass
            self._cache[k] = payload
            self._pending.discard(k)

    def _enqueue_key_blocking(self, key: Tuple[int, int]) -> None:
        """Guarantee that 'key' enters the worker queue (drain if needed, then block)."""
        if key in self._cache or key in self._pending:
            return
        try:
            self._task_q.put_nowait(key)
            self._pending.add(key)
            return
        except Exception:
            self._drain_results(non_blocking=True)
        self._task_q.put(key)  # may block only on queue backpressure
        self._pending.add(key)

    def _maybe_enqueue_file(self, file_start_seq: int) -> None:
        self._maybe_enqueue_key(_file_key_from_dp_seq(file_start_seq))

    def _maybe_enqueue_key(self, key: Tuple[int, int]) -> None:
        """Best-effort enqueue; if full, drain and try once, otherwise skip (prefetch only)."""
        if key in self._cache or key in self._pending:
            return
        try:
            self._task_q.put_nowait(key)
            self._pending.add(key)
        except Exception:
            self._drain_results(non_blocking=True)
            try:
                self._task_q.put_nowait(key)
                self._pending.add(key)
            except Exception:
                # Not critical; current path will block only when/if we reach it.
                pass

    def _drain_results(self, non_blocking: bool = True) -> None:
        """Move completed loads from result_q into the local cache."""
        import queue as _q
        while True:
            try:
                k, payload = self._result_q.get_nowait() if non_blocking else self._result_q.get()
            except _q.Empty:
                break
            if isinstance(payload, Exception):
                self._pending.discard(k)
                continue
            # Pin tensors upon admission for async H2D later
            for v in payload.values():
                if isinstance(v, torch.Tensor):
                    try:
                        v.pin_memory()
                    except Exception:
                        pass
            self._cache[k] = payload
            self._pending.discard(k)

    # -------- Rank-aware future computation & strict eviction --------
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

    def _expected_keys_around(self, file_start_seq: int) -> Set[Tuple[int, int]]:
        """Keep current + next prefetch_ahead files ALONG THIS RANK'S PATH (rank-aware)."""
        if self._last_dp_rank is None or self._last_dp_world_size is None:
            # Fallback to old behavior only during very first call bootstrap.
            return {
                _file_key_from_dp_seq(file_start_seq + i * SEQS_PER_FILE)
                for i in range(0, self.prefetch_ahead + 1)
            }
        return {
            _file_key_from_dp_seq(
                self._kth_future_file_start_seq(
                    file_start_seq, i, self._last_dp_rank, self._last_dp_world_size
                )
            )
            for i in range(0, self.prefetch_ahead + 1)
        }

    def _evict_strict(self) -> None:
        """Enforce memory cap: keep only {current file + next prefetch_ahead files} in cache/pending."""
        if self._anchor_file_start_seq is None:
            return
        keep = self._expected_keys_around(self._anchor_file_start_seq)

        # Drop anything not in keep set
        for k in list(self._cache.keys()):
            if k not in keep:
                self._cache.pop(k, None)
        for k in list(self._pending):
            if k not in keep:
                self._pending.discard(k)
