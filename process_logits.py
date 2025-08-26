import os
import argparse
import json
import multiprocessing as mp
from typing import Dict, Tuple, Any, Optional, Set
from pathlib import Path
from itertools import chain
from gc import collect

import os
import re
from collections import defaultdict
from tqdm import tqdm

import torch
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict
from torch.distributed.checkpoint.format_utils import FileSystemReader, _EmptyStateDictLoadPlanner
from torch.distributed.checkpoint import CheckpointException

from logging import getLogger
logger = getLogger(__name__)


import tempfile, json, os
def _atomic_json_write(path: Path, obj):
    with tempfile.NamedTemporaryFile('w', delete=False, dir=path.parent, prefix=path.name+'.tmp.') as f:
        json.dump(obj, f)
        tmp = Path(f.name)
    os.replace(tmp, path)


# ---------------------- USER-DEFINED CONSTANTS ----------------------
RANK = int(os.environ["RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
PREFETCH_AHEAD_FILES = 2
N_THREADS = 2

SEQLEN = 4096
TOPK = 256
SRC_PATH = "/capstor/scratch/cscs/asolergi/main_run_70B_megatron/Megatron-LM/logs/Meg-Runs/main-runs-v1/apertus3-70b-512-nodes-1e-5lr/70b-probs-tensors"
DST_PATH = "/capstor/scratch/cscs/blacksamorez/70B_processed_logits"
SEQS_PER_FILE = 32          # 32 sequences per dp file
FILES_PER_ITER = 128        # 128 dp files per iteration
SEQS_PER_ITER = SEQS_PER_FILE * FILES_PER_ITER  # 4096 sequences per iteration
# -------------------------------------------------------------------


def get_iter_dp_to_jobid(src_path: os.PathLike):
    # get all files in the directory
    files = os.listdir(src_path)

    # Filter files that are `<jobid>-iter-<iteration>-dp-<dp_rank>`
    files = [file for file in files if re.match(r'^\d+-iter-\d+-dp-\d+$', file)]

    # Parse file names and group by (iteration, dp_rank)
    parsed_files = []
    for file in files:
        match = re.match(r'^(\d+)-iter-(\d+)-dp-(\d+)$', file)
        if match:
            jobid, iteration, dp_rank = map(int, match.groups())
            parsed_files.append((jobid, iteration, dp_rank))

    # Group by (iteration, dp_rank) and keep the one with largest jobid
    grouped = defaultdict(list)
    for jobid, iteration, dp_rank in parsed_files:
        grouped[(iteration, dp_rank)].append(jobid)
        
    iteration_dp_to_jobid = defaultdict(dict)
    for (iteration, dp_rank), jobids in grouped.items():
        jobids = sorted(jobids, reverse=True)
        iteration_dp_to_jobid[str(iteration)][str(dp_rank)] = jobids
        
    return iteration_dp_to_jobid


# ---- loader worker functions ----
def _filepaths_for_key(orig_iter: int, orig_dp: int, src_path: os.PathLike, iteration_dp_to_jobid: Dict[str, Dict[str, list[int]]]) -> list[str]:
    return [os.path.join(src_path, f"{jobid}-iter-{orig_iter}-dp-{orig_dp}") for jobid in iteration_dp_to_jobid[str(orig_iter)][str(orig_dp)]]


def _load_one_file(orig_iter: int, orig_dp: int, src_path: os.PathLike, iteration_dp_to_jobid: Dict[str, Dict[str, list[int]]]) -> Dict[str, torch.Tensor]:
    """Load a single DP file from disk (CPU tensors) and return the three buffers."""
    file_paths = _filepaths_for_key(orig_iter, orig_dp, src_path, iteration_dp_to_jobid)
    tensor_sd: Dict[str, Any] = {}
    for file_path in file_paths:
        try:
            _load_state_dict(
                tensor_sd,
                storage_reader=FileSystemReader(file_path),
                planner=_EmptyStateDictLoadPlanner(),
                no_dist=True,
            )
            break
        except CheckpointException:
            continue
    else:
        return {
            "failed": True,
        }

    # Expected shapes after processing:
    # - input_ids: [32, T]  (seq, tokens)
    # - exp_logits: [T, 32, 4*TOPK]
    # - index: [T, 32, 4*TOPK] (after offsets applied)
    labels_buffer = tensor_sd["labels"].transpose(0, 1).contiguous()  # [32, T]
    input_ids_buffer = torch.cat([torch.full((32, 1), 1, dtype=labels_buffer.dtype), labels_buffer[:,:-1].clone()], dim=1)
    exp_logits_buffer = tensor_sd["exp_logits"].contiguous()             # [T, 32, 4*TOPK]
    index_buffer = tensor_sd["index"].contiguous()                       # [T, 32, 4*TOPK]

    # Apply banked offsets in-place (CPU)
    index_buffer[:, :, :TOPK] += 0
    index_buffer[:, :, TOPK:2 * TOPK] += 32768
    index_buffer[:, :, 2 * TOPK:3 * TOPK] += 32768 * 2
    index_buffer[:, :, 3 * TOPK:4 * TOPK] += 32768 * 3

    loss_mask = torch.ones(32, SEQLEN, dtype=torch.bool)

    return {
        "input_ids": input_ids_buffer,
        "labels": labels_buffer,
        "exp_logits": exp_logits_buffer,
        "index": index_buffer,
        "loss_mask": loss_mask,
    }


def _loader_worker(task_q: mp.Queue, result_q: mp.Queue, src_path: os.PathLike):
    from concurrent.futures import ThreadPoolExecutor
    
    iteration_dp_to_jobid = get_iter_dp_to_jobid(src_path)

    def _run_one(key):
        try:
            payload = _load_one_file(key[0], key[1], src_path, iteration_dp_to_jobid)
            result_q.put((key, payload))
        except Exception as e:
            result_q.put((key, e))

    with ThreadPoolExecutor(max_workers=N_THREADS) as pool:
        while True:
            msg = task_q.get()
            if msg is None:  # sentinel
                break
            key = tuple(msg)
            pool.submit(_run_one, key)


# ---- saver worker functions ----
def _save_file(current_chunk_to_save: int, payload: Dict[str, torch.Tensor], dst_path: os.PathLike):
    torch.save(
        payload,
        os.path.join(dst_path, f"{current_chunk_to_save:010d}.pt")
    )


def _save_worker(processed_q: mp.Queue, status_q: mp.Queue, dst_path: os.PathLike):
    from concurrent.futures import ThreadPoolExecutor
    
    def _run_one(msg): # ((iter_to_save, dp_to_save), chunk_to_save, payload)
        try:
            if "failed" in msg[2]:
                status_q.put((msg[0], {"failed": True}))
                return
            
            _save_file(msg[1], msg[2], dst_path)
            status_q.put((msg[0], {"chunk_saved": msg[1]}))
        except Exception as e:
            status_q.put((msg[0], e))

    with ThreadPoolExecutor(max_workers=N_THREADS) as pool:
        while True:
            msg = processed_q.get()
            if msg is None:  # sentinel
                break
            pool.submit(_run_one, msg)
            
# ---- main functions ----
def file_step_forward(iter: int, dp: int, world_size: int) -> Tuple[int, int]:
    current_filecount = iter * FILES_PER_ITER + dp
    new_filecount = current_filecount + world_size
    
    return new_filecount // FILES_PER_ITER, new_filecount % FILES_PER_ITER


def file_step_backward(iter: int, dp: int, world_size: int) -> Tuple[int, int]:
    current_filecount = iter * FILES_PER_ITER + dp
    new_filecount = current_filecount - world_size
    
    return new_filecount // FILES_PER_ITER, new_filecount % FILES_PER_ITER


class LogitsProcessor:
    def __init__(self, src_path: os.PathLike, dst_path: os.PathLike, rank: int, world_size: int, prefetch_ahead_files: int = 128, start_method: str = "spawn"):
        # Prefetch config
        self.prefetch_ahead: int = prefetch_ahead_files
        self.src_path: os.PathLike = src_path
        self.dst_path: os.PathLike = dst_path
        self.rank: int = rank
        self.world_size: int = world_size

        # IPC (avoid inheriting CUDA state; do not touch CUDA in worker)
        ctx = mp.get_context(start_method)
        self._iter_dp_q: mp.Queue = ctx.Queue(maxsize=self.prefetch_ahead * 2)
        self._loader_payload_q: mp.Queue = ctx.Queue(maxsize=self.prefetch_ahead * 2)
        self._load_worker: mp.Process = ctx.Process(
            target=_loader_worker, args=(self._iter_dp_q, self._loader_payload_q, self.src_path), daemon=True
        )
        self._load_worker.start()
        
        self._saver_payload_q: mp.Queue = ctx.Queue(maxsize=self.prefetch_ahead * 2)
        self._saver_status_q: mp.Queue = ctx.Queue(maxsize=self.prefetch_ahead * 2)
        self._save_worker: mp.Process = ctx.Process(
            target=_save_worker, args=(self._saver_payload_q, self._saver_status_q, self.dst_path), daemon=True
        )
        self._save_worker.start()
        
        self.mapping = {}
        self.progress = {}
        
        self.iter_to_load = 0
        self.dp_to_load = 0
        
        self.iter_to_confirm = 0
        self.dp_to_confirm = 0
        
        self.chunk_to_save = 0
        
        self.loaded = {}
        self.confirmed = {}
        
    def pre_start(self, start_iter: int, start_dp: int):
        if not os.path.exists(self.src_path):
            raise FileNotFoundError(f"Source path {self.src_path} does not exist")
        
        # Create dst path if it doesn't exist
        if not os.path.exists(self.dst_path):
            os.makedirs(self.dst_path)
            
        # Load and verify src->dst mapping if exists
        if os.path.exists(self.dst_path.joinpath(f"consumed_mapping-{self.rank}.json")):
            with open(self.dst_path.joinpath(f"consumed_mapping-{self.rank}.json"), "r") as f:
                mapping = json.load(f)
            with open(self.dst_path.joinpath(f"progress-{self.rank}.json"), "r") as f:
                progress = json.load(f)
            self.mapping = mapping
            self.progress = progress
            logger.warning(f"Resuming from {self.progress['last_iter_dp']}")
        else:
            iter_to_load, dp_to_load = file_step_forward(start_iter, start_dp, self.rank)
            self.progress = {
                "last_iter_dp": file_step_backward(iter_to_load, dp_to_load, self.world_size),
                "chunk_to_save": self.rank,
            }
            logger.warning(f"Starting from {self.progress['last_iter_dp']}")
        self.iter_to_load, self.dp_to_load = file_step_forward(*self.progress["last_iter_dp"], self.world_size)
        self.iter_to_save, self.dp_to_save = self.iter_to_load, self.dp_to_load
        self.iter_to_confirm, self.dp_to_confirm = self.iter_to_load, self.dp_to_load
        self.chunk_to_save = self.progress["chunk_to_save"]
            
    def step_iter_dp_to_load(self):
        self.iter_to_load, self.dp_to_load = file_step_forward(self.iter_to_load, self.dp_to_load, self.world_size)
    
    def step_iter_dp_to_save(self):
        self.iter_to_save, self.dp_to_save = file_step_forward(self.iter_to_save, self.dp_to_save, self.world_size)
            
    def step_chunk_to_save(self):
        self.chunk_to_save += self.world_size
        
    def step_iter_dp_to_confirm(self):
        self.iter_to_confirm, self.dp_to_confirm = file_step_forward(self.iter_to_confirm, self.dp_to_confirm, self.world_size)
            
    def load_one_file(self):
        self._iter_dp_q.put_nowait((self.iter_to_load, self.dp_to_load))
        self.step_iter_dp_to_load()

    def pipe_one_file(self):
        iter_dp_to_save = (self.iter_to_save, self.dp_to_save)

        while iter_dp_to_save not in self.loaded:
            k, payload = self._loader_payload_q.get()
            if isinstance(payload, Exception):
                payload.add_note(f"Failed to load file {k}")
                raise payload
            
            self.loaded[k] = payload
        
        payload = self.loaded[iter_dp_to_save]
        del self.loaded[iter_dp_to_save]
        
        self._saver_payload_q.put_nowait(((self.iter_to_save, self.dp_to_save), self.chunk_to_save, payload))
        
        self.step_iter_dp_to_save()
        if "failed" in payload:
            logger.warning(f"Failed to load file {iter_dp_to_save}, skipping")
        else:
            self.step_chunk_to_save()
            
    def fill_pipeline(self):
        for _ in range(self.prefetch_ahead):
            self.load_one_file()
            
        for _ in range(self.prefetch_ahead):
            self.pipe_one_file()
            
        for _ in range(self.prefetch_ahead):
            self.load_one_file()
            
    def flush_pipeline(self):
        for _ in range(self.prefetch_ahead):
            self.pipe_one_file()
            
        for _ in range(self.prefetch_ahead * 2):
            self.save_one_file()
            
        self.dump_progress()
            
    def step_pipeline(self):
        self.save_one_file()
        self.pipe_one_file()
        self.load_one_file()
        collect()
    
    def save_one_file(self):
        iter_dp_to_confirm = (self.iter_to_confirm, self.dp_to_confirm)
        
        while iter_dp_to_confirm not in self.confirmed:
            iter_dp_confirmed, result = self._saver_status_q.get()
            
            if isinstance(result, Exception):
                result.add_note(f"Failed to save file {iter_dp_confirmed}")
                raise result
                
            self.confirmed[iter_dp_confirmed] = result
        
        iter_dp_confirmed = iter_dp_to_confirm
        payload = self.confirmed[iter_dp_confirmed]
        del self.confirmed[iter_dp_confirmed]
        self.step_iter_dp_to_confirm()
        
        if "failed" in payload:
            return
        
        self.progress["chunk_to_save"] = payload["chunk_saved"] + self.world_size
        self.progress["last_iter_dp"] = iter_dp_confirmed
        self.mapping[f"{iter_dp_confirmed[0]}-{iter_dp_confirmed[1]}"] = payload["chunk_saved"]
        
    
    def dump_progress(self):
        logger.warning(f"Dumping progress: {self.progress}")
        
        _atomic_json_write(self.dst_path.joinpath(f"progress-{self.rank}.json"), self.progress)
        _atomic_json_write(self.dst_path.joinpath(f"consumed_mapping-{self.rank}.json"), self.mapping)

    def close(self):
        for q in (self._iter_dp_q, self._saver_payload_q):
            try: q.put_nowait(None)
            except Exception: pass

        for p in (self._load_worker, self._save_worker):
            if hasattr(self, p.name) and p.is_alive():
                p.join(timeout=5)

    def __del__(self):
        self.close()


def main(rank: int, world_size: int):
    processor = LogitsProcessor(
        Path(SRC_PATH), Path(DST_PATH),
        rank, world_size,
        prefetch_ahead_files=PREFETCH_AHEAD_FILES,
    )
    processor.pre_start(1096333, 0)
    
    total_steps = 128 * 1024
    
    if rank == 0:
        pbar = tqdm(total=total_steps, desc="Processing")
    
    processor.fill_pipeline()
    for i in range(total_steps):
        processor.step_pipeline()
        if rank == 0:
            pbar.update(1)
        if i % 100 == 0:
            processor.dump_progress()
    if rank == 0:
        pbar.close()
    processor.flush_pipeline()


if __name__ == "__main__":
    if RANK < WORLD_SIZE:
        main(RANK, WORLD_SIZE)
