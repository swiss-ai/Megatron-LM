"""
Maps BlendedDataset training steps back to original source documents and tokens.

A single training sequence can span multiple documents (packed sequences), so
this returns all source documents (parquet_path + doc_key/parquet_row) and the
exact token IDs that were fed to the model at each step.

Lookup chain:
  step
    → blend dataset_index / sample_index  (.npy files)
    → GPTDataset shuffle_index            (datasets/cache/)
    → GPTDataset sample_index             (datasets/cache/)
    → GPTDataset document_index           (datasets/cache/)
    → .meta.parquet sidecar               (alongside .bin files)
    → (parquet_path, doc_key|parquet_row)
    → .bin / .idx indexed dataset         (actual token IDs)

Usage:
    python3 scripts/tools/lookup_training_sample.py \
        --blend-metadata datasets/cache/1c7ca61109a69a7d2975edb01b361cb9-BlendedDataset-train \
        --cache-dir datasets/cache \
        --step-start 0 --step-end 9 \
        --output results.json
"""

import argparse
import json
import struct
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


# ── Indexed dataset reader ────────────────────────────────────────────────────

_INDEX_HEADER = b"MMIDIDX\x00\x00"

_DTYPE_MAP = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float32,
    7: np.float64,
    8: np.uint16,
}


class IndexedDatasetReader:
    """Lightweight reader for a Megatron MMapIndexedDataset (.bin/.idx pair).

    Keeps the .bin file memory-mapped for O(1) random token access.
    """

    def __init__(self, dataset_path: str) -> None:
        idx_path = dataset_path + ".idx"
        bin_path = dataset_path + ".bin"

        with open(idx_path, "rb") as f:
            assert f.read(9) == _INDEX_HEADER, f"Invalid .idx header: {idx_path}"
            f.read(8)  # version
            dtype_code = struct.unpack("<B", f.read(1))[0]
            self.dtype = _DTYPE_MAP[dtype_code]
            self.itemsize: int = self.dtype().itemsize
            n_seqs = struct.unpack("<Q", f.read(8))[0]
            struct.unpack("<Q", f.read(8))[0]  # n_docs (unused)
            offset = f.tell()

        idx_mmap = np.memmap(idx_path, mode="r", order="C")
        buf = memoryview(idx_mmap)
        self.sequence_lengths  = np.frombuffer(buf, dtype=np.int32, count=n_seqs, offset=offset)
        self.sequence_pointers = np.frombuffer(
            buf, dtype=np.int64, count=n_seqs,
            offset=offset + self.sequence_lengths.nbytes,
        )
        self._bin = np.memmap(bin_path, dtype=self.dtype, mode="r")

    def read_tokens(
        self,
        document_index: np.ndarray,
        j_start: int,
        k_start: int,
        j_end: int,
        k_end: int,
        add_extra_token: int = 1,
    ) -> list[int]:
        """Return the exact token IDs for one packed training sample.

        Mirrors GPTDataset.__getitem__: for each document j in [j_start, j_end],
        read from k_start (first doc) or 0 (middle docs) to k_end+add_extra_token
        (last doc) or end-of-document (middle docs).
        """
        parts: list[np.ndarray] = []
        for j in range(j_start, j_end + 1):
            doc_idx = int(document_index[j])
            ptr_elem = int(self.sequence_pointers[doc_idx]) // self.itemsize
            doc_len  = int(self.sequence_lengths[doc_idx])
            tok_lo = k_start if j == j_start else 0
            tok_hi = k_end + add_extra_token if j == j_end else doc_len
            parts.append(self._bin[ptr_elem + tok_lo : ptr_elem + tok_hi])
        return np.concatenate(parts).tolist() if parts else []


class TrainingDataLookup:
    def __init__(self, blend_prefix: str, cache_dir: str):
        self.cache_dir = Path(cache_dir)

        blend = Path(blend_prefix)
        self.blend_desc        = json.load(open(str(blend) + "-description.txt"))
        self.blend_dataset_idx = np.load(str(blend) + "-dataset_index.npy",        mmap_mode="r")
        self.blend_sample_idx  = np.load(str(blend) + "-dataset_sample_index.npy", mmap_mode="r")

        # Populated on first access per dataset
        self._gpt_prefix: dict[str, Path]           = {}
        self._gpt_arrays: dict[int, tuple]          = {}
        self._meta:       dict[str, dict]           = {}
        self._readers:    dict[str, IndexedDatasetReader] = {}

    def _find_gpt_prefix(self, dataset_path: str) -> Path:
        if dataset_path not in self._gpt_prefix:
            for f in self.cache_dir.glob("*-GPTDataset-train-description.txt"):
                if json.load(open(f)).get("dataset_path") == dataset_path:
                    self._gpt_prefix[dataset_path] = f.parent / f.name.replace("-description.txt", "")
                    break
            else:
                raise FileNotFoundError(f"No GPTDataset cache for {dataset_path}")
        return self._gpt_prefix[dataset_path]

    def _gpt_arrays_for(self, dataset_idx: int) -> tuple:
        if dataset_idx not in self._gpt_arrays:
            path = self.blend_desc["datasets"][dataset_idx]["dataset_path"]
            p = self._find_gpt_prefix(path)
            # No mmap: these are < 15 MB each; mmap causes one network roundtrip per
            # random page fault on Lustre (~10 ms each), which dominates at scale.
            self._gpt_arrays[dataset_idx] = (
                np.load(str(p) + "-shuffle_index.npy"),
                np.load(str(p) + "-sample_index.npy"),
                np.load(str(p) + "-document_index.npy"),
            )
        return self._gpt_arrays[dataset_idx]

    def _reader_for(self, dataset_path: str) -> IndexedDatasetReader:
        if dataset_path not in self._readers:
            self._readers[dataset_path] = IndexedDatasetReader(dataset_path)
        return self._readers[dataset_path]

    def _meta_for(self, dataset_path: str) -> dict:
        if dataset_path not in self._meta:
            meta_path = dataset_path + ".meta.parquet"
            if not Path(meta_path).exists():
                raise FileNotFoundError(
                    f"Missing metadata sidecar: {meta_path}\n"
                    f"Generate it with:\n"
                    f"  python3 scripts/tokenization/generate_metadata.py \\\n"
                    f"      --tokenized-folder <tokenized_folder> \\\n"
                    f"      --data-folder <raw_data_folder>"
                )
            self._meta[dataset_path] = pq.read_table(meta_path).to_pydict()
        return self._meta[dataset_path]

    def lookup_range(self, step_start: int, step_end: int) -> list[dict]:
        n = step_end - step_start + 1
        t0 = time.perf_counter()

        # Contiguous slice → one sequential read, no per-element page faults
        dataset_idxs = np.asarray(self.blend_dataset_idx[step_start:step_end + 1])
        sample_idxs  = np.asarray(self.blend_sample_idx[step_start:step_end + 1])

        results = [None] * n
        for unique_di in np.unique(dataset_idxs):
            unique_di     = int(unique_di)
            mask          = dataset_idxs == unique_di
            local_offsets = np.where(mask)[0]
            si            = sample_idxs[mask]

            dataset_path = self.blend_desc["datasets"][unique_di]["dataset_path"]
            shuffle_index, sample_index, document_index = self._gpt_arrays_for(unique_di)
            meta   = self._meta_for(dataset_path)
            reader = self._reader_for(dataset_path)

            pos_arr = shuffle_index[si]
            starts  = sample_index[pos_arr]
            ends    = sample_index[pos_arr + 1]

            for offset, sample_idx, pos, start, end in zip(
                local_offsets, si, pos_arr, starts, ends
            ):
                j_start, k_start = int(start[0]), int(start[1])
                j_end,   k_end   = int(end[0]),   int(end[1])

                docs = []
                for j in range(j_start, j_end + 1):
                    doc_idx = int(document_index[j])
                    entry = {
                        "doc_idx":      doc_idx,
                        "parquet_path": meta["parquet_path"][doc_idx],
                    }
                    if "doc_key" in meta:
                        entry["doc_key"] = meta["doc_key"][doc_idx]
                    else:
                        entry["parquet_row"] = meta["parquet_row"][doc_idx]
                    docs.append(entry)

                tokens = reader.read_tokens(document_index, j_start, k_start, j_end, k_end)

                results[offset] = {
                    "step":         step_start + int(offset),
                    "dataset_path": dataset_path,
                    "tokens":       tokens,
                    "docs":         docs,
                }

        t1 = time.perf_counter()
        print(f"[timing] {n} steps in {t1-t0:.3f}s")
        return results


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend-metadata", required=True,
                        help="Prefix of BlendedDataset cache files")
    parser.add_argument("--cache-dir", required=True,
                        help="Directory containing GPTDataset cache files")
    parser.add_argument("--step-start", type=int, default=0,
                        help="First training step to look up (default: 0)")
    parser.add_argument("--step-end", type=int, default=None,
                        help="Last training step inclusive (default: same as --step-start)")
    parser.add_argument("--output", required=True,
                        help="Path to write results JSON file")
    return parser.parse_args()


def main(args):
    step_end = args.step_end if args.step_end is not None else args.step_start
    lookup = TrainingDataLookup(args.blend_metadata, args.cache_dir)
    results = lookup.lookup_range(args.step_start, step_end)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {len(results)} results → {args.output}")


if __name__ == "__main__":
    main(get_args())
