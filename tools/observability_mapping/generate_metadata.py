"""
Post-hoc provenance metadata generator.

Reads the same parquet files that were tokenized and produces a .meta.parquet
file for each .bin file, mapping doc_index → (parquet_path, doc_key|parquet_row).

Without --key-col: parquet_row is the ABSOLUTE raw row index in the parquet file,
enabling O(1) lookup: pq.read_table(parquet_path).slice(parquet_row, 1)

With --key-col: doc_key stores the value of the specified unique-identifier column
from the original dataset (e.g. an HuggingFace dataset ID field), allowing
provenance lookup via the dataset's own primary key rather than a positional index.

Two-phase approach:
  Phase 1 — rebuild dump assignments by replicating prepare_dumps.py's greedy
             bin-packing (sort parquets by size descending, assign each to the
             smallest-so-far dump). The --filter-in and --filter-out parameters
             can also be specified if those were used when initially creating the
             dumps. This helps with getting the correct number of dumps for phase 2.
  Phase 2 — for each dump, stride-assign files to ranks (same as DataTrove)
             and emit .meta.parquet sidecars.

Usage:
    python3 scripts/generate_metadata.py \
        --tokenized-folder /path/to/datasets_tokenized/MyDataset \
        --data-folder /path/to/raw/MyDataset \
        --text-col text

    # Map to a unique key column instead of a raw row index:
    python3 scripts/generate_metadata.py \
        --tokenized-folder /path/to/datasets_tokenized/MyDataset \
        --data-folder /path/to/raw/MyDataset \
        --text-col text \
        --key-col id

    # With filters matching the original prepare_dumps.py call:
    python3 scripts/generate_metadata.py \
        --tokenized-folder /path/to/datasets_tokenized/MyDataset \
        --data-folder /path/to/raw/MyDataset \
        --filter-in train \
        --filter-out test valid
"""

import argparse
import bisect
import os
import struct
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# Must match Rehydrater.upsampling_weights exactly
_REHYDRATE_LIMITS = [1, 2, 3, 5, 100, 1000]
_REHYDRATE_WEIGHTS = {1: 1, 2: 2, 3: 3, 5: 5, 100: 8, 1000: 1}


def rehydrate_weight(minhash_cluster_size: int) -> int:
    key = _REHYDRATE_LIMITS[bisect.bisect_right(_REHYDRATE_LIMITS, minhash_cluster_size) - 1]
    return _REHYDRATE_WEIGHTS[key]


_INDEX_HEADER = b"MMIDIDX\x00\x00"


def read_num_sequences(idx_path: str) -> int:
    with open(idx_path, "rb") as f:
        assert f.read(9) == _INDEX_HEADER, f"Invalid .idx header in {idx_path}"
        f.read(8)  # version
        f.read(1)  # dtype
        return struct.unpack("<Q", f.read(8))[0]


# ── Phase 1: rebuild dump→file assignment ────────────────────────────────────

def rebuild_dump_paths(
    data_folder: str,
    n_dumps: int,
    filter_in: list[str] | None,
    filter_out: list[str] | None,
    extension: str,
) -> list[list[str]]:
    """
    Replicate prepare_dumps.py's greedy bin-packing.
    Returns dump_paths[i] = list of relative paths (from data_folder) for dump i.
    """
    all_files = []
    for dirpath, _, filenames in os.walk(data_folder, followlinks=True):
        for fn in filenames:
            if fn.lower().endswith(extension):
                all_files.append(os.path.join(dirpath, fn))

    if not all_files:
        raise ValueError(f"No *{extension} files found in {data_folder}")

    if filter_in:
        filtered = []
        for term in filter_in:
            filtered.extend(f for f in all_files if term in f and f not in filtered)
        all_files = filtered

    if filter_out:
        for term in filter_out:
            all_files = [f for f in all_files if term not in f]

    if not all_files:
        raise ValueError(f"No files remain after filtering in {data_folder}")

    sizes = [os.path.getsize(f) for f in all_files]
    # Same sort as prepare_dumps.py: largest file first
    files_sizes = sorted(zip(all_files, sizes), key=lambda x: x[1], reverse=True)

    dump_files: list[list[str]] = [[] for _ in range(n_dumps)]
    dump_sizes = [0] * n_dumps

    for f, size in files_sizes:
        min_idx = dump_sizes.index(min(dump_sizes))
        dump_files[min_idx].append(os.path.relpath(f, data_folder))
        dump_sizes[min_idx] += size

    for i, (files, size) in enumerate(zip(dump_files, dump_sizes)):
        print(f"  dump-{i}: {len(files)} files, {size / 1e9:.2f} GB", flush=True)

    return dump_files


# ── Phase 2: generate metadata per rank ──────────────────────────────────────

def get_rank_paths(dump_paths: list[str], rank: int, n_tasks: int) -> list[str]:
    """Stride-based assignment matching DataTrove's LocalPipelineExecutor."""
    return [p for i, p in enumerate(dump_paths) if (i - rank) % n_tasks == 0]


def generate_rank_metadata(
    rank: int,
    n_tasks: int,
    tokenized_folder: str,
    data_folder: str,
    dump_paths: list[str],
    text_col: str,
    output_folder: str,
    rehydrate: bool = False,
    key_col: str | None = None,
) -> None:
    bin_path = os.path.join(tokenized_folder, f"{rank:05d}_tokens.bin")
    idx_path = os.path.join(tokenized_folder, f"{rank:05d}_tokens.idx")
    meta_path = os.path.join(output_folder, f"{rank:05d}_tokens.meta.parquet")

    if not os.path.exists(bin_path):
        return

    expected = read_num_sequences(idx_path)
    assigned = get_rank_paths(dump_paths, rank, n_tasks)

    doc_indices: list[int] = []
    parquet_paths: list[str] = []
    # Stores parquet row indices (no key_col) or unique key values (key_col set)
    id_values: list = []

    doc_index = 0
    read_cols = [text_col] + (["minhash_cluster_size"] if rehydrate else [])
    if key_col and key_col not in read_cols:
        read_cols.append(key_col)

    for relative_path in assigned:
        abs_path = os.path.abspath(os.path.join(data_folder, relative_path))
        table = pq.read_table(abs_path, columns=read_cols)

        non_empty_mask = pc.greater(pc.utf8_length(table.column(text_col)), 0)
        raw_rows = np.nonzero(non_empty_mask.to_numpy())[0]

        if key_col:
            keys = table.column(key_col).filter(non_empty_mask).to_pylist()

        if not rehydrate:
            n = len(raw_rows)
            doc_indices.extend(range(doc_index, doc_index + n))
            parquet_paths.extend([abs_path] * n)
            id_values.extend(keys if key_col else raw_rows.tolist())
            doc_index += n
        else:
            cluster_sizes = table.column("minhash_cluster_size").filter(non_empty_mask).to_pylist()
            for i, (raw_row, cluster_size) in enumerate(zip(raw_rows.tolist(), cluster_sizes)):
                weight = rehydrate_weight(cluster_size)
                id_val = keys[i] if key_col else raw_row
                for _ in range(weight):
                    doc_indices.append(doc_index)
                    parquet_paths.append(abs_path)
                    id_values.append(id_val)
                    doc_index += 1

    if doc_index != expected:
        raise ValueError(
            f"rank {rank}: doc count mismatch — generated {doc_index} entries "
            f"but .idx reports {expected} sequences in {tokenized_folder}. "
            f"The rebuilt dump assignment may not match the original tokenization run."
        )

    os.makedirs(output_folder, exist_ok=True)
    if key_col:
        out_table = pa.table({
            "doc_index":    pa.array(doc_indices,   pa.int64()),
            "parquet_path": pa.array(parquet_paths, pa.string()),
            "doc_key":      pa.array(id_values),
        })
    else:
        out_table = pa.table({
            "doc_index":    pa.array(doc_indices,   pa.int64()),
            "parquet_path": pa.array(parquet_paths, pa.string()),
            "parquet_row":  pa.array(id_values,     pa.int64()),
        })
    pq.write_table(out_table, meta_path)
    print(f"rank {rank:05d}: wrote {doc_index} entries → {meta_path}", flush=True)

def _worker(kwargs: dict) -> None:
    generate_rank_metadata(**kwargs)


def _infer_n_tasks(tokenized_folder: str) -> int:
    return len(list(Path(tokenized_folder).glob("*_tokens.bin")))


def _build_worker_kwargs(
    tokenized_folder: str,
    data_folder: str,
    dump_paths: list[str],
    n_tasks_arg: int | None,
    text_col: str,
    output_folder_override: str | None,
    rehydrate: bool,
    key_col: str | None,
) -> list[dict]:
    n_tasks = n_tasks_arg or _infer_n_tasks(tokenized_folder)
    if n_tasks == 0:
        raise ValueError(f"No *_tokens.bin files in {tokenized_folder} and --n-tasks not set")
    out = output_folder_override or tokenized_folder
    return [
        dict(rank=rank, n_tasks=n_tasks, tokenized_folder=tokenized_folder,
             data_folder=data_folder, dump_paths=dump_paths,
             text_col=text_col, output_folder=out, rehydrate=rehydrate,
             key_col=key_col)
        for rank in range(n_tasks)
    ]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenized-folder", required=True,
                        help="Root folder with dump-* subdirs (or a single dump folder)")
    parser.add_argument("--data-folder", required=True,
                        help="Root folder of raw parquet files")
    parser.add_argument("--n-dumps", type=int, default=None,
                        help="Number of dumps (default: inferred from dump-* subdir count)")
    parser.add_argument("--n-tasks", type=int, default=None,
                        help="DataTrove tasks per dump (default: inferred from .bin file count)")
    parser.add_argument("--text-col", default="text",
                        help="Parquet column containing document text (default: text)")
    parser.add_argument("--filter-in", nargs="+", default=None,
                        help="Keep only parquet paths containing these substrings (same as prepare_dumps.py)")
    parser.add_argument("--filter-out", nargs="+", default=None,
                        help="Exclude parquet paths containing these substrings (same as prepare_dumps.py)")
    parser.add_argument("--extension", default=".parquet",
                        help="File extension to scan for (default: .parquet)")
    parser.add_argument("--output-folder", default=None,
                        help="Where to write .meta.parquet files (default: same as dump folder)")
    parser.add_argument("--key-col", default=None,
                        help="Parquet column whose value uniquely identifies a document "
                             "(e.g. 'id' for HuggingFace datasets). When set, the metadata "
                             "stores the column's value as 'doc_key' instead of 'parquet_row'.")
    parser.add_argument("--rehydrate", action="store_true",
                        help="Replicate Rehydrater duplication via minhash_cluster_size")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel workers (default: total rank count across all dumps)")
    return parser.parse_args()


def main(args):
    tokenized_root = Path(args.tokenized_folder)
    dump_dirs = sorted(tokenized_root.glob("dump-*"))

    if dump_dirs:
        n_dumps = args.n_dumps or len(dump_dirs)
    else:
        # Single flat dump folder — treat as dump-0
        dump_dirs = [tokenized_root]
        n_dumps = args.n_dumps or 1

    # ── Phase 1: rebuild dump→file assignment ────────────────────────────────
    print(f"Phase 1: rebuilding dump assignment for {n_dumps} dumps from {args.data_folder} ...", flush=True)
    all_dump_paths = rebuild_dump_paths(
        args.data_folder, n_dumps, args.filter_in, args.filter_out, args.extension,
    )

    # ── Phase 2: build worker kwargs for every rank across all dumps ─────────
    print("Phase 2: generating metadata ...", flush=True)
    all_kwargs: list[dict] = []
    for dump_dir in dump_dirs:
        dump_num_str = dump_dir.name[len("dump-"):] if dump_dir.name.startswith("dump-") else "0"
        try:
            dump_idx = int(dump_num_str)
        except ValueError:
            print(f"WARNING: cannot parse dump index from {dump_dir.name}, skipping", flush=True)
            continue
        if dump_idx >= len(all_dump_paths):
            print(f"WARNING: dump-{dump_idx} has no rebuilt paths (only {len(all_dump_paths)} dumps), skipping", flush=True)
            continue
        all_kwargs.extend(_build_worker_kwargs(
            str(dump_dir), args.data_folder, all_dump_paths[dump_idx],
            args.n_tasks, args.text_col, args.output_folder, args.rehydrate,
            args.key_col,
        ))

    n_workers = args.workers or len(all_kwargs)
    with Pool(n_workers) as pool:
        pool.map(_worker, all_kwargs)

    print("Done.", flush=True)


if __name__ == "__main__":
    main(get_args())
