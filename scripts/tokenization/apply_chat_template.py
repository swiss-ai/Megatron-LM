#!/usr/bin/env python3
"""
to_apertus_parquet.py
─────────────────────
Converts a linearised HuggingFace dataset (or JSON file) into a parquet file
by applying the Apertus chat template to each conversation.

Key design: workers receive only file paths + row ranges (not the data itself),
so each worker reads its own slice independently. This keeps IPC minimal and
all CPU cores busy throughout.

Usage
─────
python to_apertus_parquet.py \
    --input      /path/to/linearised_dataset \
    --output     output.parquet \
    --tokenizer  swiss-ai/Apertus-8B-Instruct-2509 \
    [--split     train]            \
    [--num-proc  287]              \
    [--tasks-per-worker 8]
"""

import argparse
import glob
import json
import os
import queue
import threading
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

_tokenizer = None

def _init_worker(tokenizer_path: str) -> None:
    """Load the tokenizer once per worker process (not once per task)."""
    global _tokenizer
    from transformers import AutoTokenizer
    _tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)


def _render_conversation(record: dict) -> dict:
    """
    Extract messages from a record, apply the chat template, and return
    the record with a new "text" field (replacing "messages").
    """
    bos = _tokenizer.bos_token or ""
    messages = record.get("messages", [])

    # Pull settings out of the special "developer" message, then discard it
    enable_thinking = False
    tools = None
    regular_messages = []

    for msg in messages:
        if msg["role"] == "developer":
            dev = msg.get("content", {})
            enable_thinking = bool(dev.get("has_thinking", False))
            raw_tools = dev.get("tools", "")
            if raw_tools:
                tools = json.loads(raw_tools) if isinstance(raw_tools, str) else raw_tools
        else:
            regular_messages.append(msg)

    text = _tokenizer.apply_chat_template(
        regular_messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
        tools=tools,
    )

    # Strip leading BOS token (the trainer adds it back later)
    if bos and text.startswith(bos):
        text = text[len(bos):]

    return {**{k: v for k, v in record.items() if k != "messages"}, "text": text}

def _process_arrow_slice(task: tuple[str, int, int]) -> list[dict]:
    """
    Worker task for Arrow shards.
    Opens the shard file, reads only the assigned row range, renders each row.
    """
    arrow_path, row_start, row_end = task

    try:
        table = pa.ipc.open_stream(arrow_path).read_all()
    except pa.lib.ArrowInvalid:
        table = pa.ipc.open_file(arrow_path).read_all()

    table = table.slice(row_start, row_end - row_start)

    # Convert to list-of-dicts for easy iteration
    columns = {name: table.column(name).to_pylist() for name in table.schema.names}
    records = [
        {name: columns[name][i] for name in table.schema.names}
        for i in range(row_end - row_start)
    ]

    return [_render_conversation(rec) for rec in records]


def _process_json_chunk(records: list[dict]) -> list[dict]:
    """Worker task for JSON input."""
    return [_render_conversation(rec) for rec in records]


def _find_arrow_files(input_path: str, split: str) -> list[str]:
    """Search common HuggingFace dataset directory layouts for .arrow files."""
    patterns = [
        os.path.join(input_path, split, "data-*.arrow"),
        os.path.join(input_path, split, "*.arrow"),
        os.path.join(input_path, "data-*.arrow"),
        os.path.join(input_path, "*.arrow"),
    ]
    for pat in patterns:
        files = sorted(glob.glob(pat))
        if files:
            return files
    return []


def _count_rows(arrow_path: str) -> int:
    """Return total row count for an Arrow shard (handles both IPC formats)."""
    try:
        reader = pa.ipc.open_stream(arrow_path)
        return sum(batch.num_rows for batch in reader)
    except pa.lib.ArrowInvalid:
        reader = pa.ipc.open_file(arrow_path)
        return sum(reader.get_batch(i).num_rows for i in range(reader.num_record_batches))


def _build_arrow_tasks(
    arrow_files: list[str],
    num_proc: int,
    tasks_per_worker: int,
    sample: int | None,
) -> list[tuple[str, int, int]]:
    """
    Split Arrow shards into (file, row_start, row_end) slices — one slice per task.
    More tasks than workers = better load balancing (fast workers pick up extras).
    """
    print("  Counting rows per shard...", flush=True)
    shard_sizes = []
    total_rows = 0
    for path in arrow_files:
        n = _count_rows(path)
        shard_sizes.append((path, n))
        total_rows += n
        print(f"    {os.path.basename(path)}: {n:,} rows", flush=True)

    total_rows = min(total_rows, sample) if sample else total_rows
    slice_size = max(1, total_rows // (num_proc * tasks_per_worker))

    print(f"\n  Total rows  : {total_rows:,}")
    print(f"  Slice size  : {slice_size:,}  ({num_proc * tasks_per_worker} target tasks)\n", flush=True)

    tasks = []
    rows_assigned = 0
    for path, shard_rows in shard_sizes:
        for start in range(0, shard_rows, slice_size):
            if sample and rows_assigned >= sample:
                break
            end = min(start + slice_size, shard_rows)
            if sample:
                end = min(end, start + (sample - rows_assigned))
            tasks.append((path, start, end))
            rows_assigned += end - start

    print(f"  Built {len(tasks)} tasks across {len(arrow_files)} shards\n", flush=True)
    return tasks


_STOP = object()  # sentinel value to shut down the writer thread

def _writer_thread(out_path: str, work_queue: queue.Queue, write_batch: int,
                   rows_written: list[int], t0: float) -> None:
    """
    Runs in a background thread. Pulls rendered rows from work_queue,
    buffers them, and flushes to parquet in batches to avoid tiny writes.
    """
    writer = None
    schema = None
    buffer = []

    def flush_to_parquet(rows: list[dict]) -> None:
        nonlocal writer, schema
        if not rows:
            return

        col_keys = list(rows[0].keys())
        table = pa.table({k: [r.get(k) for r in rows] for k in col_keys})

        if writer is None:                      # first flush -> infer schema
            schema = table.schema
            writer = pq.ParquetWriter(out_path, schema)
        else:
            table = table.cast(schema)          # keep schema consistent

        writer.write_table(table)
        rows_written[0] += len(rows)
        elapsed = time.time() - t0
        rate = rows_written[0] / elapsed if elapsed else 0
        print(f"  {rows_written[0]:>12,} rows written  ({rate:,.0f} rows/s)", flush=True)

    while True:
        item = work_queue.get()
        if item is _STOP:
            break
        buffer.extend(item)
        if len(buffer) >= write_batch:
            flush_to_parquet(buffer)
            buffer = []

    flush_to_parquet(buffer)    # write whatever remains
    if writer:
        writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply Apertus chat template and save to parquet"
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input",      help="HF dataset directory (save_to_disk format)")
    src.add_argument("--input-json", help="JSON file containing a list of conversations")

    parser.add_argument("--output",           required=True)
    parser.add_argument("--tokenizer",        required=True)
    parser.add_argument("--split",            default="train")
    parser.add_argument("--sample",           type=int, default=None,
                        help="Limit to this many rows (for testing)")
    parser.add_argument("--num-proc",         type=int,
                        default=max(1, (os.cpu_count() or 4) - 1))
    parser.add_argument("--tasks-per-worker", type=int, default=8,
                        help="Task slices per worker — more = better load balancing")
    parser.add_argument("--write-batch",      type=int, default=200_000,
                        help="Rows per parquet flush")
    args = parser.parse_args()

    print(f"\n{args.num_proc} workers | tasks_per_worker={args.tasks_per_worker} | "
          f"write_batch={args.write_batch:,}")
    print(f"Tokenizer : {args.tokenizer}\n")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Build work items and choose the right worker function
    if args.input_json:
        print(f"Loading JSON: {args.input_json}")
        all_records = json.load(open(args.input_json))
        if args.sample:
            all_records = all_records[:args.sample]
        slice_size = max(1, len(all_records) // (args.num_proc * args.tasks_per_worker))
        work_items = [all_records[s:s + slice_size] for s in range(0, len(all_records), slice_size)]
        worker_fn = _process_json_chunk
        print(f"  {len(all_records):,} records -> {len(work_items)} tasks\n")
    else:
        print(f"Loading Arrow dataset: {args.input}")
        arrow_files = _find_arrow_files(args.input, args.split)
        if not arrow_files:
            raise RuntimeError(
                f"No .arrow files found under {args.input!r} (split={args.split!r}). "
                "Check --split or use --input-json."
            )
        print(f"  Found {len(arrow_files)} shard(s)")
        work_items = _build_arrow_tasks(arrow_files, args.num_proc, args.tasks_per_worker, args.sample)
        worker_fn = _process_arrow_slice

    # Start the background writer thread
    rows_written = [0]
    t0 = time.time()
    work_queue: queue.Queue = queue.Queue(maxsize=args.num_proc * 2)
    writer = threading.Thread(
        target=_writer_thread,
        args=(args.output, work_queue, args.write_batch, rows_written, t0),
        daemon=True,
    )
    writer.start()

    # Dispatch tasks to the worker pool; feed results to the writer as they arrive
    with Pool(processes=args.num_proc, initializer=_init_worker, initargs=(args.tokenizer,)) as pool:
        for result in pool.imap(worker_fn, work_items, chunksize=1):
            work_queue.put(result)

    work_queue.put(_STOP)   # tell the writer there's nothing more coming
    writer.join()

    # Summary + quick sanity check
    n, elapsed = rows_written[0], time.time() - t0
    print(f"\nDone. {n:,} rows -> {args.output}")
    print(f"Time: {elapsed:.1f}s  ({n / elapsed:,.0f} rows/s avg)")

    sample_text = pq.read_table(args.output, columns=["text"]).slice(0, 1)["text"][0].as_py()
    print("\n--- First row (first 600 chars) ---")
    print(sample_text[:600])


if __name__ == "__main__":
    main()