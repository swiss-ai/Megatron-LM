#!/usr/bin/env python3
# Copyright (c) 2026, Swiss AI. All rights reserved.

"""Three-way verification for the Apertus 8B -> 70B multimodal splice.

Compares:
  * source 8B checkpoint: provides multimodal rows for the first 4096 dims.
  * original 70B checkpoint: all text rows, padding rows, second 4096 dims,
    and unrelated model weights must remain identical.
  * patched 70B checkpoint: candidate output to verify.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

from splice_8b_mm_embeddings_into_70b import (
    DEFAULT_SOURCE_HIDDEN_SIZE,
    DEFAULT_SOURCE_VOCAB_SIZE,
    DEFAULT_TARGET_HIDDEN_SIZE,
    DEFAULT_TARGET_PADDED_VOCAB_SIZE,
    DEFAULT_TARGET_VOCAB_SIZE,
    OUTPUT_LAYER_KEY,
    PATCH_CHUNK_ROWS,
    READ_CHUNK_BYTES,
    WORD_EMBED_KEY,
    DcpTable,
    Table,
    _assert_same_bytes,
    _assert_spliced_bytes,
    checkpoint_root_from_iter,
    discover_source_table,
    discover_table,
    find_iteration_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify a patched 70B checkpoint against original 70B and source 8B."
    )
    parser.add_argument("--source-8b", required=True)
    parser.add_argument("--original-70b", required=True)
    parser.add_argument("--patched-70b", required=True)
    parser.add_argument("--source-vocab-size", type=int, default=DEFAULT_SOURCE_VOCAB_SIZE)
    parser.add_argument("--target-vocab-size", type=int, default=DEFAULT_TARGET_VOCAB_SIZE)
    parser.add_argument(
        "--target-padded-vocab-size",
        type=int,
        default=DEFAULT_TARGET_PADDED_VOCAB_SIZE,
    )
    parser.add_argument("--source-hidden-size", type=int, default=DEFAULT_SOURCE_HIDDEN_SIZE)
    parser.add_argument("--target-hidden-size", type=int, default=DEFAULT_TARGET_HIDDEN_SIZE)
    parser.add_argument(
        "--skip-model-weight-compare",
        action="store_true",
        help="Skip the full byte comparison of all non-spliced checkpoint payloads.",
    )
    return parser.parse_args()


def compare_file_bytes(left: Path, right: Path, *, label: str) -> None:
    if left.stat().st_size != right.stat().st_size:
        raise AssertionError(
            f"{label}: size differs: {left.stat().st_size} != {right.stat().st_size}"
        )
    with open(left, "rb") as left_fh, open(right, "rb") as right_fh:
        offset = 0
        while True:
            left_chunk = left_fh.read(READ_CHUNK_BYTES)
            right_chunk = right_fh.read(READ_CHUNK_BYTES)
            if left_chunk != right_chunk:
                raise AssertionError(f"{label}: byte mismatch near offset {offset}")
            if not left_chunk:
                return
            offset += len(left_chunk)


def compare_zip_member_bytes(
    left_zip: zipfile.ZipFile,
    right_zip: zipfile.ZipFile,
    member: str,
    *,
    label: str,
) -> None:
    left_info = left_zip.getinfo(member)
    right_info = right_zip.getinfo(member)
    if left_info.file_size != right_info.file_size:
        raise AssertionError(
            f"{label}:{member}: size differs {left_info.file_size} != {right_info.file_size}"
        )
    with left_zip.open(member) as left_fh, right_zip.open(member) as right_fh:
        offset = 0
        while True:
            left_chunk = left_fh.read(READ_CHUNK_BYTES)
            right_chunk = right_fh.read(READ_CHUNK_BYTES)
            if left_chunk != right_chunk:
                raise AssertionError(f"{label}:{member}: byte mismatch near offset {offset}")
            if not left_chunk:
                return
            offset += len(left_chunk)


def compare_zip_except_members(left: Path, right: Path, ignored: set[str], *, label: str) -> None:
    with zipfile.ZipFile(left) as left_zip, zipfile.ZipFile(right) as right_zip:
        left_names = set(left_zip.namelist())
        right_names = set(right_zip.namelist())
        if left_names != right_names:
            raise AssertionError(f"{label}: zip member sets differ")
        for member in sorted(left_names):
            if member in ignored:
                continue
            compare_zip_member_bytes(left_zip, right_zip, member, label=label)


def table_patched_relpaths(
    table: Table | None,
    root: Path,
    *,
    source_vocab_size: int,
    target_vocab_size: int,
) -> dict[Path, set[str]]:
    if table is None:
        return {}
    ignored: dict[Path, set[str]] = {}
    for shard in table.shards:
        global_start = shard.tp_rank * table.rows_per_tp
        global_end = global_start + table.rows_per_tp
        patch_start = max(source_vocab_size, global_start)
        patch_end = min(target_vocab_size, global_end)
        if patch_start >= patch_end:
            continue
        rel = shard.path.relative_to(root)
        ignored.setdefault(rel, set()).add(shard.tensor.storage_member)
    return ignored


def merge_ignored_members(*items: dict[Path, set[str]]) -> dict[Path, set[str]]:
    merged: dict[Path, set[str]] = {}
    for item in items:
        for path, members in item.items():
            merged.setdefault(path, set()).update(members)
    return merged


def verify_non_spliced_payloads(
    *,
    original_root: Path,
    patched_root: Path,
    ignored_members: dict[Path, set[str]],
) -> None:
    original_files = {
        path.relative_to(original_root): path for path in original_root.rglob("*") if path.is_file()
    }
    patched_files = {
        path.relative_to(patched_root): path for path in patched_root.rglob("*") if path.is_file()
    }
    if set(original_files) != set(patched_files):
        missing = sorted(set(original_files) - set(patched_files))
        extra = sorted(set(patched_files) - set(original_files))
        raise AssertionError(f"checkpoint file sets differ: missing={missing[:10]}, extra={extra[:10]}")

    full_file_count = 0
    zip_file_count = 0
    for rel in sorted(original_files):
        left = original_files[rel]
        right = patched_files[rel]
        if rel in ignored_members:
            compare_zip_except_members(left, right, ignored_members[rel], label=str(rel))
            zip_file_count += 1
        else:
            compare_file_bytes(left, right, label=str(rel))
            full_file_count += 1
    print(
        "Verified non-spliced payloads: "
        f"{full_file_count} files byte-identical, {zip_file_count} checkpoint files "
        "identical except expected embedding/output storage members."
    )


def verify_table_rows(
    *,
    label: str,
    source: Table | DcpTable,
    original: Table,
    patched: Table,
    source_vocab_size: int,
    target_vocab_size: int,
    target_padded_vocab_size: int,
    source_hidden_size: int,
) -> None:
    source_row_bytes = source_hidden_size * source.element_size
    target_row_bytes = patched.hidden * patched.element_size

    if original.padded_vocab_size != patched.padded_vocab_size:
        raise AssertionError(f"{label}: original and patched padded vocab differ")
    if patched.padded_vocab_size != target_padded_vocab_size:
        raise AssertionError(
            f"{label}: patched padded vocab {patched.padded_vocab_size} != {target_padded_vocab_size}"
        )

    print(f"Verifying {label} text rows against original 70B...")
    verify_identical_range(
        label=f"{label} text",
        left=original,
        right=patched,
        start=0,
        end=source_vocab_size,
    )

    print(f"Verifying {label} multimodal rows against 8B/original 70B...")
    row = source_vocab_size
    while row < target_vocab_size:
        nrows = min(PATCH_CHUNK_ROWS, target_vocab_size - row)
        source_bytes = source.read_global_rows(row, nrows)
        original_bytes = original.read_global_rows(row, nrows)
        patched_bytes = patched.read_global_rows(row, nrows)
        _assert_spliced_bytes(
            label=f"{label} multimodal rows [{row},{row + nrows})",
            output_bytes=patched_bytes,
            source_bytes=source_bytes,
            original_bytes=original_bytes,
            nrows=nrows,
            source_row_bytes=source_row_bytes,
            target_row_bytes=target_row_bytes,
        )
        row += nrows

    print(f"Verifying {label} padding rows against original 70B...")
    verify_identical_range(
        label=f"{label} padding",
        left=original,
        right=patched,
        start=target_vocab_size,
        end=target_padded_vocab_size,
    )
    print(f"Verified {label}.")


def verify_identical_range(
    *,
    label: str,
    left: Table,
    right: Table,
    start: int,
    end: int,
) -> None:
    row = start
    while row < end:
        nrows = min(PATCH_CHUNK_ROWS, end - row)
        _assert_same_bytes(
            f"{label} rows [{row},{row + nrows})",
            right.read_global_rows(row, nrows),
            left.read_global_rows(row, nrows),
        )
        row += nrows


def main() -> None:
    args = parse_args()
    source_iter = find_iteration_dir(Path(args.source_8b).resolve())
    original_iter = find_iteration_dir(Path(args.original_70b).resolve())
    patched_iter = find_iteration_dir(Path(args.patched_70b).resolve())
    original_root = checkpoint_root_from_iter(original_iter)
    patched_root = checkpoint_root_from_iter(patched_iter)

    source_embed = discover_source_table(source_iter, WORD_EMBED_KEY, prefer_pp="min")
    original_embed = discover_table(original_iter, WORD_EMBED_KEY, prefer_pp="min")
    patched_embed = discover_table(patched_iter, WORD_EMBED_KEY, prefer_pp="min")
    if source_embed is None or original_embed is None or patched_embed is None:
        raise SystemExit("Could not discover embedding table in all three checkpoints.")

    source_output = discover_source_table(source_iter, OUTPUT_LAYER_KEY, prefer_pp="max")
    original_output = discover_table(original_iter, OUTPUT_LAYER_KEY, prefer_pp="max")
    patched_output = discover_table(patched_iter, OUTPUT_LAYER_KEY, prefer_pp="max")

    verify_table_rows(
        label="embedding.word_embeddings.weight",
        source=source_embed,
        original=original_embed,
        patched=patched_embed,
        source_vocab_size=args.source_vocab_size,
        target_vocab_size=args.target_vocab_size,
        target_padded_vocab_size=args.target_padded_vocab_size,
        source_hidden_size=args.source_hidden_size,
    )

    if original_output is not None or patched_output is not None:
        if original_output is None or patched_output is None:
            raise AssertionError("original/patched output_layer presence differs")
        output_source = source_output if source_output is not None else source_embed
        verify_table_rows(
            label="output_layer.weight",
            source=output_source,
            original=original_output,
            patched=patched_output,
            source_vocab_size=args.source_vocab_size,
            target_vocab_size=args.target_vocab_size,
            target_padded_vocab_size=args.target_padded_vocab_size,
            source_hidden_size=args.source_hidden_size,
        )

    if not args.skip_model_weight_compare:
        ignored_members = merge_ignored_members(
            table_patched_relpaths(
                patched_embed,
                patched_root,
                source_vocab_size=args.source_vocab_size,
                target_vocab_size=args.target_vocab_size,
            ),
            table_patched_relpaths(
                patched_output,
                patched_root,
                source_vocab_size=args.source_vocab_size,
                target_vocab_size=args.target_vocab_size,
            ),
        )
        verify_non_spliced_payloads(
            original_root=original_root,
            patched_root=patched_root,
            ignored_members=ignored_members,
        )

    print("RESULT: three-way splice verification passed.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
