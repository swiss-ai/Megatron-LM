#!/usr/bin/env python3
# Copyright (c) 2026, Swiss AI. All rights reserved.

"""One-off splice of Apertus 8B multimodal embeddings into Apertus 70B.

This script patches a legacy Megatron torch checkpoint directory. It copies the
70B checkpoint to a new output directory, then only changes multimodal vocab
rows in the word embedding and, when present, the untied output layer:

    70B[row, :4096] = 8B[row, :]
    70B[row, 4096:] = original 70B[row, 4096:]

Text rows and padding rows are verified to remain bitwise unchanged.

The script intentionally does not touch torch_dist checkpoints, optimizer
state, common.pt, metadata.json, or Megatron core code.
"""

from __future__ import annotations

import argparse
import os
import pickletools
import re
import shutil
import struct
import sys
import zlib
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Literal


WORD_EMBED_KEY = "embedding.word_embeddings.weight"
OUTPUT_LAYER_KEY = "output_layer.weight"
CHECKPOINT_BASENAME = "model_optim_rng.pt"

DEFAULT_SOURCE_VOCAB_SIZE = 131_072
DEFAULT_TARGET_VOCAB_SIZE = 266_440
DEFAULT_TARGET_PADDED_VOCAB_SIZE = 266_752
DEFAULT_SOURCE_HIDDEN_SIZE = 4096
DEFAULT_TARGET_HIDDEN_SIZE = 8192

READ_CHUNK_BYTES = 32 * 1024 * 1024
PATCH_CHUNK_ROWS = 512

_PICKLE_STRING_OPS = frozenset({"BINUNICODE", "SHORT_BINUNICODE", "UNICODE"})
_PICKLE_INT_OPS = frozenset({"BININT", "BININT1", "BININT2", "LONG1", "LONG4", "INT"})

# Maps torch.dtype string forms (e.g. "bfloat16") to the storage class names that
# torch.save serializes via GLOBAL/STACK_GLOBAL ops (e.g. "BFloat16Storage").
_TORCH_DTYPE_TO_STORAGE_CLASS = {
    "float64": "DoubleStorage",
    "float32": "FloatStorage",
    "float16": "HalfStorage",
    "bfloat16": "BFloat16Storage",
    "int64": "LongStorage",
    "int32": "IntStorage",
    "int16": "ShortStorage",
    "int8": "CharStorage",
    "uint8": "ByteStorage",
    "bool": "BoolStorage",
}

PreferPP = Literal["min", "max"]


@dataclass(frozen=True)
class TensorInfo:
    storage_member: str
    shape: tuple[int, int]
    strides: tuple[int, int]
    storage_offset: int
    element_size: int
    data_offset: int
    dtype: str

    @property
    def rows(self) -> int:
        return self.shape[0]

    @property
    def hidden(self) -> int:
        return self.shape[1]

    @property
    def row_bytes(self) -> int:
        return self.hidden * self.element_size

    @property
    def storage_bytes(self) -> int:
        return self.rows * self.row_bytes


@dataclass(frozen=True)
class RankShard:
    path: Path
    tp_rank: int
    pp_rank: int
    ep_rank: int
    tensor: TensorInfo


class _RowReadable:
    """Mixin providing `read_global_rows` over an implementation-defined sub-region locator.

    Subclasses declare ``padded_vocab_size`` and implement ``_read_at(cursor, max_take)``
    returning ``(bytes, rows_advanced)`` for one sub-region starting at ``cursor``.
    """

    padded_vocab_size: int

    def _read_at(self, cursor: int, max_take: int) -> tuple[bytes, int]:
        raise NotImplementedError

    def read_global_rows(self, start: int, nrows: int) -> bytes:
        if nrows < 0:
            raise ValueError("nrows must be non-negative")
        if nrows == 0:
            return b""
        end = start + nrows
        if start < 0 or end > self.padded_vocab_size:
            raise ValueError(
                f"row range [{start}, {end}) is outside padded vocab {self.padded_vocab_size}"
            )

        pieces: list[bytes] = []
        cursor = start
        while cursor < end:
            chunk_bytes, took = self._read_at(cursor, end - cursor)
            pieces.append(chunk_bytes)
            cursor += took
        return b"".join(pieces)


@dataclass(frozen=True)
class Table(_RowReadable):
    key: str
    shards: tuple[RankShard, ...]
    padded_vocab_size: int
    tp_size: int
    pp_rank: int
    hidden: int
    element_size: int
    dtype: str

    @property
    def rows_per_tp(self) -> int:
        return self.shards[0].tensor.rows

    def shard_for_global_row(self, row: int) -> RankShard:
        if row < 0 or row >= self.padded_vocab_size:
            raise ValueError(f"row {row} is outside padded vocab {self.padded_vocab_size}")
        tp_rank = row // self.rows_per_tp
        return self.shards[tp_rank]

    def _read_at(self, cursor: int, max_take: int) -> tuple[bytes, int]:
        shard = self.shard_for_global_row(cursor)
        shard_start = shard.tp_rank * self.rows_per_tp
        local_start = cursor - shard_start
        take = min(max_take, shard.tensor.rows - local_start)
        return read_tensor_rows(shard.path, shard.tensor, local_start, take), take


@dataclass(frozen=True)
class DcpChunk:
    path: Path
    archive_offset: int
    archive_length: int
    data_offset: int
    global_start: int
    rows: int
    hidden: int
    element_size: int

    @property
    def row_bytes(self) -> int:
        return self.hidden * self.element_size


@dataclass(frozen=True)
class DcpTable(_RowReadable):
    key: str
    chunks: tuple[DcpChunk, ...]
    padded_vocab_size: int
    hidden: int
    element_size: int
    dtype: str

    def chunk_for_global_row(self, row: int) -> DcpChunk:
        for chunk in self.chunks:
            if chunk.global_start <= row < chunk.global_start + chunk.rows:
                return chunk
        raise ValueError(f"row {row} is not covered by {self.key} DCP chunks")

    def _read_at(self, cursor: int, max_take: int) -> tuple[bytes, int]:
        chunk = self.chunk_for_global_row(cursor)
        local_start = cursor - chunk.global_start
        take = min(max_take, chunk.rows - local_start)
        return read_dcp_chunk_rows(chunk, local_start, take), take


class BoundedFile:
    """Seekable view over a byte range inside a larger file."""

    def __init__(self, path: Path, offset: int, length: int):
        self._fh = open(path, "rb")
        self._offset = offset
        self._length = length
        self._pos = 0

    def close(self) -> None:
        self._fh.close()

    def tell(self) -> int:
        return self._pos

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            new_pos = offset
        elif whence == os.SEEK_CUR:
            new_pos = self._pos + offset
        elif whence == os.SEEK_END:
            new_pos = self._length + offset
        else:
            raise ValueError(f"unknown whence: {whence}")
        if new_pos < 0:
            raise ValueError("negative seek position")
        self._pos = new_pos
        return self._pos

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self._length - self._pos
        size = min(size, self._length - self._pos)
        if size <= 0:
            return b""
        self._fh.seek(self._offset + self._pos)
        data = self._fh.read(size)
        self._pos += len(data)
        return data

    def __enter__(self) -> "BoundedFile":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Splice Apertus 8B multimodal embedding rows into an Apertus 70B checkpoint."
    )
    parser.add_argument("--source-8b", required=True, help="8B extended Megatron torch checkpoint")
    parser.add_argument("--target-70b", required=True, help="70B extended Megatron torch checkpoint")
    parser.add_argument("--output-70b", required=True, help="Output checkpoint directory to create")
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
        "--overwrite-output",
        action="store_true",
        help="Delete --output-70b first if it already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover and validate checkpoint layout without copying or patching.",
    )
    return parser.parse_args()


def find_iteration_dir(checkpoint: Path) -> Path:
    if checkpoint.name.startswith("iter_") or checkpoint.name == "release":
        return checkpoint

    tracker = checkpoint / "latest_checkpointed_iteration.txt"
    if tracker.exists():
        value = tracker.read_text(encoding="utf-8").strip()
        if value == "release":
            return checkpoint / "release"
        return checkpoint / f"iter_{int(value):07d}"

    iter_dirs = sorted(p for p in checkpoint.iterdir() if p.is_dir() and p.name.startswith("iter_"))
    if len(iter_dirs) == 1:
        return iter_dirs[0]
    if not iter_dirs:
        raise FileNotFoundError(f"Could not find an iter_* directory under {checkpoint}")
    raise ValueError(
        f"Multiple iter_* directories found under {checkpoint}; provide the exact iter_* path."
    )


def checkpoint_root_from_iter(iter_dir: Path) -> Path:
    if iter_dir.name.startswith("iter_") or iter_dir.name == "release":
        return iter_dir.parent
    return iter_dir


def iter_rank_files(iter_dir: Path) -> Iterable[Path]:
    yield from sorted(iter_dir.glob(f"mp_rank_*/{CHECKPOINT_BASENAME}"))


def is_legacy_torch_checkpoint(iter_dir: Path) -> bool:
    return any(iter_rank_files(iter_dir))


def is_torch_dist_checkpoint(iter_dir: Path) -> bool:
    return (iter_dir / ".metadata").exists() and any(iter_dir.glob("*.distcp"))


def parse_rank_dir(rank_dir: str) -> tuple[int, int, int]:
    match = re.fullmatch(r"mp_rank_(\d+)(?:_(\d+))?(?:_(\d+))?", rank_dir)
    if match is None:
        raise ValueError(f"Unexpected rank directory name: {rank_dir}")
    tp_rank = int(match.group(1))
    pp_rank = int(match.group(2)) if match.group(2) is not None else 0
    ep_rank = int(match.group(3)) if match.group(3) is not None else 0
    return tp_rank, pp_rank, ep_rank


def _op_is_string(op_name: str) -> bool:
    return op_name in _PICKLE_STRING_OPS


def _op_is_int(op_name: str) -> bool:
    return op_name in _PICKLE_INT_OPS


def _read_pickle_payload(zip_path: Path) -> tuple[str, bytes]:
    with zipfile.ZipFile(zip_path) as zf:
        return _read_pickle_payload_from_zip(zf, str(zip_path))


def _read_pickle_payload_from_zip(zf: zipfile.ZipFile, label: str) -> tuple[str, bytes]:
    pkl_members = [name for name in zf.namelist() if name.endswith("/data.pkl")]
    if len(pkl_members) != 1:
        raise ValueError(f"Expected one data.pkl in {label}, found {pkl_members}")
    member = pkl_members[0]
    return member.rsplit("/", 1)[0], zf.read(member)


def _read_int_seq_until_tuple(ops, idx: int) -> tuple[list[int], int]:
    values: list[int] = []
    while idx < len(ops):
        op, arg, _pos = ops[idx]
        if _op_is_int(op.name):
            values.append(int(arg))
        elif op.name.startswith("TUPLE"):
            return values, idx + 1
        idx += 1
    return values, idx


def _memoized_storage_classes(ops) -> dict[int, str]:
    """Return pickle memo ids that resolve to torch storage classes."""
    memo: dict[int, str] = {}
    pending_storage_class: str | None = None
    for op, arg, _pos in ops:
        if _op_is_string(op.name) and isinstance(arg, str):
            if arg.endswith("Storage") and arg != "Storage":
                pending_storage_class = arg
            continue

        if op.name == "GLOBAL" and isinstance(arg, str):
            parts = arg.split(maxsplit=1)
            if len(parts) == 2 and parts[1].endswith("Storage"):
                pending_storage_class = parts[1]
            continue

        if pending_storage_class is not None and op.name in {"BINPUT", "LONG_BINPUT"}:
            memo[int(arg)] = pending_storage_class
            pending_storage_class = None
            continue

        if op.name not in {"BINPUT", "LONG_BINPUT"}:
            # The storage class object is normally memoized immediately. If any
            # unrelated op appears first, avoid attaching the dtype to the wrong
            # pickle memo slot.
            pending_storage_class = None
    return memo


def _parse_tensor_persid(
    ops,
    *,
    search_from: int,
    label: str,
    storage_class_memo: dict[int, str],
    max_lookahead: int | None = None,
) -> tuple[str, str, int, tuple[int, int], tuple[int, int]]:
    """Parse a torch.save tensor's BINPERSID block.

    Walks ops starting at ``search_from`` until a BINPERSID op is found, then parses
    the trailing storage_offset, shape tuple, and strides tuple. Returns the storage
    id (the last digit-string seen before BINPERSID), the storage class name (e.g.
    "BFloat16Storage"), storage_offset, 2D shape, and 2D strides.
    """
    end = len(ops) if max_lookahead is None else min(search_from + max_lookahead, len(ops))
    binpersid_idx = None
    strings: list[str] = []
    storage_class: str | None = None
    for idx in range(search_from, end):
        op, arg, _pos = ops[idx]
        if _op_is_string(op.name) and isinstance(arg, str):
            strings.append(arg)
            if storage_class is None and arg.endswith("Storage") and arg != "Storage":
                # STACK_GLOBAL form: BINUNICODE 'torch'; BINUNICODE 'FloatStorage'; STACK_GLOBAL.
                storage_class = arg
        elif op.name == "GLOBAL" and isinstance(arg, str):
            # GLOBAL form: arg is "<module> <classname>" (space-separated by pickletools).
            parts = arg.split(maxsplit=1)
            if storage_class is None and len(parts) == 2 and parts[1].endswith("Storage"):
                storage_class = parts[1]
        elif op.name in {"BINGET", "LONG_BINGET"}:
            storage_class = storage_class or storage_class_memo.get(int(arg))
        if op.name == "BINPERSID":
            binpersid_idx = idx
            break
    if binpersid_idx is None:
        raise ValueError(f"Could not find tensor storage metadata for {label}")
    if storage_class is None:
        raise ValueError(f"Could not find tensor storage dtype class for {label}")

    storage_id = next((v for v in reversed(strings) if v.isdigit()), None)
    if storage_id is None:
        raise ValueError(f"Could not find storage id for {label}")

    idx = binpersid_idx + 1
    if not _op_is_int(ops[idx][0].name):
        raise ValueError(f"Could not parse storage offset for {label}")
    storage_offset = int(ops[idx][1])
    idx += 1

    dims, idx = _read_int_seq_until_tuple(ops, idx)
    if len(dims) != 2:
        raise ValueError(f"Expected 2D tensor for {label}, got shape {tuple(dims)}")

    strides, _ = _read_int_seq_until_tuple(ops, idx)
    if len(strides) != 2:
        raise ValueError(f"Expected 2D strides for {label}, got {tuple(strides)}")

    return storage_id, storage_class, storage_offset, (dims[0], dims[1]), (strides[0], strides[1])


def _element_size_for_shape(
    zf: zipfile.ZipFile, storage_member: str, shape: tuple[int, int], *, label: str
) -> int:
    info = zf.getinfo(storage_member)
    tensor_numel = shape[0] * shape[1]
    if tensor_numel == 0 or info.file_size % tensor_numel != 0:
        raise ValueError(
            f"Storage {storage_member} size {info.file_size} is incompatible with "
            f"shape {shape} in {label}"
        )
    return info.file_size // tensor_numel


def read_tensor_info(zip_path: Path, key: str) -> TensorInfo | None:
    prefix, payload = _read_pickle_payload(zip_path)
    ops = list(pickletools.genops(payload))
    storage_class_memo = _memoized_storage_classes(ops)
    matches = [
        i
        for i, (op, arg, _pos) in enumerate(ops)
        if _op_is_string(op.name) and arg == key
    ]
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"Found {len(matches)} occurrences of {key!r} in {zip_path}")

    label = f"{key!r} in {zip_path}"
    storage_id, storage_class, storage_offset, shape, strides = _parse_tensor_persid(
        ops,
        search_from=matches[0] + 1,
        label=label,
        storage_class_memo=storage_class_memo,
        max_lookahead=100,
    )

    storage_member = f"{prefix}/data/{storage_id}"
    with zipfile.ZipFile(zip_path) as zf:
        element_size = _element_size_for_shape(zf, storage_member, shape, label=label)
        data_offset = zip_member_data_offset(zf, storage_member)

    return TensorInfo(
        storage_member=storage_member,
        shape=shape,
        strides=strides,
        storage_offset=storage_offset,
        element_size=element_size,
        data_offset=data_offset,
        dtype=storage_class,
    )


def read_single_tensor_info_from_zip(
    zf: zipfile.ZipFile,
    *,
    label: str,
    archive_base_offset: int,
) -> tuple[TensorInfo, int]:
    prefix, payload = _read_pickle_payload_from_zip(zf, label)
    ops = list(pickletools.genops(payload))
    storage_class_memo = _memoized_storage_classes(ops)
    storage_id, storage_class, storage_offset, shape, strides = _parse_tensor_persid(
        ops, search_from=0, label=label, storage_class_memo=storage_class_memo
    )

    storage_member = f"{prefix}/data/{storage_id}"
    element_size = _element_size_for_shape(zf, storage_member, shape, label=label)
    if storage_offset != 0 or strides != (shape[1], 1):
        raise ValueError(
            f"Only contiguous tensor chunks are supported in {label}; "
            f"offset={storage_offset}, strides={strides}"
        )

    data_offset = archive_base_offset + zip_member_data_offset(zf, storage_member)
    tensor = TensorInfo(
        storage_member=storage_member,
        shape=shape,
        strides=strides,
        storage_offset=storage_offset,
        element_size=element_size,
        data_offset=data_offset,
        dtype=storage_class,
    )
    return tensor, data_offset


def discover_table(iter_dir: Path, key: str, *, prefer_pp: PreferPP) -> Table | None:
    if prefer_pp not in ("min", "max"):
        raise ValueError(f"prefer_pp must be 'min' or 'max', got {prefer_pp!r}")
    shards_by_pp: dict[int, list[RankShard]] = {}
    for rank_file in iter_rank_files(iter_dir):
        tensor = read_tensor_info(rank_file, key)
        if tensor is None:
            continue
        tp_rank, pp_rank, ep_rank = parse_rank_dir(rank_file.parent.name)
        if ep_rank != 0:
            raise NotImplementedError("Expert-parallel checkpoint layouts are not supported.")
        shards_by_pp.setdefault(pp_rank, []).append(
            RankShard(rank_file, tp_rank, pp_rank, ep_rank, tensor)
        )

    if not shards_by_pp:
        return None

    pp_selector = min if prefer_pp == "min" else max
    pp_rank = pp_selector(shards_by_pp)
    shards = tuple(sorted(shards_by_pp[pp_rank], key=lambda shard: shard.tp_rank))
    tp_ranks = [shard.tp_rank for shard in shards]
    if tp_ranks != list(range(len(shards))):
        raise ValueError(f"{key} shards have non-contiguous TP ranks: {tp_ranks}")

    first = shards[0].tensor
    for shard in shards:
        tensor = shard.tensor
        if tensor.rows != first.rows:
            raise ValueError(f"{key} shards do not have uniform row counts")
        if tensor.hidden != first.hidden:
            raise ValueError(f"{key} shards do not have uniform hidden sizes")
        if tensor.element_size != first.element_size:
            raise ValueError(f"{key} shards do not have uniform element sizes")
        if tensor.dtype != first.dtype:
            raise ValueError(
                f"{key} shards do not have uniform dtypes: "
                f"tp=0 {first.dtype} vs tp={shard.tp_rank} {tensor.dtype}"
            )
        if tensor.storage_offset != 0 or tensor.strides != (tensor.hidden, 1):
            raise ValueError(
                f"{key} in {shard.path} must be contiguous with storage_offset=0, "
                f"got offset={tensor.storage_offset}, strides={tensor.strides}"
            )

    return Table(
        key=key,
        shards=shards,
        padded_vocab_size=first.rows * len(shards),
        tp_size=len(shards),
        pp_rank=pp_rank,
        hidden=first.hidden,
        element_size=first.element_size,
        dtype=first.dtype,
    )


def discover_dcp_table(iter_dir: Path, key: str) -> DcpTable | None:
    try:
        import torch.distributed.checkpoint as dcp
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch.distributed.checkpoint is required to read torch_dist source checkpoints") from exc

    reader = dcp.FileSystemReader(str(iter_dir))
    metadata = reader.read_metadata()
    if key not in metadata.state_dict_metadata:
        return None

    tensor_metadata = metadata.state_dict_metadata[key]
    size = tuple(int(x) for x in tensor_metadata.size)
    if len(size) != 2:
        raise ValueError(f"{key} in {iter_dir} must be 2D, got shape {size}")

    dtype_str = str(tensor_metadata.properties.dtype).removeprefix("torch.")
    if dtype_str not in _TORCH_DTYPE_TO_STORAGE_CLASS:
        raise ValueError(f"{key} in {iter_dir}: unsupported dtype {tensor_metadata.properties.dtype}")
    storage_class = _TORCH_DTYPE_TO_STORAGE_CLASS[dtype_str]

    chunks: list[DcpChunk] = []
    for index, storage in metadata.storage_data.items():
        if index.fqn != key:
            continue
        if index.offset is None:
            raise ValueError(f"{key} has a storage entry without a tensor offset")
        row_offset = int(index.offset[0])
        col_offset = int(index.offset[1])
        if col_offset != 0:
            raise ValueError(f"{key} has column-sharded DCP chunk at offset {index.offset}")

        storage_path = iter_dir / storage.relative_path
        with BoundedFile(storage_path, int(storage.offset), int(storage.length)) as bounded:
            with zipfile.ZipFile(bounded) as zf:
                tensor, data_offset = read_single_tensor_info_from_zip(
                    zf,
                    label=f"{storage_path}@{storage.offset}",
                    archive_base_offset=int(storage.offset),
                )
        if tensor.hidden != size[1]:
            raise ValueError(
                f"{key} chunk {storage.relative_path}@{storage.offset} has hidden "
                f"{tensor.hidden}, expected {size[1]}"
            )
        if tensor.dtype != storage_class:
            raise ValueError(
                f"{key} chunk {storage.relative_path}@{storage.offset} has dtype "
                f"{tensor.dtype}, expected {storage_class} (from DCP metadata)"
            )
        chunks.append(
            DcpChunk(
                path=storage_path,
                archive_offset=int(storage.offset),
                archive_length=int(storage.length),
                data_offset=data_offset,
                global_start=row_offset,
                rows=tensor.rows,
                hidden=tensor.hidden,
                element_size=tensor.element_size,
            )
        )

    if not chunks:
        return None

    chunks = sorted(chunks, key=lambda chunk: chunk.global_start)
    cursor = 0
    for chunk in chunks:
        if chunk.global_start != cursor:
            raise ValueError(
                f"{key} DCP chunks are not contiguous: expected row {cursor}, "
                f"found {chunk.global_start}"
            )
        cursor += chunk.rows
    if cursor != size[0]:
        raise ValueError(f"{key} DCP chunks cover {cursor} rows, metadata says {size[0]}")

    first = chunks[0]
    for chunk in chunks:
        if chunk.hidden != first.hidden or chunk.element_size != first.element_size:
            raise ValueError(f"{key} DCP chunks have inconsistent shape/dtype")

    return DcpTable(
        key=key,
        chunks=tuple(chunks),
        padded_vocab_size=size[0],
        hidden=size[1],
        element_size=first.element_size,
        dtype=storage_class,
    )


def discover_source_table(iter_dir: Path, key: str, *, prefer_pp: PreferPP) -> Table | DcpTable | None:
    if is_legacy_torch_checkpoint(iter_dir):
        return discover_table(iter_dir, key, prefer_pp=prefer_pp)
    if is_torch_dist_checkpoint(iter_dir):
        return discover_dcp_table(iter_dir, key)
    raise ValueError(f"Unsupported source checkpoint format under {iter_dir}")


def zip_member_data_offset(zf: zipfile.ZipFile, member: str) -> int:
    info = zf.getinfo(member)
    if zf.fp is None:
        raise ValueError("zip file is closed")
    zf.fp.seek(info.header_offset)
    header = zf.fp.read(30)
    if len(header) != 30:
        raise ValueError(f"Could not read local zip header for {member}")
    signature, *_rest, filename_len, extra_len = struct.unpack("<IHHHHHIIIHH", header)
    if signature != 0x04034B50:
        raise ValueError(f"Bad local zip header for {member}")
    return info.header_offset + 30 + filename_len + extra_len


def read_tensor_rows(zip_path: Path, tensor: TensorInfo, local_start: int, nrows: int) -> bytes:
    if nrows == 0:
        return b""
    if local_start < 0 or local_start + nrows > tensor.rows:
        raise ValueError(
            f"Local row range [{local_start}, {local_start + nrows}) outside tensor {tensor.shape}"
        )
    with open(zip_path, "rb") as fh:
        fh.seek(tensor.data_offset + local_start * tensor.row_bytes)
        data = fh.read(nrows * tensor.row_bytes)
    expected = nrows * tensor.row_bytes
    if len(data) != expected:
        raise IOError(f"Short read from {zip_path}:{tensor.storage_member}")
    return data


def read_dcp_chunk_rows(chunk: DcpChunk, local_start: int, nrows: int) -> bytes:
    if nrows == 0:
        return b""
    if local_start < 0 or local_start + nrows > chunk.rows:
        raise ValueError(
            f"Local row range [{local_start}, {local_start + nrows}) outside DCP chunk "
            f"[0,{chunk.rows})"
        )
    with open(chunk.path, "rb") as fh:
        fh.seek(chunk.data_offset + local_start * chunk.row_bytes)
        data = fh.read(nrows * chunk.row_bytes)
    expected = nrows * chunk.row_bytes
    if len(data) != expected:
        raise IOError(f"Short read from DCP chunk {chunk.path}@{chunk.archive_offset}")
    return data


def _stream_storage_crc(zip_path: Path, tensor: TensorInfo) -> int:
    crc = 0
    remaining = tensor.storage_bytes
    with open(zip_path, "rb") as fh:
        fh.seek(tensor.data_offset)
        while remaining:
            chunk = fh.read(min(READ_CHUNK_BYTES, remaining))
            if not chunk:
                raise IOError(f"Short read while computing CRC for {zip_path}:{tensor.storage_member}")
            crc = zlib.crc32(chunk, crc)
            remaining -= len(chunk)
    return crc & 0xFFFFFFFF


def _update_zip_crc(zip_path: Path, member: str, crc: int) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        info = zf.getinfo(member)
        start_dir = zf.start_dir
        data_offset = zip_member_data_offset(zf, member)
    filename = member.encode("utf-8")
    with open(zip_path, "r+b") as fh:
        # Local header CRC field. PyTorch writes data descriptors, but keeping
        # this field consistent is harmless and makes the archive easier to inspect.
        fh.seek(info.header_offset + 14)
        fh.write(struct.pack("<I", crc))

        # Data descriptor CRC field when general-purpose bit 3 is set.
        if info.flag_bits & 0x08:
            descriptor_offset = data_offset + info.compress_size
            fh.seek(descriptor_offset)
            descriptor_head = fh.read(24)
            if descriptor_head[:4] == struct.pack("<I", 0x08074B50):
                fh.seek(descriptor_offset + 4)
            else:
                fh.seek(descriptor_offset)
            fh.write(struct.pack("<I", crc))

        # Central directory CRC field.
        fh.seek(start_dir)
        while True:
            entry_start = fh.tell()
            header = fh.read(46)
            if len(header) < 4:
                break
            signature = struct.unpack("<I", header[:4])[0]
            if signature == 0x06054B50:
                break
            if signature != 0x02014B50:
                raise ValueError(f"Bad central directory signature in {zip_path}")
            fields = struct.unpack("<IHHHHHHIIIHHHHHII", header)
            filename_len = fields[10]
            extra_len = fields[11]
            comment_len = fields[12]
            name = fh.read(filename_len)
            if name == filename:
                fh.seek(entry_start + 16)
                fh.write(struct.pack("<I", crc))
                return
            fh.seek(extra_len + comment_len, os.SEEK_CUR)

    raise ValueError(f"Could not find {member} in central directory of {zip_path}")


def _assert_same_bytes(label: str, left: bytes, right: bytes) -> None:
    if left != right:
        raise AssertionError(f"{label} changed unexpectedly")


def _assert_spliced_bytes(
    *,
    label: str,
    output_bytes: bytes,
    source_bytes: bytes,
    original_bytes: bytes,
    nrows: int,
    source_row_bytes: int,
    target_row_bytes: int,
) -> None:
    for row in range(nrows):
        source_start = row * source_row_bytes
        target_start = row * target_row_bytes
        target_mid = target_start + source_row_bytes
        target_end = target_start + target_row_bytes
        if output_bytes[target_start:target_mid] != source_bytes[
            source_start : source_start + source_row_bytes
        ]:
            raise AssertionError(f"{label}: first half mismatch at local row {row}")
        if output_bytes[target_mid:target_end] != original_bytes[target_mid:target_end]:
            raise AssertionError(f"{label}: second half changed at local row {row}")


def patch_table(
    *,
    label: str,
    source: Table | DcpTable,
    original_target: Table,
    output_target: Table,
    source_vocab_size: int,
    target_vocab_size: int,
    source_hidden_size: int,
) -> None:
    if source.dtype != output_target.dtype:
        raise ValueError(
            f"{label}: source dtype {source.dtype} does not match "
            f"target dtype {output_target.dtype}; refusing raw byte splice "
            f"(byte encodings differ even when element sizes match, e.g. fp16 vs bf16)."
        )
    if source.element_size != output_target.element_size:
        raise ValueError(
            f"{label}: source element size {source.element_size} does not match "
            f"target element size {output_target.element_size}; refusing raw byte splice."
        )
    source_row_bytes = source_hidden_size * source.element_size
    target_row_bytes = output_target.hidden * output_target.element_size

    print(f"Patching {label}:")
    for shard in output_target.shards:
        global_start = shard.tp_rank * output_target.rows_per_tp
        global_end = global_start + output_target.rows_per_tp
        patch_start = max(source_vocab_size, global_start)
        patch_end = min(target_vocab_size, global_end)
        if patch_start >= patch_end:
            continue

        original_shard = original_target.shards[shard.tp_rank]
        local_start = patch_start - global_start
        nrows = patch_end - patch_start
        print(
            f"  tp={shard.tp_rank}: rows [{patch_start},{patch_end}) "
            f"local [{local_start},{local_start + nrows})"
        )

        data_offset = shard.tensor.data_offset
        done = 0
        with open(shard.path, "r+b") as fh:
            while done < nrows:
                rows = min(PATCH_CHUNK_ROWS, nrows - done)
                local = local_start + done
                global_row = patch_start + done

                fh.seek(data_offset + local * target_row_bytes)
                target_bytes = fh.read(rows * target_row_bytes)
                if len(target_bytes) != rows * target_row_bytes:
                    raise IOError(f"Short read while patching {shard.path}")

                source_bytes = source.read_global_rows(global_row, rows)
                patched = bytearray(target_bytes)
                for row in range(rows):
                    src_row = row * source_row_bytes
                    tgt_row = row * target_row_bytes
                    patched[tgt_row : tgt_row + source_row_bytes] = source_bytes[
                        src_row : src_row + source_row_bytes
                    ]

                fh.seek(data_offset + local * target_row_bytes)
                fh.write(patched)
                done += rows

        crc = _stream_storage_crc(shard.path, shard.tensor)
        _update_zip_crc(shard.path, shard.tensor.storage_member, crc)

        verify_table_shard(
            label=label,
            source=source,
            original_shard=original_shard,
            output_shard=shard,
            source_vocab_size=source_vocab_size,
            target_vocab_size=target_vocab_size,
            source_row_bytes=source_row_bytes,
            target_row_bytes=target_row_bytes,
        )


def verify_table_shard(
    *,
    label: str,
    source: Table | DcpTable,
    original_shard: RankShard,
    output_shard: RankShard,
    source_vocab_size: int,
    target_vocab_size: int,
    source_row_bytes: int,
    target_row_bytes: int,
) -> None:
    global_start = output_shard.tp_rank * output_shard.tensor.rows
    global_end = global_start + output_shard.tensor.rows
    patch_start = max(source_vocab_size, global_start)
    patch_end = min(target_vocab_size, global_end)
    if patch_start >= patch_end:
        return

    local_patch_start = patch_start - global_start
    patch_rows = patch_end - patch_start

    if local_patch_start > 0:
        original_prefix = read_tensor_rows(original_shard.path, original_shard.tensor, 0, local_patch_start)
        output_prefix = read_tensor_rows(output_shard.path, output_shard.tensor, 0, local_patch_start)
        _assert_same_bytes(f"{label} tp={output_shard.tp_rank} prefix rows", output_prefix, original_prefix)

    if local_patch_start + patch_rows < output_shard.tensor.rows:
        suffix_start = local_patch_start + patch_rows
        suffix_rows = output_shard.tensor.rows - suffix_start
        original_suffix = read_tensor_rows(
            original_shard.path, original_shard.tensor, suffix_start, suffix_rows
        )
        output_suffix = read_tensor_rows(output_shard.path, output_shard.tensor, suffix_start, suffix_rows)
        _assert_same_bytes(f"{label} tp={output_shard.tp_rank} suffix rows", output_suffix, original_suffix)

    done = 0
    while done < patch_rows:
        rows = min(PATCH_CHUNK_ROWS, patch_rows - done)
        local = local_patch_start + done
        global_row = patch_start + done
        output_bytes = read_tensor_rows(output_shard.path, output_shard.tensor, local, rows)
        original_bytes = read_tensor_rows(original_shard.path, original_shard.tensor, local, rows)
        source_bytes = source.read_global_rows(global_row, rows)
        _assert_spliced_bytes(
            label=f"{label} tp={output_shard.tp_rank} rows [{global_row},{global_row + rows})",
            output_bytes=output_bytes,
            source_bytes=source_bytes,
            original_bytes=original_bytes,
            nrows=rows,
            source_row_bytes=source_row_bytes,
            target_row_bytes=target_row_bytes,
        )
        done += rows
    print(f"  verified tp={output_shard.tp_rank}")


def validate_tables(
    *,
    source_table: Table | DcpTable,
    target_table: Table,
    label: str,
    source_hidden_size: int,
    target_hidden_size: int,
    target_padded_vocab_size: int,
    target_vocab_size: int,
) -> None:
    if source_table.hidden != source_hidden_size:
        raise ValueError(f"{label}: source hidden size is {source_table.hidden}, expected {source_hidden_size}")
    if target_table.hidden != target_hidden_size:
        raise ValueError(f"{label}: target hidden size is {target_table.hidden}, expected {target_hidden_size}")
    if target_table.padded_vocab_size != target_padded_vocab_size:
        raise ValueError(
            f"{label}: target padded vocab is {target_table.padded_vocab_size}, "
            f"expected {target_padded_vocab_size}"
        )
    if source_table.padded_vocab_size < target_vocab_size:
        raise ValueError(
            f"{label}: source padded vocab {source_table.padded_vocab_size} does not cover "
            f"target vocab {target_vocab_size}"
        )
    if source_table.element_size != target_table.element_size:
        raise ValueError(
            f"{label}: source element size {source_table.element_size} != "
            f"target element size {target_table.element_size}"
        )
    if source_table.dtype != target_table.dtype:
        raise ValueError(
            f"{label}: source dtype {source_table.dtype} != target dtype {target_table.dtype}"
        )


def print_table_summary(name: str, table: Table | DcpTable | None) -> None:
    if table is None:
        print(f"{name}: absent")
        return
    if isinstance(table, DcpTable):
        chunk_rows = ",".join(str(chunk.rows) for chunk in table.chunks)
        print(
            f"{name}: key={table.key}, format=torch_dist, chunks={len(table.chunks)} "
            f"({chunk_rows}), padded_vocab={table.padded_vocab_size}, "
            f"hidden={table.hidden}, dtype={table.dtype}, element_size={table.element_size}"
        )
        return
    print(
        f"{name}: key={table.key}, format=torch, tp={table.tp_size}, pp={table.pp_rank}, "
        f"rows_per_tp={table.rows_per_tp}, padded_vocab={table.padded_vocab_size}, "
        f"hidden={table.hidden}, dtype={table.dtype}, element_size={table.element_size}"
    )


def copy_checkpoint_tree(source: Path, output: Path, *, overwrite: bool) -> None:
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"{output} already exists; pass --overwrite-output to replace it.")
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Copying 70B checkpoint: {source} -> {output}")
    shutil.copytree(source, output, copy_function=shutil.copy2)


def main() -> None:
    args = parse_args()
    source_8b_root = Path(args.source_8b).resolve()
    target_70b_root = Path(args.target_70b).resolve()
    output_70b_root = Path(args.output_70b).resolve()

    if source_8b_root == target_70b_root:
        raise SystemExit("--source-8b and --target-70b must be different paths.")
    if output_70b_root in (source_8b_root, target_70b_root):
        raise SystemExit("--output-70b must be distinct from both input paths.")
    if args.source_vocab_size >= args.target_vocab_size:
        raise SystemExit("--source-vocab-size must be smaller than --target-vocab-size.")
    if args.target_vocab_size > args.target_padded_vocab_size:
        raise SystemExit("--target-vocab-size must be <= --target-padded-vocab-size.")
    if args.source_hidden_size >= args.target_hidden_size:
        raise SystemExit("--source-hidden-size must be smaller than --target-hidden-size.")

    source_iter = find_iteration_dir(source_8b_root)
    target_iter = find_iteration_dir(target_70b_root)
    target_root = checkpoint_root_from_iter(target_iter)
    output_root = output_70b_root
    print(f"8B source iter: {source_iter}")
    print(f"70B target iter: {target_iter}")
    print(f"70B output root: {output_root}")

    source_embed = discover_source_table(source_iter, WORD_EMBED_KEY, prefer_pp="min")
    target_embed = discover_table(target_iter, WORD_EMBED_KEY, prefer_pp="min")
    if source_embed is None:
        raise SystemExit(f"Could not find {WORD_EMBED_KEY} in {source_iter}")
    if target_embed is None:
        raise SystemExit(f"Could not find {WORD_EMBED_KEY} in {target_iter}")

    source_output = discover_source_table(source_iter, OUTPUT_LAYER_KEY, prefer_pp="max")
    target_output = discover_table(target_iter, OUTPUT_LAYER_KEY, prefer_pp="max")

    print_table_summary("8B embedding", source_embed)
    print_table_summary("70B embedding", target_embed)
    print_table_summary("8B output_layer", source_output)
    print_table_summary("70B output_layer", target_output)

    validate_tables(
        source_table=source_embed,
        target_table=target_embed,
        label="embedding",
        source_hidden_size=args.source_hidden_size,
        target_hidden_size=args.target_hidden_size,
        target_padded_vocab_size=args.target_padded_vocab_size,
        target_vocab_size=args.target_vocab_size,
    )
    if target_output is not None:
        output_source = source_output if source_output is not None else source_embed
        validate_tables(
            source_table=output_source,
            target_table=target_output,
            label="output_layer",
            source_hidden_size=args.source_hidden_size,
            target_hidden_size=args.target_hidden_size,
            target_padded_vocab_size=args.target_padded_vocab_size,
            target_vocab_size=args.target_vocab_size,
        )

    if source_output is None and target_output is not None:
        print("8B output_layer is absent; using 8B word embeddings as output_layer source.")
    if target_output is None:
        print("70B output_layer is absent; only word embeddings will be patched.")

    if args.dry_run:
        print("Dry run complete; no files copied or patched.")
        return

    copy_checkpoint_tree(target_root, output_root, overwrite=args.overwrite_output)

    # Mirror the same iter_* selection that target_root resolved to; do not re-discover,
    # which could pick a different iter via latest_checkpointed_iteration.txt.
    output_iter = output_root / target_iter.relative_to(target_root)
    output_embed = discover_table(output_iter, WORD_EMBED_KEY, prefer_pp="min")
    if output_embed is None:
        raise SystemExit(f"Could not find {WORD_EMBED_KEY} in copied output checkpoint")
    patch_table(
        label="embedding",
        source=source_embed,
        original_target=target_embed,
        output_target=output_embed,
        source_vocab_size=args.source_vocab_size,
        target_vocab_size=args.target_vocab_size,
        source_hidden_size=args.source_hidden_size,
    )

    if target_output is not None:
        output_source = source_output if source_output is not None else source_embed
        output_output = discover_table(output_iter, OUTPUT_LAYER_KEY, prefer_pp="max")
        if output_output is None:
            raise SystemExit(f"Could not find {OUTPUT_LAYER_KEY} in copied output checkpoint")
        patch_table(
            label="output_layer",
            source=output_source,
            original_target=target_output,
            output_target=output_output,
            source_vocab_size=args.source_vocab_size,
            target_vocab_size=args.target_vocab_size,
            source_hidden_size=args.source_hidden_size,
        )

    print("Done. Patched checkpoint:")
    print(output_root)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
