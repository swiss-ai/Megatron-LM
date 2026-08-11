# Copyright (c) 2026, SwissAI. All rights reserved.

"""Preflight Apertus SFT packing counts (BFD or greedy strategy).

Counts the packed samples per dataset without launching Megatron, using the
same packing logic as ApertusSFTDataset:

- ``bfd``: Best-Fit Decreasing. Order-independent, so the count is exact for
  any seed. Pass ``--max-docs-per-bin`` if training uses
  ``--max-docs-per-bin-sft``.
- ``greedy``: Packs documents in the seeded shuffle order used at training
  time, so ``--seed`` must match the training run's ``--seed`` for the count
  to be exact.

Both counts assume the whole dataset lands in the train split
(``--split 100,0,0``) and that each counted ``.idx`` file is its own
``--data-path`` entry in training. For ``--ap-sft-load-loss-mask`` datasets
(documents store ``[tokens, loss_mask]``), pass the same doubled
``--seq-length`` the training run uses.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

MEGATRON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MEGATRON_DIR))

from megatron.core.datasets.indexed_dataset import _IndexReader  # noqa: E402
from megatron.training.datasets.apertus_sft_dataset import (  # noqa: E402
    _build_sample_idx_bfd,
    _build_sample_idx_greedy,
)


def parse_epochs(raw_epochs: str) -> int:
    try:
        return int(raw_epochs)
    except ValueError:
        if "." in raw_epochs:
            raise ValueError(
                "fractional epochs are not supported yet; use an integer epoch count"
            ) from None
        raise ValueError(f"epoch count must be an integer, got: {raw_epochs}") from None


def read_data_path_file(path: Path) -> list[tuple[Path, int]]:
    datasets = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"expected '<path> <epochs>', got: {line}")

        epochs = parse_epochs(parts[1])
        if epochs < 1:
            raise ValueError("epoch count must be >= 1")
        datasets.append((Path(parts[0]), epochs))
    return datasets


def idx_files_for(path: Path) -> list[Path]:
    if path.is_dir():
        idx_files = sorted(path.glob("*.idx"))
    else:
        prefix = str(path)
        if prefix.endswith(".idx"):
            prefix = prefix[:-4]
        idx_files = [Path(prefix + ".idx")]

    if not idx_files:
        raise FileNotFoundError(f"no .idx files found for {path}")
    for idx_path in idx_files:
        if not idx_path.is_file():
            raise FileNotFoundError(f"missing indexed dataset file: {idx_path}")
    return idx_files


def shuffled_document_order(n_docs: int, seed: int) -> np.ndarray:
    """Replicate the one-epoch document shuffle from
    ApertusSFTDataset (_build_document_index with num_epochs=1)."""
    document_index = np.arange(n_docs, dtype=np.int32)
    np.random.RandomState(seed).shuffle(document_index)
    return document_index


def packed_count_for_idx(
    idx_path: Path,
    seq_length: int,
    add_extra_token: int,
    strategy: str = "bfd",
    seed: int = 1234,
    max_docs_per_bin: int = 0,
) -> tuple[int, int, int]:
    idx_prefix = str(idx_path)
    if idx_prefix.endswith(".idx"):
        idx_prefix = idx_prefix[:-4]

    reader = _IndexReader(
        idx_prefix + ".idx",
        multimodal=False,
        sequences_per_dataset=None,
        dtype_code=None,
    )
    seq_lens = np.asarray(reader.sequence_lengths)
    n_docs = len(seq_lens)
    n_tokens = int(seq_lens.sum())
    if strategy == "bfd":
        doc_idx = np.arange(n_docs, dtype=np.int32)
        _reordered, sample_idx = _build_sample_idx_bfd(
            sequence_lengths=seq_lens,
            document_index=doc_idx,
            seq_length=seq_length,
            add_extra_token=add_extra_token,
            max_docs_per_bin=max_docs_per_bin,
        )
        n_samples = int(sample_idx.shape[0] - 1)
    elif strategy == "greedy":
        document_order = shuffled_document_order(n_docs, seed)
        sample_idx = _build_sample_idx_greedy(
            seq_lens, document_order, seq_length, add_extra_token
        )
        n_samples = int(sample_idx.shape[0] - 1)
    else:
        raise ValueError(f"unknown packing strategy: {strategy}")
    return n_docs, n_tokens, n_samples


def summarize_datasets(
    datasets: list[tuple[Path, int]],
    seq_length: int,
    add_extra_token: int,
    strategy: str = "bfd",
    seed: int = 1234,
    max_docs_per_bin: int = 0,
    count_fn=packed_count_for_idx,
) -> tuple[list[dict], int]:
    rows = []
    grand_samples = 0
    for path, epochs in datasets:
        idx_files = idx_files_for(path)
        n_files = len(idx_files)
        tot_docs, tot_toks, tot_samp = 0, 0, 0
        for idx_path in idx_files:
            nd, nt, ns = count_fn(
                idx_path,
                seq_length,
                add_extra_token,
                strategy=strategy,
                seed=seed,
                max_docs_per_bin=max_docs_per_bin,
            )
            tot_docs += nd
            tot_toks += nt
            tot_samp += ns

        weighted = tot_samp * epochs
        grand_samples += weighted
        rows.append(
            {
                "path": str(path),
                "files": n_files,
                "docs": tot_docs,
                "tokens": tot_toks,
                "samples": tot_samp,
                "epochs": epochs,
                "weighted_samples": weighted,
            }
        )
    return rows, grand_samples


def compute_step_counts(
    total_packed_samples: int, global_batch_size: int
) -> dict[str, int]:
    if global_batch_size <= 0:
        raise ValueError("--global-batch-size must be positive")

    floor_steps = total_packed_samples // global_batch_size
    ceil_steps = math.ceil(total_packed_samples / global_batch_size)
    return {
        "floor_steps": floor_steps,
        "ceil_steps": ceil_steps,
        "floor_leftover_samples": total_packed_samples
        - floor_steps * global_batch_size,
        "ceil_shortfall_samples": ceil_steps * global_batch_size
        - total_packed_samples,
    }


def format_summary(
    rows: list[dict],
    grand_samples: int,
    steps: dict[str, int],
    seq_length: int,
    add_extra_token: int,
    global_batch_size: int,
    strategy: str = "bfd",
    seed: int = 1234,
) -> str:
    capacity = seq_length + add_extra_token
    strategy_line = f"PACKING_STRATEGY = {strategy}"
    if strategy == "greedy":
        strategy_line += f" (SEED = {seed}; must match training --seed)"
    lines = [
        strategy_line,
        f"Capacity = SEQ_LENGTH + ADD_EXTRA_TOKEN = {seq_length} + {add_extra_token} = {capacity}",
        f"GBS = {global_batch_size}",
        "",
        (
            f"{'dataset':10s} {'files':>5s} {'docs':>14s} {'tokens':>18s} "
            f"{'samples/epoch':>16s} {'epochs':>6s} {'samples*epochs':>17s}"
        ),
        "-" * 100,
    ]
    for row in rows:
        label = Path(row["path"]).name
        lines.append(
            f"{label[:10]:10s} {row['files']:>5d} {row['docs']:>14,} "
            f"{row['tokens']:>18,} {row['samples']:>16,} {row['epochs']:>6d} "
            f"{row['weighted_samples']:>17,}"
        )
    lines.extend(
        [
            "-" * 100,
            f"{'TOTAL':10s} {'':>5s} {'':>14s} {'':>18s} {'':>16s} {'':>6s} {grand_samples:>17,}",
            "",
            f"TRAINING_STEPS_FLOOR = floor({grand_samples:,} / {global_batch_size}) = {steps['floor_steps']}",
            f"LEFTOVER_PACKED_SAMPLES = {steps['floor_leftover_samples']}",
            f"TRAINING_STEPS_CEIL = ceil({grand_samples:,} / {global_batch_size}) = {steps['ceil_steps']}",
            f"CEIL_SHORTFALL_SAMPLES = {steps['ceil_shortfall_samples']}",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path-file", type=Path, required=True)
    parser.add_argument("--seq-length", type=int, required=True)
    parser.add_argument("--global-batch-size", type=int, required=True)
    parser.add_argument("--add-extra-token-to-sequence", type=int, default=1)
    parser.add_argument(
        "--packing-strategy",
        choices=("bfd", "greedy"),
        required=True,
        help="must match the training run's --ap-sft-packing-strategy",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="greedy only: must match the training run's --seed",
    )
    parser.add_argument(
        "--max-docs-per-bin",
        type=int,
        default=0,
        help="bfd only: must match the training run's --max-docs-per-bin-sft (0 = unlimited)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows, grand_samples = summarize_datasets(
        read_data_path_file(args.data_path_file),
        seq_length=args.seq_length,
        add_extra_token=args.add_extra_token_to_sequence,
        strategy=args.packing_strategy,
        seed=args.seed,
        max_docs_per_bin=args.max_docs_per_bin,
    )
    steps = compute_step_counts(grand_samples, args.global_batch_size)
    print(
        format_summary(
            rows,
            grand_samples,
            steps,
            args.seq_length,
            args.add_extra_token_to_sequence,
            args.global_batch_size,
            strategy=args.packing_strategy,
            seed=args.seed,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
