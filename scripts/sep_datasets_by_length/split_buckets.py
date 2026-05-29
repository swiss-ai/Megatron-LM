#!/usr/bin/env python3
"""
Split MMap / SFT indexed datasets into token-length buckets.

Two dataset-discovery modes
---------------------------
--input-dirs DIR [DIR ...]
    Scans each directory for flat .idx/.bin pairs and splits every file found.

--named-datasets NAME:PREFIX [NAME:PREFIX ...]
    Explicitly names each source.  NAME becomes the output stem; PREFIX is the
    path prefix shared by the .bin and .idx files.

Two built-in bucket sets (or supply your own with --buckets)
-------------------------------------------------------------
--bucket-set vision     lower_16k / 16k_64k / 64k_128k / 128k_256k / 256k_plus
--buckets NAME:LO:HI [NAME:LO:HI ...]
    Fully custom buckets; HI may be omitted / left empty for unbounded.

Output layout
-------------
Examples
--------
# SFT named datasets
python split_buckets.py \\
    --named-datasets EnvScaler:/path/to/EnvScaler_tokens \\
                     OpenSeeker:/path/to/OpenSeeker_tokens \\
    --bucket-set sft \\
    --output-dir /output/sft

# Vision directory scan
python split_buckets.py \\
    --input-dirs /path/to/vision \\
    --bucket-set vision \\
    --output-dir /output/vision

# Dry-run to check stats only (no files written)
python split_buckets.py --input-dirs /data --bucket-set vision --output-dir /tmp --dry-run
"""

import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np

from megatron.core.datasets.indexed_dataset import IndexedDataset, IndexedDatasetBuilder

logger = logging.getLogger(__name__)

MIN_BUCKET_TOKENS: int = 262_144
BUCKET_SETS = {
    "sft": [
        ("lower_16k",   0,       16_384),
        ("16k_64k",    16_384,   65_536),
        ("64k_128k",   65_536,  131_072),
        ("128k_256k",  131_072, 262_144),
        ("256k_plus",  262_144, None),
    ],
    "vision": [
        ("lower_16k",   0,       16_384),
        ("16k_64k",    16_384,   65_536),
        ("64k_128k",   65_536,  131_072),
        ("128k_256k",  131_072, 262_144),
        ("256k_plus",  262_144, None),
    ],
}


def fmt_tokens(n: int) -> str:
    if n >= 1e9:  return f"{n / 1e9:.3f}B"
    if n >= 1e6:  return f"{n / 1e6:.1f}M"
    if n >= 1e3:  return f"{n / 1e3:.1f}K"
    return str(n)


def compute_doc_token_counts(ds: IndexedDataset) -> np.ndarray:
    """Return an int64 array with the token count for each document."""
    starts = ds.document_indices[:-1].astype(np.intp)
    return np.add.reduceat(ds.sequence_lengths.astype(np.int64), starts)


def discover_flat_prefixes(data_dir: str) -> list[str]:
    """Find all .idx/.bin pairs directly under *data_dir* (flat layout)."""
    root = Path(data_dir)
    prefixes = []
    for idx_file in sorted(root.glob("*.idx")):
        prefix = str(idx_file)[:-4]
        if Path(prefix + ".bin").exists():
            prefixes.append(prefix)
        else:
            logger.warning("Skipping %s: missing .bin counterpart", idx_file)
    return prefixes


def parse_buckets(specs: list[str]) -> list[tuple[str, int, int | None]]:
    """
    Parse a list of 'NAME:LO:HI' strings into bucket tuples.
    HI is optional; omit it or leave it empty for an unbounded bucket.
    """
    buckets = []
    for spec in specs:
        parts = spec.split(":")
        if len(parts) < 2 or len(parts) > 3:
            raise argparse.ArgumentTypeError(
                f"Invalid bucket spec '{spec}'. Expected NAME:LO or NAME:LO:HI"
            )
        name = parts[0]
        lo   = int(parts[1])
        hi   = int(parts[2]) if len(parts) == 3 and parts[2] else None
        buckets.append((name, lo, hi))
    return buckets


def _close_builder(b: dict) -> None:
    """
    Close the open .bin file handle held by an IndexedDatasetBuilder.

    IndexedDatasetBuilder keeps self.data_file open until finalize() is
    called.  On network / parallel filesystems (Lustre, GPFS) unlinking an
    open file can silently fail or leave the directory entry intact, so we
    must close the handle explicitly before attempting os.remove().
    """
    builder = b.get("builder")
    if builder is None:
        return
    try:
        builder.data_file.close()  # public attribute per IndexedDatasetBuilder source
    except Exception:
        pass
    finally:
        b["builder"] = None   # drop the reference; let GC finalize the object


def split_source(
    source_name: str,
    prefix: str,
    output_dir: str,
    buckets: list[tuple[str, int, int | None]],
    dry_run: bool,
) -> None:
    """Read one source shard and fan its documents into bucket output files."""

    if not IndexedDataset.exists(prefix):
        logger.error("[%s] dataset not found at prefix: %s", source_name, prefix)
        return

    t0 = time.perf_counter()
    ds = IndexedDataset(prefix)
    doc_counts = compute_doc_token_counts(ds)
    n_docs = len(doc_counts)
    logger.info("[%s] loaded %d documents (%.1fs)", source_name, n_docs,
                time.perf_counter() - t0)

    # Build per-bucket masks and log stats
    bucket_masks: dict[str, np.ndarray] = {}
    for bucket_name, lo, hi in buckets:
        mask = (doc_counts >= lo) & (doc_counts < hi) if hi is not None \
               else (doc_counts >= lo)
        bucket_masks[bucket_name] = mask
        n    = int(mask.sum())
        toks = int(doc_counts[mask].sum())
        logger.info("  %-14s  %7d docs  %s tokens", bucket_name, n, fmt_tokens(toks))

    if dry_run:
        logger.info("[%s] dry-run: skipping write\n", source_name)
        del ds
        return

    dtype   = ds.index.dtype
    doc_idx = ds.document_indices  # shape (n_docs + 1,)

    # Open one builder per non-empty bucket
    builders: dict[str, dict] = {}
    for bucket_name, lo, hi in buckets:
        mask = bucket_masks[bucket_name]
        if not mask.any():
            continue
        out_dir    = Path(output_dir) / bucket_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_prefix = str(out_dir / source_name)
        builders[bucket_name] = {
            "builder":  IndexedDatasetBuilder(out_prefix + ".bin", dtype=dtype),
            "prefix":   out_prefix,
            "mask":     mask,
            "n_docs":   0,
            "n_tokens": 0,
        }

    # One sequential pass per bucket: fetch each document independently using
    # absolute sequence indices from doc_idx.

    for bucket_name, lo, hi in buckets:
        if bucket_name not in builders:
            continue

        b   = builders[bucket_name]
        ids = np.where(b["mask"])[0]

        for d in ids:
            seq_lo  = int(doc_idx[d])
            seq_hi  = int(doc_idx[d + 1])
            seqs    = ds[seq_lo:seq_hi]   # absolute indices - unambiguous
            lengths = [len(s) for s in seqs]
            data    = np.concatenate(seqs)
            b["builder"].add_document(data, lengths)
            b["n_docs"]   += 1
            b["n_tokens"] += sum(lengths)

    del ds

    for bucket_name, b in builders.items():
        if b["n_tokens"] >= MIN_BUCKET_TOKENS:
            b["builder"].finalize(b["prefix"] + ".idx")
            logger.info("[%s/%s] wrote %d docs, %s tokens -> %s",
                        source_name, bucket_name,
                        b["n_docs"], fmt_tokens(b["n_tokens"]), b["prefix"])
        else:
            # Close the open file handle BEFORE attempting removal.
            # IndexedDatasetBuilder only closes data_file inside finalize();
            # on network filesystems (Lustre/GPFS) unlinking an open fd can
            # silently fail, leaving an orphaned .bin with no matching .idx.
            _close_builder(b)

            bin_path = b["prefix"] + ".bin"
            try:
                if os.path.exists(bin_path):
                    os.remove(bin_path)
            except OSError as exc:
                logger.warning(
                    "[%s/%s] DISCARDED — only %s tokens (< %s minimum); "
                    "could not remove orphaned .bin (%s) — delete manually: %s",
                    source_name, bucket_name,
                    fmt_tokens(b["n_tokens"]), fmt_tokens(MIN_BUCKET_TOKENS),
                    exc, bin_path,
                )
            else:
                logger.warning(
                    "[%s/%s] DISCARDED — only %s tokens (< %s minimum)",
                    source_name, bucket_name,
                    fmt_tokens(b["n_tokens"]), fmt_tokens(MIN_BUCKET_TOKENS),
                )

    logger.info("[%s] done in %.1fs\n", source_name, time.perf_counter() - t0)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Split indexed datasets into token-length buckets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Dataset discovery (mutually exclusive)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--input-dirs", nargs="+", metavar="DIR",
        help="Directories containing flat .idx/.bin pairs (auto-discovered).",
    )
    src.add_argument(
        "--named-datasets", nargs="+", metavar="NAME:PREFIX",
        help="Explicit 'name:prefix' pairs, one per dataset.",
    )

    # Bucket definition (mutually exclusive)
    bkt = ap.add_mutually_exclusive_group(required=True)
    bkt.add_argument(
        "--bucket-set", choices=list(BUCKET_SETS),
        help="Use a built-in bucket set.",
    )
    bkt.add_argument(
        "--buckets", nargs="+", metavar="NAME:LO[:HI]",
        help="Custom bucket specs.  HI is optional (omit for unbounded).",
    )

    ap.add_argument("--output-dir", required=True,
                    help="Root output directory.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print statistics only; do not write any files.")
    return ap


def main() -> None:
    args = build_parser().parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Resolve buckets
    if args.bucket_set:
        buckets = BUCKET_SETS[args.bucket_set]
    else:
        buckets = parse_buckets(args.buckets)

    # Resolve datasets → list of (name, prefix)
    sources: list[tuple[str, str]] = []
    if args.input_dirs:
        for d in args.input_dirs:
            prefixes = discover_flat_prefixes(d)
            if not prefixes:
                logger.warning("No .idx/.bin pairs found under %s", d)
            else:
                logger.info("Found %d dataset(s) under %s", len(prefixes), d)
            sources.extend((Path(p).stem, p) for p in prefixes)
    else:
        for spec in args.named_datasets:
            if ":" not in spec:
                raise SystemExit(f"Invalid --named-datasets entry '{spec}'. "
                                 f"Expected NAME:PREFIX.")
            name, _, prefix = spec.partition(":")
            sources.append((name, prefix))

    if not sources:
        logger.error("No datasets found. Exiting.")
        return

    output_dir = str(Path(args.output_dir).resolve())
    if not args.dry_run:
        os.makedirs(output_dir, exist_ok=True)

    if args.dry_run:
        logger.info("DRY RUN — no files will be written.\n")

    logger.info("Bucket set:")
    for bucket_name, lo, hi in buckets:
        hi_str = str(hi) if hi is not None else "∞"
        logger.info("  %-14s  [%d, %s)", bucket_name, lo, hi_str)
    logger.info("")

    t_global = time.perf_counter()
    for source_name, prefix in sources:
        logger.info("=" * 70)
        logger.info("Processing: %s", source_name)
        split_source(source_name, prefix, output_dir, buckets, dry_run=args.dry_run)

    logger.info("All done in %.1fs", time.perf_counter() - t_global)


if __name__ == "__main__":
    main()