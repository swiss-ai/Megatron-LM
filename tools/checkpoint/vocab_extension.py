# Copyright (c) 2026, Swiss AI. All rights reserved.

"""Utilities for deterministic vocab-table extension during checkpoint conversion.

Vocab-extension converter family — split across two directories:

  tools/vocab_extension/                (CLI entry points)
    extend_megatron_vocab.py    Megatron ckpt -> Megatron with extended vocab
    extend_apertus_hf_vocab.py  Apertus HF ckpt -> Megatron with extended vocab

  tools/checkpoint/                     (shared converter machinery)
    vocab_extension.py    this file: deterministic row append + padding,
                          ``vpp_saver_dispatch``, ``pop_xielu_params``
    topology.py           layer routing + uniform-layout guard
    saver_vpp.py          saver plugin loaded as ``--saver vpp`` when the
                          target topology has virtual pipeline parallelism

The CLI scripts dispatch through ``convert.py`` with ``--saver core`` (no VPP)
or ``--saver vpp`` (when ``--target-num-layers-per-virtual-pipeline-stage`` is
set). VPP routing lives in ``saver_vpp.py`` as a sibling subclass of
``MegatronCheckpointSaverLLM``; shared saver code in ``saver_base.py`` /
``saver_core.py`` only owns format/protocol behavior that applies to both.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch


DEFAULT_PADDING_SEED_OFFSET = 1_000_000
VALID_INIT_METHODS = ("mean-std", "megatron-original")


def vpp_saver_dispatch(
    target_num_layers_per_virtual_pipeline_stage: Optional[int],
) -> Tuple[str, List[str]]:
    """Validate the VPP knob and return (saver_name, extra_argv) for convert.py.

    Single source of truth for both ``extend_megatron_vocab.py`` and
    ``extend_apertus_hf_vocab.py``. ``extra_argv`` is empty when VPP is disabled.
    """
    if target_num_layers_per_virtual_pipeline_stage is None:
        return "core", []
    if target_num_layers_per_virtual_pipeline_stage <= 0:
        raise SystemExit(
            "--target-num-layers-per-virtual-pipeline-stage must be positive."
        )
    return "vpp", [
        "--target-num-layers-per-virtual-pipeline-stage",
        str(target_num_layers_per_virtual_pipeline_stage),
    ]


@dataclass(frozen=True)
class VocabExtensionConfig:
    """Resolved vocab-extension parameters for a saver run.

    Single source of truth so callers don't reach into args via getattr per site.
    """

    source_vocab_size: int
    target_vocab_size: int
    seed: int
    init_method: str = "mean-std"


@dataclass(frozen=True)
class VocabExtensionPlan:
    """Describes how a vocab table was resized."""

    original_vocab_size: int
    source_vocab_size: int
    target_vocab_size: int
    padded_vocab_size: int
    trimmed_rows: int
    real_added_rows: int
    padding_added_rows: int


def make_generator(seed: int, device: torch.device | str) -> torch.Generator:
    """Create a local generator so callers do not mutate global RNG state."""

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def _column_mean_std(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-column mean/std as fp32, dropping the upcast tensor before returning."""

    stats_f32 = weight.float()
    mean = stats_f32.mean(dim=0, keepdim=True)
    std = stats_f32.std(dim=0, keepdim=True)
    return mean, std


def _sample_mean_std_rows(
    n_new: int,
    hidden: int,
    *,
    mean: torch.Tensor,
    std: torch.Tensor,
    out_dtype: torch.dtype,
    device: torch.device,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    """Sample ``n_new`` rows from N(mean, std) using a local generator."""

    rows = torch.randn(
        (n_new, hidden), dtype=torch.float32, device=device, generator=generator
    )
    rows.mul_(std).add_(mean)
    return rows.to(out_dtype)


def append_mean_std_rows(
    weight: torch.Tensor,
    new_vocab_size: int,
    *,
    stats_weight: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Append rows initialized from per-dimension mean/std statistics.

    Existing rows are preserved exactly. ``stats_weight`` controls which rows
    provide the mean/std statistics; if omitted, ``weight`` is used.
    """

    old_vocab_size = weight.shape[0]
    if new_vocab_size < old_vocab_size:
        raise ValueError(
            f"new_vocab_size={new_vocab_size} is smaller than current vocab size "
            f"({old_vocab_size}); shrinking is not supported by append_mean_std_rows."
        )
    if new_vocab_size == old_vocab_size:
        return weight

    if stats_weight is None:
        stats_weight = weight
    if stats_weight.shape[0] == 0:
        raise ValueError("stats_weight must contain at least one row")

    mean, std = _column_mean_std(stats_weight)
    new_rows = _sample_mean_std_rows(
        new_vocab_size - old_vocab_size,
        weight.shape[1],
        mean=mean.to(weight.device),
        std=std.to(weight.device),
        out_dtype=weight.dtype,
        device=weight.device,
        generator=generator,
    )
    return torch.cat([weight, new_rows], dim=0)


def append_megatron_original_rows(weight: torch.Tensor, new_vocab_size: int) -> torch.Tensor:
    """Append rows using Megatron's original padding behavior.

    Existing Megatron converter padding expands vocab tables by repeating the
    final row. This helper applies the same behavior to explicit vocab
    extension when requested.
    """

    old_vocab_size = weight.shape[0]
    if new_vocab_size < old_vocab_size:
        raise ValueError(
            f"new_vocab_size={new_vocab_size} is smaller than current vocab size "
            f"({old_vocab_size}); shrinking is not supported by append_megatron_original_rows."
        )
    if new_vocab_size == old_vocab_size:
        return weight
    if old_vocab_size == 0:
        raise ValueError("Cannot extend an empty vocab table")

    padding_size = new_vocab_size - old_vocab_size
    return torch.cat([weight, weight[-1].unsqueeze(0).expand(padding_size, -1)])


def resize_vocab_table(
    weight: torch.Tensor,
    *,
    source_vocab_size: int,
    target_vocab_size: int,
    padded_vocab_size: int,
    seed: int,
    init_method: str = "mean-std",
    padding_seed_offset: int = DEFAULT_PADDING_SEED_OFFSET,
) -> tuple[torch.Tensor, VocabExtensionPlan]:
    """Resize a full vocab table using Megatron-style source/target/padded ranges.

    Ranges:
      * [0:source_vocab_size) is preserved bitwise.
      * [source_vocab_size:target_vocab_size) receives deterministic real-token rows.
      * [target_vocab_size:padded_vocab_size) receives deterministic padding rows.

    ``init_method="mean-std"`` samples from source-row mean/std statistics.
    ``init_method="megatron-original"`` repeats the final source row, matching
    Megatron's historical dummy-padding behavior.
    """

    original_vocab_size = weight.shape[0]
    if init_method not in VALID_INIT_METHODS:
        raise ValueError(
            f"Unsupported vocab extension init_method={init_method!r}; "
            f"expected one of {VALID_INIT_METHODS}."
        )
    if source_vocab_size <= 0:
        raise ValueError(f"source_vocab_size must be positive, got {source_vocab_size}")
    if target_vocab_size < source_vocab_size:
        raise ValueError(
            f"target_vocab_size ({target_vocab_size}) cannot be smaller than "
            f"source_vocab_size ({source_vocab_size})"
        )
    if padded_vocab_size < target_vocab_size:
        raise ValueError(
            f"padded_vocab_size ({padded_vocab_size}) cannot be smaller than "
            f"target_vocab_size ({target_vocab_size})"
        )
    if original_vocab_size < source_vocab_size:
        raise ValueError(
            f"Input table has only {original_vocab_size} rows but source_vocab_size "
            f"is {source_vocab_size}"
        )

    base_weight = weight[:source_vocab_size]
    n_real = target_vocab_size - source_vocab_size
    n_pad = padded_vocab_size - target_vocab_size

    if init_method == "mean-std":
        # Compute stats once and reuse across both real and padding row sampling.
        mean, std = _column_mean_std(base_weight)
        device = base_weight.device
        mean = mean.to(device)
        std = std.to(device)
        chunks = [base_weight]
        if n_real > 0:
            chunks.append(_sample_mean_std_rows(
                n_real, base_weight.shape[1],
                mean=mean, std=std, out_dtype=base_weight.dtype, device=device,
                generator=make_generator(seed, device),
            ))
        if n_pad > 0:
            chunks.append(_sample_mean_std_rows(
                n_pad, base_weight.shape[1],
                mean=mean, std=std, out_dtype=base_weight.dtype, device=device,
                generator=make_generator(seed + padding_seed_offset, device),
            ))
        # Single allocation for the final table; avoids the intermediate copy
        # an incremental two-step torch.cat would create.
        resized = torch.cat(chunks, dim=0) if len(chunks) > 1 else base_weight
    else:
        resized = append_megatron_original_rows(base_weight, target_vocab_size)
        resized = append_megatron_original_rows(resized, padded_vocab_size)

    plan = VocabExtensionPlan(
        original_vocab_size=original_vocab_size,
        source_vocab_size=source_vocab_size,
        target_vocab_size=target_vocab_size,
        padded_vocab_size=padded_vocab_size,
        trimmed_rows=max(0, original_vocab_size - source_vocab_size),
        real_added_rows=n_real,
        padding_added_rows=n_pad,
    )
    return resized, plan
