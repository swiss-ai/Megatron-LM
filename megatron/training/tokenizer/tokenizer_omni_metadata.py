# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Modifications by Yixuan Xu (ETH AI Center).

"""Helpers for tokenizer-provided omni metadata propagation.

Expected metadata is read from tokenizer init kwargs, typically:
`tokenizer._tokenizer.init_kwargs` (legacy wrappers) or nested variants like
`tokenizer._tokenizer.tokenizer.init_kwargs` (core wrappers).

Example `omnimodal_config` payload:
{
  "omni_special_token_offset": 131072,
  "modalities": [
    {"name": "vision", "offset": 131272, "vocab_size": 131072},
    {"name": "audio", "offset": 262344, "vocab_size": 4096}
  ]
}
"""

import math
from typing import Any, Callable, Dict, Optional, Tuple


def _vocab_size_with_padding(orig_vocab_size: int, args) -> int:
    """Pad vocab size for tensor-parallel-friendly embedding table sizing."""
    multiple = args.make_vocab_size_divisible_by * args.tensor_model_parallel_size
    return int(math.ceil(orig_vocab_size / multiple) * multiple)


def _iter_tokenizer_wrappers(tokenizer: Any):
    """Yield tokenizer and nested wrapper objects linked by `_tokenizer`/`tokenizer` attrs.

    Concrete wrapper chains seen in this repo include:
    - legacy path: wrapper -> `_tokenizer` -> HF tokenizer (has `init_kwargs`)
    - core path: wrapper -> `_tokenizer` -> library wrapper -> `tokenizer` -> HF tokenizer
    """
    queue = [tokenizer]
    seen = set()
    while queue:
        obj = queue.pop(0)
        if obj is None:
            continue
        obj_id = id(obj)
        if obj_id in seen:
            continue
        seen.add(obj_id)
        yield obj
        for attr in ("_tokenizer", "tokenizer"):
            child = getattr(obj, attr, None)
            if child is not None:
                queue.append(child)


def extract_tokenizer_init_kwargs(tokenizer: Any) -> Dict[str, Any]:
    """Find first `init_kwargs` dict across tokenizer wrapper chain.

    Returns `{}` when no `init_kwargs` dictionary is available.
    """
    for obj in _iter_tokenizer_wrappers(tokenizer):
        init_kwargs = getattr(obj, "init_kwargs", None)
        if isinstance(init_kwargs, dict):
            return init_kwargs
    return {}


def compute_goldfish_exemption_range(
    base_vocab_size: Optional[int], omnimodal_config: Optional[Dict[str, Any]]
) -> Optional[Tuple[int, int]]:
    """Compute goldfish exemption range `[start, end)` for omni special tokens.

    We exempt the contiguous span between text vocab end (`base_vocab_size`) and the
    first modality offset from `omnimodal_config.modalities`.
    """
    if base_vocab_size is None or omnimodal_config is None:
        return None

    min_offset = min(
        (
            int(modality.get("offset"))
            for modality in omnimodal_config.get("modalities", [])
            if modality.get("offset") is not None
        ),
        default=None,
    )
    if min_offset is None or int(base_vocab_size) >= min_offset:
        return None
    return int(base_vocab_size), int(min_offset)


def populate_omni_metadata_from_tokenizer(
    args,
    tokenizer: Any,
    base_padded_vocab_size_fn: Optional[Callable[[int], int]] = None,
    print_loaded_modalities: bool = False,
    reset_goldfish_exemption: bool = False,
) -> None:
    """Populate args/tokenizer with omni metadata used by dataset weighting and goldfish masking.

    Populated fields:
    - `args.base_vocab_size`
    - `args.base_padded_vocab_size`
    - `args.omnimodal_config`
    - `args.{name}_token_offset`, `args.{name}_vocab_size` for each modality entry
    - `tokenizer.goldfish_exemption_range`
    - `tokenizer.sft_assistant_begin_sequence`, `tokenizer.sft_assistant_end_sequence`

    Metadata source:
    - tokenizer `init_kwargs` keys: `base_vocab_size`, `omnimodal_config`, `sft_assistant_begin_sequence`, `sft_assistant_end_sequence`
    - falls back to already-populated `args.*` when direct kwargs are unavailable
    """
    init_kwargs = extract_tokenizer_init_kwargs(tokenizer)

    base_vocab_size = init_kwargs.get("base_vocab_size", getattr(args, "base_vocab_size", None))
    if base_vocab_size is not None:
        base_vocab_size = int(base_vocab_size)
        args.base_vocab_size = base_vocab_size
        if base_padded_vocab_size_fn is None:
            args.base_padded_vocab_size = _vocab_size_with_padding(base_vocab_size, args)
        else:
            args.base_padded_vocab_size = base_padded_vocab_size_fn(base_vocab_size)

    omnimodal_config = init_kwargs.get("omnimodal_config", getattr(args, "omnimodal_config", None))
    if omnimodal_config is not None:
        args.omnimodal_config = omnimodal_config
        for modality in omnimodal_config.get("modalities", []):
            name = modality.get("name")
            if not name:
                continue
            offset = modality.get("offset")
            vocab_size = modality.get("vocab_size")
            if offset is not None:
                setattr(args, f"{name}_token_offset", int(offset))
            if vocab_size is not None:
                setattr(args, f"{name}_vocab_size", int(vocab_size))
        if print_loaded_modalities and getattr(args, "rank", None) == 0:
            print(
                " > loaded omnimodal_config with modalities: "
                f"{[m.get('name') for m in omnimodal_config.get('modalities', []) if m.get('name')]}",
                flush=True,
            )

    if reset_goldfish_exemption:
        tokenizer.goldfish_exemption_range = None
    if getattr(tokenizer, "goldfish_exemption_range", None) is None:
        exemption_range = compute_goldfish_exemption_range(base_vocab_size, omnimodal_config)
        if exemption_range is not None:
            tokenizer.goldfish_exemption_range = exemption_range
