# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import logging
from enum import Enum
from typing import List, Optional, Tuple

import numpy

from ..utils import log_single_rank

logger = logging.getLogger(__name__)


class Split(Enum):
    train = 0
    valid = 1
    test = 2


def compile_helpers():
    """Compile C++ helper functions at runtime. Make sure this is invoked on a single process."""
    import os
    import subprocess

    command = ["make", "-C", os.path.abspath(os.path.dirname(__file__))]
    if subprocess.run(command).returncode != 0:
        import sys

        log_single_rank(logger, logging.ERROR, "Failed to compile the C++ dataset helper functions")
        sys.exit(1)


def normalize(weights: List[float]) -> List[float]:
    """Do non-exponentiated normalization

    Args:
        weights (List[float]): The weights

    Returns:
        List[float]: The normalized weights
    """

    w = numpy.array(weights, dtype=numpy.float64)
    w_sum = numpy.sum(w)
    w = (w / w_sum).tolist()
    return w


def get_blend_from_list(
    blend: Optional[List[str]],
) -> Optional[Tuple[List[str], Optional[List[float]]]]:
    # pylint: disable=line-too-long
    """Get the blended_megatron_dataset_config.BlendedMegatronDatasetConfig blend
    from the blend list

    Args:
        blend (Optional[List[str]]): The blend list, which can be either
            (1) a list of prefixes, e.g. ["path/to/dataset_1_prefix", "path/to/dataset_2_prefix"], or
            (2) a flattened, zipped list of weights and prefixes, e.g. ["30", "path/to/dataset_1_prefix", "70", "path/to/dataset_2_prefix"]

    Returns:
        Optional[Tuple[List[str], Optional[List[float]]]]: The blend, consisting of a list of dataset prefixes and optionally a list of dataset weights, e.g. [["path/to/dataset_1_prefix", "path/to/dataset_2_prefix"], [30.0, 70.0]].
    """
    # pylint: enable=line-too-long
    if blend is None:
        return None

    if len(blend) % 2 == 1:
        weight_per_dataset = None
        raw_prefix_per_dataset = blend
    else:
        raw_weight_per_dataset, raw_prefix_per_dataset = zip(
            *[(blend[i], blend[i + 1]) for i in range(0, len(blend), 2)]
        )

        weight_per_dataset = []
        for rwpd in raw_weight_per_dataset:
            try:
                weight = float(rwpd)
            except ValueError:
                weight = None
            weight_per_dataset.append(weight)

        is_none = map(lambda _: _ is None, weight_per_dataset)
        if any(is_none):
            assert all(is_none)
            weight_per_dataset = None
            raw_prefix_per_dataset = blend

    prefix_per_dataset = [rppd.strip() for rppd in raw_prefix_per_dataset]

    return prefix_per_dataset, weight_per_dataset


DATASET_TYPE_MARKERS = ("sft:", "pretrain:")


def split_dataset_type_marker(prefix: Optional[str]) -> Tuple[str, Optional[str]]:
    """Split a blend prefix into (dataset_type, clean_path).

    Recognises the reserved markers ``sft:`` and ``pretrain:`` (matched
    case-insensitively, so ``SFT:/x`` and ``sft:/x`` behave identically).
    Returns ``("default", original_path)`` when no marker is present.
    A ``None`` prefix (mock datasets) round-trips as ``("default", None)``.
    Only the leading marker is stripped, so ``"sft:sft:/x"`` resolves to
    ``("sft", "sft:/x")``. An empty path after a marker raises an
    AssertionError so typos like ``"sft:"`` fail fast.
    """
    if prefix is None:
        return ("default", None)
    lowered = prefix.lower()
    for marker in DATASET_TYPE_MARKERS:
        if lowered.startswith(marker):
            clean = prefix[len(marker):]
            assert clean, (
                f"Dataset path is empty after dataset-type marker "
                f"'{prefix[:len(marker)]}'"
            )
            return (marker[:-1], clean)
    return ("default", prefix)
