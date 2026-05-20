# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for the per-entry SFT/pretrain marker in blend paths."""

import warnings
from types import SimpleNamespace

import pytest

from megatron.core.datasets.blended_megatron_dataset_builder import (
    BlendedMegatronDatasetBuilder,
)
from megatron.core.datasets.gpt_dataset import GPTDataset, MockGPTDataset
from megatron.core.datasets.utils import split_dataset_type_marker
from megatron.training.datasets.apertus_sft_dataset import ApertusSFTDataset


# ---------------------------------------------------------------------------
# split_dataset_type_marker
# ---------------------------------------------------------------------------


def test_marker_sft():
    assert split_dataset_type_marker("sft:/x/y") == ("sft", "/x/y")


def test_marker_pretrain():
    assert split_dataset_type_marker("pretrain:/x") == ("pretrain", "/x")


def test_marker_none_prefix():
    assert split_dataset_type_marker("/plain/path") == ("default", "/plain/path")


def test_marker_none_value():
    assert split_dataset_type_marker(None) == ("default", None)


def test_marker_strips_only_once():
    assert split_dataset_type_marker("sft:sft:/x") == ("sft", "sft:/x")


def test_marker_case_insensitive():
    assert split_dataset_type_marker("SFT:/x") == ("sft", "/x")
    assert split_dataset_type_marker("Pretrain:/y") == ("pretrain", "/y")


def test_marker_empty_path_after_marker_raises():
    with pytest.raises(AssertionError, match="empty"):
        split_dataset_type_marker("sft:")


# ---------------------------------------------------------------------------
# BlendedMegatronDatasetBuilder._resolve_dataset_class
# ---------------------------------------------------------------------------


def _make_builder(*, mock=False, ap_sft_auto_tag=False):
    """Build a barebones builder instance without running __init__ (which
    requires a fully populated GPTDatasetConfig)."""
    builder = BlendedMegatronDatasetBuilder.__new__(BlendedMegatronDatasetBuilder)
    builder.config = SimpleNamespace(mock=mock, ap_sft_auto_tag=ap_sft_auto_tag)
    return builder


def test_resolve_explicit_sft_marker():
    cls, path = _make_builder()._resolve_dataset_class("sft:/data/a")
    assert cls is ApertusSFTDataset
    assert path == "/data/a"


def test_resolve_explicit_pretrain_marker():
    cls, path = _make_builder()._resolve_dataset_class("pretrain:/data/b")
    assert cls is GPTDataset
    assert path == "/data/b"


def test_resolve_auto_tag_unmarked_path():
    cls, path = _make_builder(ap_sft_auto_tag=True)._resolve_dataset_class("/data/c")
    assert cls is ApertusSFTDataset
    assert path == "/data/c"


def test_resolve_legacy_substring_emits_deprecation():
    builder = _make_builder()
    with pytest.warns(DeprecationWarning, match="apertus_sft"):
        cls, path = builder._resolve_dataset_class("/data/legacy_apertus_sft_set")
    assert cls is ApertusSFTDataset
    assert path == "/data/legacy_apertus_sft_set"


def test_resolve_default_pretrain():
    cls, path = _make_builder()._resolve_dataset_class("/data/fineweb")
    assert cls is GPTDataset
    assert path == "/data/fineweb"


def test_resolve_mock_config_short_circuits():
    cls, path = _make_builder(mock=True)._resolve_dataset_class("/data/anything")
    assert cls is MockGPTDataset
    assert path is None


def test_resolve_none_path_short_circuits():
    cls, path = _make_builder()._resolve_dataset_class(None)
    assert cls is MockGPTDataset
    assert path is None


def test_explicit_marker_overrides_auto_tag():
    cls, path = _make_builder(ap_sft_auto_tag=True)._resolve_dataset_class(
        "pretrain:/data/d"
    )
    assert cls is GPTDataset
    assert path == "/data/d"


def test_explicit_marker_overrides_legacy_substring():
    """A path with 'apertus_sft' in it but tagged pretrain: must NOT warn
    and must dispatch as GPTDataset."""
    builder = _make_builder()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        cls, path = builder._resolve_dataset_class("pretrain:/data/apertus_sft_legacy")
    assert cls is GPTDataset
    assert path == "/data/apertus_sft_legacy"
