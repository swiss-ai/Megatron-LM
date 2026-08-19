# Copyright (c) 2026, Swiss AI. All rights reserved.

import os
import sys

import pytest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CHECKPOINT_TOOLS = os.path.join(REPO_ROOT, "tools", "checkpoint")
sys.path.insert(0, CHECKPOINT_TOOLS)

from types import SimpleNamespace

from topology import (  # noqa: E402
    UNSUPPORTED_LAYOUT_MARGS,
    assert_uniform_pipeline_layout,
    build_layer_route,
    derive_vp_size,
    is_post_process_chunk,
    is_pre_process_chunk,
    post_process_chunk_index,
)


def test_no_vpp_route_is_per_pp_block():
    route = build_layer_route(num_layers=8, pp_size=4, vp_size=1)
    assert route == [
        (0, 0, 0), (0, 0, 1),
        (1, 0, 0), (1, 0, 1),
        (2, 0, 0), (2, 0, 1),
        (3, 0, 0), (3, 0, 1),
    ]


def test_vpp_route_pp2_vp2_layers8():
    """Matches the existing test_checkpoint_saver_vpp expected_by_chunk."""
    route = build_layer_route(num_layers=8, pp_size=2, vp_size=2)
    chunks = {}
    for global_idx, (pp, vp, local) in enumerate(route):
        chunks.setdefault((pp, vp), []).append((local, global_idx))
    chunks = {key: [g for _, g in sorted(value)] for key, value in chunks.items()}
    assert chunks == {
        (0, 0): [0, 1],
        (1, 0): [2, 3],
        (0, 1): [4, 5],
        (1, 1): [6, 7],
    }


def test_apertus_70b_config():
    """num_layers=80, pp=8, vp=5 — the real continued-pretrain shape."""
    route = build_layer_route(num_layers=80, pp_size=8, vp_size=5)
    assert len(route) == 80
    chunks = {}
    for global_idx, (pp, vp, local) in enumerate(route):
        chunks.setdefault((pp, vp), []).append(global_idx)
    assert len(chunks) == 8 * 5
    for global_indices in chunks.values():
        assert len(global_indices) == 2  # 80 / (8 * 5)
    assert chunks[(0, 0)] == [0, 1]
    assert chunks[(7, 0)] == [14, 15]
    assert chunks[(0, 1)] == [16, 17]
    assert chunks[(7, 4)] == [78, 79]


def test_route_covers_every_layer_exactly_once():
    route = build_layer_route(num_layers=24, pp_size=4, vp_size=3)
    seen = {}
    for global_idx, slot in enumerate(route):
        assert slot not in seen, f"slot {slot} reused at {seen[slot]} and {global_idx}"
        seen[slot] = global_idx
    assert len(seen) == 24


def test_post_process_chunk_is_last_pp_last_vp():
    """The chunk owning the highest global layer is (pp_size-1, vp_size-1)."""
    route = build_layer_route(num_layers=80, pp_size=8, vp_size=5)
    last_pp, last_vp, _ = route[-1]
    assert last_pp == 7
    assert last_vp == 4


def test_pre_process_chunk_is_first_pp_first_vp():
    """The chunk owning global layer 0 is (0, 0)."""
    route = build_layer_route(num_layers=80, pp_size=8, vp_size=5)
    first_pp, first_vp, first_local = route[0]
    assert (first_pp, first_vp, first_local) == (0, 0, 0)


def test_indivisible_layers_raises():
    with pytest.raises(ValueError, match="must be divisible"):
        build_layer_route(num_layers=10, pp_size=4, vp_size=1)
    with pytest.raises(ValueError, match="must be divisible"):
        build_layer_route(num_layers=80, pp_size=8, vp_size=3)


def test_zero_or_negative_inputs_raise():
    with pytest.raises(ValueError, match="must be positive"):
        build_layer_route(num_layers=8, pp_size=0, vp_size=1)
    with pytest.raises(ValueError, match="must be positive"):
        build_layer_route(num_layers=8, pp_size=1, vp_size=-1)
    with pytest.raises(ValueError, match="must be positive"):
        build_layer_route(num_layers=0, pp_size=1, vp_size=1)


def test_derive_vp_size_apertus_70b():
    """num_layers=80, pp=8, layers_per_vp=2 → vp_size=5."""
    assert (
        derive_vp_size(num_layers=80, pp_size=8, num_layers_per_virtual_pipeline_stage=2)
        == 5
    )


def test_derive_vp_size_no_vpp_equivalent():
    """When layers_per_vp == layers_per_pp, vp_size collapses to 1."""
    assert (
        derive_vp_size(num_layers=80, pp_size=8, num_layers_per_virtual_pipeline_stage=10)
        == 1
    )


def test_derive_vp_size_rejects_non_divisible():
    with pytest.raises(ValueError, match="divisible by pp_size"):
        derive_vp_size(
            num_layers=10, pp_size=4, num_layers_per_virtual_pipeline_stage=1
        )
    with pytest.raises(ValueError, match="divisible by"):
        derive_vp_size(
            num_layers=80, pp_size=8, num_layers_per_virtual_pipeline_stage=3
        )


def test_pre_process_chunk_predicate():
    assert is_pre_process_chunk(0, 0)
    assert not is_pre_process_chunk(0, 1)
    assert not is_pre_process_chunk(1, 0)


def test_assert_uniform_pipeline_layout_passes_for_uniform():
    margs = SimpleNamespace(
        pipeline_model_parallel_layout=None,
        num_layers_in_first_pipeline_stage=None,
        num_layers_in_last_pipeline_stage=None,
        account_for_embedding_in_pipeline_split=False,
        account_for_loss_in_pipeline_split=False,
    )
    assert_uniform_pipeline_layout(margs)  # no raise


def test_assert_uniform_pipeline_layout_rejects_each_flag():
    base = {name: sentinel for name, sentinel in UNSUPPORTED_LAYOUT_MARGS}
    for name, sentinel in UNSUPPORTED_LAYOUT_MARGS:
        bad = dict(base)
        bad[name] = "set" if sentinel is None else not sentinel
        with pytest.raises(NotImplementedError, match=name):
            assert_uniform_pipeline_layout(SimpleNamespace(**bad))


def test_assert_uniform_pipeline_layout_tolerates_missing_attrs():
    # margs from older Megatron versions may not have all flags; defaults stand in.
    assert_uniform_pipeline_layout(SimpleNamespace())  # no raise


def test_post_process_chunk_predicate_and_index():
    assert is_post_process_chunk(7, 4, pp_size=8, vp_size=5)
    assert not is_post_process_chunk(7, 0, pp_size=8, vp_size=5)
    assert not is_post_process_chunk(0, 4, pp_size=8, vp_size=5)
    assert post_process_chunk_index(pp_size=8, vp_size=5) == (7, 4)
    # Non-VPP collapses to (last_pp, 0).
    assert post_process_chunk_index(pp_size=4, vp_size=1) == (3, 0)
    assert is_post_process_chunk(3, 0, pp_size=4, vp_size=1)
