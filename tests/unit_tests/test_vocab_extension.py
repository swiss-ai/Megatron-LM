# Copyright (c) 2026, Swiss AI. All rights reserved.

import pytest
import torch

from tools.checkpoint.vocab_extension import resize_vocab_table


def test_resize_preserves_source_rows_and_is_deterministic():
    weight = torch.arange(18, dtype=torch.float32).view(6, 3)

    resized_a, plan_a = resize_vocab_table(
        weight,
        source_vocab_size=4,
        target_vocab_size=7,
        padded_vocab_size=8,
        seed=123,
    )
    resized_b, plan_b = resize_vocab_table(
        weight,
        source_vocab_size=4,
        target_vocab_size=7,
        padded_vocab_size=8,
        seed=123,
    )

    assert resized_a.shape == (8, 3)
    assert torch.equal(resized_a, resized_b)
    assert plan_a == plan_b
    assert torch.equal(resized_a[:4], weight[:4])
    assert plan_a.trimmed_rows == 2
    assert plan_a.real_added_rows == 3
    assert plan_a.padding_added_rows == 1


@pytest.mark.parametrize("seed", [123, 124])
def test_tp_shards_preserve_all_source_rows_when_source_crosses_shard_boundary(seed):
    source_vocab_size = 10
    target_vocab_size = 19
    padded_vocab_size = 24
    tensor_parallel_size = 4
    weight = torch.arange(source_vocab_size * 3, dtype=torch.float32).view(
        source_vocab_size, 3
    )

    resized, _ = resize_vocab_table(
        weight,
        source_vocab_size=source_vocab_size,
        target_vocab_size=target_vocab_size,
        padded_vocab_size=padded_vocab_size,
        seed=seed,
    )

    shard_size = padded_vocab_size // tensor_parallel_size
    shards = torch.chunk(resized, tensor_parallel_size, dim=0)

    for tp_rank, shard in enumerate(shards):
        global_start = tp_rank * shard_size
        global_end = global_start + shard.shape[0]
        text_start = min(global_start, source_vocab_size)
        text_end = min(global_end, source_vocab_size)
        text_rows = text_end - text_start

        if text_rows == 0:
            continue

        local_start = text_start - global_start
        local_end = local_start + text_rows
        assert torch.equal(shard[local_start:local_end], weight[text_start:text_end])


def test_different_seed_changes_added_rows_but_not_source_rows():
    weight = torch.arange(18, dtype=torch.float32).view(6, 3)

    resized_a, _ = resize_vocab_table(
        weight,
        source_vocab_size=4,
        target_vocab_size=7,
        padded_vocab_size=8,
        seed=123,
    )
    resized_b, _ = resize_vocab_table(
        weight,
        source_vocab_size=4,
        target_vocab_size=7,
        padded_vocab_size=8,
        seed=124,
    )

    assert torch.equal(resized_a[:4], resized_b[:4])
    assert not torch.equal(resized_a[4:], resized_b[4:])


def test_real_rows_do_not_depend_on_padding_size():
    weight = torch.arange(18, dtype=torch.float32).view(6, 3)

    padded_8, _ = resize_vocab_table(
        weight,
        source_vocab_size=4,
        target_vocab_size=7,
        padded_vocab_size=8,
        seed=123,
    )
    padded_10, _ = resize_vocab_table(
        weight,
        source_vocab_size=4,
        target_vocab_size=7,
        padded_vocab_size=10,
        seed=123,
    )

    assert torch.equal(padded_8[:7], padded_10[:7])
    assert padded_10.shape == (10, 3)


def test_resize_does_not_mutate_global_rng_state():
    weight = torch.arange(18, dtype=torch.float32).view(6, 3)
    rng_state = torch.get_rng_state()

    resize_vocab_table(
        weight,
        source_vocab_size=4,
        target_vocab_size=7,
        padded_vocab_size=8,
        seed=123,
    )

    assert torch.equal(torch.get_rng_state(), rng_state)


def test_megatron_original_init_repeats_final_source_row():
    weight = torch.arange(18, dtype=torch.float32).view(6, 3)

    resized, plan = resize_vocab_table(
        weight,
        source_vocab_size=4,
        target_vocab_size=7,
        padded_vocab_size=9,
        seed=123,
        init_method="megatron-original",
    )

    assert resized.shape == (9, 3)
    assert torch.equal(resized[:4], weight[:4])
    assert torch.equal(resized[4:], weight[3].unsqueeze(0).expand(5, -1))
    assert plan.trimmed_rows == 2
    assert plan.real_added_rows == 3
    assert plan.padding_added_rows == 2


@pytest.mark.parametrize(
    ("source_vocab_size", "target_vocab_size", "padded_vocab_size"),
    [
        (0, 7, 8),
        (5, 4, 8),
        (4, 9, 8),
    ],
)
def test_invalid_sizes_raise(source_vocab_size, target_vocab_size, padded_vocab_size):
    weight = torch.arange(18, dtype=torch.float32).view(6, 3)

    with pytest.raises(ValueError):
        resize_vocab_table(
            weight,
            source_vocab_size=source_vocab_size,
            target_vocab_size=target_vocab_size,
            padded_vocab_size=padded_vocab_size,
            seed=123,
        )


def test_invalid_init_method_raises():
    weight = torch.arange(18, dtype=torch.float32).view(6, 3)

    with pytest.raises(ValueError):
        resize_vocab_table(
            weight,
            source_vocab_size=4,
            target_vocab_size=7,
            padded_vocab_size=8,
            seed=123,
            init_method="unknown",
        )
