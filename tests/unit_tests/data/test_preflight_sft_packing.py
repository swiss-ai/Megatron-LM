from pathlib import Path

import numpy as np
import pytest

from tools import preflight_sft_packing


def test_read_data_path_file_reads_path_and_epochs(tmp_path: Path):
    path_file = tmp_path / "prefixes.txt"
    path_file.write_text(
        "\n"
        "# path epochs\n"
        "/data/text 2\n"
        "  /data/vision 1\n"
    )

    assert preflight_sft_packing.read_data_path_file(path_file) == [
        (Path("/data/text"), 2),
        (Path("/data/vision"), 1),
    ]


def test_read_data_path_file_rejects_fractional_epochs(tmp_path: Path):
    path_file = tmp_path / "prefixes.txt"
    path_file.write_text("/data/text 0.5\n")

    with pytest.raises(ValueError, match="fractional epochs are not supported yet"):
        preflight_sft_packing.read_data_path_file(path_file)


def test_summarize_datasets_expands_directory_once_and_applies_epochs(tmp_path: Path):
    text_dir = tmp_path / "text"
    text_dir.mkdir()
    (text_dir / "b.idx").touch()
    (text_dir / "a.idx").touch()

    calls = []

    def fake_count(
        idx_path: Path,
        seq_length: int,
        add_extra_token: int,
        strategy: str = "bfd",
        seed: int = 1234,
        max_docs_per_bin: int = 0,
    ):
        calls.append((idx_path.name, seq_length, add_extra_token))
        if idx_path.name == "a.idx":
            return 10, 100, 7
        return 20, 200, 11

    rows, total_packed_samples = preflight_sft_packing.summarize_datasets(
        [(text_dir, 2)],
        seq_length=8192,
        add_extra_token=1,
        count_fn=fake_count,
    )

    assert calls == [("a.idx", 8192, 1), ("b.idx", 8192, 1)]
    assert total_packed_samples == 36
    assert rows == [
        {
            "path": str(text_dir),
            "files": 2,
            "docs": 30,
            "tokens": 300,
            "samples": 18,
            "epochs": 2,
            "weighted_samples": 36,
        }
    ]


def test_compute_step_counts_uses_floor_for_recommended_steps():
    steps = preflight_sft_packing.compute_step_counts(
        total_packed_samples=25, global_batch_size=8
    )

    assert steps == {
        "floor_steps": 3,
        "ceil_steps": 4,
        "floor_leftover_samples": 1,
        "ceil_shortfall_samples": 7,
    }


def _greedy_python_count(lengths: np.ndarray, order: np.ndarray, capacity: int) -> int:
    from megatron.training.datasets.apertus_sft_dataset import (
        _build_sample_idx_greedy_python,
    )

    sample_idx = _build_sample_idx_greedy_python(
        lengths, order, capacity, add_extra_token=0
    )
    return int(sample_idx.shape[0] - 1)


def test_greedy_packs_in_order():
    lengths = np.array([4, 4, 4], dtype=np.int32)
    order = np.arange(3, dtype=np.int32)

    # capacity 10: [4, 4] fills to 8, the third 4 starts a new sample
    assert _greedy_python_count(lengths, order, 10) == 2


def test_greedy_exact_fit_closes_sample():
    lengths = np.array([5, 5, 5], dtype=np.int32)
    order = np.arange(3, dtype=np.int32)

    assert _greedy_python_count(lengths, order, 10) == 2


def test_greedy_oversized_doc_gets_own_sample():
    lengths = np.array([12, 3], dtype=np.int32)
    order = np.arange(2, dtype=np.int32)

    assert _greedy_python_count(lengths, order, 10) == 2


def test_greedy_respects_document_order():
    lengths = np.array([6, 6, 4, 4], dtype=np.int32)

    # in-order: [6], [6, 4], [4] -> 3 samples; interleaved: [6, 4], [6, 4] -> 2
    in_order = np.array([0, 1, 2, 3], dtype=np.int32)
    interleaved = np.array([0, 2, 1, 3], dtype=np.int32)
    assert _greedy_python_count(lengths, in_order, 10) == 3
    assert _greedy_python_count(lengths, interleaved, 10) == 2


def test_greedy_python_matches_cpp_kernel():
    helpers = pytest.importorskip(
        "megatron.core.datasets.helpers", reason="compiled helpers_cpp unavailable"
    )
    from megatron.training.datasets.apertus_sft_dataset import (
        _build_sample_idx_greedy_python,
    )

    rng = np.random.RandomState(0)
    lengths = rng.randint(1, 300, size=500).astype(np.int32)
    order = np.arange(500, dtype=np.int32)
    rng.shuffle(order)

    python_idx = _build_sample_idx_greedy_python(
        lengths, order, 128, add_extra_token=1
    )
    cpp_idx = helpers.build_sample_idx_packed_whole_docs(
        lengths, order, 128, add_extra_token_to_sequence=1
    )
    assert np.array_equal(python_idx, cpp_idx)


def test_shuffled_document_order_matches_training_shuffle():
    from megatron.training.datasets.apertus_sft_dataset import _build_document_index

    seed = 28
    expected = _build_document_index(
        1, np.arange(50, dtype=np.int32), np.random.RandomState(seed)
    )

    assert np.array_equal(
        preflight_sft_packing.shuffled_document_order(50, seed), expected
    )
