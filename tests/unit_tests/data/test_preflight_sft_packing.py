from pathlib import Path

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

    def fake_count(idx_path: Path, seq_length: int, add_extra_token: int):
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
