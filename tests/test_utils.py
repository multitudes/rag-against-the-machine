"""Tests for src/utils.py helpers."""

from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from core.schemas import (
    MinimalAnswer,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from utils import (
    calculate_overlap_percentage,
    save_search_results_and_answer_to_json,
    write_search_to_file,
)


def test_overlap_full_containment() -> None:
    """Retrieved span fully covering ground truth is 100%."""
    assert calculate_overlap_percentage(0, 10, 0, 10) == 100.0


def test_overlap_partial() -> None:
    """Half overlap relative to the ground-truth length is 50%."""
    assert calculate_overlap_percentage(0, 10, 5, 15) == 50.0


def test_overlap_none() -> None:
    """Disjoint ranges return 0.0."""
    assert calculate_overlap_percentage(0, 10, 20, 30) == 0.0


def test_overlap_zero_length_ground_truth() -> None:
    """Empty ground-truth span returns 0.0."""
    assert calculate_overlap_percentage(5, 5, 0, 10) == 0.0


def test_write_search_to_file(
    tmp_path: Path,
    sample_student_results: StudentSearchResults,
) -> None:
    """write_search_to_file writes a dated JSON under output_dir."""
    fixed = datetime(2026, 1, 2, 3, 4, 5)
    with patch("utils.datetime") as mock_dt:
        mock_dt.now.return_value = fixed
        write_search_to_file(sample_student_results, str(tmp_path))
    expected = tmp_path / "search_results_2026-01-02_03-04-05.json"
    assert expected.exists()
    text = expected.read_text(encoding="utf-8")
    assert "search_results" in text


def test_save_search_results_and_answer_to_json(
    tmp_path: Path,
    sample_answer: MinimalAnswer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """save_search_results_and_answer_to_json writes under cwd."""
    monkeypatch.chdir(tmp_path)
    result = StudentSearchResultsAndAnswer(
        search_results=[sample_answer],
        k=1,
    )
    fixed = datetime(2026, 1, 2, 3, 4, 5)
    with patch("utils.datetime") as mock_dt:
        mock_dt.now.return_value = fixed
        save_search_results_and_answer_to_json(result)
    expected = (
        tmp_path
        / "data"
        / "output"
        / "search_results"
        / "Dataset_2026-01-02_03-04-05_valid.json"
    )
    assert expected.exists()
