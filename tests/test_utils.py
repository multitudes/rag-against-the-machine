"""Tests for src/utils.py helpers."""

from utils import calculate_overlap_percentage


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
