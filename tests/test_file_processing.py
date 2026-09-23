"""Tests for ingestion.file_processing."""

from pathlib import Path

import pytest

from ingestion.file_processing import (
    extract_files_from_questions,
    get_all_files,
    read_file,
)


def test_read_file_success(text_file: Path) -> None:
    """read_file returns the UTF-8 contents of an existing file."""
    assert read_file(str(text_file)) == "abcdefghij"


def test_read_file_missing_returns_empty(tmp_path: Path) -> None:
    """read_file returns '' when the path does not exist."""
    assert read_file(str(tmp_path / "missing.txt")) == ""


def test_extract_files_from_questions(tmp_path: Path) -> None:
    """extract_files_from_questions collects unique file_path values."""
    tsv = tmp_path / "questions.tsv"
    tsv.write_text(
        "question\tfile_path\n"
        "q1\tsrc/a.py\n"
        "q2\tsrc/b.py\n"
        "q3\tsrc/a.py\n",
        encoding="utf-8",
    )
    paths = extract_files_from_questions(str(tsv))
    assert set(paths) == {"src/a.py", "src/b.py"}


def test_extract_files_missing_header(tmp_path: Path) -> None:
    """A TSV without file_path raises ValueError."""
    tsv = tmp_path / "bad.tsv"
    tsv.write_text("question\tanswer\nq1\ta1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="file_path"):
        extract_files_from_questions(str(tsv))


def test_get_all_files_skips_excluded(tmp_path: Path) -> None:
    """get_all_files skips hidden files and excluded directories."""
    (tmp_path / "keep.py").write_text("x", encoding="utf-8")
    (tmp_path / ".hidden").write_text("x", encoding="utf-8")
    git = tmp_path / ".git"
    git.mkdir()
    (git / "config").write_text("x", encoding="utf-8")
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "mod.pyc").write_text("x", encoding="utf-8")
    nested = tmp_path / "pkg"
    nested.mkdir()
    (nested / "ok.py").write_text("x", encoding="utf-8")

    files = get_all_files(str(tmp_path))
    names = {Path(p).name for p in files}
    assert names == {"keep.py", "ok.py"}
