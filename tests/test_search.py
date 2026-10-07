"""Tests for retrieval.search (no network)."""

import logging
from pathlib import Path

import pytest

from core.schemas import (
    ChunkSource,
    MinimalSearchResults,
    MinimalSource,
    UnansweredQuestion,
)
from ingestion.indexing import create_bm25_index
from retrieval.search import (
    Searcher,
    _read_source_text,
    retrieve_context_from_sources,
)


def test_read_source_text_span(text_file: Path) -> None:
    """_read_source_text returns the requested character span."""
    source = MinimalSource(
        file_path=str(text_file),
        first_character_index=2,
        last_character_index=6,
    )
    assert _read_source_text(source) == "cdef"


def test_read_source_text_utf8_span(tmp_path: Path) -> None:
    """Offsets are characters, not bytes (é is two UTF-8 bytes)."""
    path = tmp_path / "cafe.txt"
    path.write_text("caféXYZ", encoding="utf-8")
    source = MinimalSource(
        file_path=str(path),
        first_character_index=4,
        last_character_index=7,
    )
    assert _read_source_text(source) == "XYZ"


def test_read_source_text_missing_returns_none(tmp_path: Path) -> None:
    """A missing file yields None instead of raising."""
    source = MinimalSource(
        file_path=str(tmp_path / "gone.txt"),
        first_character_index=0,
        last_character_index=4,
    )
    assert _read_source_text(source) is None


def test_retrieve_context_from_sources(text_file: Path) -> None:
    """Standalone helper reads each source span."""
    result = MinimalSearchResults(
        question_id="q",
        question="what?",
        retrieved_sources=[
            MinimalSource(
                file_path=str(text_file),
                first_character_index=0,
                last_character_index=5,
            ),
        ],
    )
    chunks = retrieve_context_from_sources(result)
    assert chunks == ["abcde"]


def test_searcher_missing_index_dir(tmp_path: Path) -> None:
    """Searcher raises FileNotFoundError when the index is absent."""
    with pytest.raises(FileNotFoundError, match="Index directory"):
        Searcher(index_dir=str(tmp_path / "nope"))


def test_searcher_does_not_force_root_debug(tmp_path: Path) -> None:
    """Searcher leaves the process log level (CLI INFO) unchanged."""
    root = logging.getLogger()
    previous = root.level
    root.setLevel(logging.INFO)
    try:
        with pytest.raises(FileNotFoundError):
            Searcher(index_dir=str(tmp_path / "nope"))
        assert root.level == logging.INFO
    finally:
        root.setLevel(previous)


def test_searcher_missing_metadata(tmp_path: Path) -> None:
    """Searcher raises when metadata.json is missing."""
    tmp_path.mkdir(exist_ok=True)
    with pytest.raises(FileNotFoundError, match="Metadata file"):
        Searcher(index_dir=str(tmp_path))


def _build_index(tmp_path: Path) -> Path:
    """
    Create a tiny on-disk BM25 index for search tests.

    Args:
        tmp_path: pytest temporary directory.

    Returns:
        Path to the index directory.

    """
    index_dir = tmp_path / "idx"
    chunks = [
        ChunkSource(
            text="the cat sat on the mat near the fireplace",
            source=MinimalSource(
                file_path="cat.py",
                first_character_index=0,
                last_character_index=41,
            ),
        ),
        ChunkSource(
            text="dogs bark loudly at the moon every night",
            source=MinimalSource(
                file_path="dog.py",
                first_character_index=0,
                last_character_index=40,
            ),
        ),
        ChunkSource(
            text="bm25 ranks documents by term frequency",
            source=MinimalSource(
                file_path="bm25.py",
                first_character_index=0,
                last_character_index=38,
            ),
        ),
    ]
    create_bm25_index(chunks, index_dir=str(index_dir))
    return index_dir


def test_search_one_and_dataset(tmp_path: Path) -> None:
    """search_one and search_dataset return the expected shapes."""
    index_dir = _build_index(tmp_path)
    searcher = Searcher(index_dir=str(index_dir))
    question = UnansweredQuestion(
        question_id="q-cat",
        question="where did the cat sit",
    )
    one = searcher.search_one(unanswered_question=question, k=2)
    assert one.question_id == "q-cat"
    assert one.question == question.question
    assert 1 <= len(one.retrieved_sources) <= 2
    assert all(s.file_path.endswith(".py") for s in one.retrieved_sources)

    dataset = searcher.search_dataset(questions=[question], k=2)
    assert dataset.k == 2
    assert len(dataset.search_results) == 1
    assert dataset.search_results[0].question_id == "q-cat"
