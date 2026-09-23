"""Shared fixtures for the RAG test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.schemas import (
    ChunkSource,
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    StudentSearchResults,
    UnansweredQuestion,
)


@pytest.fixture
def sample_source() -> MinimalSource:
    """
    Return a MinimalSource pointing at a fake file span.

    Returns:
        A MinimalSource with a 10-character span.

    """
    return MinimalSource(
        file_path="data/raw/example.py",
        first_character_index=0,
        last_character_index=10,
    )


@pytest.fixture
def sample_chunk(sample_source: MinimalSource) -> ChunkSource:
    """
    Return a ChunkSource wrapping sample_source.

    Args:
        sample_source: Source fixture.

    Returns:
        ChunkSource with short text.

    """
    return ChunkSource(text="hello world", source=sample_source)


@pytest.fixture
def sample_question() -> UnansweredQuestion:
    """
    Return an UnansweredQuestion with a fixed id.

    Returns:
        UnansweredQuestion used across search/answer tests.

    """
    return UnansweredQuestion(
        question_id="q-1",
        question="What is BM25?",
    )


@pytest.fixture
def sample_search_result(
    sample_question: UnansweredQuestion,
    sample_source: MinimalSource,
) -> MinimalSearchResults:
    """
    Return a MinimalSearchResults for a single question.

    Args:
        sample_question: Question fixture.
        sample_source: Source fixture.

    Returns:
        MinimalSearchResults with one retrieved source.

    """
    return MinimalSearchResults(
        question_id=sample_question.question_id,
        question=sample_question.question,
        retrieved_sources=[sample_source],
    )


@pytest.fixture
def sample_student_results(
    sample_search_result: MinimalSearchResults,
) -> StudentSearchResults:
    """
    Return StudentSearchResults wrapping one search result.

    Args:
        sample_search_result: Single-question search result.

    Returns:
        StudentSearchResults with k=1.

    """
    return StudentSearchResults(
        search_results=[sample_search_result],
        k=1,
    )


@pytest.fixture
def sample_answer(
    sample_search_result: MinimalSearchResults,
) -> MinimalAnswer:
    """
    Return a MinimalAnswer built from sample_search_result.

    Args:
        sample_search_result: Search result fixture.

    Returns:
        MinimalAnswer with a stub answer string.

    """
    return MinimalAnswer(
        question_id=sample_search_result.question_id,
        question=sample_search_result.question,
        retrieved_sources=sample_search_result.retrieved_sources,
        answer="BM25 is a ranking function.",
    )


@pytest.fixture
def text_file(tmp_path: Path) -> Path:
    """
    Write a short UTF-8 text file used by file-reading tests.

    Args:
        tmp_path: pytest temporary directory.

    Returns:
        Path to the written file.

    """
    path = tmp_path / "sample.txt"
    path.write_text("abcdefghij", encoding="utf-8")
    return path
