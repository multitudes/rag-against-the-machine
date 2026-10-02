"""Tests for MiniLM semantic retrieval (model is mocked)."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from core.schemas import ChunkSource, MinimalSource, UnansweredQuestion
from ingestion.indexing import create_bm25_index
from retrieval.search import Searcher
from retrieval.semantic import (
    create_semantic_index,
    embeddings_path,
    search_semantic_ids,
    top_k_indices,
)


def _build_index(tmp_path: Path) -> Path:
    """
    Create a tiny on-disk BM25 index for semantic tests.

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


def test_top_k_indices_orders_by_cosine() -> None:
    """Closest vector (after L2) is returned first."""
    matrix = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    query = np.array([0.0, 2.0, 0.0], dtype=np.float32)
    ids = top_k_indices(query, matrix, k=2)
    assert list(ids)[0] == 1
    assert len(ids) == 2


def test_create_semantic_index_writes_npy(tmp_path: Path) -> None:
    """create_semantic_index saves a matrix aligned with the corpus."""
    fake = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    def _encode(texts: list[str]) -> np.ndarray:
        assert len(texts) == 2
        return fake

    with patch("retrieval.semantic.encode_texts", side_effect=_encode):
        create_semantic_index(["alpha", "beta"], str(tmp_path))
    path = embeddings_path(str(tmp_path))
    assert path.exists()
    loaded = np.load(path)
    assert loaded.shape == (2, 2)


def test_search_semantic_ids_missing_file(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Search without embeddings.npy logs an error and returns []."""
    ids = search_semantic_ids("q", str(tmp_path), k=3)
    assert ids == []
    assert "Semantic index not found" in caplog.text


def test_search_one_semantic_uses_vectors(tmp_path: Path) -> None:
    """search_one(semantic=True) returns the nearest fake embedding."""
    index_dir = _build_index(tmp_path)
    matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    np.save(embeddings_path(str(index_dir)), matrix)

    def _encode(texts: list[str]) -> np.ndarray:
        return np.array([[0.0, 0.0, 1.0]], dtype=np.float32)

    searcher = Searcher(index_dir=str(index_dir))
    question = UnansweredQuestion(question_id="q", question="bm25?")
    with patch("retrieval.semantic.encode_texts", side_effect=_encode):
        result = searcher.search_one(
            unanswered_question=question,
            k=1,
            semantic=True,
        )
    assert len(result.retrieved_sources) == 1
    assert result.retrieved_sources[0].file_path == "bm25.py"


def test_search_one_default_stays_bm25(tmp_path: Path) -> None:
    """Without semantic=True, missing embeddings.npy is ignored."""
    index_dir = _build_index(tmp_path)
    searcher = Searcher(index_dir=str(index_dir))
    question = UnansweredQuestion(
        question_id="q-cat",
        question="where did the cat sit",
    )
    result = searcher.search_one(unanswered_question=question, k=2)
    assert result.retrieved_sources
    assert all(s.file_path.endswith(".py") for s in result.retrieved_sources)


def test_search_one_semantic_missing_embeddings(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """semantic=True without embeddings.npy returns no sources."""
    index_dir = _build_index(tmp_path)
    searcher = Searcher(index_dir=str(index_dir))
    question = UnansweredQuestion(question_id="q", question="anything")
    result = searcher.search_one(
        unanswered_question=question,
        k=3,
        semantic=True,
    )
    assert result.retrieved_sources == []
    assert "Semantic index not found" in caplog.text


def test_search_semantic_ids_ranks_top_k(tmp_path: Path) -> None:
    """search_semantic_ids returns ids ordered by cosine to the query."""
    matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.2, 0.8, 0.0],
        ],
        dtype=np.float32,
    )
    np.save(embeddings_path(str(tmp_path)), matrix)

    def _encode(texts: list[str]) -> np.ndarray:
        return np.array([[0.0, 1.0, 0.0]], dtype=np.float32)

    with patch("retrieval.semantic.encode_texts", side_effect=_encode):
        ids = search_semantic_ids("dogs", str(tmp_path), k=2)
    assert ids[0] == 1
    assert len(ids) == 2


def test_top_k_respects_corpus_size() -> None:
    """k larger than the matrix still returns every row."""
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    query = np.array([1.0, 0.0], dtype=np.float32)
    ids = top_k_indices(query, matrix, k=10)
    assert len(ids) == 2


def test_semantic_index_empty_skips(tmp_path: Path) -> None:
    """No texts means no embeddings file."""
    create_semantic_index([], str(tmp_path))
    assert not embeddings_path(str(tmp_path)).exists()
