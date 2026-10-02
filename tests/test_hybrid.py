"""Tests for hybrid BM25 + MiniLM retrieval (model is mocked)."""

from pathlib import Path
from unittest.mock import patch

from core.schemas import ChunkSource, MinimalSource, UnansweredQuestion
from ingestion.indexing import create_bm25_index
from retrieval.search import Searcher
from retrieval.semantic import rrf_fuse


def _build_index(tmp_path: Path) -> Path:
    """
    Create a tiny on-disk BM25 index for hybrid tests.

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


def test_rrf_prefers_ids_in_both_lists() -> None:
    """An id ranked high in both lists beats a single-list winner."""
    fused = rrf_fuse([[0, 1, 2], [2, 0, 3]], k=3)
    assert fused[0] == 0
    assert 2 in fused[:2]
    assert len(fused) == 3


def test_rrf_empty_lists() -> None:
    """No rankings means no fused ids."""
    assert rrf_fuse([[], []], k=5) == []


def test_rrf_caps_at_k() -> None:
    """Fusion returns at most k unique ids."""
    fused = rrf_fuse([[0, 1, 2, 3], [4, 5, 6]], k=2)
    assert len(fused) == 2


def test_search_one_hybrid_fuses_lists(tmp_path: Path) -> None:
    """hybrid=True RRF-merges BM25 ids with mocked MiniLM ids."""
    index_dir = _build_index(tmp_path)
    searcher = Searcher(index_dir=str(index_dir))
    question = UnansweredQuestion(question_id="q", question="cat")
    with (
        patch.object(
            searcher,
            "_bm25_chunk_ids",
            return_value=[0, 1, 2],
        ),
        patch(
            "retrieval.semantic.search_semantic_ids",
            return_value=[2, 0],
        ),
    ):
        result = searcher.search_one(
            unanswered_question=question,
            k=2,
            hybrid=True,
        )
    paths = [s.file_path for s in result.retrieved_sources]
    assert paths[0] == "cat.py"
    assert "bm25.py" in paths
    assert len(paths) == 2


def test_hybrid_wins_over_semantic_flag(tmp_path: Path) -> None:
    """When both flags are set, search_one uses RRF not MiniLM-only."""
    index_dir = _build_index(tmp_path)
    searcher = Searcher(index_dir=str(index_dir))
    question = UnansweredQuestion(question_id="q", question="cat")
    with (
        patch.object(
            searcher,
            "_bm25_chunk_ids",
            return_value=[0, 1],
        ) as lexical,
        patch(
            "retrieval.semantic.search_semantic_ids",
            return_value=[1, 2],
        ) as semantic_ids,
        patch.object(
            searcher,
            "_search_one_semantic",
        ) as semantic_only,
    ):
        result = searcher.search_one(
            unanswered_question=question,
            k=2,
            semantic=True,
            hybrid=True,
        )
    lexical.assert_called()
    semantic_ids.assert_called()
    semantic_only.assert_not_called()
    assert result.retrieved_sources


def test_hybrid_without_embeddings_falls_back_to_bm25(
    tmp_path: Path,
) -> None:
    """Missing embeddings.npy still returns BM25 hits under --hybrid."""
    index_dir = _build_index(tmp_path)
    searcher = Searcher(index_dir=str(index_dir))
    question = UnansweredQuestion(
        question_id="q-cat",
        question="where did the cat sit",
    )
    result = searcher.search_one(
        unanswered_question=question,
        k=1,
        hybrid=True,
    )
    assert result.retrieved_sources
    assert result.retrieved_sources[0].file_path == "cat.py"
