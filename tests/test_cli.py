"""Tests for RagCLI validation and helpers (no live network)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from core.schemas import (
    AnsweredQuestion,
    ChunkSource,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    UnansweredQuestion,
)
from ingestion.indexing import create_bm25_index


def _load_cli() -> Any:
    """
    Import src/__main__.py as rag_main (not pytest's __main__).

    Returns:
        The loaded module object.

    """
    path = Path(__file__).resolve().parents[1] / "src" / "__main__.py"
    spec = importlib.util.spec_from_file_location("rag_main", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cli() -> Any:
    """
    Return a RagCLI instance from the loaded module.

    Returns:
        RagCLI instance.

    """
    return _load_cli().RagCLI()


def test_index_rejects_oversized_chunk(cli: Any, caplog: Any) -> None:
    """index returns early when max_chunk_size exceeds the cap."""
    cli.index(max_chunk_size=5000, repo_path="unused")
    assert "cannot exceed" in caplog.text


def test_index_rejects_non_positive_chunk(cli: Any, caplog: Any) -> None:
    """index returns early when max_chunk_size is not positive."""
    cli.index(max_chunk_size=0, repo_path="unused")
    assert "positive integer" in caplog.text


def test_index_rejects_missing_repo(
    cli: Any,
    tmp_path: Path,
    caplog: Any,
) -> None:
    """index returns early when repo_path does not exist."""
    cli.index(max_chunk_size=200, repo_path=str(tmp_path / "gone"))
    assert "does not exist" in caplog.text


def test_search_rejects_empty_query(cli: Any, caplog: Any) -> None:
    """search returns early on a blank query."""
    cli.search(query="   ", k=5)
    assert "cannot be empty" in caplog.text


def test_search_rejects_bad_k(cli: Any, caplog: Any) -> None:
    """search returns early when k is not positive."""
    cli.search(query="hello", k=0)
    assert "positive integer" in caplog.text


def test_serve_rejects_missing_index(
    cli: Any,
    tmp_path: Path,
    caplog: Any,
) -> None:
    """serve returns early when the index directory is missing."""
    cli.serve(index_dir=str(tmp_path / "nope"), port=8000)
    assert "not found" in caplog.text


def test_serve_rejects_bad_port(cli: Any, caplog: Any) -> None:
    """serve returns early when the port is out of range."""
    cli.serve(port=0)
    assert "1..65535" in caplog.text


def test_search_rejects_missing_index(
    cli: Any,
    tmp_path: Path,
    caplog: Any,
) -> None:
    """search returns early when the index directory is missing."""
    cli.search(query="hello", k=5, index_dir=str(tmp_path / "nope"))
    assert "not found" in caplog.text


def test_search_dataset_rejects_bad_k(cli: Any, caplog: Any) -> None:
    """search_dataset validates k after path checks."""
    cli.search_dataset(
        dataset_path="missing.json",
        k=0,
        index_dir="missing-index",
    )
    assert "not found" in caplog.text


def test_answer_rejects_empty_query(cli: Any, caplog: Any) -> None:
    """answer returns early on an empty query (no Ollama call)."""
    cli.answer(query="", k=5)
    assert "cannot be empty" in caplog.text


def test_check_ollama_success_is_mocked(cli: Any) -> None:
    """_check_ollama returns True when the health request succeeds."""
    fake = MagicMock()
    fake.raise_for_status.return_value = None
    with patch("requests.get", return_value=fake):
        assert cli._check_ollama() is True


def test_check_ollama_failure_is_mocked(cli: Any) -> None:
    """_check_ollama returns False when the health request fails."""
    import requests

    with patch(
        "requests.get",
        side_effect=requests.exceptions.ConnectionError(),
    ):
        assert cli._check_ollama() is False


def test_evaluate_computes_recall(
    cli: Any,
    tmp_path: Path,
    capsys: Any,
) -> None:
    """evaluate prints recall when student and GT sources overlap."""
    src = MinimalSource(
        file_path="a.py",
        first_character_index=0,
        last_character_index=20,
    )
    student = StudentSearchResults(
        search_results=[
            MinimalSearchResults(
                question_id="q1",
                question="q",
                retrieved_sources=[src],
            ),
        ],
        k=1,
    )
    ground = RagDataset(
        rag_questions=[
            AnsweredQuestion(
                question_id="q1",
                question="q",
                sources=[src],
                answer="a",
            ),
        ],
    )
    student_path = tmp_path / "student.json"
    gt_path = tmp_path / "gt.json"
    student_path.write_text(student.model_dump_json(), encoding="utf-8")
    gt_path.write_text(ground.model_dump_json(), encoding="utf-8")

    cli.evaluate(str(student_path), str(gt_path))
    captured = capsys.readouterr()
    assert "Recall@1" in captured.out
    assert "100.0%" in captured.out


def test_evaluate_missing_files(cli: Any, tmp_path: Path, caplog: Any) -> None:
    """evaluate returns early when input files are missing."""
    cli.evaluate(str(tmp_path / "no.json"), str(tmp_path / "no2.json"))
    assert "not found" in caplog.text


def test_answer_dataset_missing_results(
    cli: Any,
    tmp_path: Path,
    caplog: Any,
) -> None:
    """answer_dataset returns early without calling Ollama."""
    cli.answer_dataset(str(tmp_path / "missing.json"))
    assert "not found" in caplog.text


def test_search_semantic_without_vectors(
    cli: Any,
    tmp_path: Path,
    caplog: Any,
) -> None:
    """CLI --semantic with no embeddings.npy does not crash."""
    index_dir = tmp_path / "idx"
    create_bm25_index(
        [
            ChunkSource(
                text="hello world test chunk about cats",
                source=MinimalSource(
                    file_path="a.py",
                    first_character_index=0,
                    last_character_index=34,
                ),
            ),
            ChunkSource(
                text="another document about dogs barking",
                source=MinimalSource(
                    file_path="b.py",
                    first_character_index=0,
                    last_character_index=35,
                ),
            ),
        ],
        str(index_dir),
    )
    cli.search(
        query="hello",
        k=1,
        index_dir=str(index_dir),
        semantic=True,
    )
    assert "Semantic index not found" in caplog.text


def test_cli_hybrid_without_embeddings_still_prints(
    cli: Any,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI --hybrid with no embeddings.npy still prints BM25 hits."""
    index_dir = tmp_path / "idx"
    create_bm25_index(
        [
            ChunkSource(
                text="hello world test chunk about cats",
                source=MinimalSource(
                    file_path="a.py",
                    first_character_index=0,
                    last_character_index=34,
                ),
            ),
            ChunkSource(
                text="another document about dogs barking",
                source=MinimalSource(
                    file_path="b.py",
                    first_character_index=0,
                    last_character_index=35,
                ),
            ),
        ],
        str(index_dir),
    )
    cli.search(
        query="hello cats",
        k=1,
        index_dir=str(index_dir),
        hybrid=True,
    )
    captured = capsys.readouterr()
    assert "a.py" in captured.out


def test_cli_incremental_noop_prints_up_to_date(
    cli: Any,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI --incremental with no edits prints that the index is current."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "keep.py").write_text(
        "def keep():\n    return 'alpha cat sat'\n",
        encoding="utf-8",
    )
    (repo / "edit.py").write_text(
        "def edit():\n    return 'beta dog bark'\n",
        encoding="utf-8",
    )
    index_dir = tmp_path / "idx"
    cli.index(
        max_chunk_size=200,
        repo_path=str(repo),
        index_dir=str(index_dir),
    )
    assert (index_dir / "files.json").exists()
    capsys.readouterr()
    cli.index(
        max_chunk_size=200,
        repo_path=str(repo),
        index_dir=str(index_dir),
        incremental=True,
    )
    captured = capsys.readouterr()
    assert "already up to date" in captured.out


def test_cli_cache_hit_skips_searcher(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Second search --cache reprints hits without loading BM25."""
    module = _load_cli()
    cli = module.RagCLI()
    index_dir = tmp_path / "idx"
    create_bm25_index(
        [
            ChunkSource(
                text="hello world test chunk about cats",
                source=MinimalSource(
                    file_path="a.py",
                    first_character_index=0,
                    last_character_index=34,
                ),
            ),
            ChunkSource(
                text="another document about dogs barking",
                source=MinimalSource(
                    file_path="b.py",
                    first_character_index=0,
                    last_character_index=35,
                ),
            ),
        ],
        str(index_dir),
    )
    cli.search(
        query="hello cats",
        k=1,
        index_dir=str(index_dir),
        cache=True,
    )
    assert (index_dir / "query_cache.json").exists()
    capsys.readouterr()
    with (
        patch.object(module, "Searcher") as ctor,
        patch.object(module, "get_cached_searcher") as cached,
    ):
        cli.search(
            query="hello cats",
            k=1,
            index_dir=str(index_dir),
            cache=True,
        )
    ctor.assert_not_called()
    cached.assert_not_called()
    captured = capsys.readouterr()
    assert "a.py" in captured.out


def test_cli_index_clears_query_cache(
    cli: Any,
    tmp_path: Path,
) -> None:
    """A successful index deletes query_cache.json."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "keep.py").write_text(
        "def keep():\n    return 'alpha cat sat'\n",
        encoding="utf-8",
    )
    index_dir = tmp_path / "idx"
    cli.index(
        max_chunk_size=200,
        repo_path=str(repo),
        index_dir=str(index_dir),
    )
    cache_file = index_dir / "query_cache.json"
    cache_file.write_text('{"entries": {}}', encoding="utf-8")
    cli.index(
        max_chunk_size=200,
        repo_path=str(repo),
        index_dir=str(index_dir),
    )
    assert not cache_file.exists()


def test_unanswered_question_used_in_cli() -> None:
    """Sanity: UnansweredQuestion is the CLI search input type."""
    q = UnansweredQuestion(question="hello")
    assert q.question == "hello"
