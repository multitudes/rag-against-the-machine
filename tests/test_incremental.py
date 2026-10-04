"""Tests for incremental indexing (bonus 3)."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from core.schemas import ChunkSource, MinimalSource, UnansweredQuestion
from ingestion.chunking import chunk_content
from ingestion.file_processing import get_all_files
from ingestion.incremental import (
    chunk_all_files,
    collect_chunks,
    collect_chunks_incremental,
    load_chunks_from_index,
    write_file_manifest,
)
from ingestion.indexing import create_bm25_index
from retrieval.search import Searcher
from retrieval.semantic import embeddings_path, merge_semantic_index


def _tiny_repo(tmp_path: Path) -> Path:
    """
    Create a two-file corpus under tmp_path/repo.

    Args:
        tmp_path: pytest temporary directory.

    Returns:
        Path to the repo root.

    """
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
    return repo


def _path_ending(files: list[str], name: str) -> str:
    """
    Return the corpus path whose basename matches name.

    Args:
        files: Paths from get_all_files.
        name: File name suffix, e.g. ``edit.py``.

    Returns:
        Matching path string.

    """
    for path in files:
        if path.endswith(name):
            return path
    msg = f"no file ending with {name!r}"
    raise AssertionError(msg)


def _full_index(
    repo: Path,
    index_dir: Path,
    max_chunk_size: int = 200,
) -> list[str]:
    """
    Build a BM25 index and files.json for a tiny repo.

    Args:
        repo: Corpus root.
        index_dir: Output index directory.
        max_chunk_size: Chunk size for this index.

    Returns:
        Paths passed to the indexer.

    """
    files = get_all_files(str(repo))
    chunks = chunk_all_files(files, max_chunk_size)
    create_bm25_index(chunks, str(index_dir))
    write_file_manifest(files, max_chunk_size, str(index_dir))
    return files


def test_full_index_writes_manifest(tmp_path: Path) -> None:
    """A full index writes files.json next to metadata.json."""
    repo = _tiny_repo(tmp_path)
    index_dir = tmp_path / "idx"
    _full_index(repo, index_dir)
    assert (index_dir / "files.json").exists()
    assert (index_dir / "metadata.json").exists()


def test_incremental_noop_skips_chunking(tmp_path: Path) -> None:
    """Second --incremental with no edits reuses the old chunks."""
    repo = _tiny_repo(tmp_path)
    index_dir = tmp_path / "idx"
    files = _full_index(repo, index_dir)
    with patch(
        "ingestion.incremental.chunk_content",
        wraps=chunk_content,
    ) as spy:
        result = collect_chunks_incremental(files, 200, str(index_dir))
    assert result is not None
    assert result.nothing_changed is True
    spy.assert_not_called()


def test_incremental_rechunks_only_changed_file(
    tmp_path: Path,
) -> None:
    """Editing one file re-chunks that file only."""
    repo = _tiny_repo(tmp_path)
    index_dir = tmp_path / "idx"
    files = _full_index(repo, index_dir)
    edit_path = _path_ending(files, "edit.py")
    Path(edit_path).write_text(
        "def edit():\n    return 'gamma whale sings'\n",
        encoding="utf-8",
    )
    with patch(
        "ingestion.incremental.chunk_content",
        wraps=chunk_content,
    ) as spy:
        result = collect_chunks_incremental(files, 200, str(index_dir))
    assert result is not None
    assert result.nothing_changed is False
    called = [c.args[0] for c in spy.call_args_list]
    assert called == [edit_path]
    texts = [c.text for c in result.chunks]
    assert any("gamma whale" in t for t in texts)
    assert any("alpha cat" in t for t in texts)


def test_incremental_rebuilds_searchable_index(
    tmp_path: Path,
) -> None:
    """After a merge, BM25 search finds the new wording."""
    repo = _tiny_repo(tmp_path)
    index_dir = tmp_path / "idx"
    files = _full_index(repo, index_dir)
    edit_path = _path_ending(files, "edit.py")
    Path(edit_path).write_text(
        "def edit():\n    return 'gamma whale sings'\n",
        encoding="utf-8",
    )
    result = collect_chunks_incremental(files, 200, str(index_dir))
    assert result is not None
    create_bm25_index(result.chunks, str(index_dir))
    write_file_manifest(files, 200, str(index_dir))
    searcher = Searcher(index_dir=str(index_dir))
    question = UnansweredQuestion(question="gamma whale")
    hits = searcher.search_one(question, k=1)
    assert hits.retrieved_sources
    assert hits.retrieved_sources[0].file_path.endswith("edit.py")


def test_incremental_drops_deleted_file(tmp_path: Path) -> None:
    """Removing a file drops its chunks from the merge."""
    repo = _tiny_repo(tmp_path)
    index_dir = tmp_path / "idx"
    files = _full_index(repo, index_dir)
    edit_path = _path_ending(files, "edit.py")
    Path(edit_path).unlink()
    remaining = [p for p in files if p != edit_path]
    result = collect_chunks_incremental(
        remaining,
        200,
        str(index_dir),
    )
    assert result is not None
    paths = {c.source.file_path for c in result.chunks}
    assert edit_path not in paths
    assert any(p.endswith("keep.py") for p in paths)


def test_incremental_adds_new_file(tmp_path: Path) -> None:
    """A new file is chunked and appended."""
    repo = _tiny_repo(tmp_path)
    index_dir = tmp_path / "idx"
    _full_index(repo, index_dir)
    (repo / "extra.py").write_text(
        "def extra():\n    return 'omega extra'\n",
        encoding="utf-8",
    )
    files = get_all_files(str(repo))
    extra_path = _path_ending(files, "extra.py")
    result = collect_chunks_incremental(files, 200, str(index_dir))
    assert result is not None
    paths = {c.source.file_path for c in result.chunks}
    assert extra_path in paths


def test_collect_chunks_full_has_empty_old_list(
    tmp_path: Path,
) -> None:
    """Without --incremental we wrap a full chunk pass."""
    repo = _tiny_repo(tmp_path)
    files = get_all_files(str(repo))
    result = collect_chunks(files, 200, str(tmp_path / "idx"))
    assert result.nothing_changed is False
    assert result.old_chunks == []
    assert result.unchanged_files == set()
    assert result.chunks


def test_collect_chunks_fallback_matches_full(
    tmp_path: Path,
) -> None:
    """Missing baseline still returns a full ChunkBuildResult."""
    repo = _tiny_repo(tmp_path)
    files = get_all_files(str(repo))
    result = collect_chunks(
        files,
        200,
        str(tmp_path / "missing"),
        incremental=True,
    )
    assert result.nothing_changed is False
    assert result.old_chunks == []
    assert result.chunks


def test_incremental_falls_back_without_index(
    tmp_path: Path,
) -> None:
    """No previous index means collect returns None (full reindex)."""
    repo = _tiny_repo(tmp_path)
    files = get_all_files(str(repo))
    result = collect_chunks_incremental(
        files,
        200,
        str(tmp_path / "missing"),
    )
    assert result is None


def test_incremental_falls_back_on_chunk_size_change(
    tmp_path: Path,
) -> None:
    """A different max_chunk_size forces a full reindex."""
    repo = _tiny_repo(tmp_path)
    index_dir = tmp_path / "idx"
    files = _full_index(repo, index_dir, max_chunk_size=200)
    result = collect_chunks_incremental(files, 100, str(index_dir))
    assert result is None


def test_load_chunks_roundtrip(tmp_path: Path) -> None:
    """load_chunks_from_index restores texts and paths."""
    repo = _tiny_repo(tmp_path)
    index_dir = tmp_path / "idx"
    _full_index(repo, index_dir)
    loaded = load_chunks_from_index(str(index_dir))
    assert loaded
    assert all(c.text for c in loaded)
    assert all(c.source.file_path.endswith(".py") for c in loaded)


def test_load_chunks_rejects_non_object_corpus_line(
    tmp_path: Path,
) -> None:
    """A corpus line that is not {id, text} is an error."""
    repo = _tiny_repo(tmp_path)
    index_dir = tmp_path / "idx"
    _full_index(repo, index_dir)
    corpus = index_dir / "corpus.jsonl"
    corpus.write_text('"just a string"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="text"):
        load_chunks_from_index(str(index_dir))


def test_merge_semantic_encodes_only_new_texts(
    tmp_path: Path,
) -> None:
    """Unchanged files keep their MiniLM rows; new texts are encoded."""
    old_chunks = [
        ChunkSource(
            text="keep me",
            source=MinimalSource(
                file_path="keep.py",
                first_character_index=0,
                last_character_index=7,
            ),
        ),
        ChunkSource(
            text="old edit",
            source=MinimalSource(
                file_path="edit.py",
                first_character_index=0,
                last_character_index=8,
            ),
        ),
    ]
    new_chunks = [
        old_chunks[0],
        ChunkSource(
            text="new edit",
            source=MinimalSource(
                file_path="edit.py",
                first_character_index=0,
                last_character_index=8,
            ),
        ),
    ]
    old_matrix = np.array(
        [[1.0, 0.0], [0.0, 1.0]],
        dtype=np.float32,
    )
    np.save(embeddings_path(str(tmp_path)), old_matrix)
    encoded = np.array([[0.5, 0.5]], dtype=np.float32)

    def _encode(texts: list[str]) -> np.ndarray:
        assert texts == ["new edit"]
        return encoded

    with patch("retrieval.semantic.encode_texts", side_effect=_encode):
        merge_semantic_index(
            old_chunks,
            new_chunks,
            {"keep.py"},
            str(tmp_path),
        )
    matrix = np.load(embeddings_path(str(tmp_path)))
    assert list(matrix[0]) == [1.0, 0.0]
    assert list(matrix[1]) == [0.5, 0.5]
