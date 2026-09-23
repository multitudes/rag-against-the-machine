"""Tests for BM25 index creation."""

from pathlib import Path

from core.schemas import ChunkSource, MinimalSource
from ingestion.indexing import create_bm25_index


def _chunk(text: str, path: str, start: int) -> ChunkSource:
    """
    Build a ChunkSource for indexing tests.

    Args:
        text: Chunk text.
        path: Fake source file path.
        start: First character index.

    Returns:
        ChunkSource with offsets derived from text length.

    """
    return ChunkSource(
        text=text,
        source=MinimalSource(
            file_path=path,
            first_character_index=start,
            last_character_index=start + len(text),
        ),
    )


def test_create_bm25_index_empty_skips(tmp_path: Path) -> None:
    """No chunks means no files are written."""
    create_bm25_index([], index_dir=str(tmp_path / "empty"))
    assert not (tmp_path / "empty" / "metadata.json").exists()


def test_create_bm25_index_writes_metadata(tmp_path: Path) -> None:
    """A tiny corpus writes metadata.json and bm25s index files."""
    index_dir = tmp_path / "idx"
    chunks = [
        _chunk("the cat sat on the mat", "a.py", 0),
        _chunk("dogs bark at the moon tonight", "b.py", 0),
        _chunk("retrieval uses bm25 ranking scores", "c.py", 0),
    ]
    create_bm25_index(chunks, index_dir=str(index_dir))
    assert (index_dir / "metadata.json").exists()
    assert (index_dir / "params.index.json").exists()
    assert (index_dir / "corpus.jsonl").exists()
