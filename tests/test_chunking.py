"""Tests for ingestion.chunking helpers and chunk_content."""

from pathlib import Path

from chonkie import CodeChunker, RecursiveChunker, SentenceChunker

from core.schemas import ChunkSource, MinimalSource
from ingestion.chunking import (
    _enforce_max_size,
    _resolve_ext,
    _select_chunker,
    chunk_content,
)


def test_resolve_ext_regular() -> None:
    """A dotted filename yields the lowercase extension."""
    assert _resolve_ext("Foo/Bar.PY") == "py"


def test_resolve_ext_cmakelists() -> None:
    """CMakeLists.txt is treated as the cmake special case."""
    assert _resolve_ext("proj/CMakeLists.txt") == "cmakelists.txt"


def test_resolve_ext_no_dot() -> None:
    """A filename without a dot is returned lowercased."""
    assert _resolve_ext("Dockerfile") == "dockerfile"


def test_select_chunker_code() -> None:
    """Python files use CodeChunker."""
    chunker = _select_chunker("py", chunk_size=200, overlap=20)
    assert isinstance(chunker, CodeChunker)


def test_select_chunker_markdown() -> None:
    """Markdown files use RecursiveChunker."""
    chunker = _select_chunker("md", chunk_size=200, overlap=20)
    assert isinstance(chunker, RecursiveChunker)


def test_select_chunker_text() -> None:
    """Plain-text files use SentenceChunker."""
    chunker = _select_chunker("txt", chunk_size=200, overlap=20)
    assert isinstance(chunker, SentenceChunker)


def test_select_chunker_unknown() -> None:
    """Unsupported extensions return None."""
    assert _select_chunker("exe", chunk_size=200, overlap=20) is None


def test_enforce_max_size_splits_and_keeps_offsets() -> None:
    """Oversized chunks are split; character indices stay consistent."""
    chunk = ChunkSource(
        text="abcdefghij",
        source=MinimalSource(
            file_path="a.py",
            first_character_index=100,
            last_character_index=110,
        ),
    )
    parts = _enforce_max_size([chunk], max_size=4)
    assert [p.text for p in parts] == ["abcd", "efgh", "ij"]
    assert parts[0].source.first_character_index == 100
    assert parts[0].source.last_character_index == 104
    assert parts[-1].source.first_character_index == 108
    assert parts[-1].source.last_character_index == 110
    assert all(len(p.text) <= 4 for p in parts)


def test_enforce_max_size_keeps_small_chunks() -> None:
    """Chunks already under the limit are unchanged."""
    chunk = ChunkSource(
        text="hi",
        source=MinimalSource(
            file_path="a.py",
            first_character_index=0,
            last_character_index=2,
        ),
    )
    assert _enforce_max_size([chunk], max_size=10) == [chunk]


def test_chunk_content_ignores_binary_extension(tmp_path: Path) -> None:
    """Ignored extensions return an empty list without reading."""
    path = tmp_path / "image.png"
    path.write_bytes(b"\x00\x01")
    assert chunk_content(str(path)) == []


def test_chunk_content_python_file(tmp_path: Path) -> None:
    """A small Python file produces at least one ChunkSource."""
    path = tmp_path / "mod.py"
    path.write_text(
        "def greet(name):\n    return f'hello {name}'\n",
        encoding="utf-8",
    )
    chunks = chunk_content(str(path), chunk_size=200)
    assert chunks
    assert all(isinstance(c, ChunkSource) for c in chunks)
    assert all(c.source.file_path == str(path) for c in chunks)
    assert all(len(c.text) <= 200 for c in chunks)


def test_chunk_content_missing_file_returns_empty(tmp_path: Path) -> None:
    """A missing file is logged and yields no chunks."""
    assert chunk_content(str(tmp_path / "nope.py")) == []
