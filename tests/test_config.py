"""Tests for core.config constants."""

from core.config import (
    DEFAULT_INDEX_DIR,
    MAX_CHUNK_SIZE,
    OLLAMA_API_URL,
    OLLAMA_HEALTH_TIMEOUT,
    OLLAMA_HEALTH_URL,
)


def test_default_index_dir() -> None:
    """CLI, Searcher, and BM25 write share one default folder."""
    assert DEFAULT_INDEX_DIR == "data/processed"


def test_max_chunk_size_is_moulinette_limit() -> None:
    """MAX_CHUNK_SIZE must match the moulinette 2000-char cap."""
    assert MAX_CHUNK_SIZE == 2000


def test_ollama_urls_point_at_localhost() -> None:
    """Ollama endpoints must target the local server."""
    assert OLLAMA_API_URL.startswith("http://localhost:11434/")
    assert OLLAMA_HEALTH_URL.startswith("http://localhost:11434/")


def test_ollama_health_timeout_allows_model_load() -> None:
    """Health check waits longer than a tight 2s ping."""
    assert OLLAMA_HEALTH_TIMEOUT >= 10
