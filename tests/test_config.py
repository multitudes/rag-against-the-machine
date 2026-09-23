"""Tests for core.config constants."""

from core.config import (
    MAX_CHUNK_SIZE,
    OLLAMA_API_URL,
    OLLAMA_HEALTH_URL,
)


def test_max_chunk_size_is_moulinette_limit() -> None:
    """MAX_CHUNK_SIZE must match the moulinette 2000-char cap."""
    assert MAX_CHUNK_SIZE == 2000


def test_ollama_urls_point_at_localhost() -> None:
    """Ollama endpoints must target the local server."""
    assert OLLAMA_API_URL.startswith("http://localhost:11434/")
    assert OLLAMA_HEALTH_URL.startswith("http://localhost:11434/")
