"""Tests for the local HTTP API (no live Ollama)."""

from __future__ import annotations

import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from api.server import (
    _make_handler,
    handle_answer,
    handle_search,
    health_payload,
    ollama_available,
)
from core.schemas import ChunkSource, MinimalAnswer, MinimalSource
from ingestion.indexing import create_bm25_index


def _tiny_index(tmp_path: Path) -> Path:
    """
    Build a two-chunk BM25 index under tmp_path/idx.

    Args:
        tmp_path: pytest temporary directory.

    Returns:
        Path to the index directory.

    """
    index_dir = tmp_path / "idx"
    chunks = [
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
    ]
    create_bm25_index(chunks, str(index_dir))
    return index_dir


def test_health_reports_index_ready(tmp_path: Path) -> None:
    """ /health is 200 and index_ready when metadata.json exists."""
    index_dir = _tiny_index(tmp_path)
    status, body = health_payload(str(index_dir))
    assert status == 200
    assert body["status"] == "ok"
    assert body["index_ready"] is True


def test_handle_search_empty_query(tmp_path: Path) -> None:
    """Empty query is HTTP 400."""
    index_dir = _tiny_index(tmp_path)
    status, body = handle_search(
        str(index_dir), "", 1, False, False, False,
    )
    assert status == 400
    assert "query" in body["error"]


def test_handle_search_returns_sources(tmp_path: Path) -> None:
    """handle_search returns BM25 hits for a tiny corpus."""
    index_dir = _tiny_index(tmp_path)
    status, body = handle_search(
        str(index_dir), "cats", 1, False, False, False,
    )
    assert status == 200
    assert body["question"] == "cats"
    assert body["cached"] is False
    assert body["retrieved_sources"]
    assert body["retrieved_sources"][0]["file_path"] == "a.py"


def test_handle_search_cache_hit(tmp_path: Path) -> None:
    """Second handle_search with cache=True sets cached true."""
    index_dir = _tiny_index(tmp_path)
    handle_search(str(index_dir), "cats", 1, False, False, True)
    status, body = handle_search(
        str(index_dir), "cats", 1, False, False, True,
    )
    assert status == 200
    assert body["cached"] is True
    assert body["retrieved_sources"][0]["file_path"] == "a.py"


def test_ollama_available_success_is_mocked() -> None:
    """ollama_available is True when GET /api/tags succeeds."""
    fake = MagicMock()
    fake.raise_for_status.return_value = None
    with patch("api.server.requests.get", return_value=fake):
        assert ollama_available() is True


def test_ollama_available_failure_is_mocked() -> None:
    """ollama_available is False when the health request fails."""
    import requests

    with patch(
        "api.server.requests.get",
        side_effect=requests.exceptions.ConnectionError(),
    ):
        assert ollama_available() is False


def test_handle_answer_requires_ollama(tmp_path: Path) -> None:
    """ /answer is 503 when Ollama is down."""
    index_dir = _tiny_index(tmp_path)
    with patch("api.server.ollama_available", return_value=False):
        status, body = handle_answer(
            str(index_dir), "cats", 1, False, False, False,
        )
    assert status == 503
    assert "Ollama" in body["error"]


def test_handle_answer_mocked_llm(tmp_path: Path) -> None:
    """ /answer returns the mocked LLM text plus sources."""
    index_dir = _tiny_index(tmp_path)
    fake = MinimalAnswer(
        question_id="q",
        question="cats",
        retrieved_sources=[
            MinimalSource(
                file_path="a.py",
                first_character_index=0,
                last_character_index=34,
            ),
        ],
        answer="cats sit",
    )
    with (
        patch("api.server.ollama_available", return_value=True),
        patch(
            "api.server.answer_from_search_result",
            return_value=fake,
        ),
    ):
        status, body = handle_answer(
            str(index_dir), "cats", 1, False, False, False,
        )
    assert status == 200
    assert body["answer"] == "cats sit"
    assert body["retrieved_sources"][0]["file_path"] == "a.py"


def test_http_server_search_roundtrip(tmp_path: Path) -> None:
    """A real loopback server answers GET /search."""
    index_dir = _tiny_index(tmp_path)
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        _make_handler(str(index_dir)),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        with urlopen(
            f"http://{host}:{port}/health",
            timeout=2,
        ) as response:
            health = json.loads(response.read().decode("utf-8"))
        assert health["status"] == "ok"
        with urlopen(
            f"http://{host}:{port}/search?query=cats&k=1",
            timeout=5,
        ) as response:
            body = json.loads(response.read().decode("utf-8"))
        assert body["retrieved_sources"][0]["file_path"] == "a.py"
        req = Request(
            f"http://{host}:{port}/search",
            data=json.dumps({"query": "cats", "k": 1}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=5) as response:
            posted = json.loads(response.read().decode("utf-8"))
        assert posted["question"] == "cats"
    finally:
        server.shutdown()
        server.server_close()


def test_http_server_rejects_empty_query(tmp_path: Path) -> None:
    """GET /search without query is HTTP 400."""
    index_dir = _tiny_index(tmp_path)
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        _make_handler(str(index_dir)),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        try:
            urlopen(f"http://{host}:{port}/search?query=", timeout=2)
        except HTTPError as exc:
            assert exc.code == 400
        else:
            raise AssertionError("expected HTTP 400")
    finally:
        server.shutdown()
        server.server_close()
