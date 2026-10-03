"""
Local HTTP API (bonus 5).

Stdlib ``http.server`` so the index can be queried and answered
without the CLI. The process keeps one Searcher warm (bonus 4).
"""

from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests

from answering.answer import answer_from_search_result
from core.config import OLLAMA_HEALTH_URL
from core.schemas import (
    MinimalSearchResults,
    MinimalSource,
    UnansweredQuestion,
)
from retrieval.cache import get_cached_searcher, lookup_query, store_query

logger = logging.getLogger(__name__)


def ollama_available() -> bool:
    """
    Return True if the local Ollama server responds.

    Returns:
        True when GET /api/tags succeeds.

    """
    try:
        response = requests.get(OLLAMA_HEALTH_URL, timeout=2)
        response.raise_for_status()
    except requests.exceptions.RequestException:
        logger.exception("Ollama is not running or not accessible.")
        return False
    return True


def health_payload(index_dir: str) -> tuple[int, dict[str, Any]]:
    """
    Build the /health response.

    Args:
        index_dir: BM25 index directory.

    Returns:
        HTTP status and JSON body.

    """
    ready = (Path(index_dir) / "metadata.json").exists()
    return 200, {
        "status": "ok",
        "index_dir": index_dir,
        "index_ready": ready,
    }


def handle_search(
    index_dir: str,
    query: str,
    k: int,
    semantic: bool,
    hybrid: bool,
    cache: bool,
) -> tuple[int, dict[str, Any]]:
    """
    Run one search and return a JSON-ready body.

    Args:
        index_dir: BM25 index directory.
        query: Question text.
        k: Number of sources.
        semantic: MiniLM-only ranking.
        hybrid: RRF of BM25 and MiniLM.
        cache: Use the on-disk query cache.

    Returns:
        HTTP status and JSON body.

    """
    if not query or not query.strip():
        return 400, {"error": "query cannot be empty"}
    if k <= 0:
        return 400, {"error": "k must be a positive integer"}

    cached = False
    sources: list[MinimalSource]
    if cache:
        hit = lookup_query(query, k, index_dir, semantic, hybrid)
        if hit is not None:
            sources = hit
            cached = True
            return 200, _search_body(
                query, k, semantic, hybrid, cache, cached, sources
            )
    try:
        searcher = get_cached_searcher(index_dir)
        result = searcher.search_one(
            unanswered_question=UnansweredQuestion(question=query),
            k=k,
            semantic=semantic,
            hybrid=hybrid,
        )
    except FileNotFoundError:
        logger.exception("Index files not found")
        return 503, {"error": "index not found; run index first"}
    except Exception:
        logger.exception("Search failed")
        return 500, {"error": "search failed"}
    sources = result.retrieved_sources
    if cache:
        store_query(query, k, index_dir, semantic, hybrid, sources)
    return 200, _search_body(
        query, k, semantic, hybrid, cache, cached, sources
    )


def handle_answer(
    index_dir: str,
    query: str,
    k: int,
    semantic: bool,
    hybrid: bool,
    cache: bool,
) -> tuple[int, dict[str, Any]]:
    """
    Search then generate an answer with Qwen via Ollama.

    Args:
        index_dir: BM25 index directory.
        query: Question text.
        k: Number of sources.
        semantic: MiniLM-only ranking.
        hybrid: RRF of BM25 and MiniLM.
        cache: Use the on-disk query cache for retrieval.

    Returns:
        HTTP status and JSON body.

    """
    if not ollama_available():
        return 503, {
            "error": "Ollama is not running; start it before /answer",
        }
    status, body = handle_search(
        index_dir, query, k, semantic, hybrid, cache
    )
    if status != 200:
        return status, body
    sources = [
        MinimalSource.model_validate(item)
        for item in body["retrieved_sources"]
    ]
    result = MinimalSearchResults(
        question_id="",
        question=query,
        retrieved_sources=sources,
    )
    try:
        answer = answer_from_search_result(result)
    except Exception:
        logger.exception("Answer failed")
        return 500, {"error": "answer failed"}
    return 200, {
        "question": answer.question,
        "k": k,
        "retrieved_sources": [
            src.model_dump() for src in answer.retrieved_sources
        ],
        "answer": answer.answer,
    }


def run_server(host: str, port: int, index_dir: str) -> None:
    """
    Serve /health, /search, and /answer until interrupted.

    Args:
        host: Bind address (default loopback).
        port: TCP port.
        index_dir: BM25 index directory.

    Returns:
        None.

    """
    server = ThreadingHTTPServer(
        (host, port),
        _make_handler(index_dir),
    )
    bound_host, bound_port = server.server_address[:2]
    logger.info("RAG API listening on http://%s:%s", bound_host, bound_port)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def _search_body(
    query: str,
    k: int,
    semantic: bool,
    hybrid: bool,
    cache: bool,
    cached: bool,
    sources: list[MinimalSource],
) -> dict[str, Any]:
    """
    Shape a /search JSON body.

    Args:
        query: Question text.
        k: Number of sources.
        semantic: MiniLM-only flag.
        hybrid: RRF flag.
        cache: Whether the client asked for the cache.
        cached: Whether this response came from the cache.
        sources: Retrieved source rows.

    Returns:
        JSON-serialisable dict.

    """
    return {
        "question": query,
        "k": k,
        "semantic": semantic,
        "hybrid": hybrid,
        "cache": cache,
        "cached": cached,
        "retrieved_sources": [src.model_dump() for src in sources],
    }


def _as_bool(value: Any, default: bool = False) -> bool:
    """
    Coerce a query/JSON value to bool.

    Args:
        value: Raw value (bool, str, or None).
        default: Fallback when value is None.

    Returns:
        Boolean.

    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _as_int(value: Any, default: int) -> int | None:
    """
    Coerce a query/JSON value to int.

    Args:
        value: Raw value.
        default: Fallback when value is None or "".

    Returns:
        Integer, or None if the value is not a number.

    """
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_args(
    data: dict[str, Any],
) -> tuple[str, int, bool, bool, bool] | str:
    """
    Read query / k / flags from a request dict.

    Args:
        data: Combined query-string or JSON fields.

    Returns:
        Parsed tuple, or an error string.

    """
    query = str(data.get("query") or data.get("q") or "").strip()
    k = _as_int(data.get("k"), 5)
    if k is None:
        return "k must be an integer"
    semantic = _as_bool(data.get("semantic"))
    hybrid = _as_bool(data.get("hybrid"))
    cache = _as_bool(data.get("cache"))
    return query, k, semantic, hybrid, cache


def _first(
    query: dict[str, list[str]],
    key: str,
) -> str | None:
    """
    Return the first query-string value for key.

    Args:
        query: parse_qs mapping.
        key: Parameter name.

    Returns:
        First value, or None.

    """
    values = query.get(key)
    if not values:
        return None
    return values[0]


def _make_handler(index_dir: str) -> type[BaseHTTPRequestHandler]:
    """
    Build a request handler closed over index_dir.

    Args:
        index_dir: BM25 index directory.

    Returns:
        BaseHTTPRequestHandler subclass.

    """

    class RagHandler(BaseHTTPRequestHandler):
        """Route GET/POST to health, search, and answer."""

        def log_message(self, fmt: str, *args: Any) -> None:
            """
            Send access logs through the project logger.

            Args:
                fmt: Stdlib log format string.
                *args: Values for fmt.

            """
            logger.info("%s - %s", self.address_string(), fmt % args)

        def do_GET(self) -> None:
            """Handle GET / /health /search /answer."""
            parsed = urlparse(self.path)
            route = parsed.path.rstrip("/") or "/"
            qs = parse_qs(parsed.query)
            data = {
                "query": _first(qs, "query") or _first(qs, "q"),
                "k": _first(qs, "k"),
                "semantic": _first(qs, "semantic"),
                "hybrid": _first(qs, "hybrid"),
                "cache": _first(qs, "cache"),
            }
            if route == "/":
                self._send(
                    200,
                    {
                        "name": "rage-against-the-machine",
                        "endpoints": [
                            "GET /health",
                            "GET|POST /search",
                            "GET|POST /answer",
                        ],
                    },
                )
                return
            if route == "/health":
                self._send(*health_payload(index_dir))
                return
            if route == "/search":
                self._dispatch_search(data)
                return
            if route == "/answer":
                self._dispatch_answer(data)
                return
            self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            """Handle POST /search and /answer."""
            parsed = urlparse(self.path)
            route = parsed.path.rstrip("/") or "/"
            body = self._read_json()
            if body is None:
                self._send(400, {"error": "invalid JSON body"})
                return
            if route == "/search":
                self._dispatch_search(body)
                return
            if route == "/answer":
                self._dispatch_answer(body)
                return
            self._send(404, {"error": "not found"})

        def _dispatch_search(self, data: dict[str, Any]) -> None:
            parsed_args = _parse_args(data)
            if isinstance(parsed_args, str):
                self._send(400, {"error": parsed_args})
                return
            query, k, semantic, hybrid, cache = parsed_args
            self._send(
                *handle_search(
                    index_dir, query, k, semantic, hybrid, cache
                )
            )

        def _dispatch_answer(self, data: dict[str, Any]) -> None:
            parsed_args = _parse_args(data)
            if isinstance(parsed_args, str):
                self._send(400, {"error": parsed_args})
                return
            query, k, semantic, hybrid, cache = parsed_args
            self._send(
                *handle_answer(
                    index_dir, query, k, semantic, hybrid, cache
                )
            )

        def _read_json(self) -> dict[str, Any] | None:
            length_raw = self.headers.get("Content-Length", "0")
            try:
                length = int(length_raw)
            except ValueError:
                return None
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            try:
                data = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                logger.exception("Invalid JSON body")
                return None
            if not isinstance(data, dict):
                return None
            return data

        def _send(self, status: int, body: dict[str, Any]) -> None:
            payload = json.dumps(body, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    return RagHandler
