"""
Index and query caches (bonus 4).

An in-process Searcher is reused while the index fingerprint is
unchanged. Query hits are stored in query_cache.json so a repeated
``search --cache`` can skip loading BM25 entirely.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from core.config import QUERY_CACHE_FILENAME
from core.schemas import MinimalSource
from retrieval.search import Searcher

logger = logging.getLogger(__name__)

_INDEX_FINGERPRINT_FILES = (
    "metadata.json",
    "params.index.json",
    "embeddings.npy",
    "files.json",
)

_searchers: dict[str, tuple[str, Searcher]] = {}


def query_cache_path(index_dir: str) -> Path:
    """
    Return the path of the on-disk query cache.

    Args:
        index_dir: Index directory (same as BM25).

    Returns:
        Path to query_cache.json.

    """
    return Path(index_dir) / str(QUERY_CACHE_FILENAME)


def index_fingerprint(index_dir: str) -> str:
    """
    Build a fingerprint from the index files on disk.

    Args:
        index_dir: Directory that holds the BM25 index.

    Returns:
        Stable string that changes when the index is rewritten.

    """
    parts: list[str] = []
    root = Path(index_dir)
    for name in _INDEX_FINGERPRINT_FILES:
        path = root / name
        if not path.exists():
            continue
        try:
            st = path.stat()
        except OSError:
            logger.exception("Could not stat %s", path)
            continue
        parts.append(
            f"{name}:{int(st.st_mtime_ns)}:{int(st.st_size)}"
        )
    return "|".join(parts)


def get_cached_searcher(index_dir: str) -> Searcher:
    """
    Return a Searcher, reusing one already loaded for this index.

    Args:
        index_dir: Directory that holds the BM25 index.

    Returns:
        A Searcher whose fingerprint still matches the files on disk.

    """
    fingerprint = index_fingerprint(index_dir)
    cached = _searchers.get(index_dir)
    if cached is not None and cached[0] == fingerprint:
        logger.debug("Reusing cached Searcher for %s", index_dir)
        return cached[1]
    searcher = Searcher(index_dir=index_dir)
    _searchers[index_dir] = (fingerprint, searcher)
    logger.debug("Cached Searcher for %s", index_dir)
    return searcher


def clear_searcher_cache() -> None:
    """
    Drop in-process Searcher instances.

    Returns:
        None.

    """
    _searchers.clear()


def clear_query_cache(index_dir: str) -> None:
    """
    Delete query_cache.json if it exists.

    Args:
        index_dir: Directory that holds the BM25 index.

    Returns:
        None.

    """
    path = query_cache_path(index_dir)
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.exception("Could not remove query cache at %s", path)
        return
    logger.info("Cleared query cache at %s", path)


def clear_index_caches(index_dir: str) -> None:
    """
    Drop both the in-process Searcher and the on-disk query cache.

    Args:
        index_dir: Directory that holds the BM25 index.

    Returns:
        None.

    """
    clear_searcher_cache()
    clear_query_cache(index_dir)


def lookup_query(
    query: str,
    k: int,
    index_dir: str,
    semantic: bool,
    hybrid: bool,
) -> list[MinimalSource] | None:
    """
    Return cached sources for this query, or None on a miss.

    Args:
        query: Search query string.
        k: Number of results requested.
        index_dir: Index directory.
        semantic: Whether the original search used MiniLM only.
        hybrid: Whether the original search used RRF.

    Returns:
        Cached MinimalSource list, or None.

    """
    fingerprint = index_fingerprint(index_dir)
    key = _entry_key(query, k, semantic, hybrid, fingerprint)
    entries = _load_entries(index_dir)
    raw = entries.get(key)
    if not isinstance(raw, dict):
        return None
    sources_raw = raw.get("sources")
    if not isinstance(sources_raw, list):
        return None
    try:
        sources = [
            MinimalSource.model_validate(item) for item in sources_raw
        ]
    except (TypeError, ValueError):
        logger.exception("Invalid cached sources for query %s", query)
        return None
    logger.info("Query cache hit for %s", query)
    return sources


def store_query(
    query: str,
    k: int,
    index_dir: str,
    semantic: bool,
    hybrid: bool,
    sources: list[MinimalSource],
) -> None:
    """
    Persist sources for a query under the current index fingerprint.

    Args:
        query: Search query string.
        k: Number of results requested.
        index_dir: Index directory.
        semantic: Whether this search used MiniLM only.
        hybrid: Whether this search used RRF.
        sources: Retrieved sources to store.

    Returns:
        None.

    """
    fingerprint = index_fingerprint(index_dir)
    key = _entry_key(query, k, semantic, hybrid, fingerprint)
    entries = _load_entries(index_dir)
    entries[key] = {
        "query": query,
        "k": k,
        "semantic": semantic,
        "hybrid": hybrid,
        "fp": fingerprint,
        "sources": [source.model_dump() for source in sources],
    }
    path = query_cache_path(index_dir)
    try:
        Path(index_dir).mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump({"entries": entries}, f, indent=4)
    except OSError:
        logger.exception("Could not write query cache at %s", path)
        return
    logger.debug("Stored query cache entry for %s", query)


def _entry_key(
    query: str,
    k: int,
    semantic: bool,
    hybrid: bool,
    fingerprint: str,
) -> str:
    """
    Hash the cache coordinates into a stable key.

    Args:
        query: Search query string.
        k: Number of results requested.
        semantic: MiniLM-only flag.
        hybrid: RRF flag.
        fingerprint: Current index fingerprint.

    Returns:
        Hex SHA-256 digest.

    """
    payload = json.dumps(
        {
            "query": query,
            "k": k,
            "semantic": semantic,
            "hybrid": hybrid,
            "fp": fingerprint,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_entries(index_dir: str) -> dict[str, Any]:
    """
    Load the entries map from query_cache.json.

    Args:
        index_dir: Index directory.

    Returns:
        Entries dict, or empty if the file is missing or invalid.

    """
    path = query_cache_path(index_dir)
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        logger.exception("Could not read query cache at %s", path)
        return {}
    if not isinstance(data, dict):
        return {}
    entries = data.get("entries", {})
    if not isinstance(entries, dict):
        return {}
    return entries
