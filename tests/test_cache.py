"""Tests for index / query caching (bonus 4)."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from core.schemas import ChunkSource, MinimalSource
from ingestion.indexing import create_bm25_index
from retrieval.cache import (
    clear_index_caches,
    clear_searcher_cache,
    get_cached_searcher,
    index_fingerprint,
    lookup_query,
    query_cache_path,
    store_query,
)


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


def test_store_then_lookup_same_query(tmp_path: Path) -> None:
    """A stored query is returned with the same sources."""
    index_dir = _tiny_index(tmp_path)
    sources = [
        MinimalSource(
            file_path="a.py",
            first_character_index=0,
            last_character_index=34,
        ),
    ]
    store_query("cats", 1, str(index_dir), False, False, sources)
    hit = lookup_query("cats", 1, str(index_dir), False, False)
    assert hit is not None
    assert hit[0].file_path == "a.py"
    assert query_cache_path(str(index_dir)).exists()


def test_lookup_misses_if_stored_fp_tampered(
    tmp_path: Path,
) -> None:
    """A stored hit whose body fp no longer matches is a miss."""
    index_dir = _tiny_index(tmp_path)
    sources = [
        MinimalSource(
            file_path="a.py",
            first_character_index=0,
            last_character_index=34,
        ),
    ]
    store_query("cats", 1, str(index_dir), False, False, sources)
    path = query_cache_path(str(index_dir))
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    for entry in data["entries"].values():
        entry["fp"] = "tampered"
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f)
    assert lookup_query("cats", 1, str(index_dir), False, False) is None


def test_store_query_concurrent_keeps_both(tmp_path: Path) -> None:
    """Two threaded stores both survive (API uses ThreadingHTTPServer)."""
    index_dir = _tiny_index(tmp_path)
    src_a = [
        MinimalSource(
            file_path="a.py",
            first_character_index=0,
            last_character_index=34,
        ),
    ]
    src_b = [
        MinimalSource(
            file_path="b.py",
            first_character_index=0,
            last_character_index=35,
        ),
    ]

    def _store_cats() -> None:
        store_query("cats", 1, str(index_dir), False, False, src_a)

    def _store_dogs() -> None:
        store_query("dogs", 1, str(index_dir), False, False, src_b)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(_store_cats)
        second = pool.submit(_store_dogs)
        first.result()
        second.result()
    assert lookup_query("cats", 1, str(index_dir), False, False) is not None
    assert lookup_query("dogs", 1, str(index_dir), False, False) is not None


def test_lookup_misses_on_different_k(tmp_path: Path) -> None:
    """k is part of the cache key."""
    index_dir = _tiny_index(tmp_path)
    sources = [
        MinimalSource(
            file_path="a.py",
            first_character_index=0,
            last_character_index=34,
        ),
    ]
    store_query("cats", 1, str(index_dir), False, False, sources)
    assert lookup_query("cats", 2, str(index_dir), False, False) is None


def test_lookup_misses_after_fingerprint_change(
    tmp_path: Path,
) -> None:
    """Rewriting the index invalidates previous query entries."""
    index_dir = _tiny_index(tmp_path)
    sources = [
        MinimalSource(
            file_path="a.py",
            first_character_index=0,
            last_character_index=34,
        ),
    ]
    store_query("cats", 1, str(index_dir), False, False, sources)
    assert lookup_query("cats", 1, str(index_dir), False, False) is not None
    create_bm25_index(
        [
            ChunkSource(
                text="brand new corpus about whales singing",
                source=MinimalSource(
                    file_path="c.py",
                    first_character_index=0,
                    last_character_index=37,
                ),
            ),
        ],
        str(index_dir),
    )
    assert lookup_query("cats", 1, str(index_dir), False, False) is None


def test_clear_index_caches_removes_file(tmp_path: Path) -> None:
    """clear_index_caches deletes query_cache.json."""
    index_dir = _tiny_index(tmp_path)
    store_query(
        "cats",
        1,
        str(index_dir),
        False,
        False,
        [
            MinimalSource(
                file_path="a.py",
                first_character_index=0,
                last_character_index=34,
            ),
        ],
    )
    assert query_cache_path(str(index_dir)).exists()
    clear_index_caches(str(index_dir))
    assert not query_cache_path(str(index_dir)).exists()


def test_get_cached_searcher_reuses_instance(tmp_path: Path) -> None:
    """The same Searcher is returned while the fingerprint is stable."""
    index_dir = _tiny_index(tmp_path)
    clear_searcher_cache()
    first = get_cached_searcher(str(index_dir))
    second = get_cached_searcher(str(index_dir))
    assert first is second
    clear_searcher_cache()


def test_index_fingerprint_changes_on_rewrite(tmp_path: Path) -> None:
    """Rebuilding BM25 changes the fingerprint string."""
    index_dir = _tiny_index(tmp_path)
    before = index_fingerprint(str(index_dir))
    create_bm25_index(
        [
            ChunkSource(
                text="rewritten chunk about foxes jumping",
                source=MinimalSource(
                    file_path="d.py",
                    first_character_index=0,
                    last_character_index=35,
                ),
            ),
        ],
        str(index_dir),
    )
    after = index_fingerprint(str(index_dir))
    assert before != after
