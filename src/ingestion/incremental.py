"""
Incremental indexing (bonus 3).

Re-chunk only files whose size or mtime changed since the last index.
The BM25 matrix is still rebuilt from the merged chunk list (bm25s has
no document-update API). Unchanged files skip chonkie entirely.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tqdm import tqdm

from core.config import FILES_MANIFEST_FILENAME
from ingestion.chunking import chunk_content
from ingestion.indexing import load_chunks_from_index

# because ruff was complaining we import the class only to use as type hint
if TYPE_CHECKING:
    from core.schemas import ChunkSource

logger = logging.getLogger(__name__)


@dataclass
class ChunkBuildResult:
    """
    Outcome of an incremental chunk pass.

    Attributes:
        chunks: Merged chunks to write into the BM25 index.
        old_chunks: Chunks loaded from the previous index.
        unchanged_files: Paths whose chunks were reused as-is.
        nothing_changed: True if the index is already current.

    """

    chunks: list[ChunkSource]
    old_chunks: list[ChunkSource]
    unchanged_files: set[str]
    nothing_changed: bool


def manifest_path(index_dir: str) -> Path:
    """
    Return the path of the per-file fingerprint manifest.

    Args:
        index_dir: Index directory (same as BM25).

    Returns:
        Path to files.json.

    """
    return Path(index_dir) / str(FILES_MANIFEST_FILENAME)


def file_fingerprint(path: str) -> dict[str, int] | None:
    """
    Read size and mtime for one file.

    Args:
        path: File to stat.

    Returns:
        Dict with mtime_ns and size, or None if stat failed.

    """
    try:
        st = Path(path).stat()
    except OSError:
        logger.exception("Could not stat %s", path)
        return None
    return {"mtime_ns": int(st.st_mtime_ns), "size": int(st.st_size)}


def write_file_manifest(
    files: list[str],
    max_chunk_size: int,
    index_dir: str,
) -> None:
    """
    Persist fingerprints so the next --incremental run can diff.

    Args:
        files: Corpus paths that were indexed.
        max_chunk_size: Chunk size used for this index.
        index_dir: Directory that holds the BM25 index.

    Returns:
        None.

    """
    files_meta: dict[str, dict[str, int]] = {}
    for file_path in files:
        fingerprint = file_fingerprint(file_path)
        if fingerprint is not None:
            files_meta[file_path] = fingerprint
    payload: dict[str, Any] = {
        "max_chunk_size": max_chunk_size,
        "files": files_meta,
    }
    path = manifest_path(index_dir)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
    logger.info(
        "Wrote file manifest (%d files) to %s",
        len(files_meta),
        path,
    )


def _load_manifest(index_dir: str) -> dict[str, Any] | None:
    """
    Load files.json or return None if it is missing/invalid.

    Args:
        index_dir: Index directory.

    Returns:
        Parsed manifest dict, or None.

    """
    path = manifest_path(index_dir)
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        logger.exception("Could not read incremental manifest at %s", path)
        return None
    if not isinstance(data, dict):
        return None
    return data


def classify_files(
    current_files: list[str],
    manifest: dict[str, Any],
) -> tuple[list[str], list[str], list[str], list[str]]:
    """
    Split corpus paths into added, changed, deleted, unchanged.

    Args:
        current_files: Paths from get_all_files.
        manifest: Previously written files.json payload.

    Returns:
        Four lists: added, changed, deleted, unchanged.

    """
    fingerprints = manifest.get("files", {})
    if not isinstance(fingerprints, dict):
        fingerprints = {}
    old_files = set(fingerprints)
    current_set = set(current_files)
    added = [p for p in current_files if p not in old_files]
    deleted = sorted(old_files - current_set)
    changed: list[str] = []
    unchanged: list[str] = []
    for path in current_files:
        if path not in fingerprints:
            continue
        fingerprint = file_fingerprint(path)
        if fingerprint is None:
            changed.append(path)
            continue
        old = fingerprints[path]
        if not isinstance(old, dict):
            changed.append(path)
            continue
        try:
            same = (
                int(old.get("mtime_ns", -1)) == fingerprint["mtime_ns"]
                and int(old.get("size", -1)) == fingerprint["size"]
            )
        except (TypeError, ValueError):
            changed.append(path)
            continue
        if same:
            unchanged.append(path)
        else:
            changed.append(path)
    return added, changed, deleted, unchanged


def chunk_all_files(
    files: list[str],
    max_chunk_size: int,
) -> list[ChunkSource]:
    """
    Chunk every file (full index path).

    Args:
        files: Corpus paths.
        max_chunk_size: Maximum characters per chunk.

    Returns:
        All chunks in file order.

    """
    chunks: list[ChunkSource] = []
    for file_path in tqdm(files, desc="Chunking files"):
        chunks.extend(chunk_content(file_path, max_chunk_size))
    return chunks


def collect_chunks(
    files: list[str],
    max_chunk_size: int,
    index_dir: str,
    incremental: bool = False,
) -> ChunkBuildResult:
    """
    Chunk the corpus, incrementally when requested and possible.

    Args:
        files: Corpus paths from get_all_files.
        max_chunk_size: Maximum characters per chunk.
        index_dir: Directory of the previous index.
        incremental: If True, reuse unchanged files when a baseline
            exists.

    Returns:
        A ChunkBuildResult. ``nothing_changed`` is True when the
        index is already current. ``old_chunks`` is empty on a full
        reindex (flag off, or incremental fallback).

    """
    if incremental:
        result = collect_chunks_incremental(
            files,
            max_chunk_size,
            index_dir,
        )
        if result is not None:
            return result
    chunks = chunk_all_files(files, max_chunk_size)
    return ChunkBuildResult(
        chunks=chunks,
        old_chunks=[],
        unchanged_files=set(),
        nothing_changed=False,
    )


def collect_chunks_incremental(
    files: list[str],
    max_chunk_size: int,
    index_dir: str,
) -> ChunkBuildResult | None:
    """
    Re-chunk only added/changed files and merge with kept chunks.

    Args:
        files: Current corpus paths from get_all_files.
        max_chunk_size: Maximum characters per chunk.
        index_dir: Directory of the previous index.

    Returns:
        A ChunkBuildResult, or None to signal a full reindex.

    """
    metadata_path = Path(index_dir) / "metadata.json"
    corpus_path = Path(index_dir) / "corpus.jsonl"
    if (
        not metadata_path.exists()
        or not corpus_path.exists()
    ):
        logger.info(
            "No incremental baseline at %s. Full reindex.",
            index_dir,
        )
        return None
    manifest = _load_manifest(index_dir)
    if manifest is None:
        logger.info(
            "No incremental manifest at %s. Full reindex.",
            index_dir,
        )
        return None
    if manifest.get("max_chunk_size") != max_chunk_size:
        logger.info(
            "max_chunk_size changed (%s → %d). Full reindex.",
            manifest.get("max_chunk_size"),
            max_chunk_size,
        )
        return None
    try:
        old_chunks = load_chunks_from_index(index_dir)
    except (OSError, ValueError, KeyError, TypeError):
        logger.exception("Could not load previous chunks. Full reindex.")
        return None

    added, changed, deleted, unchanged = classify_files(files, manifest)
    logger.info(
        "Incremental: %d changed, %d added, %d deleted, %d unchanged",
        len(changed),
        len(added),
        len(deleted),
        len(unchanged),
    )
    unchanged_set = set(unchanged)
    dirty = added + changed
    if not dirty and not deleted:
        return ChunkBuildResult(
            chunks=old_chunks,
            old_chunks=old_chunks,
            unchanged_files=unchanged_set,
            nothing_changed=True,
        )

    kept = [
        chunk
        for chunk in old_chunks
        if chunk.source.file_path in unchanged_set
    ]
    new_parts: list[ChunkSource] = []
    for path in tqdm(dirty, desc="Chunking changed files"):
        new_parts.extend(chunk_content(path, max_chunk_size))

    by_file: dict[str, list[ChunkSource]] = defaultdict(list)
    for chunk in kept + new_parts:
        by_file[chunk.source.file_path].append(chunk)
    merged: list[ChunkSource] = []
    for path in files:
        merged.extend(by_file.get(path, []))
    return ChunkBuildResult(
        chunks=merged,
        old_chunks=old_chunks,
        unchanged_files=unchanged_set,
        nothing_changed=False,
    )
