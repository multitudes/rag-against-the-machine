"""
Semantic embedding index (bonus 1) and RRF fusion (bonus 2).

Stores a MiniLM vector per chunk next to the BM25 index. Used when the
CLI ``--semantic`` or ``--hybrid`` flag is set.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import Any

import numpy as np
from numpy.typing import NDArray

from core.config import EMBEDDING_MODEL, EMBEDDINGS_FILENAME, RRF_K
from core.schemas import ChunkSource

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_model() -> Any:
    """
    Load MiniLM once on CPU.

    Returns:
        A sentence-transformers SentenceTransformer instance.

    """

    logger.info("Loading embedding model %s on cpu", EMBEDDING_MODEL)
    return SentenceTransformer(EMBEDDING_MODEL, device="cpu")


def encode_texts(texts: list[str]) -> NDArray[np.float32]:
    """
    Embed a list of strings with MiniLM (L2-normalised).

    Args:
        texts: Chunk or query strings.

    Returns:
        Array of shape (len(texts), 384).

    """
    model = _load_model()
    vectors = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=len(texts) > 1,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(vectors, dtype=np.float32)


def embeddings_path(index_dir: str) -> Path:
    """
    Return the path of the saved embedding matrix.

    Args:
        index_dir: Index directory (same as BM25).

    Returns:
        Path to embeddings.npy.

    """
    return Path(index_dir).joinpath(EMBEDDINGS_FILENAME)


def create_semantic_index(
    texts: list[str],
    index_dir: str,
) -> None:
    """
    Embed all chunk texts and save embeddings.npy under index_dir.

    Row i of the matrix corresponds to metadata.json / BM25 corpus row i.

    Args:
        texts: Chunk texts in corpus order.
        index_dir: Directory that already holds the BM25 index.

    Returns:
        None.

    """
    if not texts:
        logger.warning("No texts provided for semantic index. Skipping.")
        return
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    logger.info("Embedding %d chunks with %s …", len(texts), EMBEDDING_MODEL)
    matrix = encode_texts(texts)
    path = embeddings_path(index_dir)
    np.save(path, matrix)
    logger.info(
        "Semantic index saved to %s (shape %s)", path, matrix.shape,
    )


def merge_semantic_index(
    old_chunks: list[ChunkSource],
    new_chunks: list[ChunkSource],
    unchanged_files: set[str],
    index_dir: str,
) -> None:
    """
    Reuse MiniLM rows for unchanged files; encode only new chunks.

    Args:
        old_chunks: Chunks from the previous index (row-aligned).
        new_chunks: Merged chunks about to be searched.
        unchanged_files: Paths whose old vectors can be copied.
        index_dir: Directory containing embeddings.npy.

    Returns:
        None.

    """
    path = embeddings_path(index_dir)
    if not path.exists():
        create_semantic_index(
            [chunk.text for chunk in new_chunks],
            index_dir,
        )
        return
    try:
        old_matrix = np.load(path)
    except OSError:
        logger.exception("Could not load embeddings at %s", path)
        create_semantic_index(
            [chunk.text for chunk in new_chunks],
            index_dir,
        )
        return
    if len(old_matrix) != len(old_chunks):
        logger.info(
            "Embedding rows (%d) != old chunks (%d). Re-encoding.",
            len(old_matrix),
            len(old_chunks),
        )
        create_semantic_index(
            [chunk.text for chunk in new_chunks],
            index_dir,
        )
        return

    old_rows_by_file: dict[str, list[int]] = {}
    for i, chunk in enumerate(old_chunks):
        old_rows_by_file.setdefault(
            chunk.source.file_path,
            [],
        ).append(i)
    cursor: dict[str, int] = {}
    rows: list[NDArray[np.float32] | None] = []
    to_encode: list[str] = []
    encode_at: list[int] = []
    try:
        for chunk in new_chunks:
            file_path = chunk.source.file_path
            if file_path in unchanged_files:
                idx_in_file = cursor.get(file_path, 0)
                cursor[file_path] = idx_in_file + 1
                old_i = old_rows_by_file[file_path][idx_in_file]
                rows.append(old_matrix[old_i])
            else:
                encode_at.append(len(rows))
                rows.append(None)
                to_encode.append(chunk.text)
    except (KeyError, IndexError):
        logger.exception(
            "Could not align embeddings with chunks. Re-encoding.",
        )
        create_semantic_index(
            [chunk.text for chunk in new_chunks],
            index_dir,
        )
        return

    if to_encode:
        encoded = encode_texts(to_encode)
        for slot, vec in zip(encode_at, encoded, strict=True):
            rows[slot] = vec
    filled: list[NDArray[np.float32]] = []
    for row in rows:
        if row is None:
            logger.error("Unfilled embedding row. Re-encoding.")
            create_semantic_index(
                [chunk.text for chunk in new_chunks],
                index_dir,
            )
            return
        filled.append(row)
    if not filled:
        logger.warning("No embedding rows to save. Skipping.")
        return
    matrix = np.stack(filled).astype(np.float32)
    np.save(path, matrix)
    logger.info(
        "Semantic index merged at %s (shape %s, encoded %d)",
        path,
        matrix.shape,
        len(to_encode),
    )


def top_k_indices(
    query_vec: NDArray[np.float32],
    matrix: NDArray[np.float32],
    k: int,
) -> NDArray[np.intp]:
    """
    Return indices of the k vectors closest to query_vec (cosine).

    Args:
        query_vec: Query embedding, shape (dim,) or (1, dim).
        matrix: Chunk embeddings, shape (n_chunks, dim).
        k: Number of hits to return.

    Returns:
        Integer indices of the top-k rows, best first.

    """
    if query_vec.ndim == 2:
        query_vec = query_vec[0]
    q_norm = float(np.linalg.norm(query_vec))
    if q_norm == 0.0:
        q_norm = 1.0
    query = query_vec / q_norm
    row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    row_norms = np.where(row_norms == 0.0, 1.0, row_norms)
    scores = (matrix / row_norms) @ query
    k_eff = min(k, int(scores.shape[0]))
    if k_eff <= 0:
        return np.array([], dtype=np.intp)
    rough = np.argpartition(-scores, kth=k_eff - 1)[:k_eff]
    return rough[np.argsort(-scores[rough])]


def search_semantic_ids(
    query: str,
    index_dir: str,
    k: int,
) -> list[int]:
    """
    Rank chunk ids by cosine similarity to the query embedding.

    Args:
        query: Question text.
        index_dir: Directory containing embeddings.npy.
        k: Number of results.

    Returns:
        Chunk ids (rows in metadata.json), best first.

    """
    path = embeddings_path(index_dir)
    try:
        matrix = np.load(path)
    except FileNotFoundError:
        logger.exception(
            "Semantic index not found at %s. "
            "Re-run index with --semantic.",
            path,
        )
        return []
    query_vec = encode_texts([query])
    ids = top_k_indices(query_vec, matrix, k)
    return [int(i) for i in ids]


def rrf_fuse(
    ranked_lists: list[list[int]],
    k: int,
    rrf_k: int = RRF_K,
) -> list[int]:
    """
    Merge ranked id lists with Reciprocal Rank Fusion.

    Args:
        ranked_lists: Each list is chunk ids best-first (BM25, MiniLM, …).
        k: Number of fused ids to return.
        rrf_k: Smoothing constant in 1 / (rrf_k + rank).

    Returns:
        Unique chunk ids ordered by fused score, length at most k.

    """
    scores: dict[int, float] = {}
    for ranking in ranked_lists:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (
                rrf_k + rank
            )
    ordered = sorted(
        scores,
        key=lambda chunk_id: scores[chunk_id],
        reverse=True,
    )
    return ordered[:k]
