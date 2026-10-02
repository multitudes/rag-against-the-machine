"""
Semantic embedding index (bonus 1).

Stores a MiniLM vector per chunk next to the BM25 index. Used only when
the CLI ``--semantic`` flag is set.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import Any

import numpy as np
from numpy.typing import NDArray

from core.config import EMBEDDING_MODEL, EMBEDDINGS_FILENAME

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
        "Semantic index saved to %s (shape %s)", path, matrix.shape
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
