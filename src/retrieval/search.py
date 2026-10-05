import json
import logging
from pathlib import Path

import bm25s
import Stemmer

from core.schemas import (
    MinimalSearchResults,
    MinimalSource,
    StudentSearchResults,
    UnansweredQuestion,
)

logger = logging.getLogger(__name__)

DEFAULT_INDEX_DIR = "data/processed"


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------

def _read_source_text(source: MinimalSource) -> str | None:
    """
    Read the text span described by a MinimalSource.

    Args:
        source: Source location (file path + character offsets).

    Returns:
        The extracted text, or None if the file could not be read.

    """
    try:
        with Path(source.file_path).open(encoding="utf-8") as f:
            f.seek(source.first_character_index)
            length = (
                source.last_character_index - source.first_character_index
            )
            return f.read(length)
    except Exception:
        logger.exception("Error reading file %s", source.file_path)
        return None


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

class Searcher:
    """A class to handle loading a BM25 index and perform searches."""

    def __init__(self, index_dir: str = DEFAULT_INDEX_DIR) -> None:
        """
        Initialize the Searcher, loading the BM25 index and metadata.

        Args:
            index_dir: Directory where the BM25 index files are stored.

        """
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        logger.debug("Using memory-mapped index (mmap) to reduce memory usage.")
        if not Path(index_dir).exists():
            msg = (
                f"Index directory '{index_dir}' not found. "
                "Please run the 'index' command first."
            )
            raise FileNotFoundError(msg)
        metadata_path = Path(index_dir) / "metadata.json"
        if not metadata_path.exists():
            msg = (
                f"Metadata file not found at {metadata_path}. "
                "Please run the 'index' command first."
            )
            raise FileNotFoundError(msg)

        self.index_dir = index_dir
        self.retriever = bm25s.BM25.load(index_dir, mmap=True, load_corpus=True)
        self.stemmer = Stemmer.Stemmer("english")
        self.corpus = self.retriever.corpus
        logger.debug("BM25 index loaded with %d documents.", len(self.corpus))

        with metadata_path.open(encoding="utf-8") as f:
            self.metadata = json.load(f)
        logger.debug("Loaded metadata for %d chunks.", len(self.metadata))

    def search_one(
        self,
        unanswered_question: UnansweredQuestion,
        k: int = 5,
        semantic: bool = False,
        hybrid: bool = False,
    ) -> MinimalSearchResults:
        """
        Perform a single search and return a MinimalSearchResults object.

        Args:
            unanswered_question: The question to search for.
            k: Number of top results to return.
            semantic: If True, rank with MiniLM cosine instead of BM25.
            hybrid: If True, fuse BM25 and MiniLM ranks (wins over semantic).

        Returns:
            MinimalSearchResults with the top-k sources.

        """
        logger.debug(
            "Retrieving top-%d results for: '%s' "
            "(semantic=%s hybrid=%s)",
            k,
            unanswered_question.question,
            semantic,
            hybrid,
        )
        if hybrid:
            return self._search_one_hybrid(unanswered_question, k)
        if semantic:
            return self._search_one_semantic(unanswered_question, k)
        return self._search_one_bm25(unanswered_question, k)

    def _sources_from_chunk_ids(
        self,
        chunk_ids: list[int],
    ) -> list[MinimalSource]:
        """
        Map corpus row ids to MinimalSource via metadata.json.

        Args:
            chunk_ids: Indices into self.metadata.

        Returns:
            List of MinimalSource for those rows.

        """
        retrieved_sources: list[MinimalSource] = []
        for chunk_id in chunk_ids:
            chunk = self.metadata[chunk_id]
            retrieved_sources.append(
                MinimalSource(
                    file_path=chunk["file_path"],
                    first_character_index=chunk["first_character_index"],
                    last_character_index=chunk["last_character_index"],
                ),
            )
        return retrieved_sources

    def _search_one_semantic(
        self,
        unanswered_question: UnansweredQuestion,
        k: int,
    ) -> MinimalSearchResults:
        """
        Rank with MiniLM embeddings stored next to the BM25 index.

        Args:
            unanswered_question: The question to search for.
            k: Number of top results to return.

        Returns:
            MinimalSearchResults with the top-k sources.

        """
        from retrieval.semantic import search_semantic_ids

        chunk_ids = search_semantic_ids(
            unanswered_question.question,
            self.index_dir,
            k,
        )
        logger.debug("Semantic retrieved %d sources.", len(chunk_ids))
        return MinimalSearchResults(
            question_id=unanswered_question.question_id,
            question=unanswered_question.question,
            retrieved_sources=self._sources_from_chunk_ids(chunk_ids),
        )

    def _bm25_chunk_ids(self, question: str, k: int) -> list[int]:
        """
        Rank chunk ids with BM25, best first.

        Args:
            question: Query text.
            k: Number of ids to return (capped at corpus size).

        Returns:
            Chunk ids aligned with metadata.json rows.

        """
        n_docs = len(self.metadata)
        k = min(k, n_docs)
        if k <= 0:
            return []
        query_tokens = bm25s.tokenize(question, stemmer=self.stemmer)
        results, scores = self.retriever.retrieve(query_tokens, k=k)
        chunk_ids: list[int] = []
        for i in range(results.shape[1]):
            logger.debug("Score %d: %s", i + 1, scores[0, i])
            chunk_ids.append(int(results[0, i]["id"]))
        return chunk_ids

    def _search_one_hybrid(
        self,
        unanswered_question: UnansweredQuestion,
        k: int,
    ) -> MinimalSearchResults:
        """
        Fuse BM25 and MiniLM ranks with Reciprocal Rank Fusion.

        Args:
            unanswered_question: The question to search for.
            k: Number of top results to return.

        Returns:
            MinimalSearchResults with the fused top-k sources.

        """
        from core.config import HYBRID_POOL
        from retrieval.semantic import rrf_fuse, search_semantic_ids

        n_docs = len(self.metadata)
        pool = min(max(k, HYBRID_POOL), n_docs)
        query = unanswered_question.question
        lexical_ids = self._bm25_chunk_ids(query, pool)
        semantic_ids = search_semantic_ids(query, self.index_dir, pool)
        chunk_ids = rrf_fuse([lexical_ids, semantic_ids], k=k)
        logger.debug(
            "Hybrid fused %d sources (pool=%d).",
            len(chunk_ids),
            pool,
        )
        return MinimalSearchResults(
            question_id=unanswered_question.question_id,
            question=unanswered_question.question,
            retrieved_sources=self._sources_from_chunk_ids(chunk_ids),
        )

    def _search_one_bm25(
        self,
        unanswered_question: UnansweredQuestion,
        k: int,
    ) -> MinimalSearchResults:
        """
        Rank with the BM25 index (mandatory lexical path).

        Args:
            unanswered_question: The question to search for.
            k: Number of top results to return.

        Returns:
            MinimalSearchResults with the top-k sources.

        """
        chunk_ids = self._bm25_chunk_ids(unanswered_question.question, k)
        logger.debug("Retrieved %d sources.", len(chunk_ids))
        return MinimalSearchResults(
            question_id=unanswered_question.question_id,
            question=unanswered_question.question,
            retrieved_sources=self._sources_from_chunk_ids(chunk_ids),
        )

    def search_dataset(
        self,
        questions: list[UnansweredQuestion],
        k: int = 5,
    ) -> StudentSearchResults:
        """
        Perform searches for a list of questions.

        Args:
            questions: List of UnansweredQuestion objects.
            k: Number of top results per question.

        Returns:
            StudentSearchResults containing all search results.

        """
        logger.debug("Searching the dataset...")
        search_results = []
        for question in questions:
            result = self.search_one(unanswered_question=question, k=k)
            search_results.append(result)
        logger.debug("Found results for %d questions.", len(search_results))
        return StudentSearchResults(
            search_results=search_results,
            k=k,
        )

    def retrieve_context(
        self,
        search_results: MinimalSearchResults,
    ) -> list[str]:
        """
        Read file content for each source in search results.

        Args:
            search_results: MinimalSearchResults with source locations.

        Returns:
            List of text chunks read from the source files.

        """
        context_chunks = []
        for source in search_results.retrieved_sources:
            content = _read_source_text(source)
            if content is not None:
                context_chunks.append(content)
        if not context_chunks:
            logger.error("Could not retrieve any context content.")
        return context_chunks


def retrieve_context_from_sources(
    search_result: MinimalSearchResults,
) -> list[str]:
    """
    Standalone helper to read file content for each source.

    Does not require a loaded BM25 index.

    Args:
        search_result: MinimalSearchResults with source locations.

    Returns:
        List of text chunks read from the source files.

    """
    context_chunks = []
    for source in search_result.retrieved_sources:
        content = _read_source_text(source)
        if content is not None:
            context_chunks.append(content)
    return context_chunks
