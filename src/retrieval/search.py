import json
import logging
import os

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


class Searcher:
    """
    A class to handle loading a BM25 index and perform searches."""

    def __init__(self, index_dir: str = DEFAULT_INDEX_DIR) -> None:
        """
        Initialize the Searcher, loading the BM25 index and metadata.

        Args:
            index_dir: Directory where the BM25 index files are stored.

        """
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.WARNING)
        logger.debug("Using memory-mapped index (mmap) to reduce memory usage.")
        if not os.path.exists(index_dir):
            raise FileNotFoundError(
                f"Index directory '{index_dir}' not found. "
                "Please run the 'index' command first."
            )
        metadata_path = os.path.join(index_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Metadata file not found at {metadata_path}. "
                "Please run the 'index' command first."
            )
        self.retriever = bm25s.BM25.load(index_dir, mmap=True, load_corpus=True)
        self.stemmer = Stemmer.Stemmer("english")
        self.corpus = self.retriever.corpus
        logger.debug("BM25 index loaded with %d documents.", len(self.corpus))

        with open(metadata_path, encoding="utf-8") as f:
            self.metadata = json.load(f)
        logger.debug("Loaded metadata for %d chunks.", len(self.metadata))

    def search_one(
        self,
        unansweredQuestion: UnansweredQuestion,
        k: int = 5,
    ) -> MinimalSearchResults:
        """
        Perform a single search and return a MinimalSearchResults object.

        Args:
            unansweredQuestion: The question to search for.
            k: Number of top results to return.

        Returns:
            MinimalSearchResults with the top-k sources.

        """
        logger.debug(
            "Retrieving top-%d results for: '%s'",
            k,
            unansweredQuestion.question,
        )
        query_tokens = bm25s.tokenize(
            unansweredQuestion.question, stemmer=self.stemmer
        )

        results, scores = self.retriever.retrieve(query_tokens, k=k)

        retrieved_sources: list[MinimalSource] = []
        for i in range(results.shape[1]):
            logger.debug("Score %d: %s", i + 1, scores[0, i])
            meta_idx = results[0, i]["id"]
            meta = self.metadata[meta_idx]
            meta_text = results[0, i]["text"]
            logger.debug("Rank %d: %s", i + 1, meta_text[:40])

            min_src = MinimalSource(
                file_path=meta["file_path"],
                first_character_index=meta["first_character_index"],
                last_character_index=meta["last_character_index"],
            )
            retrieved_sources.append(min_src)

        logger.debug("Retrieved %d sources.", len(retrieved_sources))
        return MinimalSearchResults(
            question_id=unansweredQuestion.question_id,
            question=unansweredQuestion.question,
            retrieved_sources=retrieved_sources,
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
            result = self.search_one(unansweredQuestion=question, k=k)
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
            try:
                with open(source.file_path, encoding="utf-8") as f:
                    f.seek(source.first_character_index)
                    length = (
                        source.last_character_index
                        - source.first_character_index
                    )
                    content = f.read(length)
                    context_chunks.append(content)
            except Exception as e:
                logger.info("Error reading file %s: %s", source.file_path, e)
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
        try:
            with open(source.file_path, encoding="utf-8") as f:
                f.seek(source.first_character_index)
                content = f.read(
                    source.last_character_index - source.first_character_index,
                )
                context_chunks.append(content)
        except Exception as e:
            logger.info("Error reading file %s: %s", source.file_path, e)
    return context_chunks
