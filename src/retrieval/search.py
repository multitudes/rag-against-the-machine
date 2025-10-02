import os
import json
import bm25s
import Stemmer
import logging
from core.schemas import MinimalSource, MinimalSearchResults
from core.schemas import StudentSearchResults


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Searcher:
    """
    A class to handle loading a BM25 index and perform searches
    """

    def __init__(self, index_dir: str = "bm25s_indices"):
        """
        """
        logger.info("Using memory-mapped index (mmap) to reduce memory usage.")
        if not os.path.exists(index_dir):
            raise FileNotFoundError("BM25 files missing")
        metadata_path = os.path.join(index_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Metadata file not found at {metadata_path}. Please run the 'index' command first.")
        self.retriever = bm25s.BM25.load(
            index_dir, mmap=True, load_corpus=True)
        self.stemmer = Stemmer.Stemmer("english")
        self.corpus = self.retriever.corpus
        logger.info(
            f"BM25 index and corpus with {len(self.corpus)} documents loaded.")

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        logger.info(f"Loaded metadata for {len(self.metadata)} chunks.")

    def search(self, query: str, k: int = 5) -> StudentSearchResults:
        """
        Performs a search and returns a structured StudentSearchResults object.
        """
        logger.info(f"Retrieving top-{k} results for query: '{query}'")
        query_tokens = bm25s.tokenize(query, stemmer=self.stemmer)

        # Get top-k results as a tuple of (doc ids, scores).
        results, scores = self.retriever.retrieve(query_tokens, k=k)

        # The documents are returned as a numpy array of shape (n_queries, k)
        retrieved_sources = []
        for i in range(results.shape[1]):
            logger.info(f"score {i+1}: {scores[0, i]}")
            logger.info(f"Rank {i+1}: {results[0, i]}")
            meta_idx = results[0, i]['id']
            meta = self.metadata[meta_idx]

            min_src = MinimalSource(
                file_path=meta['file_path'],
                first_character_index=meta['first_character_index'],
                last_character_index=meta['last_character_index']
            )
            retrieved_sources.append(min_src)
        logger.info(f"min srcs are {len(retrieved_sources)}")
        min_search_res = MinimalSearchResults(
            question_id="",
            retrieved_sources=retrieved_sources
        )
        return StudentSearchResults(
            search_results=[min_search_res],
            k=k
        )
