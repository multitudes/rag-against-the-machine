import bm25s
import Stemmer
import json
import os
from typing import List, Dict, Any
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Searcher:
    """
    A class to handle loading a BM25 index and perform searches
    """
    def __init__(self, index_dir: str = "bm25s_indices"):
        """
        """
        self.retriever = bm25s.BM25.load(index_dir, mmap=True, load_corpus=True)
        self.stemmer = Stemmer.Stemmer("english")
        self.metadata = []
        self.corpus = []

        if not os.path.exists(index_dir):
            raise FileNotFoundError("BM25 files missing")

        # 3. Load the corresponding metadata
        metadata_path = os.path.join(index_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}. Please run the 'index' command first.")
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        logger.info(f"Loaded metadata for {len(self.metadata)} chunks.")

    def search(self, query: str, k: int = 5) -> List[Dict[str, any]]:
        """
        """
        logger.info(f"Retrieving top-{k} results for query: '{query}'")
        # --- Running Test Query ---
        logger.info("--- Running Test Query ---")
        query = "What command is used to start the vLLM OpenAI-compatible server?"

        query_tokens = bm25s.tokenize(query, stemmer=self.stemmer)

        # Get top-k results as a tuple of (doc ids, scores).
        results, scores = self.retriever.retrieve(query_tokens, k=5)

        print(f"\nTop {len(results[0])} results for query: '{query}'\n")
        for i, doc_index in enumerate(results[0]):
            score = scores[0][i]

            # Use the doc_index to look up the original text and metadata
            doc_text = self.corpus[doc_index]
            doc_metadata = self.metadata[doc_index]

            print(f"Rank {i+1} (Score: {score:.2f})")
            print(
                f"Source: {doc_metadata['file_path']} (chars {doc_metadata['first_character_index']}-{doc_metadata['last_character_index']})")
            print(f"Text: {doc_text[:200]}...")  # Print a snippet of the text
            print("-" * 20)
