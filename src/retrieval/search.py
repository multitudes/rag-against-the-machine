import os
import json
import bm25s
import Stemmer
import logging
from typing import List
from core.schemas import MinimalSource, MinimalSearchResults
from core.schemas import StudentSearchResults, UnansweredQuestion

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
                f"Metadata file not found at {metadata_path}.\
                Please run the 'index' command first.")
        self.retriever = bm25s.BM25.load(
            index_dir, mmap=True, load_corpus=True)
        self.stemmer = Stemmer.Stemmer("english")
        self.corpus = self.retriever.corpus
        logger.info(
            f"BM25 index and corpus with {len(self.corpus)} documents loaded.")

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        logger.info(f"Loaded metadata for {len(self.metadata)} chunks.")

    def search_one(self,
                   unansweredQuestion: UnansweredQuestion,
                   k: int = 5
                   ) -> MinimalSearchResults:
        """
        Performs a search and returns a structured StudentSearchResults object.
        """
        logger.info(f"Retrieving top-{k} results for query: ")
        logger.info(f"'{unansweredQuestion.question}'")
        query_tokens = bm25s.tokenize(unansweredQuestion.question, 
                                      stemmer=self.stemmer)

        # Get top-k results as a tuple of (doc ids, scores).
        results, scores = self.retriever.retrieve(query_tokens, k=k)

        # The documents are returned as a numpy array of shape (n_queries, k)
        retrieved_sources = []
        for i in range(results.shape[1]):
            logger.info(f"score {i+1}: {scores[0, i]}")
            # logger.info(f"Rank {i+1}: {results[0, i]['text'][:20]}...")
            meta_idx = results[0, i]['id']
            meta = self.metadata[meta_idx]
            meta_text = results[0, i]['text']
            logger.info(f"Rank {i+1}: {meta_text[:40]}")

            min_src = MinimalSource(
                file_path=meta['file_path'],
                first_character_index=meta['first_character_index'],
                last_character_index=meta['last_character_index']
            )
            retrieved_sources.append(min_src)
        logger.info(f"min srcs are {len(retrieved_sources)}")
        return MinimalSearchResults(
            question_id=unansweredQuestion.question_id,
            retrieved_sources=retrieved_sources
        )

    def search_dataset(
            self,
            questions: UnansweredQuestion,
            k: int = 5
    ) -> StudentSearchResults:
        """
        Performs a search and returns a structured StudentSearchResults object.
        """
        logger.info("searching the dataset... ")
        search_results = []
        for question in questions:
            logger.info(
                f"Retrieving top-{k} results for question: "
                f"'{question.question}'")
            result = self.search_one(query=question.question, k=k)
            result.question_id = question.question_id
            search_results.append(result)
        logger.info(f"Found {len(search_results)} results")
        return StudentSearchResults(
            search_results=search_results,
            k=k
        )
    
    def retrieve_context(self, search_results: MinimalSearchResults) -> List[str]:
        """
        Reads the content of chunks from files based on search results.
        Used to create the context for a prompt
        """
        context_chunks = []
        for source in search_results.retrieved_sources:
            try:
                with open(source.file_path, 'r', encoding='utf-8') as f:
                    f.seek(source.first_character_index)
                    content = f.read(
                        source.last_character_index
                        - source.first_character_index)
                    context_chunks.append(content)
            except Exception as e:
                logger.error(f"Error reading file {source.file_path}: {e}")
        if not context_chunks:
            logger.error("Could not retrieve any content. Abort")
        return context_chunks
