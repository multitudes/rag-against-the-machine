import bm25s
import Stemmer
import logging
from typing import List
import os


logger = logging.getLogger(__name__)


def create_bm25_index(all_chunks: List, index_dir: str):
    """
    Creates and saves a BM25 index from a list of text chunks.

    Args:
        all_chunks: A list of chunk objects from chonkie.
        index_path: The path to save the serialized index file.
    """
    if not all_chunks:
        logger.warning("No chunks provided to create BM25 index. Skipping.")
        return
    # Ensure the output directory exists
    output_dir = os.path.dirname(index_dir)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Extract the text from each chunk object
    corpus = [chunk.text for chunk in all_chunks]
    logger.info(f"Creating BM25 index from {len(corpus)} total chunks...")
    
    # optional: create a stemmer
    stemmer = Stemmer.Stemmer("english")
    
    # Tokenize the corpus and only keep the ids (faster and saves memory)
    corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer, show_progress=True)
    retriever = bm25s.BM25()
    logger.info("Indexing documents with BM25...")
    retriever.index(corpus_tokens)
    tokenizer = bm25s.tokenization.Tokenizer(stemmer=stemmer)
    retriever.save(index_dir)
    tokenizer.save_vocab(index_dir)
    tokenizer.save_stopwords(index_dir)
    logger.info(f"Saving BM25 index to {index_dir}...")
    
    # get memory usage
    mem_use = bm25s.utils.benchmark.get_max_memory_usage()
    logger.info(f"Peak memory usage: {mem_use:.2f} GB")

    
    logger.info("BM25 index saved successfully.")
