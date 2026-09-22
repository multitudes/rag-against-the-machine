import json
import logging
from pathlib import Path

import bm25s
import Stemmer

from core.schemas import ChunkSource

logger = logging.getLogger(__name__)


DEFAULT_INDEX_DIR = "data/processed"


def create_bm25_index(
    all_chunks: list[ChunkSource],
    index_dir: str = DEFAULT_INDEX_DIR,
) -> None:
    """
    Creates and saves a BM25 index from a list of text chunks.

    Args:
        all_chunks: A list of chunk objects from chonkie.
        index_dir: The path to save the serialized index files.

    """
    if not all_chunks:
        logger.warning("No chunks provided to create BM25 index. Skipping.")
        return
    # Ensure the output directory exists
    Path(index_dir).mkdir(parents=True, exist_ok=True)

    # Extract the text from each chunk object
    corpus = [chunk.text for chunk in all_chunks]
    metadata = [chunk.source.model_dump() for chunk in all_chunks]
    # Save the metadata to a JSON file
    metadata_path = Path(index_dir) / "metadata.json"
    with Path(metadata_path).open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    logger.info(
        "Metadata for %d chunks saved to %s", len(metadata), metadata_path
    )

    logger.info("Creating BM25 index from %d total chunks...", len(corpus))

    # optional: create a stemmer
    stemmer = Stemmer.Stemmer("english")

    # Tokenize the corpus and only keep the ids (faster and saves memory)
    corpus_tokens = bm25s.tokenize(
        corpus,
        stopwords="en",
        stemmer=stemmer,
        show_progress=True,
    )
    retriever = bm25s.BM25()
    logger.info("Indexing documents with BM25...")
    retriever.index(corpus_tokens)
    tokenizer = bm25s.tokenization.Tokenizer(stemmer=stemmer)
    retriever.save(index_dir, corpus=corpus)
    tokenizer.save_vocab(index_dir)
    tokenizer.save_stopwords(index_dir)
    logger.info("Saving BM25 index to %s...", index_dir)

    # get memory usage
    mem_use = bm25s.utils.benchmark.get_max_memory_usage()
    logger.info("Peak memory usage: %.2f GB", mem_use)
    logger.info("BM25 index saved successfully.")
