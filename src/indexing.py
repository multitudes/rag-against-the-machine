import bm25s
import Stemmer
import logging
import json
from typing import List
from schemas import MinimalSource
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
    metadata = [chunk.source.model_dump() for chunk in all_chunks]
    # Save the metadata to a JSON file
    metadata_path = os.path.join(index_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Metadata for {len(metadata)} chunks saved to {metadata_path}")


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

    logger.info("query test")
    # --- Running Test Query ---
    logger.info("--- Running Test Query ---")
    query = "What command is used to start the vLLM OpenAI-compatible server?"
    query_tokens = bm25s.tokenize(query, stemmer=stemmer)

    # Get top-k results as a tuple of (doc ids, scores).
    results, scores = retriever.retrieve(query_tokens, k=5)

    print(f"\nTop {len(results[0])} results for query: '{query}'\n")
    for i, doc_index in enumerate(results[0]):
        score = scores[0][i]
        
        # Use the doc_index to look up the original text and metadata
        doc_text = corpus[doc_index]
        doc_metadata = metadata[doc_index]
        
        print(f"Rank {i+1} (Score: {score:.2f})")
        print(f"Source: {doc_metadata['file_path']} (chars {doc_metadata['first_character_index']}-{doc_metadata['last_character_index']})")
        print(f"Text: {doc_text[:200]}...") # Print a snippet of the text
        print("-" * 20)
    # query = "What command is used to start the vLLM OpenAI-compatible server?"
    # query_tokens = bm25s.tokenize(query, stemmer=stemmer)

    # # Get top-k results as a tuple of (doc ids, scores). Both are arrays of shape (n_queries, k).
    # # To return docs instead of IDs, set the `corpus=corpus` parameter.
    # results, scores = retriever.retrieve(query_tokens, k=2)

    # for i in range(results.shape[1]):
    #     doc, score = results[0, i], scores[0, i]
    #     print(f"Rank {i+1} (score: {score:.2f}): {doc}")

