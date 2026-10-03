"""BM25 index creation and persistence."""

import json
import logging
from pathlib import Path

import bm25s
import Stemmer

from core.schemas import ChunkSource, MinimalSource

logger = logging.getLogger(__name__)


DEFAULT_INDEX_DIR = "data/processed"


def create_bm25_index(
    all_chunks: list[ChunkSource],
    index_dir: str = DEFAULT_INDEX_DIR,
) -> None:
    """
    Create and save a BM25 index from a list of text chunks.

    Args:
        all_chunks: Chunk objects produced by the chunking pipeline.
        index_dir: Directory to save the serialized index files.

    Returns:
        None.

    """
    if not all_chunks:
        logger.warning("No chunks provided to create BM25 index. Skipping.")
        return
    # Ensure the output directory exists
    Path(index_dir).mkdir(parents=True, exist_ok=True)

    # Extract the text from each chunk object
    corpus = [chunk.text for chunk in all_chunks]
    metadata = [chunk.source.model_dump() for chunk in all_chunks]

    # --- our code writes metadata.json ---
    # Maps each corpus position → MinimalSource (file_path + char offsets).
    # Needed at search time to turn BM25 hits into moulinette-compatible
    # source locations (bm25s only stores the chunk text, not the path).
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

    # --- bm25s.BM25.save() writes ---
    #   data.csc.index.npy      BM25 scores (CSC sparse matrix data)
    #   indices.csc.index.npy   row indices of the CSC matrix
    #   indptr.csc.index.npy    column pointers of the CSC matrix
    #   vocab.index.json        term → id map used by the index
    #   params.index.json       BM25 hyperparameters (k1, b, …)
    #   corpus.jsonl            raw chunk texts (one JSON object per line)
    #   corpus.mmindex.json     byte offsets into corpus.jsonl for mmap
    retriever.save(index_dir, corpus=corpus)

    # --- Tokenizer.save_vocab() writes vocab.tokenizer.json ---
    # Stemmed/token vocabulary of the Tokenizer class (distinct from
    # vocab.index.json above, which belongs to the BM25 index itself).
    tokenizer.save_vocab(index_dir)

    # --- Tokenizer.save_stopwords() writes stopwords.tokenizer.json ---
    # English stopwords list used during tokenization.
    tokenizer.save_stopwords(index_dir)

    logger.info("Saving BM25 index to %s...", index_dir)

    # get memory usage
    mem_use = bm25s.utils.benchmark.get_max_memory_usage()
    logger.info("Peak memory usage: %.2f GB", mem_use)
    logger.info("BM25 index saved successfully.")


def load_chunks_from_index(index_dir: str) -> list[ChunkSource]:
    """
    Rebuild ChunkSource rows from metadata.json and corpus.jsonl.

    Args:
        index_dir: Directory that holds a previous BM25 index.

    Returns:
        Chunks aligned with the saved corpus order.

    Raises:
        FileNotFoundError: If metadata or corpus is missing.
        ValueError: If the two files disagree on length.
        KeyError: If a metadata row is missing required keys.

    """
    metadata_path = Path(index_dir) / "metadata.json"
    corpus_path = Path(index_dir) / "corpus.jsonl"
    with metadata_path.open(encoding="utf-8") as f:
        metadata = json.load(f)
    texts: list[str] = []
    with corpus_path.open(encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            doc = json.loads(raw)
            if isinstance(doc, dict):
                texts.append(str(doc.get("text", "")))
            else:
                texts.append(str(doc))
    if len(texts) != len(metadata):
        msg = (
            f"corpus.jsonl has {len(texts)} rows but "
            f"metadata.json has {len(metadata)}"
        )
        raise ValueError(msg)
    chunks: list[ChunkSource] = []
    for text, src in zip(texts, metadata, strict=True):
        chunks.append(
            ChunkSource(
                text=text,
                source=MinimalSource(
                    file_path=src["file_path"],
                    first_character_index=src["first_character_index"],
                    last_character_index=src["last_character_index"],
                ),
            )
        )
    return chunks
