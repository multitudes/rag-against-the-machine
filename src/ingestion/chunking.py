from chonkie import TextChef, MarkdownChef
from typing import List
import logging
from chonkie import RecursiveChunker
from chonkie import CodeChunker
from core.schemas import MinimalSource, ChunkSource


logger = logging.getLogger(__name__)


def get_docs_for_file(file_path):
    ext = file_path.lower().rsplit('.', 1)[-1]
    chef = TextChef()
    if ext == 'md':
        chef = MarkdownChef()
    return chef.process(file_path), ext


def chunk_content(
        file_path: str,
        chunk_size: int = 2048,
        overlap: int = 200
        ) -> List[ChunkSource]:
    """
    Chunk text content into smaller pieces using chonkie.

    Args:
        file_path: Path to the file to chunk.
        chunk_size: Size of each chunk.
        overlap: Overlap between chunks.

    Returns:
        List of chunk dictionaries.
    """
    logger.info(f"Processing file with Chonkie: {file_path}")
    chunks = []
    try:
        doc, ext = get_docs_for_file(file_path)
        doc = doc.content
        # The chef has already read the file content into doc.content
        if not doc:
            logger.warning(f"No content extracted from {file_path}. Skipping.")
            return []

        # Select the appropriate chunker recipe based on file type
        if ext == 'md':
            chunker = RecursiveChunker.from_recipe("markdown", lang="en")
            logger.info(f"Using markdown chunking recipe for {file_path}")
        elif ext == 'txt':
            chunker = RecursiveChunker.from_recipe(lang="en")
            logger.info(f"Using text chunking recipe for {file_path}")
        else:
            logger.info(f"Using code chunking recipe for {file_path}")
            chunker = CodeChunker(
                language="python",
                tokenizer_or_token_counter="character",
                chunk_size=chunk_size,  # Maximum tokens per chunk
                include_nodes=False  # Optionally include AST nodes in output
            )

        # Chunk the content that the chef extracted
        chunks = chunker.chunk(doc)

        # You can now process the chunks
        # For now, let's just log the number of chunks found
        logger.info(f"Found {len(chunks)} chunks in {file_path}.")
        complete_chunks = []
        for chunk in chunks:
            source_obj = MinimalSource(
                file_path=file_path,
                first_character_index=chunk.start_index,
                last_character_index=chunk.end_index
            )
            complete_chunks.append(
                ChunkSource(
                    text=(chunk.text),
                    source=source_obj)
            )
        return complete_chunks

    except Exception as e:
        logger.error(f"Could not process file {file_path}: {e} ")

    return chunks
