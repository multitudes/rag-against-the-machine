from chonkie import TextChef, MarkdownChef
from typing import List, Dict, Any
import logging
import json
from file_processing import read_file
from chonkie import RecursiveChunker
from chonkie import CodeChunker
from chonkie import ChromaHandshake



logger = logging.getLogger(__name__)


def get_docs_for_file(file_path):
    ext = file_path.lower().rsplit('.', 1)[-1]
    chef = TextChef()
    if ext == 'md':
        chef = MarkdownChef()
    return chef.process(file_path), ext


def chunk_content(file_path: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
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
                language="python",                 # Specify the programming language
                tokenizer_or_token_counter="character", # Default tokenizer (or use "gpt2", etc.)
                chunk_size=2048,                    # Maximum tokens per chunk
                include_nodes=False                # Optionally include AST nodes in output
            )
        
        # Chunk the content that the chef extracted
        chunks = chunker.chunk(doc)
        
        # You can now process the chunks
        # For now, let's just log the number of chunks found
        logger.info(f"Found {len(chunks)} chunks in {file_path}.")

        # --- Let's inspect the first chunk to see its structure ---
        if chunks:
            first_chunk = chunks[0]
            logger.info("--- Inspecting the first chunk ---")
            logger.info(f"Chunk ID: {first_chunk.id}")
            logger.info(f"Chunk Text: ...\n{first_chunk.text[:20]}...\n\n")
            logger.info(f"Chunk Start Index: {first_chunk.start_index}\n\n")
            logger.info(f"Chunk End Index: {first_chunk.end_index}\n\n")
            logger.info(f"Chunk Token Count: {first_chunk.token_count}\n\n")
            
            # The 'context' can sometimes be None if not generated
            if first_chunk.context:
                logger.info(f"Chunk Context: {first_chunk.context}")

            # You can easily convert the chunk to a dictionary
            chunk_dict = first_chunk.to_dict()
            pretty_json = json.dumps(chunk_dict, indent=4)
            logger.info(f"Chunk as pretty dictionary:\n{pretty_json}")
            logger.info("------------------------------------")
    

    except Exception as e:
        logger.error(f"Could not process file {file_path}: {e} ")
    
    return chunks

