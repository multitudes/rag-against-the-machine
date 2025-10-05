from chonkie import TextChef, MarkdownChef, SentenceChunker
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
        # Improved extension handling
        parts = file_path.lower().rsplit('.', 1)
        if len(parts) == 2:
            ext = parts[1]
            name = parts[0]
            if name.endswith("cmakelists"):
                ext = "cmakelists.txt"
        else:
            # Handle files with no extension like 'Dockerfile'
            ext = file_path.split('/')[-1].lower()

        # Define language mapping for CodeChunker
        code_languages = {
            "py": "python", "pyi": "python",
            "sh": "bash",
            "cu": "cpp", "cuh": "cpp", "cpp": "cpp", "h": "cpp",
            "css": "css",
            "cmake": "cmake", "cmakelists.txt": "cmake"
        }

        # Define extensions to ignore
        ignore_extensions = [
            "pdf", "zip", "so", "svg", "png", "eot", "ttf", "woff", "woff2",
            "pylintrc"
        ]

        # Define text/markdown extensions
        text_extensions = [
            "txt", "toml", "yaml", "yml", "json", "license", "dco",
            "manifest.in", "in", "j2"
        ]
        markdown_extensions = ["md", "html", "rst"]

        doc, _ = get_docs_for_file(file_path)
        doc_content = doc.content

        if not doc_content:
            logger.warning(f"No content extracted from {file_path}. Skipping.")
            return []

        # Select the appropriate chunker based on file type
        if ext in code_languages:
            language = code_languages[ext]
            logger.info(f"Using CodeChunker for {language} in {file_path}")
            chunker = CodeChunker(
                language=language,
                tokenizer_or_token_counter="character",
                chunk_size=chunk_size,
                include_nodes=False
            )
        elif ext in markdown_extensions:
            logger.info(f"Using RecursiveChunker for markdown in {file_path}")
            chunker = RecursiveChunker.from_recipe("markdown", lang="en")
        elif ext in text_extensions or ext == 'dockerfile':
            logger.info(f"Using SentenceChunker for text in {file_path}")
            chunker = SentenceChunker(
                tokenizer_or_token_counter="character",
                chunk_size=chunk_size,
                chunk_overlap=128,
                min_sentences_per_chunk=1
            )
        elif ext in ignore_extensions:
            logger.info(f"Ignoring binary/config file: {file_path}")
            return []
        else:
            logger.warning(
                f"No specific chunker for '{ext}'. Using default text chunker."
            )
            chunker = RecursiveChunker(chunk_size=chunk_size, overlap=overlap)

        # Chunk the content
        chunks = chunker.chunk(doc_content)

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
