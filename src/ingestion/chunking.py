import os
import logging
from typing import Any, List, Optional, Tuple, Union
from chonkie import CodeChunker
from chonkie import RecursiveChunker
from chonkie import (
    TextChef,
    MarkdownChef,
    SentenceChunker
)
from core.schemas import MinimalSource, ChunkSource


logger = logging.getLogger(__name__)

IGNORE_EXTENSIONS = [
    "pdf", "zip", "so", "svg", "png", "eot", "ttf", "woff", "woff2",
    "pylintrc", "ico", "jpg", "neuron", "nightly_torch", "ppc64le",
    "rocm", "rocm_base", "s390x", "tpu", "xpu", "typed"
]

# Define language mapping for CodeChunker
CODE_LANGUAGES = {
    "py": "python", "pyi": "python",
    "sh": "bash",
    "cu": "cpp", "cuh": "cpp", "cpp": "cpp", "h": "cpp",
    "hpp": "cpp", "inl": "cpp",
    "css": "css",
    "cmake": "cmake", "cmakelists.txt": "cmake",
    "js": "javascript"
}
TEXT_EXTENSIONS = [
    "txt", "toml", "yaml", "yml", "json", "license", "dco",
    "manifest.in", "in", "j2", "jinja", "tpl", "jsonl", "patch", "env"
]
MARKDOWN_EXTENSIONS = ["md", "html", "rst"]


def get_docs_for_file(
    file_path: str,
) -> Tuple[Optional[Any], str]:
    """Load a file via chonkie's TextChef or MarkdownChef.

    Args:
        file_path: Path to the file to load.

    Returns:
        Tuple of (document object or None, file extension string).
    """
    ext = file_path.lower().rsplit('.', 1)[-1]
    chef: Union[TextChef, MarkdownChef] = TextChef()
    if ext == 'md':
        chef = MarkdownChef()
    try:
        return chef.process(file_path), ext
    except UnicodeDecodeError:
        logger.info(f"File {file_path} is not valid UTF-8. Skipping.")
        return None, ext


def _enforce_max_size(
    chunks: List[ChunkSource], max_size: int
) -> List[ChunkSource]:
    """Split any chunk whose text exceeds max_size into smaller pieces.

    This is a hard safety net for cases where the primary chunker cannot
    split an AST node or paragraph smaller than max_size (e.g. a very
    long function body). Character indices are adjusted to stay consistent
    with the original file so the moulinette overlap check still works.

    Args:
        chunks: Chunks produced by the primary chunker.
        max_size: Maximum allowed character length per chunk.

    Returns:
        List of chunks all guaranteed to be <= max_size characters.
    """
    result: List[ChunkSource] = []
    for chunk in chunks:
        if len(chunk.text) <= max_size:
            result.append(chunk)
            continue
        # Naive character split — preserves correct file offsets
        base = chunk.source.first_character_index
        text = chunk.text
        for i in range(0, len(text), max_size):
            sub_text = text[i:i + max_size]
            result.append(ChunkSource(
                text=sub_text,
                source=MinimalSource(
                    file_path=chunk.source.file_path,
                    first_character_index=base + i,
                    last_character_index=base + i + len(sub_text),
                ),
            ))
    return result


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
    logger.debug(f"Processing file with Chonkie: {file_path}")
    chunks = []
    try:
        parts = file_path.lower().rsplit('.', 1)
        if len(parts) == 2:
            ext = parts[1]
            name = parts[0]
            if name.endswith("cmakelists"):
                ext = "cmakelists.txt"
        else:
            filename = os.path.basename(file_path).lower()
            ext = filename

        if ext in IGNORE_EXTENSIONS:
            logger.debug(f"Ignoring binary/config file: {file_path}")
            return []
        doc, _ = get_docs_for_file(file_path)
        if doc is None:
            # get_docs_for_file already logged the error
            return []
        doc_content = doc.content

        if not doc_content:
            logger.debug(f"No content extracted from {file_path}. Skipping.")
            return []

        # Select the appropriate chunker based on file type
        chunker: Union[CodeChunker, RecursiveChunker, SentenceChunker]
        if ext in CODE_LANGUAGES:
            language = CODE_LANGUAGES[ext]
            logger.debug(f"Using CodeChunker for {language} in {file_path}")
            chunker = CodeChunker(
                language=language,
                tokenizer_or_token_counter="character",
                chunk_size=chunk_size,
                include_nodes=False
            )
        elif ext in MARKDOWN_EXTENSIONS:
            logger.debug(
                f"Using RecursiveChunker for markdown in {file_path}")
            chunker = RecursiveChunker(
                tokenizer_or_token_counter="character",
                chunk_size=chunk_size,
                min_characters_per_chunk=1,
            )
        elif ext in TEXT_EXTENSIONS or ext == 'dockerfile':
            logger.debug(f"Using SentenceChunker for text in {file_path}")
            chunker = SentenceChunker(
                tokenizer_or_token_counter="character",
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                min_sentences_per_chunk=1
            )
        else:
            logger.debug(f"No specific chunker for '{ext}'.")
            return []

        chunks = chunker.chunk(doc_content)

        logger.debug(f"Found {len(chunks)} chunks in {file_path}.")
        complete_chunks = []
        for chunk in chunks:
            source_obj = MinimalSource(
                file_path=file_path,
                first_character_index=chunk.start_index,
                last_character_index=chunk.end_index
            )
            complete_chunks.append(
                ChunkSource(text=chunk.text, source=source_obj)
            )

        # Hard safety net: split any chunk that still exceeds chunk_size.
        # CodeChunker preserves AST nodes so a single large function may
        # exceed the limit — the moulinette rejects the whole output if
        # any source is longer than max_context_length (2000 chars).
        return _enforce_max_size(complete_chunks, chunk_size)

    except Exception as e:
        logger.error(f"Could not process file {file_path}: {e} ")

    return chunks
