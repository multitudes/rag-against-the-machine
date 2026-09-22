import logging
from pathlib import Path
from typing import Any

from chonkie import (  # type: ignore[attr-defined]
    CodeChunker,
    MarkdownChef,
    RecursiveChunker,
    SentenceChunker,
    TextChef,
)

from core.config import MAX_CHUNK_SIZE
from core.schemas import ChunkSource, MinimalSource

logger = logging.getLogger(__name__)

IGNORE_EXTENSIONS = [
    "pdf", "zip", "so", "svg", "png", "eot", "ttf", "woff", "woff2",
    "pylintrc", "ico", "jpg", "neuron", "nightly_torch", "ppc64le",
    "rocm", "rocm_base", "s390x", "tpu", "xpu", "typed",
]

CODE_LANGUAGES = {
    "py": "python", "pyi": "python",
    "sh": "bash",
    "cu": "cpp", "cuh": "cpp", "cpp": "cpp",
    "h": "cpp", "hpp": "cpp", "inl": "cpp",
    "css": "css",
    "cmake": "cmake", "cmakelists.txt": "cmake",
    "js": "javascript",
}
TEXT_EXTENSIONS = [
    "txt", "toml", "yaml", "yml", "json", "license", "dco",
    "manifest.in", "in", "j2", "jinja", "tpl", "jsonl", "patch", "env",
]
MARKDOWN_EXTENSIONS = ["md", "html", "rst"]


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------

def _resolve_ext(file_path: str) -> str:
    """
    Return the logical extension for a file path.

    Handles the special case of CMakeLists.txt and files with no dot.

    Args:
        file_path: Path to the file to resolve.

    Returns:
        Lowercase extension string, e.g. ``"py"``, ``"cmakelists.txt"``.

    """
    parts = file_path.lower().rsplit(".", 1)
    if len(parts) == 2:
        name, ext = parts
        if name.endswith("cmakelists"):
            return "cmakelists.txt"
        return ext
    return Path(file_path).name.lower()


def _load_doc(file_path: str, ext: str) -> Any | None:
    """
    Load a file via chonkie's TextChef or MarkdownChef.

    Args:
        file_path: Path to the file to load.
        ext: Already-resolved file extension (from _resolve_ext).

    Returns:
        Chonkie document object, or None if the file is not valid UTF-8.

    """
    chef: TextChef | MarkdownChef = (
        MarkdownChef() if ext == "md" else TextChef()
    )
    try:
        return chef.process(file_path)
    except UnicodeDecodeError:
        logger.info("File %s is not valid UTF-8. Skipping.", file_path)
        return None


def _select_chunker(
    ext: str,
    chunk_size: int,
    overlap: int,
) -> CodeChunker | RecursiveChunker | SentenceChunker | None:
    """
    Return the right chonkie chunker for *ext*, or None to skip.

    Args:
        ext: Logical file extension (from _resolve_ext).
        chunk_size: Maximum characters per chunk.
        overlap: Overlap between chunks (SentenceChunker only).

    Returns:
        A configured chunker instance, or None if ext is unsupported.

    """
    if ext in CODE_LANGUAGES:
        language = CODE_LANGUAGES[ext]
        logger.debug("Using CodeChunker for %s", language)
        return CodeChunker(
            language=language,
            tokenizer_or_token_counter="character",
            chunk_size=chunk_size,
            include_nodes=False,
        )
    if ext in MARKDOWN_EXTENSIONS:
        logger.debug("Using RecursiveChunker for markdown")
        return RecursiveChunker(
            tokenizer_or_token_counter="character",
            chunk_size=chunk_size,
            min_characters_per_chunk=1,
        )
    if ext in TEXT_EXTENSIONS or ext == "dockerfile":
        logger.debug("Using SentenceChunker for text")
        return SentenceChunker(
            tokenizer_or_token_counter="character",
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            min_sentences_per_chunk=1,
        )
    logger.debug("No specific chunker for '%s'.", ext)
    return None


def _enforce_max_size(
    chunks: list[ChunkSource],
    max_size: int,
) -> list[ChunkSource]:
    """
    Split any chunk whose text exceeds max_size into smaller pieces.

    Hard safety net for cases where the primary chunker cannot split an
    AST node or paragraph smaller than max_size (e.g. a very long
    function body). Character indices are adjusted to stay consistent
    with the original file so the moulinette overlap check still works.

    Args:
        chunks: Chunks produced by the primary chunker.
        max_size: Maximum allowed character length per chunk.

    Returns:
        List of chunks all guaranteed to be <= max_size characters.

    """
    result: list[ChunkSource] = []
    for chunk in chunks:
        if len(chunk.text) <= max_size:
            result.append(chunk)
            continue
        base = chunk.source.first_character_index
        text = chunk.text
        for i in range(0, len(text), max_size):
            sub_text = text[i: i + max_size]
            result.append(
                ChunkSource(
                    text=sub_text,
                    source=MinimalSource(
                        file_path=chunk.source.file_path,
                        first_character_index=base + i,
                        last_character_index=base + i + len(sub_text),
                    ),
                ),
            )
    return result


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def chunk_content(
    file_path: str,
    chunk_size: int = MAX_CHUNK_SIZE,
    overlap: int = 200,
) -> list[ChunkSource]:
    """
    Chunk a file into smaller pieces using chonkie.

    Args:
        file_path: Path to the file to chunk.
        chunk_size: Maximum characters per chunk.
        overlap: Overlap between chunks (SentenceChunker only).

    Returns:
        List of ChunkSource objects with text and source location.

    """
    logger.debug("Processing file with Chonkie: %s", file_path)
    try:
        ext = _resolve_ext(file_path)

        if ext in IGNORE_EXTENSIONS:
            logger.debug("Ignoring binary/config file: %s", file_path)
            return []

        doc = _load_doc(file_path, ext)
        if doc is None or not doc.content:
            return []

        chunker = _select_chunker(ext, chunk_size, overlap)
        if chunker is None:
            return []

        raw_chunks = chunker.chunk(doc.content)
        logger.debug("Found %d chunks in %s.", len(raw_chunks), file_path)

        complete_chunks = [
            ChunkSource(
                text=chunk.text,
                source=MinimalSource(
                    file_path=file_path,
                    first_character_index=chunk.start_index,
                    last_character_index=chunk.end_index,
                ),
            )
            for chunk in raw_chunks
        ]

        # Hard safety net: CodeChunker preserves AST nodes so a single
        # large function may exceed the limit. The moulinette rejects
        # output where any source > max_context_length (2000 chars).
        return _enforce_max_size(complete_chunks, chunk_size)

    except Exception:
        logger.exception("Could not process file %s", file_path)
        return []
