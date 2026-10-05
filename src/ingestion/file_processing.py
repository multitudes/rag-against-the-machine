import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def read_file(file_path: str) -> str:
    """
    Read content from a file handling different encodings.

    Args:
        file_path: Path to the file to read.

    Returns:
        File content as a string, or an empty string on error.

    """
    try:
        with Path(file_path).open(encoding="utf-8") as f:
            return f.read()
    except Exception:
        logger.exception("Error reading file %s", file_path)
        return ""


def extract_files_from_questions(questions_file: str) -> list[str]:
    """
    Get the list of file paths referenced in a TSV questions file.

    Args:
        questions_file: Path to the TSV file containing questions.

    Returns:
        List of unique file paths extracted from the TSV.

    """
    file_paths = set()
    with Path(questions_file).open(encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        if "file_path" not in header:
            msg = "Header does not contain file_path"
            raise ValueError(msg)
        file_path_idx = header.index("file_path")

        for line in f:
            parts = line.strip().split("\t")
            if len(parts) > abs(file_path_idx):
                file_paths.add(parts[file_path_idx])

    logger.info(
        "Extracted %d unique files from questions.tsv",
        len(file_paths),
    )
    return list(file_paths)


def get_all_files(repo_path: str) -> list[str]:
    """
    Get all files from the repository, skipping common temporary directories.

    Args:
        repo_path: Root directory of the repository to walk.

    Returns:
        List of absolute file paths found under repo_path.

    """
    files: list[str] = []
    excluded_dirs = ["__pycache__", "node_modules", ".git"]

    for dirpath, dirs, filenames in Path(repo_path).walk():
        # Prune the walk by editing dirs in place (same as os.walk).
        dirs[:] = [
            d for d in dirs
            if not d.startswith(".") and d not in excluded_dirs
        ]

        for filename in filenames:
            if filename.startswith("."):
                continue
            files.append(str(dirpath / filename))

    logger.info("Found %d files to process in %s", len(files), repo_path)
    return files
