import logging
import os

logger = logging.getLogger(__name__)


def read_file(file_path: str) -> str:
    """
    Read content from a file handling different encodings."""
    try:
        with open(file_path, encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error("Error reading file %s: %s", file_path, e)
        return ""


def extract_files_from_questions(questions_file: str) -> list[str]:
    """
    Get the list of file paths referenced in a TSV questions file."""
    file_paths = set()
    try:
        with open(questions_file, encoding="utf-8") as f:
            header = f.readline().strip().split("\t")
            if "file_path" in header:
                file_path_idx = header.index("file_path")
            else:
                raise ValueError("Header does not contain file_path")

            for line in f:
                parts = line.strip().split("\t")
                if len(parts) > abs(file_path_idx):
                    file_paths.add(parts[file_path_idx])

        logger.info(
            "Extracted %d unique files from questions.tsv", len(file_paths)
        )
        return list(file_paths)
    except Exception:
        raise


def get_all_files(repo_path: str) -> list[str]:
    """
    Get all files from the repository, skipping common temporary directories.
    """
    files = []
    excluded_dirs = ["__pycache__", "node_modules", ".git"]

    for root, dirs, filenames in os.walk(repo_path):
		# pruning the walk modifying dirs in place
        dirs[:] = [
            d for d in dirs
            if not d.startswith(".") and d not in excluded_dirs
        ]

        for filename in filenames:
            if filename.startswith("."):
                continue
            files.append(os.path.join(root, filename))

    logger.info("Found %d files to process in %s", len(files), repo_path)
    return files
