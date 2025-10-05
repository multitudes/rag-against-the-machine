import os
import logging

logger = logging.getLogger(__name__)


def read_file(file_path: str) -> str:
    """
    Read content from a file handling different encodings
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return ""


def extract_files_from_questions(questions_file: str):
    """
    Get all relevant files from the repository
    """
    file_paths = set()
    try:
        with open(questions_file, 'r', encoding='utf-8') as f:
            # Read header to find the file path column index
            header = f.readline().strip().split('\t')
            print(f"Header {header}")
            if 'file_path' in header:
                file_path_idx = header.index('file_path')
            else:
                raise ValueError("Header does not contain file_path")

            # Here I if there is an error I just skip
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) > abs(file_path_idx):
                    file_paths.add(parts[file_path_idx])

        logger.info(f"Extracted {len(file_paths)} unique files\
                     from questions.tsv")
        return list(file_paths)
    except Exception:
        raise


def get_all_files(repo_path: str) -> list[str]:
    """
    Get all files from the repository, skipping common temporary directories.
    """
    files = []
    excluded_dirs = ['__pycache__', 'node_modules', '.git']

    for root, dirs, filenames in os.walk(repo_path):
        # Modify dirs in-place to skip excluded directories
        dirs[:] = [d for d in dirs if not d.startswith('.') 
                   and d not in excluded_dirs]

        for filename in filenames:
            # Skip hidden files
            if filename.startswith('.'):
                continue
            files.append(os.path.join(root, filename))

    logger.info(f"Found {len(files)} files to process in {repo_path}")
    return files
