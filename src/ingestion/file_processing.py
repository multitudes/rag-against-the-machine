import os
from pathlib import Path
import logging
# import json

logging.basicConfig(level=logging.INFO)
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


def get_all_files(repo_path):
    """
    Get all relevant files from the repository
    """
    file_extensions = {'.py', '.md', '.rst', '.txt', '.json', '.yaml',
                       '.yml', '.toml'}
    files = []

    for root, dirs, filenames in os.walk(repo_path):
        # skip directories we dont need
        dirs[:] = [d for d in dirs if not d.startswith('.') and
                   d not in ['__pycache__', 'node_modules', '.git']]

        for filename in filenames:
            file_path = os.path.join(root, filename)
            if Path(filename).suffix.lower() in file_extensions:
                files.append(file_path)

    logger.info(f"found {len(files)} files to process")
    return files
