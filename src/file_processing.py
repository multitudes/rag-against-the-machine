import os
from pathlib import Path
from typing import List
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


def extract_files_from_questions(repo_path: str):
    """
    Get all relevant files from the repository
    """
    file_extensions = {'.py', '.md', '.rst', '.txt', '.json', \
                       '.yaml', '.yml', '.toml'}
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


def get_all_files(repo_path):
    """
    Get all relevant files from the repository
    """
    file_extensions = {'.py', '.md', '.rst', '.txt', '.json', '.yaml', '.yml', '.toml'}
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
