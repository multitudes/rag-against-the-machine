"""
Utility functions for the Rage Against the Machine RAG system.

Includes helpers for saving search results, calculating overlap,
and managing output files.
"""

import logging
import os
from datetime import datetime
from pathlib import Path

from core.schemas import StudentSearchResults, StudentSearchResultsAndAnswer

logger = logging.getLogger(__name__)


def write_search_to_file(
    result: StudentSearchResults,
    output_dir: str,
) -> None:
    """
    Save StudentSearchResults to a JSON file in the output directory.

    The filename includes the current date and time for uniqueness.

    Args:
        result: The search results to save.
        output_dir: Directory to save the output file.

    Returns:
        None.

    """
    os.makedirs(output_dir, exist_ok=True)
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_filename = f"search_results_{current_date}.json"
    output_path = os.path.join(output_dir, output_filename)
    with Path(output_path).open("w", encoding="utf-8") as f:
        f.write(result.model_dump_json(indent=4))
    logger.debug("Search results saved to %s", output_path)


def calculate_overlap_percentage(
    start1: int,
    end1: int,
    start2: int,
    end2: int,
) -> float:
    """
    Calculate the percentage overlap between two character ranges.

    The overlap is measured relative to the length of the ground truth
    chunk. Returns 0.0 if there is no overlap.

    Args:
        start1: Start index of the ground truth chunk.
        end1: End index of the ground truth chunk.
        start2: Start index of the retrieved chunk.
        end2: End index of the retrieved chunk.

    Returns:
        Percentage overlap (0.0 if no overlap).

    """
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)

    overlap_length = max(0, overlap_end - overlap_start)
    if overlap_length == 0:
        return 0.0

    ground_truth_length = end1 - start1
    if ground_truth_length == 0:
        return 0.0

    return (overlap_length / ground_truth_length) * 100


def save_search_results_and_answer_to_json(
    final_result: StudentSearchResultsAndAnswer,
) -> None:
    """
    Save StudentSearchResultsAndAnswer to a JSON file.

    The filename includes the current date and time for uniqueness.

    Args:
        final_result: The results-and-answers object to save.

    Returns:
        None.

    """
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"Dataset_{date_str}_valid.json"
    output_dir = "data/output/search_results/"
    output_path = os.path.join(output_dir, filename)
    os.makedirs(output_dir, exist_ok=True)
    with Path(output_path).open("w", encoding="utf-8") as f:
        f.write(final_result.model_dump_json(indent=4))

    logger.info("results will be saved to: %s", output_path)
    logger.info("Answer generation completed!")
