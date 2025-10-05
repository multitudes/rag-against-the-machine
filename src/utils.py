import os
import logging
from datetime import datetime
from core.schemas import StudentSearchResults

logger = logging.getLogger(__name__)


def write_search_to_file(result: StudentSearchResults, output_dir: str):
    """
    """
    os.makedirs(output_dir, exist_ok=True)
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_filename = f"search_results_{current_date}.json"
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(result.model_dump_json(indent=4))
    logger.info(f"Search results saved to {output_path}")

def calculate_overlap_percentage(start1, end1, start2, end2):
    """Calculates the overlap percentage between two character ranges."""
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    
    overlap_length = max(0, overlap_end - overlap_start)
    if overlap_length == 0:
        return 0.0
        
    # a common way is to measure
    # overlap relative to the length of the ground truth chunk.
    ground_truth_length = end1 - start1
    if ground_truth_length == 0:
        return 0.0
        
    return (overlap_length / ground_truth_length) * 100
