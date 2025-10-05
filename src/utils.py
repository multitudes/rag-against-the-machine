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
    logger.info(result.model_dump_json(indent=4))
