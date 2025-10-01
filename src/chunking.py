import chonkie
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


def chunk_content(content: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
    """
    Chunk content into smaller pieces using chonkie.
    
    Args:
        content: Text content to chunk
        chunk_size: Size of each chunk
        overlap: Overlap between chunks
        
    Returns:
        List of chunk dictionaries
    """
