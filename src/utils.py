"""
Utility functions for the Rage Against the Machine RAG system.
"""


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

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
