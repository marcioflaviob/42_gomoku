import numpy as np
from scipy.ndimage import uniform_filter
from constants import EMPTY, PLAYER1, PLAYER2

def get_candidate_moves(board: np.ndarray, radius: int = 2, player: int = PLAYER2) -> list[tuple[int, int]]:
    """
    Returns empty cells within `radius` of any placed stone.
    Board is a 19x19 numpy array.
    """
    # Calculate occupied cells for the specified player
    occupied = (board == PLAYER1) | (board == PLAYER2)

    if not occupied.any():
        return [(9, 9)]

    # Build neighborhood mask using a sliding window sum
    neighborhood = uniform_filter(
        occupied.astype(float),
        size=2 * radius + 1,
        mode='constant'
    )


    # Candidate = empty AND in neighborhood of a stone
    candidate = (neighborhood > 0) & (board == EMPTY)
    candidates = list(zip(*np.where(candidate)))

    return [(int(r), int(c)) for r, c in candidates]


def board_from_json(json_board: list[list[int]]) -> np.ndarray:
    """Convert JSON board payload to numpy array immediately on receipt."""
    return np.array(json_board, dtype=np.int8)