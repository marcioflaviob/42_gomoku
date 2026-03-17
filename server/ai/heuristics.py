import numpy as np
from ai.constants import EMPTY, BOARD_SIZE

# Scores for patterns — you will tune these
SCORE_OPEN_FOUR    = 100_000
SCORE_CLOSED_FOUR  = 10_000
SCORE_OPEN_THREE   = 5_000
SCORE_CLOSED_THREE = 500
SCORE_OPEN_TWO     = 200
SCORE_CLOSED_TWO   = 50
SCORE_CAPTURE      = 3_000


def score_window(window: np.ndarray, player: int) -> int:
    """
    Scores a 1D window (slice of a line) for `player`.
    window contains values: 0=empty, 1=P1, 2=P2
    """
    opponent = 2 if player == 1 else 1
    score = 0

    player_count = np.sum(window == player)
    opponent_count = np.sum(window == opponent)
    empty_count = np.sum(window == EMPTY)

    # If opponent has stones here too, no threat possible
    if opponent_count > 0 and player_count > 0:
        return 0

    # Pure opponent window — no score for us
    if opponent_count > 0:
        return 0

    # Score based on how many player stones are in this window
    if player_count == 4:
        # Check if open or closed
        if empty_count == 1:
            score += SCORE_CLOSED_FOUR
        # open four needs context (checked in scan)

    elif player_count == 3:
        if empty_count == 2:
            score += SCORE_OPEN_THREE
        elif empty_count == 1:
            score += SCORE_CLOSED_THREE

    elif player_count == 2:
        if empty_count == 3:
            score += SCORE_OPEN_TWO
        elif empty_count == 2:
            score += SCORE_CLOSED_TWO

    elif player_count == 1:
        score += 10  # positional value

    return score

def get_lines(board: np.ndarray) -> list[np.ndarray]:
    """
    Extracts all lines from the board in 4 directions.
    Returns list of 1D arrays.
    """
    lines = []

    # Horizontal
    for r in range(BOARD_SIZE):
        lines.append(board[r, :])

    # Vertical
    for c in range(BOARD_SIZE):
        lines.append(board[:, c])

    # Diagonals (top-left to bottom-right)
    for offset in range(-(BOARD_SIZE - 5), BOARD_SIZE - 4):
        lines.append(np.diag(board, offset))

    # Anti-diagonals (top-right to bottom-left)
    flipped = np.fliplr(board)
    for offset in range(-(BOARD_SIZE - 5), BOARD_SIZE - 4):
        lines.append(np.diag(flipped, offset))

    return lines


def score_lines(board: np.ndarray, player: int) -> int:
    """
    Scans all lines in all directions and sums window scores.
    Uses a sliding window of size 5.
    """
    WINDOW_SIZE = 6
    total = 0

    for line in get_lines(board):
        length = len(line)
        if length < WINDOW_SIZE:
            continue  # line too short to matter

        for i in range(length - WINDOW_SIZE + 1):
            window = line[i:i + WINDOW_SIZE]
            total += score_window(window, player)

    return total

def score_captures(player_captures: int, opponent_captures: int) -> int:
    """
    Scores the capture situation.
    Near-win captures are worth exponentially more.
    """
    score = 0

    # Our captures
    score += player_captures * SCORE_CAPTURE

    # Exponential bonus when close to capture win (10 stones = 5 pairs)
    if player_captures >= 8:    # 4 pairs, one away from winning
        score += 500_000
    elif player_captures >= 6:  # 3 pairs
        score += 50_000

    # Penalize opponent captures
    score -= opponent_captures * SCORE_CAPTURE

    if opponent_captures >= 8:
        score -= 500_000
    elif opponent_captures >= 6:
        score -= 50_000

    return score

def evaluate_board(
    board: np.ndarray,
    player: int,
    player1_captures: int,
    player2_captures: int
) -> float:
    """
    Main heuristic entry point.
    Returns positive score if position is good for `player`.
    """
    opponent = 2 if player == 1 else 1

    # Alignment score — symmetric, from each player's perspective
    player_score = score_lines(board, player)
    opponent_score = score_lines(board, opponent)

    # Capture score
    p_captures = player1_captures if player == 1 else player2_captures
    o_captures = player2_captures if player == 1 else player1_captures
    capture_score = score_captures(p_captures, o_captures)

    return (player_score - opponent_score) + capture_score
