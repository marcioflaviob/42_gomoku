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


def get_pattern_freedom(window: np.ndarray, pattern_start: int, pattern_length: int, player: int) -> str:
    """
    Determines the freedom level of a pattern.
    Returns: 'free' (both ends open), 'half_free' (one end open), or 'flanked' (both ends blocked)
    """
    opponent = 2 if player == 1 else 1
    pattern_end = pattern_start + pattern_length - 1

    left_blocked = pattern_start > 0 and window[pattern_start - 1] == opponent
    right_blocked = pattern_end < len(window) - 1 and window[pattern_end + 1] == opponent

    if left_blocked and right_blocked:
        return 'flanked'
    elif left_blocked or right_blocked:
        return 'half_free'
    else:
        return 'free'


def has_space_to_develop(window: np.ndarray, pattern_start: int, pattern_length: int, player: int) -> bool:
    """
    Checks if a pattern has enough space on both sides to develop into a 5-in-a-row
    A pattern needs at least one open end (not blocked by opponent)
    """
    freedom = get_pattern_freedom(window, pattern_start, pattern_length, player)
    return freedom != 'flanked'


def score_window(window: np.ndarray, player: int) -> int:
    """
    Scores a 1D window (slice of a line) for `player`.
    window contains values: 0=empty, 1=P1, 2=P2
    Verifies patterns have space to develop into 5-in-a-row.
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

    # Freedom multipliers
    freedom_multiplier = {
        'free': 1.5,        # Both ends open: high priority
        'half_free': 1.0,   # One end open: normal priority
        'flanked': 0.3      # Both ends blocked: low priority
    }
    #  a "free four" is worth 150,000 points, while a "flanked four" only 3,000 points


    if player_count == 4:
        # Find where the 4 consecutive stones are
        for i in range(len(window) - 3):
            if np.all(window[i:i+4] == player):
                # Verify this 4-stone pattern has space to develop to 5
                if has_space_to_develop(window, i, 4, player):
                    freedom = get_pattern_freedom(window, i, 4, player)
                    multiplier = freedom_multiplier[freedom]
                    
                    # Check if open or closed
                    if empty_count == 1:
                        score += int(SCORE_CLOSED_FOUR * multiplier)
                    else:
                        # Open four
                        score += int(SCORE_OPEN_FOUR * multiplier)
                break

    elif player_count == 3:
        # Find the 3-stone pattern
        for i in range(len(window) - 2):
            if np.all(window[i:i+3] == player):
                if has_space_to_develop(window, i, 3, player):
                    freedom = get_pattern_freedom(window, i, 3, player)
                    multiplier = freedom_multiplier[freedom]
                    
                    if empty_count == 2:
                        score += int(SCORE_OPEN_THREE * multiplier)
                    elif empty_count == 1:
                        score += int(SCORE_CLOSED_THREE * multiplier)
                break

    elif player_count == 2:
        # For pairs, check if they have space to develop
        for i in range(len(window) - 1):
            if np.all(window[i:i+2] == player):
                if has_space_to_develop(window, i, 2, player):
                    freedom = get_pattern_freedom(window, i, 2, player)
                    multiplier = freedom_multiplier[freedom]
                    
                    if empty_count == 3:
                        score += int(SCORE_OPEN_TWO * multiplier)
                    elif empty_count == 2:
                        score += int(SCORE_CLOSED_TWO * multiplier)
                break

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

def detect_potential_captures(board: np.ndarray, player: int) -> int:
    """
    Detects potential capture opportunities.
    """
    opponent = 2 if player == 1 else 1
    potential_capture_score = 0
    WINDOW_SIZE = 5

    for line in get_lines(board):
        length = len(line)
        if length < WINDOW_SIZE:
            continue

        for i in range(length - WINDOW_SIZE + 1):
            window = line[i:i + WINDOW_SIZE]

            # Look for opponent pairs: X O O X (potential capture threat)
            for j in range(len(window) - 3):
                # Pattern: empty, opponent, opponent, empty
                if (window[j] == EMPTY and 
                    window[j+1] == opponent and 
                    window[j+2] == opponent and 
                    window[j+3] == EMPTY):
                    potential_capture_score -= 500  # Opponent has capture opportunity

                # Pattern with player on ends: player, opponent, opponent, player
                if (window[j] == player and 
                    window[j+1] == opponent and 
                    window[j+2] == opponent and 
                    window[j+3] == player):
                    potential_capture_score += 2000  # We can threatened capture

    return potential_capture_score


def score_captures(player_captures: int, opponent_captures: int) -> int:
    """
    Scores the capture situation.
    Near-win captures are worth way more.
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
    Returns positive score if position is good for `player`.
    """
    opponent = 2 if player == 1 else 1

    # Alignment score — symmetric, from each player's perspective
    player_score = score_lines(board, player)
    opponent_score = score_lines(board, opponent)

    # Capture score (actual captures)
    p_captures = player1_captures if player == 1 else player2_captures
    o_captures = player2_captures if player == 1 else player1_captures
    capture_score = score_captures(p_captures, o_captures)

    # Potential capture threats
    potential_capture_score = detect_potential_captures(board, player)

    return (player_score - opponent_score) + capture_score + potential_capture_score
