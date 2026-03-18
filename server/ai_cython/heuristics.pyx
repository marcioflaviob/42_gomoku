# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, initializedcheck=False

import numpy as np
cimport numpy as cnp
from ai_cython.constants import EMPTY, BOARD_SIZE

cnp.import_array()

# Scores for patterns — you will tune these
SCORE_OPEN_FOUR    = 100_000
SCORE_CLOSED_FOUR  = 10_000
SCORE_OPEN_THREE   = 5_000
SCORE_CLOSED_THREE = 500
SCORE_OPEN_TWO     = 200
SCORE_CLOSED_TWO   = 50
SCORE_CAPTURE      = 3_000


cdef inline int _scaled_score(int base_score, int freedom_code):
    # freedom_code: 0=free, 1=half_free, 2=flanked
    if freedom_code == 0:
        return (base_score * 3) // 2
    elif freedom_code == 1:
        return base_score
    return (base_score * 3) // 10


cdef inline int _freedom_code_from_values(int left_value, int right_value, int opponent):
    cdef bint left_blocked = left_value == opponent
    cdef bint right_blocked = right_value == opponent
    if left_blocked and right_blocked:
        return 2
    elif left_blocked or right_blocked:
        return 1
    return 0


cdef inline int _score_window6_values(
    int v0,
    int v1,
    int v2,
    int v3,
    int v4,
    int v5,
    int player,
):
    cdef int opponent = 2 if player == 1 else 1
    cdef int values[6]
    cdef int player_count = 0
    cdef int opponent_count = 0
    cdef int empty_count = 0
    cdef int i
    cdef int freedom_code

    values[0] = v0
    values[1] = v1
    values[2] = v2
    values[3] = v3
    values[4] = v4
    values[5] = v5

    for i in range(6):
        if values[i] == player:
            player_count += 1
        elif values[i] == opponent:
            opponent_count += 1
        elif values[i] == EMPTY:
            empty_count += 1

    if opponent_count > 0 and player_count > 0:
        return 0
    if opponent_count > 0:
        return 0

    if player_count == 4:
        for i in range(3):
            if (
                values[i] == player
                and values[i + 1] == player
                and values[i + 2] == player
                and values[i + 3] == player
            ):
                # Match original logic: window edges count as not blocked.
                freedom_code = _freedom_code_from_values(
                    opponent if i > 0 and values[i - 1] == opponent else EMPTY,
                    opponent if i + 3 < 5 and values[i + 4] == opponent else EMPTY,
                    opponent,
                )
                if freedom_code == 2:
                    return 0
                if empty_count == 1:
                    return _scaled_score(SCORE_CLOSED_FOUR, freedom_code)
                return _scaled_score(SCORE_OPEN_FOUR, freedom_code)
        return 0

    elif player_count == 3:
        for i in range(4):
            if values[i] == player and values[i + 1] == player and values[i + 2] == player:
                freedom_code = _freedom_code_from_values(
                    opponent if i > 0 and values[i - 1] == opponent else EMPTY,
                    opponent if i + 2 < 5 and values[i + 3] == opponent else EMPTY,
                    opponent,
                )
                if freedom_code == 2:
                    return 0
                if empty_count == 2:
                    return _scaled_score(SCORE_OPEN_THREE, freedom_code)
                elif empty_count == 1:
                    return _scaled_score(SCORE_CLOSED_THREE, freedom_code)
                return 0
        return 0

    elif player_count == 2:
        for i in range(5):
            if values[i] == player and values[i + 1] == player:
                freedom_code = _freedom_code_from_values(
                    opponent if i > 0 and values[i - 1] == opponent else EMPTY,
                    opponent if i + 1 < 5 and values[i + 2] == opponent else EMPTY,
                    opponent,
                )
                if freedom_code == 2:
                    return 0
                if empty_count == 3:
                    return _scaled_score(SCORE_OPEN_TWO, freedom_code)
                elif empty_count == 2:
                    return _scaled_score(SCORE_CLOSED_TWO, freedom_code)
                return 0
        return 0

    elif player_count == 1:
        return 10

    return 0


def get_pattern_freedom(window: np.ndarray, pattern_start: int, pattern_length: int, player: int) -> str:
    """
    Determines the freedom level of a pattern.
    Returns: 'free' (both ends open), 'half_free' (one end open), or 'flanked' (both ends blocked)
    """
    cdef int opponent = 2 if player == 1 else 1
    cdef int pattern_end = pattern_start + pattern_length - 1
    cdef bint left_blocked = pattern_start > 0 and window[pattern_start - 1] == opponent
    cdef bint right_blocked = pattern_end < len(window) - 1 and window[pattern_end + 1] == opponent

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
    cdef int n = len(window)
    if n == 6:
        return _score_window6_values(
            <int>window[0],
            <int>window[1],
            <int>window[2],
            <int>window[3],
            <int>window[4],
            <int>window[5],
            player,
        )

    # Compatibility fallback if called with a non-standard window size.
    cdef int i
    cdef int score = 0
    cdef int opponent = 2 if player == 1 else 1
    cdef int player_count = 0
    cdef int opponent_count = 0
    cdef int empty_count = 0
    cdef int freedom_code

    for i in range(n):
        if window[i] == player:
            player_count += 1
        elif window[i] == opponent:
            opponent_count += 1
        elif window[i] == EMPTY:
            empty_count += 1

    if opponent_count > 0:
        return 0

    if player_count == 4:
        for i in range(n - 3):
            if window[i] == player and window[i + 1] == player and window[i + 2] == player and window[i + 3] == player:
                freedom_code = _freedom_code_from_values(
                    opponent if i > 0 and window[i - 1] == opponent else EMPTY,
                    opponent if i + 3 < n - 1 and window[i + 4] == opponent else EMPTY,
                    opponent,
                )
                if freedom_code != 2:
                    if empty_count == 1:
                        score += _scaled_score(SCORE_CLOSED_FOUR, freedom_code)
                    else:
                        score += _scaled_score(SCORE_OPEN_FOUR, freedom_code)
                break

    elif player_count == 3:
        for i in range(n - 2):
            if window[i] == player and window[i + 1] == player and window[i + 2] == player:
                freedom_code = _freedom_code_from_values(
                    opponent if i > 0 and window[i - 1] == opponent else EMPTY,
                    opponent if i + 2 < n - 1 and window[i + 3] == opponent else EMPTY,
                    opponent,
                )
                if freedom_code != 2:
                    if empty_count == 2:
                        score += _scaled_score(SCORE_OPEN_THREE, freedom_code)
                    elif empty_count == 1:
                        score += _scaled_score(SCORE_CLOSED_THREE, freedom_code)
                break

    elif player_count == 2:
        for i in range(n - 1):
            if window[i] == player and window[i + 1] == player:
                freedom_code = _freedom_code_from_values(
                    opponent if i > 0 and window[i - 1] == opponent else EMPTY,
                    opponent if i + 1 < n - 1 and window[i + 2] == opponent else EMPTY,
                    opponent,
                )
                if freedom_code != 2:
                    if empty_count == 3:
                        score += _scaled_score(SCORE_OPEN_TWO, freedom_code)
                    elif empty_count == 2:
                        score += _scaled_score(SCORE_CLOSED_TWO, freedom_code)
                break

    elif player_count == 1:
        score += 10

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


cdef int _score_lines_fast(cnp.ndarray[cnp.int64_t, ndim=2] board, int player):
    cdef int total = 0
    cdef int r, c, i, length
    cdef int rr, cc
    cdef int line_buf[32]

    # Horizontal lines
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE - 5):
            total += _score_window6_values(
                <int>board[r, c],
                <int>board[r, c + 1],
                <int>board[r, c + 2],
                <int>board[r, c + 3],
                <int>board[r, c + 4],
                <int>board[r, c + 5],
                player,
            )

    # Vertical lines
    for c in range(BOARD_SIZE):
        for r in range(BOARD_SIZE - 5):
            total += _score_window6_values(
                <int>board[r, c],
                <int>board[r + 1, c],
                <int>board[r + 2, c],
                <int>board[r + 3, c],
                <int>board[r + 4, c],
                <int>board[r + 5, c],
                player,
            )

    # Diagonals (top-left to bottom-right)
    for c in range(BOARD_SIZE):
        rr = 0
        cc = c
        length = BOARD_SIZE - c
        if length >= 6:
            for i in range(length):
                line_buf[i] = <int>board[rr + i, cc + i]
            for i in range(length - 5):
                total += _score_window6_values(
                    line_buf[i],
                    line_buf[i + 1],
                    line_buf[i + 2],
                    line_buf[i + 3],
                    line_buf[i + 4],
                    line_buf[i + 5],
                    player,
                )
    for r in range(1, BOARD_SIZE):
        rr = r
        cc = 0
        length = BOARD_SIZE - r
        if length >= 6:
            for i in range(length):
                line_buf[i] = <int>board[rr + i, cc + i]
            for i in range(length - 5):
                total += _score_window6_values(
                    line_buf[i],
                    line_buf[i + 1],
                    line_buf[i + 2],
                    line_buf[i + 3],
                    line_buf[i + 4],
                    line_buf[i + 5],
                    player,
                )

    # Anti-diagonals (top-right to bottom-left)
    for c in range(BOARD_SIZE):
        rr = 0
        cc = c
        length = c + 1
        if length >= 6:
            for i in range(length):
                line_buf[i] = <int>board[rr + i, cc - i]
            for i in range(length - 5):
                total += _score_window6_values(
                    line_buf[i],
                    line_buf[i + 1],
                    line_buf[i + 2],
                    line_buf[i + 3],
                    line_buf[i + 4],
                    line_buf[i + 5],
                    player,
                )
    for r in range(1, BOARD_SIZE):
        rr = r
        cc = BOARD_SIZE - 1
        length = BOARD_SIZE - r
        if length >= 6:
            for i in range(length):
                line_buf[i] = <int>board[rr + i, cc - i]
            for i in range(length - 5):
                total += _score_window6_values(
                    line_buf[i],
                    line_buf[i + 1],
                    line_buf[i + 2],
                    line_buf[i + 3],
                    line_buf[i + 4],
                    line_buf[i + 5],
                    player,
                )

    return total


def score_lines(board: np.ndarray, player: int) -> int:
    """
    Scans all lines in all directions and sums window scores.
    Uses a sliding window of size 5.
    """
    return _score_lines_fast(board, player)

cdef int _detect_potential_captures_fast(cnp.ndarray[cnp.int64_t, ndim=2] board, int player):
    cdef int opponent = 2 if player == 1 else 1
    cdef int potential_capture_score = 0
    cdef int r, c, i, j, k, length
    cdef int rr, cc
    cdef int a, b, d, e
    cdef int line_buf[32]

    # Helper pattern checks are done directly over 4-cell spans.

    # Horizontal lines
    for r in range(BOARD_SIZE):
        for i in range(BOARD_SIZE):
            line_buf[i] = <int>board[r, i]
        for i in range(BOARD_SIZE - 4):
            for j in range(2):
                k = i + j
                if k + 3 >= BOARD_SIZE:
                    continue
                a = line_buf[k]
                b = line_buf[k + 1]
                d = line_buf[k + 2]
                e = line_buf[k + 3]
                if a == EMPTY and b == opponent and d == opponent and e == EMPTY:
                    potential_capture_score -= 500
                if a == player and b == opponent and d == opponent and e == player:
                    potential_capture_score += 2000

    # Vertical lines
    for c in range(BOARD_SIZE):
        for i in range(BOARD_SIZE):
            line_buf[i] = <int>board[i, c]
        for i in range(BOARD_SIZE - 4):
            for j in range(2):
                k = i + j
                if k + 3 >= BOARD_SIZE:
                    continue
                a = line_buf[k]
                b = line_buf[k + 1]
                d = line_buf[k + 2]
                e = line_buf[k + 3]
                if a == EMPTY and b == opponent and d == opponent and e == EMPTY:
                    potential_capture_score -= 500
                if a == player and b == opponent and d == opponent and e == player:
                    potential_capture_score += 2000

    # Diagonals (top-left to bottom-right)
    for c in range(BOARD_SIZE):
        rr = 0
        cc = c
        length = BOARD_SIZE - c
        for i in range(length):
            line_buf[i] = <int>board[rr + i, cc + i]
        if length >= 5:
            for i in range(length - 4):
                for j in range(2):
                    k = i + j
                    if k + 3 >= length:
                        continue
                    a = line_buf[k]
                    b = line_buf[k + 1]
                    d = line_buf[k + 2]
                    e = line_buf[k + 3]
                    if a == EMPTY and b == opponent and d == opponent and e == EMPTY:
                        potential_capture_score -= 500
                    if a == player and b == opponent and d == opponent and e == player:
                        potential_capture_score += 2000
    for r in range(1, BOARD_SIZE):
        rr = r
        cc = 0
        length = BOARD_SIZE - r
        for i in range(length):
            line_buf[i] = <int>board[rr + i, cc + i]
        if length >= 5:
            for i in range(length - 4):
                for j in range(2):
                    k = i + j
                    if k + 3 >= length:
                        continue
                    a = line_buf[k]
                    b = line_buf[k + 1]
                    d = line_buf[k + 2]
                    e = line_buf[k + 3]
                    if a == EMPTY and b == opponent and d == opponent and e == EMPTY:
                        potential_capture_score -= 500
                    if a == player and b == opponent and d == opponent and e == player:
                        potential_capture_score += 2000

    # Anti-diagonals (top-right to bottom-left)
    for c in range(BOARD_SIZE):
        rr = 0
        cc = c
        length = c + 1
        for i in range(length):
            line_buf[i] = <int>board[rr + i, cc - i]
        if length >= 5:
            for i in range(length - 4):
                for j in range(2):
                    k = i + j
                    if k + 3 >= length:
                        continue
                    a = line_buf[k]
                    b = line_buf[k + 1]
                    d = line_buf[k + 2]
                    e = line_buf[k + 3]
                    if a == EMPTY and b == opponent and d == opponent and e == EMPTY:
                        potential_capture_score -= 500
                    if a == player and b == opponent and d == opponent and e == player:
                        potential_capture_score += 2000
    for r in range(1, BOARD_SIZE):
        rr = r
        cc = BOARD_SIZE - 1
        length = BOARD_SIZE - r
        for i in range(length):
            line_buf[i] = <int>board[rr + i, cc - i]
        if length >= 5:
            for i in range(length - 4):
                for j in range(2):
                    k = i + j
                    if k + 3 >= length:
                        continue
                    a = line_buf[k]
                    b = line_buf[k + 1]
                    d = line_buf[k + 2]
                    e = line_buf[k + 3]
                    if a == EMPTY and b == opponent and d == opponent and e == EMPTY:
                        potential_capture_score -= 500
                    if a == player and b == opponent and d == opponent and e == player:
                        potential_capture_score += 2000

    return potential_capture_score


def detect_potential_captures(board: np.ndarray, player: int) -> int:
    """
    Detects potential capture opportunities.
    """
    return _detect_potential_captures_fast(board, player)


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

cdef int _evaluate_board_fast(
    cnp.ndarray[cnp.int64_t, ndim=2] board,
    int player,
    int player1_captures,
    int player2_captures,
):
    cdef int opponent = 2 if player == 1 else 1
    cdef int player_score = _score_lines_fast(board, player)
    cdef int opponent_score = _score_lines_fast(board, opponent)
    cdef int p_captures = player1_captures if player == 1 else player2_captures
    cdef int o_captures = player2_captures if player == 1 else player1_captures
    cdef int capture_score = score_captures(p_captures, o_captures)
    cdef int potential_capture_score = _detect_potential_captures_fast(board, player)

    return (player_score - opponent_score) + capture_score + potential_capture_score


def evaluate_board(
    board: np.ndarray,
    player: int,
    player1_captures: int,
    player2_captures: int
) -> float:
    """
    Returns positive score if position is good for `player`.
    """
    return _evaluate_board_fast(board, player, player1_captures, player2_captures)
