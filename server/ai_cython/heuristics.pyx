# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, initializedcheck=False

import numpy as np
cimport numpy as cnp
from ai_cython.constants import EMPTY, BOARD_SIZE

cnp.import_array()

# Scores for patterns
SCORE_OPEN_FOUR    = 100_000
SCORE_CLOSED_FOUR  = 10_000
SCORE_OPEN_THREE   = 5_000
SCORE_CLOSED_THREE = 500
SCORE_OPEN_TWO     = 200
SCORE_CLOSED_TWO   = 50
SCORE_CAPTURE      = 3_000

# ---------------------------------------------------------------------------
# Directions: (dr, dc) for horizontal, vertical, diagonal, anti-diagonal
# ---------------------------------------------------------------------------
DEF N_DIRS = 4
cdef int DIRS_R[4]
cdef int DIRS_C[4]
DIRS_R[0] = 0;  DIRS_C[0] = 1   # horizontal
DIRS_R[1] = 1;  DIRS_C[1] = 0   # vertical
DIRS_R[2] = 1;  DIRS_C[2] = 1   # diagonal
DIRS_R[3] = 1;  DIRS_C[3] = -1  # anti-diagonal


# ===========================================================================
# Low-level scoring helpers (unchanged)
# ===========================================================================

cdef inline int _scaled_score(int base_score, int freedom_code):
    if freedom_code == 0:
        return (base_score * 3) // 2
    elif freedom_code == 1:
        return base_score
    return (base_score * 3) // 10


cdef inline int _freedom_code_from_values(int left_value, int right_value, int opponent):
    cdef bint left_blocked  = left_value  == opponent
    cdef bint right_blocked = right_value == opponent
    if left_blocked and right_blocked:
        return 2
    elif left_blocked or right_blocked:
        return 1
    return 0


cdef inline int _score_window6_values(
    int v0, int v1, int v2, int v3, int v4, int v5,
    int player,
):
    cdef int opponent = 2 if player == 1 else 1
    cdef int values[6]
    cdef int player_count = 0, opponent_count = 0, empty_count = 0
    cdef int i, freedom_code

    values[0] = v0; values[1] = v1; values[2] = v2
    values[3] = v3; values[4] = v4; values[5] = v5

    for i in range(6):
        if values[i] == player:
            player_count += 1
        elif values[i] == opponent:
            opponent_count += 1
        elif values[i] == EMPTY:
            empty_count += 1

    if opponent_count > 0:
        return 0

    if player_count == 4:
        for i in range(3):
            if (values[i] == player and values[i+1] == player
                    and values[i+2] == player and values[i+3] == player):
                freedom_code = _freedom_code_from_values(
                    opponent if i > 0 and values[i-1] == opponent else EMPTY,
                    opponent if i+3 < 5 and values[i+4] == opponent else EMPTY,
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
            if values[i] == player and values[i+1] == player and values[i+2] == player:
                freedom_code = _freedom_code_from_values(
                    opponent if i > 0 and values[i-1] == opponent else EMPTY,
                    opponent if i+2 < 5 and values[i+3] == opponent else EMPTY,
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
            if values[i] == player and values[i+1] == player:
                freedom_code = _freedom_code_from_values(
                    opponent if i > 0 and values[i-1] == opponent else EMPTY,
                    opponent if i+1 < 5 and values[i+2] == opponent else EMPTY,
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


# ===========================================================================
# Incremental delta scorer
#
# For a single cell (r, c) and one direction (dr, dc), compute the sum of
# all size-6 window scores that CONTAIN (r, c).  Call this before and after
# placing/removing a stone; the difference is the score delta.
#
# A window of length 6 starting at offset k (along the line) contains (r,c)
# when 0 <= (position_of_(r,c)_in_line - k) <= 5, i.e. k in [pos-5, pos].
# We clamp to the board boundary.
# ===========================================================================

cdef int _score_cell_direction(
    cnp.ndarray[cnp.int64_t, ndim=2] board,
    int r, int c,
    int dr, int dc,
    int player,
):
    """Sum of all size-6 window scores along direction (dr,dc) that touch (r,c)."""
    cdef int total = 0
    cdef int line_buf[32]
    cdef int line_len = 0

    # Walk to the start of this line (board edge in the negative direction)
    cdef int sr = r, sc = c
    while 0 <= sr - dr < BOARD_SIZE and 0 <= sc - dc < BOARD_SIZE:
        sr -= dr
        sc -= dc

    # Fill the line buffer and find the position of (r,c) within it
    cdef int pos_in_line = -1
    cdef int rr = sr, cc = sc
    while 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE:
        if rr == r and cc == c:
            pos_in_line = line_len
        line_buf[line_len] = <int>board[rr, cc]
        line_len += 1
        rr += dr
        cc += dc

    if pos_in_line < 0 or line_len < 6:
        return 0

    # Only score the (up to 6) windows that contain pos_in_line
    cdef int k_start = pos_in_line - 5
    cdef int k_end   = pos_in_line
    if k_start < 0:
        k_start = 0
    if k_end > line_len - 6:
        k_end = line_len - 6

    cdef int k
    for k in range(k_start, k_end + 1):
        total += _score_window6_values(
            line_buf[k],   line_buf[k+1], line_buf[k+2],
            line_buf[k+3], line_buf[k+4], line_buf[k+5],
            player,
        )
    return total


cdef int _score_cell_all_dirs(
    cnp.ndarray[cnp.int64_t, ndim=2] board,
    int r, int c,
    int player,
):
    """Sum over all 4 directions of windows touching (r,c)."""
    return (
        _score_cell_direction(board, r, c, 0,  1, player) +
        _score_cell_direction(board, r, c, 1,  0, player) +
        _score_cell_direction(board, r, c, 1,  1, player) +
        _score_cell_direction(board, r, c, 1, -1, player)
    )


def score_cell_delta(
    board: np.ndarray,
    int r, int c,
    int player,
) -> int:
    """
    Public helper: score contribution of cell (r,c) for `player` across all
    directions.  Call before and after placing a stone; the difference is the
    incremental score change.
    """
    return _score_cell_all_dirs(board, r, c, player)


# ===========================================================================
# Incremental capture delta
#
# Only check the (up to 4*2) four-cell patterns that include (r,c).
# ===========================================================================

cdef int _capture_delta_cell(
    cnp.ndarray[cnp.int64_t, ndim=2] board,
    int r, int c,
    int player,
):
    """
    Potential-capture score contribution of patterns that include (r,c).
    Mirrors the logic of _detect_potential_captures_fast but restricted to
    windows containing (r,c).
    """
    cdef int opponent = 2 if player == 1 else 1
    cdef int score = 0
    cdef int dr, dc, d, i
    cdef int nr, nc, a, b, dd, e
    cdef int dirs_r[4]
    cdef int dirs_c[4]
    cdef int buf[7]
    cdef int buf_len, pos, k

    dirs_r[0] = 0;  dirs_c[0] = 1
    dirs_r[1] = 1;  dirs_c[1] = 0
    dirs_r[2] = 1;  dirs_c[2] = 1
    dirs_r[3] = 1;  dirs_c[3] = -1

    for d in range(4):
        dr = dirs_r[d]
        dc = dirs_c[d]

        # Build a 7-cell neighbourhood centred on (r,c): positions -3..+3
        buf_len = 0
        pos = -1
        for i in range(-3, 4):
            nr = r + i * dr
            nc = c + i * dc
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                if nr == r and nc == c:
                    pos = buf_len
                buf[buf_len] = <int>board[nr, nc]
                buf_len += 1

        if pos < 0:
            continue

        # Check every 4-cell window within buf that contains pos
        for k in range(buf_len - 3):
            if not (k <= pos <= k + 3):
                continue
            a = buf[k]; b = buf[k+1]; dd = buf[k+2]; e = buf[k+3]
            if a == EMPTY and b == opponent and dd == opponent and e == EMPTY:
                score -= 500
            if a == player and b == opponent and dd == opponent and e == player:
                score += 2000

    return score


def capture_delta_cell(
    board: np.ndarray,
    int r, int c,
    int player,
) -> int:
    """Public wrapper for incremental capture delta at (r,c)."""
    return _capture_delta_cell(board, r, c, player)


# ===========================================================================
# Full-board scorers (kept for initialisation / fallback)
# ===========================================================================

cdef int _score_lines_fast(cnp.ndarray[cnp.int64_t, ndim=2] board, int player):
    cdef int total = 0
    cdef int r, c, i, length
    cdef int rr, cc
    cdef int line_buf[32]

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE - 5):
            total += _score_window6_values(
                <int>board[r, c],   <int>board[r, c+1], <int>board[r, c+2],
                <int>board[r, c+3], <int>board[r, c+4], <int>board[r, c+5],
                player,
            )

    for c in range(BOARD_SIZE):
        for r in range(BOARD_SIZE - 5):
            total += _score_window6_values(
                <int>board[r, c],   <int>board[r+1, c], <int>board[r+2, c],
                <int>board[r+3, c], <int>board[r+4, c], <int>board[r+5, c],
                player,
            )

    for c in range(BOARD_SIZE):
        rr = 0; cc = c
        length = BOARD_SIZE - c
        if length >= 6:
            for i in range(length):
                line_buf[i] = <int>board[rr+i, cc+i]
            for i in range(length - 5):
                total += _score_window6_values(
                    line_buf[i], line_buf[i+1], line_buf[i+2],
                    line_buf[i+3], line_buf[i+4], line_buf[i+5],
                    player,
                )
    for r in range(1, BOARD_SIZE):
        rr = r; cc = 0
        length = BOARD_SIZE - r
        if length >= 6:
            for i in range(length):
                line_buf[i] = <int>board[rr+i, cc+i]
            for i in range(length - 5):
                total += _score_window6_values(
                    line_buf[i], line_buf[i+1], line_buf[i+2],
                    line_buf[i+3], line_buf[i+4], line_buf[i+5],
                    player,
                )

    for c in range(BOARD_SIZE):
        rr = 0; cc = c
        length = c + 1
        if length >= 6:
            for i in range(length):
                line_buf[i] = <int>board[rr+i, cc-i]
            for i in range(length - 5):
                total += _score_window6_values(
                    line_buf[i], line_buf[i+1], line_buf[i+2],
                    line_buf[i+3], line_buf[i+4], line_buf[i+5],
                    player,
                )
    for r in range(1, BOARD_SIZE):
        rr = r; cc = BOARD_SIZE - 1
        length = BOARD_SIZE - r
        if length >= 6:
            for i in range(length):
                line_buf[i] = <int>board[rr+i, cc-i]
            for i in range(length - 5):
                total += _score_window6_values(
                    line_buf[i], line_buf[i+1], line_buf[i+2],
                    line_buf[i+3], line_buf[i+4], line_buf[i+5],
                    player,
                )

    return total


cdef int _detect_potential_captures_fast(cnp.ndarray[cnp.int64_t, ndim=2] board, int player):
    cdef int opponent = 2 if player == 1 else 1
    cdef int potential_capture_score = 0
    cdef int r, c, i, j, k, length
    cdef int rr, cc
    cdef int a, b, d, e
    cdef int line_buf[32]

    for r in range(BOARD_SIZE):
        for i in range(BOARD_SIZE):
            line_buf[i] = <int>board[r, i]
        for i in range(BOARD_SIZE - 4):
            for j in range(2):
                k = i + j
                if k + 3 >= BOARD_SIZE:
                    continue
                a = line_buf[k]; b = line_buf[k+1]; d = line_buf[k+2]; e = line_buf[k+3]
                if a == EMPTY and b == opponent and d == opponent and e == EMPTY:
                    potential_capture_score -= 500
                if a == player and b == opponent and d == opponent and e == player:
                    potential_capture_score += 2000

    for c in range(BOARD_SIZE):
        for i in range(BOARD_SIZE):
            line_buf[i] = <int>board[i, c]
        for i in range(BOARD_SIZE - 4):
            for j in range(2):
                k = i + j
                if k + 3 >= BOARD_SIZE:
                    continue
                a = line_buf[k]; b = line_buf[k+1]; d = line_buf[k+2]; e = line_buf[k+3]
                if a == EMPTY and b == opponent and d == opponent and e == EMPTY:
                    potential_capture_score -= 500
                if a == player and b == opponent and d == opponent and e == player:
                    potential_capture_score += 2000

    for c in range(BOARD_SIZE):
        rr = 0; cc = c; length = BOARD_SIZE - c
        for i in range(length):
            line_buf[i] = <int>board[rr+i, cc+i]
        if length >= 5:
            for i in range(length - 4):
                for j in range(2):
                    k = i + j
                    if k + 3 >= length: continue
                    a = line_buf[k]; b = line_buf[k+1]; d = line_buf[k+2]; e = line_buf[k+3]
                    if a == EMPTY and b == opponent and d == opponent and e == EMPTY:
                        potential_capture_score -= 500
                    if a == player and b == opponent and d == opponent and e == player:
                        potential_capture_score += 2000
    for r in range(1, BOARD_SIZE):
        rr = r; cc = 0; length = BOARD_SIZE - r
        for i in range(length):
            line_buf[i] = <int>board[rr+i, cc+i]
        if length >= 5:
            for i in range(length - 4):
                for j in range(2):
                    k = i + j
                    if k + 3 >= length: continue
                    a = line_buf[k]; b = line_buf[k+1]; d = line_buf[k+2]; e = line_buf[k+3]
                    if a == EMPTY and b == opponent and d == opponent and e == EMPTY:
                        potential_capture_score -= 500
                    if a == player and b == opponent and d == opponent and e == player:
                        potential_capture_score += 2000

    for c in range(BOARD_SIZE):
        rr = 0; cc = c; length = c + 1
        for i in range(length):
            line_buf[i] = <int>board[rr+i, cc-i]
        if length >= 5:
            for i in range(length - 4):
                for j in range(2):
                    k = i + j
                    if k + 3 >= length: continue
                    a = line_buf[k]; b = line_buf[k+1]; d = line_buf[k+2]; e = line_buf[k+3]
                    if a == EMPTY and b == opponent and d == opponent and e == EMPTY:
                        potential_capture_score -= 500
                    if a == player and b == opponent and d == opponent and e == player:
                        potential_capture_score += 2000
    for r in range(1, BOARD_SIZE):
        rr = r; cc = BOARD_SIZE - 1; length = BOARD_SIZE - r
        for i in range(length):
            line_buf[i] = <int>board[rr+i, cc-i]
        if length >= 5:
            for i in range(length - 4):
                for j in range(2):
                    k = i + j
                    if k + 3 >= length: continue
                    a = line_buf[k]; b = line_buf[k+1]; d = line_buf[k+2]; e = line_buf[k+3]
                    if a == EMPTY and b == opponent and d == opponent and e == EMPTY:
                        potential_capture_score -= 500
                    if a == player and b == opponent and d == opponent and e == player:
                        potential_capture_score += 2000

    return potential_capture_score


# ===========================================================================
# IncrementalScorer — stateful object that maintains running scores
# ===========================================================================

class IncrementalScorer:
    """
    Maintains running pattern scores for both players so that evaluate_board()
    only needs to update the ~24 windows affected by the last move rather than
    scanning the entire 19×19 board.

    Usage
    -----
    scorer = IncrementalScorer(board)          # O(board) — once at game start

    # Before make_move:
    scorer.before_move(board, r, c, player)

    # After updating board:
    board[r, c] = player
    scorer.after_move(board, r, c, player)

    # Undo (before restoring board):
    scorer.before_move(board, r, c, player)    # same call — subtracts new windows
    board[r, c] = EMPTY
    scorer.after_move(board, r, c, player)     # re-adds old (now restored) windows

    # Evaluate:
    total = scorer.evaluate(player, p1_caps, p2_caps)
    """

    def __init__(self, board: np.ndarray):
        # Full-board initialisation — called once
        self._score = [0, 0, 0]  # index 1 = player1, 2 = player2
        self._capture_score = [0, 0, 0]
        for p in (1, 2):
            self._score[p] = int(_score_lines_fast(board, p))
            self._capture_score[p] = int(_detect_potential_captures_fast(board, p))

    def before_move(self, board: np.ndarray, r: int, c: int, player: int):
        """
        Subtract the current contribution of cell (r,c) from running scores.
        Call this BEFORE modifying the board.
        """
        opponent = 2 if player == 1 else 1
        for p in (player, opponent):
            self._score[p]         -= _score_cell_all_dirs(board, r, c, p)
            self._capture_score[p] -= _capture_delta_cell(board, r, c, p)

    def after_move(self, board: np.ndarray, r: int, c: int, player: int):
        """
        Add the new contribution of cell (r,c) to running scores.
        Call this AFTER modifying the board.
        """
        opponent = 2 if player == 1 else 1
        for p in (player, opponent):
            self._score[p]         += _score_cell_all_dirs(board, r, c, p)
            self._capture_score[p] += _capture_delta_cell(board, r, c, p)

    def evaluate(self, player: int, player1_captures: int, player2_captures: int) -> int:
        """
        Return the heuristic score from `player`'s perspective.
        Equivalent to the old evaluate_board() but O(1).
        """
        opponent = 2 if player == 1 else 1
        p_caps = player1_captures if player == 1 else player2_captures
        o_caps = player2_captures if player == 1 else player1_captures

        pattern_score   = self._score[player]   - self._score[opponent]
        potential_score = self._capture_score[player]
        cap_score       = score_captures(p_caps, o_caps)

        return pattern_score + cap_score + potential_score

    def full_resync(self, board: np.ndarray):
        """
        Recompute scores from scratch. Call after bulk board changes (e.g.
        after a capture removes stones) or as a periodic sanity check.
        """
        for p in (1, 2):
            self._score[p]         = int(_score_lines_fast(board, p))
            self._capture_score[p] = int(_detect_potential_captures_fast(board, p))


# ===========================================================================
# Candidate move generator
# ===========================================================================

def get_candidates(board: np.ndarray, int radius=2) -> list:
    """
    Return empty cells within `radius` steps of any occupied cell.
    Reduces the branching factor from ~300 to ~20-40 moves per turn.
    """
    cdef int r, c, dr, dc, nr, nc
    candidates = set()
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r, c] != EMPTY:
                for dr in range(-radius, radius + 1):
                    for dc in range(-radius, radius + 1):
                        nr = r + dr
                        nc = c + dc
                        if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                            if board[nr, nc] == EMPTY:
                                candidates.add((nr, nc))
    return list(candidates)


# ===========================================================================
# Public API — existing functions preserved for backward compatibility
# ===========================================================================

def get_pattern_freedom(window: np.ndarray, pattern_start: int, pattern_length: int, player: int) -> str:
    cdef int opponent = 2 if player == 1 else 1
    cdef int pattern_end = pattern_start + pattern_length - 1
    cdef bint left_blocked  = pattern_start > 0 and window[pattern_start - 1] == opponent
    cdef bint right_blocked = pattern_end < len(window) - 1 and window[pattern_end + 1] == opponent
    if left_blocked and right_blocked:
        return 'flanked'
    elif left_blocked or right_blocked:
        return 'half_free'
    else:
        return 'free'


def has_space_to_develop(window: np.ndarray, pattern_start: int, pattern_length: int, player: int) -> bool:
    return get_pattern_freedom(window, pattern_start, pattern_length, player) != 'flanked'


def score_window(window: np.ndarray, player: int) -> int:
    cdef int n = len(window)
    if n == 6:
        return _score_window6_values(
            <int>window[0], <int>window[1], <int>window[2],
            <int>window[3], <int>window[4], <int>window[5],
            player,
        )
    cdef int i, score = 0, opponent = 2 if player == 1 else 1
    cdef int player_count = 0, opponent_count = 0, empty_count = 0, freedom_code
    for i in range(n):
        if window[i] == player:       player_count   += 1
        elif window[i] == opponent:   opponent_count += 1
        elif window[i] == EMPTY:      empty_count    += 1
    if player_count == 5:
        return 10_000_000

    if opponent_count > 0 and player_count < 4:
        return 0

    if opponent_count > 1:
        return 0

    if player_count == 4:
        for i in range(n - 3):
            if window[i] == player and window[i+1] == player and window[i+2] == player and window[i+3] == player:
                freedom_code = _freedom_code_from_values(
                    opponent if i > 0 and window[i-1] == opponent else EMPTY,
                    opponent if i+3 < n-1 and window[i+4] == opponent else EMPTY,
                    opponent,
                )
                if freedom_code != 2:
                    score += _scaled_score(SCORE_CLOSED_FOUR if empty_count == 1 else SCORE_OPEN_FOUR, freedom_code)
                break
    elif player_count == 3:
        for i in range(n - 2):
            if window[i] == player and window[i+1] == player and window[i+2] == player:
                freedom_code = _freedom_code_from_values(
                    opponent if i > 0 and window[i-1] == opponent else EMPTY,
                    opponent if i+2 < n-1 and window[i+3] == opponent else EMPTY,
                    opponent,
                )
                if freedom_code != 2:
                    if empty_count == 2:   score += _scaled_score(SCORE_OPEN_THREE, freedom_code)
                    elif empty_count == 1: score += _scaled_score(SCORE_CLOSED_THREE, freedom_code)
                break
    elif player_count == 2:
        for i in range(n - 1):
            if window[i] == player and window[i+1] == player:
                freedom_code = _freedom_code_from_values(
                    opponent if i > 0 and window[i-1] == opponent else EMPTY,
                    opponent if i+1 < n-1 and window[i+2] == opponent else EMPTY,
                    opponent,
                )
                if freedom_code != 2:
                    if empty_count == 3:   score += _scaled_score(SCORE_OPEN_TWO, freedom_code)
                    elif empty_count == 2: score += _scaled_score(SCORE_CLOSED_TWO, freedom_code)
                break
    elif player_count == 1:
        score += 10
    return score


def get_lines(board: np.ndarray) -> list:
    lines = []
    for r in range(BOARD_SIZE):
        lines.append(board[r, :])
    for c in range(BOARD_SIZE):
        lines.append(board[:, c])
    for offset in range(-(BOARD_SIZE - 5), BOARD_SIZE - 4):
        lines.append(np.diag(board, offset))
    flipped = np.fliplr(board)
    for offset in range(-(BOARD_SIZE - 5), BOARD_SIZE - 4):
        lines.append(np.diag(flipped, offset))
    return lines


def score_lines(board: np.ndarray, player: int) -> int:
    """Full-board line scorer. Prefer IncrementalScorer for search loops."""
    return _score_lines_fast(board, player)


def detect_potential_captures(board: np.ndarray, player: int) -> int:
    """Full-board capture detector. Prefer IncrementalScorer for search loops."""
    return _detect_potential_captures_fast(board, player)


def score_captures(player_captures: int, opponent_captures: int) -> int:
    score = 0
    score += player_captures * SCORE_CAPTURE
    if player_captures >= 8:
        score += 500_000
    elif player_captures >= 6:
        score += 50_000
    score -= opponent_captures * SCORE_CAPTURE
    if opponent_captures >= 8:
        score -= 500_000
    elif opponent_captures >= 6:
        score -= 50_000
    return score

cdef int compteur_heuristique = 0

def evaluate_board(
    board: np.ndarray,
    player: int,
    player1_captures: int,
    player2_captures: int,
) -> float:
    """
    Full-board evaluate. Use IncrementalScorer.evaluate() inside search loops
    instead — this is kept for one-off calls and testing.
    """
    cdef int opponent = 2 if player == 1 else 1
    cdef int player_score   = _score_lines_fast(board, player)
    cdef int opponent_score = _score_lines_fast(board, opponent)
    cdef int p_caps = player1_captures if player == 1 else player2_captures
    cdef int o_caps = player2_captures if player == 1 else player1_captures
    cdef int capture_score   = score_captures(p_caps, o_caps)
    cdef int potential_score = _detect_potential_captures_fast(board, player)
    # print score 1 out of every 100 times
    global compteur_heuristique
    compteur_heuristique += 1
    # if compteur_heuristique % 1000 == 0:
    #print(f"Player Score: {player_score}, Opponent Score: {opponent_score}, Capture Score: {capture_score}, Potential Score: {potential_score}. Returning total: {(player_score - opponent_score) + capture_score + potential_score}")

    return (player_score - opponent_score) + capture_score + potential_score
