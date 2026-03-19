# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, initializedcheck=False

import numpy as np
cimport numpy as cnp

cnp.import_array()

# ── Compile-time constants ─────────────────────────────────────────────────────
# DEF makes these true C literals — the compiler substitutes the value
# everywhere, including loop bounds, removing all runtime loads.
DEF EMPTY      = 0
DEF BOARD_SIZE = 19

# ── Python-level score constants (kept for external API callers) ──────────────
SCORE_OPEN_FOUR    = 100_000
SCORE_CLOSED_FOUR  = 10_000
SCORE_OPEN_THREE   = 5_000
SCORE_CLOSED_THREE = 500
SCORE_OPEN_TWO     = 200
SCORE_CLOSED_TWO   = 50
SCORE_CAPTURE      = 3_000

# ── C-level score constants ────────────────────────────────────────────────────
# These are used in cdef/nogil hot paths.  Module-level cdef ints are plain C
# globals — one load, no Python object overhead, no GIL required.
cdef int C_OPEN4   = 100000
cdef int C_CLOSE4  = 10000
cdef int C_OPEN3   = 5000
cdef int C_CLOSE3  = 500
cdef int C_OPEN2   = 200
cdef int C_CLOSE2  = 50
cdef int C_CAPTURE = 3000

# ── Direction tables (H, V, diag, anti-diag) ─────────────────────────────────
cdef int DIRS_R[4]
cdef int DIRS_C[4]
DIRS_R[0] = 0;  DIRS_C[0] = 1   # horizontal
DIRS_R[1] = 1;  DIRS_C[1] = 0   # vertical
DIRS_R[2] = 1;  DIRS_C[2] = 1   # diagonal
DIRS_R[3] = 1;  DIRS_C[3] = -1  # anti-diagonal


# ===========================================================================
# Low-level scoring helpers  (all nogil — pure C arithmetic)
# ===========================================================================

cdef inline int _scaled_score(int base_score, int freedom_code) nogil:
    if freedom_code == 0:
        return (base_score * 3) // 2
    elif freedom_code == 1:
        return base_score
    return (base_score * 3) // 10


cdef inline int _freedom_code_from_values(int left_value, int right_value,
                                           int opponent) nogil:
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
) nogil:
    # Using C_* constants and DEF EMPTY means this function is fully nogil.
    cdef int opponent = 3 - player   # faster than "2 if player==1 else 1"
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
        else:
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
                    return _scaled_score(C_CLOSE4, freedom_code)
                return _scaled_score(C_OPEN4, freedom_code)
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
                    return _scaled_score(C_OPEN3, freedom_code)
                elif empty_count == 1:
                    return _scaled_score(C_CLOSE3, freedom_code)
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
                    return _scaled_score(C_OPEN2, freedom_code)
                elif empty_count == 2:
                    return _scaled_score(C_CLOSE2, freedom_code)
                return 0
        return 0

    elif player_count == 1:
        return 10

    return 0


cdef inline int _score_captures_c(int p_caps, int o_caps) nogil:
    """C-level capture-count scorer (no Python objects, no GIL)."""
    cdef int score = p_caps * C_CAPTURE - o_caps * C_CAPTURE
    if p_caps >= 8:
        score += 500000
    elif p_caps >= 6:
        score += 50000
    if o_caps >= 8:
        score -= 500000
    elif o_caps >= 6:
        score -= 50000
    return score


# ===========================================================================
# Incremental 4-line scorers
# ===========================================================================
# When a stone is placed at (row, col), only windows that lie on one of the
# 4 lines through that cell can change.  Captures are always inline with the
# placed stone, so those 4 lines cover ALL board changes.
#
# Re-scoring the full content of those 4 lines (instead of scanning all 361
# cells) reduces leaf-node work from ~400 window evaluations to ~56, giving
# a ~7× speed-up in the evaluation hot path.

cdef int _score_4_lines(cnp.int64_t[:, :] board, int player,
                         int row, int col) nogil:
    """Score every 6-window in the 4 lines that pass through (row, col)."""
    cdef int total = 0
    cdef int d, dr, dc, rr, cc, llen, i
    cdef int line_buf[BOARD_SIZE]  # BOARD_SIZE is a DEF — exact size, no waste

    for d in range(4):
        dr = DIRS_R[d]
        dc = DIRS_C[d]

        # Walk backwards to the start of this line
        rr = row; cc = col
        while (rr - dr >= 0 and rr - dr < BOARD_SIZE and
               cc - dc >= 0 and cc - dc < BOARD_SIZE):
            rr -= dr; cc -= dc

        # Extract the full line into a stack-allocated C array
        llen = 0
        while 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE:
            line_buf[llen] = <int>board[rr, cc]
            llen += 1
            rr += dr; cc += dc

        # Score every window of 6 in this line
        if llen >= 6:
            for i in range(llen - 5):
                total += _score_window6_values(
                    line_buf[i],   line_buf[i+1], line_buf[i+2],
                    line_buf[i+3], line_buf[i+4], line_buf[i+5],
                    player,
                )

    return total


cdef int _capture_score_4_lines(cnp.int64_t[:, :] board, int player,
                                  int row, int col) nogil:
    """Compute capture-potential score for the 4 lines through (row, col)."""
    cdef int opponent = 3 - player
    cdef int total = 0
    cdef int d, dr, dc, rr, cc, llen, i
    cdef int line_buf[BOARD_SIZE]
    cdef int a, b, dd, e

    for d in range(4):
        dr = DIRS_R[d]; dc = DIRS_C[d]
        rr = row; cc = col
        while (rr - dr >= 0 and rr - dr < BOARD_SIZE and
               cc - dc >= 0 and cc - dc < BOARD_SIZE):
            rr -= dr; cc -= dc

        llen = 0
        while 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE:
            line_buf[llen] = <int>board[rr, cc]
            llen += 1
            rr += dr; cc += dc

        # Check every 4-window for capture patterns (each window checked once)
        if llen >= 4:
            for i in range(llen - 3):
                a = line_buf[i]; b = line_buf[i+1]
                dd = line_buf[i+2]; e = line_buf[i+3]
                if a == EMPTY and b == opponent and dd == opponent and e == EMPTY:
                    total -= 500
                elif a == player and b == opponent and dd == opponent and e == player:
                    total += 2000

    return total


# ===========================================================================
# Full-board scorers  (used only once per search: to seed board_score)
# ===========================================================================

cdef int _score_lines_fast(cnp.int64_t[:, :] board, int player):
    """Scan the entire board; prefer _score_4_lines for incremental updates."""
    cdef int total = 0
    cdef int r, c, i, length
    cdef int rr, cc
    cdef int line_buf[BOARD_SIZE]

    # Horizontal
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE - 5):
            total += _score_window6_values(
                <int>board[r, c],   <int>board[r, c+1], <int>board[r, c+2],
                <int>board[r, c+3], <int>board[r, c+4], <int>board[r, c+5],
                player,
            )

    # Vertical
    for c in range(BOARD_SIZE):
        for r in range(BOARD_SIZE - 5):
            total += _score_window6_values(
                <int>board[r,   c], <int>board[r+1, c], <int>board[r+2, c],
                <int>board[r+3, c], <int>board[r+4, c], <int>board[r+5, c],
                player,
            )

    # Diagonal top-right (col offset)
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

    # Anti-diagonal
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


cdef int _detect_potential_captures_fast(cnp.int64_t[:, :] board, int player):
    cdef int opponent = 3 - player
    cdef int potential_capture_score = 0
    cdef int r, c, i, length
    cdef int rr, cc
    cdef int a, b, d, e
    cdef int line_buf[BOARD_SIZE]

    # Horizontal
    for r in range(BOARD_SIZE):
        for i in range(BOARD_SIZE):
            line_buf[i] = <int>board[r, i]
        for i in range(BOARD_SIZE - 3):
            a = line_buf[i]; b = line_buf[i+1]; d = line_buf[i+2]; e = line_buf[i+3]
            if a == EMPTY and b == opponent and d == opponent and e == EMPTY:
                potential_capture_score -= 500
            elif a == player and b == opponent and d == opponent and e == player:
                potential_capture_score += 2000

    # Vertical
    for c in range(BOARD_SIZE):
        for i in range(BOARD_SIZE):
            line_buf[i] = <int>board[i, c]
        for i in range(BOARD_SIZE - 3):
            a = line_buf[i]; b = line_buf[i+1]; d = line_buf[i+2]; e = line_buf[i+3]
            if a == EMPTY and b == opponent and d == opponent and e == EMPTY:
                potential_capture_score -= 500
            elif a == player and b == opponent and d == opponent and e == player:
                potential_capture_score += 2000

    # Diagonal
    for c in range(BOARD_SIZE):
        rr = 0; cc = c; length = BOARD_SIZE - c
        for i in range(length):
            line_buf[i] = <int>board[rr+i, cc+i]
        if length >= 4:
            for i in range(length - 3):
                a = line_buf[i]; b = line_buf[i+1]; d = line_buf[i+2]; e = line_buf[i+3]
                if a == EMPTY and b == opponent and d == opponent and e == EMPTY:
                    potential_capture_score -= 500
                elif a == player and b == opponent and d == opponent and e == player:
                    potential_capture_score += 2000
    for r in range(1, BOARD_SIZE):
        rr = r; cc = 0; length = BOARD_SIZE - r
        for i in range(length):
            line_buf[i] = <int>board[rr+i, cc+i]
        if length >= 4:
            for i in range(length - 3):
                a = line_buf[i]; b = line_buf[i+1]; d = line_buf[i+2]; e = line_buf[i+3]
                if a == EMPTY and b == opponent and d == opponent and e == EMPTY:
                    potential_capture_score -= 500
                elif a == player and b == opponent and d == opponent and e == player:
                    potential_capture_score += 2000

    # Anti-diagonal
    for c in range(BOARD_SIZE):
        rr = 0; cc = c; length = c + 1
        for i in range(length):
            line_buf[i] = <int>board[rr+i, cc-i]
        if length >= 4:
            for i in range(length - 3):
                a = line_buf[i]; b = line_buf[i+1]; d = line_buf[i+2]; e = line_buf[i+3]
                if a == EMPTY and b == opponent and d == opponent and e == EMPTY:
                    potential_capture_score -= 500
                elif a == player and b == opponent and d == opponent and e == player:
                    potential_capture_score += 2000
    for r in range(1, BOARD_SIZE):
        rr = r; cc = BOARD_SIZE - 1; length = BOARD_SIZE - r
        for i in range(length):
            line_buf[i] = <int>board[rr+i, cc-i]
        if length >= 4:
            for i in range(length - 3):
                a = line_buf[i]; b = line_buf[i+1]; d = line_buf[i+2]; e = line_buf[i+3]
                if a == EMPTY and b == opponent and d == opponent and e == EMPTY:
                    potential_capture_score -= 500
                elif a == player and b == opponent and d == opponent and e == player:
                    potential_capture_score += 2000

    return potential_capture_score


# ===========================================================================
# Public cpdef exports for minimax.pyx
# ===========================================================================

cpdef int score_4_lines_for_player(cnp.int64_t[:, :] board, int player,
                                    int row, int col):
    """Thin cpdef wrapper so minimax.pyx can call the nogil cdef."""
    return _score_4_lines(board, player, row, col)


cpdef int capture_score_4_lines_for_player(cnp.int64_t[:, :] board, int player,
                                            int row, int col):
    return _capture_score_4_lines(board, player, row, col)


cpdef int score_captures_fast(int p_caps, int o_caps):
    return _score_captures_c(p_caps, o_caps)


cpdef double evaluate_board_full_mv(cnp.int64_t[:, :] board, int player,
                                     int p_caps, int o_caps):
    """
    Full-board evaluation using a typed memoryview.
    Called ONCE per search root to seed the incremental board_score.
    """
    cdef int opponent = 3 - player
    cdef int p_lines   = _score_lines_fast(board, player)
    cdef int o_lines   = _score_lines_fast(board, opponent)
    cdef int cap_score = _score_captures_c(p_caps, o_caps)
    cdef int pot_score = _detect_potential_captures_fast(board, player)
    return <double>(p_lines - o_lines + cap_score + pot_score)


# ===========================================================================
# Candidate move generator
# ===========================================================================

def get_candidates(cnp.int64_t[:, :] board, int radius=2) -> list:
    """
    Return empty cells within `radius` steps of any occupied cell.
    Uses a typed memoryview for fast cell reads (no Python __getitem__).
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


def get_pattern_freedom(window, int pattern_start, int pattern_length,
                         int player) -> str:
    cdef int opponent    = 3 - player
    cdef int pattern_end = pattern_start + pattern_length - 1
    cdef bint left_blocked  = (pattern_start > 0 and
                                window[pattern_start - 1] == opponent)
    cdef bint right_blocked = (pattern_end < len(window) - 1 and
                                window[pattern_end + 1] == opponent)
    if left_blocked and right_blocked:
        return 'flanked'
    elif left_blocked or right_blocked:
        return 'half_free'
    else:
        return 'free'


def has_space_to_develop(window, int pattern_start, int pattern_length,
                          int player) -> bool:
    return get_pattern_freedom(window, pattern_start, pattern_length,
                               player) != 'flanked'


def score_window(window, int player) -> int:
    cdef int n = len(window)
    if n == 6:
        return _score_window6_values(
            <int>window[0], <int>window[1], <int>window[2],
            <int>window[3], <int>window[4], <int>window[5],
            player,
        )
    cdef int i, score = 0, opponent = 3 - player
    cdef int player_count = 0, opponent_count = 0, empty_count = 0, freedom_code
    for i in range(n):
        if window[i] == player:       player_count   += 1
        elif window[i] == opponent:   opponent_count += 1
        else:                         empty_count    += 1
    if player_count == 5:
        return 10_000_000

    if opponent_count > 0 and player_count < 4:
        return 0
    if opponent_count > 1:
        return 0

    if player_count == 4:
        for i in range(n - 3):
            if (window[i] == player and window[i+1] == player and
                    window[i+2] == player and window[i+3] == player):
                freedom_code = _freedom_code_from_values(
                    opponent if i > 0 and window[i-1] == opponent else EMPTY,
                    opponent if i+3 < n-1 and window[i+4] == opponent else EMPTY,
                    opponent,
                )
                if freedom_code != 2:
                    score += _scaled_score(
                        C_CLOSE4 if empty_count == 1 else C_OPEN4,
                        freedom_code,
                    )
                break
    elif player_count == 3:
        for i in range(n - 2):
            if (window[i] == player and window[i+1] == player and
                    window[i+2] == player):
                freedom_code = _freedom_code_from_values(
                    opponent if i > 0 and window[i-1] == opponent else EMPTY,
                    opponent if i+2 < n-1 and window[i+3] == opponent else EMPTY,
                    opponent,
                )
                if freedom_code != 2:
                    if empty_count == 2:   score += _scaled_score(C_OPEN3, freedom_code)
                    elif empty_count == 1: score += _scaled_score(C_CLOSE3, freedom_code)
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
                    if empty_count == 3:   score += _scaled_score(C_OPEN2, freedom_code)
                    elif empty_count == 2: score += _scaled_score(C_CLOSE2, freedom_code)
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


def score_lines(board: np.ndarray, int player) -> int:
    """Full-board line scorer. Prefer evaluate_board_full_mv for search loops."""
    cdef cnp.int64_t[:, :] bv = board
    return _score_lines_fast(bv, player)


def detect_potential_captures(board: np.ndarray, int player) -> int:
    """Full-board capture detector. Prefer evaluate_board_full_mv for search loops."""
    cdef cnp.int64_t[:, :] bv = board
    return _detect_potential_captures_fast(bv, player)


def score_captures(int player_captures, int opponent_captures) -> int:
    return _score_captures_c(player_captures, opponent_captures)


def evaluate_board(
    board: np.ndarray,
    int player,
    int player1_captures,
    int player2_captures,
) -> float:
    """
    Full-board evaluate.  Use evaluate_board_full_mv inside search loops
    (typed memoryview, called once to seed board_score) — this is kept for
    one-off calls and testing.
    """
    cdef cnp.int64_t[:, :] bv = board
    cdef int p_caps = player1_captures if player == 1 else player2_captures
    cdef int o_caps = player2_captures if player == 1 else player1_captures
    return evaluate_board_full_mv(bv, player, p_caps, o_caps)
