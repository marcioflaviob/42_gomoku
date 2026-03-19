# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, initializedcheck=False

cimport numpy as cnp
import numpy as np

cnp.import_array()

DEF EMPTY      = 0
DEF BOARD_SIZE = 19

cdef int C_OPEN4   = 100000
cdef int C_CLOSE4  = 10000
cdef int C_OPEN3   = 5000
cdef int C_CLOSE3  = 500
cdef int C_OPEN2   = 200
cdef int C_CLOSE2  = 50
cdef int C_CAPTURE = 3000

cdef int DIRS_R[4]
cdef int DIRS_C[4]
DIRS_R[0] = 0;  DIRS_C[0] = 1
DIRS_R[1] = 1;  DIRS_C[1] = 0
DIRS_R[2] = 1;  DIRS_C[2] = 1
DIRS_R[3] = 1;  DIRS_C[3] = -1

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
    cdef int opponent = 3 - player
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


cdef int _score_4_lines(cnp.int64_t[:, :] board, int player,
                         int row, int col) nogil:
    """Score every 6-window in the 4 lines that pass through (row, col)."""
    cdef int total = 0
    cdef int d, dr, dc, rr, cc, llen, i
    cdef int line_buf[BOARD_SIZE]

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

        if llen >= 4:
            for i in range(llen - 3):
                a = line_buf[i]; b = line_buf[i+1]
                dd = line_buf[i+2]; e = line_buf[i+3]
                if a == EMPTY and b == opponent and dd == opponent and e == EMPTY:
                    total -= 500
                elif a == player and b == opponent and dd == opponent and e == player:
                    total += 2000

    return total

# SCore full board
cdef int _score_lines_fast(cnp.int64_t[:, :] board, int player):
    cdef int total = 0
    cdef int r, c, i, length, rr, cc
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

    # Diagonal
    for c in range(BOARD_SIZE):
        rr = 0; cc = c; length = BOARD_SIZE - c
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
        rr = r; cc = 0; length = BOARD_SIZE - r
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
        rr = 0; cc = c; length = c + 1
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
        rr = r; cc = BOARD_SIZE - 1; length = BOARD_SIZE - r
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
    cdef int r, c, i, length, rr, cc
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

# Public
cpdef int score_4_lines_for_player(cnp.int64_t[:, :] board, int player,
                                    int row, int col):
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
    All subsequent evaluations inside minimax use O(1) deltas.
    """
    cdef int opponent  = 3 - player
    cdef int p_lines   = _score_lines_fast(board, player)
    cdef int o_lines   = _score_lines_fast(board, opponent)
    cdef int cap_score = _score_captures_c(p_caps, o_caps)
    cdef int pot_score = _detect_potential_captures_fast(board, player)
    return <double>(p_lines - o_lines + cap_score + pot_score)
