# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, initializedcheck=False

from ai.moves cimport apply_capture, check_win, check_double_three
from ai.heuristics cimport evaluate_board_full_mv
cimport numpy as cnp
from libc.stdlib cimport rand, srand, RAND_MAX
from libc.time cimport time as c_time
import time
import numpy as np

INF = float('inf')
compteur_heuristique = 0

cnp.import_array()

cdef unsigned long long ZOBRIST_TABLE[19][19][2]
cdef bint zobrist_initialized = False
cdef dict transposition_table = {}

cdef int FLAG_EXACT = 0
cdef int FLAG_ALPHA = 1
cdef int FLAG_BETA  = 2

# ── C-level score constants (mirrors heuristics.pyx — avoids cross-module call) ─
cdef int _MM_C_CAPTURE = 3000


# ===========================================================================
# Local cdef helpers  (duplicated from heuristics to avoid Python call overhead)
# Calling a cpdef across modules incurs Python dispatch; these are pure C calls.
# ===========================================================================

cdef int _mm_score_4_lines(cnp.int64_t[:, :] board, int player,
                             int row, int col) nogil:
    """Score every 6-window in the 4 lines that pass through (row, col)."""
    cdef int total = 0
    cdef int d, dr, dc, rr, cc, llen, i
    cdef int line_buf[19]

    # direction table inline (same as DIRS_R/C in heuristics)
    cdef int drs[4]
    cdef int dcs[4]
    drs[0] = 0; dcs[0] = 1
    drs[1] = 1; dcs[1] = 0
    drs[2] = 1; dcs[2] = 1
    drs[3] = 1; dcs[3] = -1

    for d in range(4):
        dr = drs[d]; dc = dcs[d]
        # Walk back to line start
        rr = row; cc = col
        while (rr - dr >= 0 and rr - dr < 19 and
               cc - dc >= 0 and cc - dc < 19):
            rr -= dr; cc -= dc
        # Extract full line
        llen = 0
        while 0 <= rr < 19 and 0 <= cc < 19:
            line_buf[llen] = <int>board[rr, cc]
            llen += 1
            rr += dr; cc += dc
        # Score every window of 6
        if llen >= 6:
            for i in range(llen - 5):
                total += _mm_score_window6(
                    line_buf[i],   line_buf[i+1], line_buf[i+2],
                    line_buf[i+3], line_buf[i+4], line_buf[i+5],
                    player,
                )
    return total


cdef int _mm_capture_score_4_lines(cnp.int64_t[:, :] board, int player,
                                     int row, int col) nogil:
    """Capture-potential score for the 4 lines through (row, col)."""
    cdef int opponent = 3 - player
    cdef int total = 0
    cdef int d, dr, dc, rr, cc, llen, i
    cdef int line_buf[19]
    cdef int a, b, dd, e

    cdef int drs[4]
    cdef int dcs[4]
    drs[0] = 0; dcs[0] = 1
    drs[1] = 1; dcs[1] = 0
    drs[2] = 1; dcs[2] = 1
    drs[3] = 1; dcs[3] = -1

    for d in range(4):
        dr = drs[d]; dc = dcs[d]
        rr = row; cc = col
        while (rr - dr >= 0 and rr - dr < 19 and
               cc - dc >= 0 and cc - dc < 19):
            rr -= dr; cc -= dc
        llen = 0
        while 0 <= rr < 19 and 0 <= cc < 19:
            line_buf[llen] = <int>board[rr, cc]
            llen += 1
            rr += dr; cc += dc
        if llen >= 4:
            for i in range(llen - 3):
                a = line_buf[i]; b = line_buf[i+1]
                dd = line_buf[i+2]; e = line_buf[i+3]
                if a == 0 and b == opponent and dd == opponent and e == 0:
                    total -= 500
                elif a == player and b == opponent and dd == opponent and e == player:
                    total += 2000
    return total


cdef inline int _mm_score_captures(int p_caps, int o_caps) nogil:
    """Inline capture-count scorer — identical to heuristics._score_captures_c."""
    cdef int score = p_caps * _MM_C_CAPTURE - o_caps * _MM_C_CAPTURE
    if p_caps >= 8:
        score += 500000
    elif p_caps >= 6:
        score += 50000
    if o_caps >= 8:
        score -= 500000
    elif o_caps >= 6:
        score -= 50000
    return score


cdef inline int _mm_score_window6(
    int v0, int v1, int v2, int v3, int v4, int v5,
    int player,
) nogil:
    """Minimal clone of heuristics._score_window6_values, all C, no GIL."""
    cdef int opponent = 3 - player
    cdef int values[6]
    cdef int pc = 0, oc = 0, ec = 0
    cdef int i, freedom_code
    cdef bint lb, rb

    values[0] = v0; values[1] = v1; values[2] = v2
    values[3] = v3; values[4] = v4; values[5] = v5

    for i in range(6):
        if values[i] == player:   pc += 1
        elif values[i] == opponent: oc += 1
        else:                     ec += 1

    if oc > 0:
        return 0

    if pc == 4:
        for i in range(3):
            if (values[i] == player and values[i+1] == player and
                    values[i+2] == player and values[i+3] == player):
                lb = i > 0 and values[i-1] == opponent
                rb = i+3 < 5 and values[i+4] == opponent
                if lb and rb: return 0
                freedom_code = 1 if (lb or rb) else 0
                if ec == 1: return _mm_scaled(10000, freedom_code)
                return _mm_scaled(100000, freedom_code)
        return 0

    elif pc == 3:
        for i in range(4):
            if values[i] == player and values[i+1] == player and values[i+2] == player:
                lb = i > 0 and values[i-1] == opponent
                rb = i+2 < 5 and values[i+3] == opponent
                if lb and rb: return 0
                freedom_code = 1 if (lb or rb) else 0
                if ec == 2:   return _mm_scaled(5000, freedom_code)
                elif ec == 1: return _mm_scaled(500, freedom_code)
                return 0
        return 0

    elif pc == 2:
        for i in range(5):
            if values[i] == player and values[i+1] == player:
                lb = i > 0 and values[i-1] == opponent
                rb = i+1 < 5 and values[i+2] == opponent
                if lb and rb: return 0
                freedom_code = 1 if (lb or rb) else 0
                if ec == 3:   return _mm_scaled(200, freedom_code)
                elif ec == 2: return _mm_scaled(50, freedom_code)
                return 0
        return 0

    elif pc == 1:
        return 10

    return 0


cdef inline int _mm_scaled(int base, int freedom_code) nogil:
    if freedom_code == 0: return (base * 3) // 2
    elif freedom_code == 1: return base
    return (base * 3) // 10


# ===========================================================================
# Zobrist
# ===========================================================================

cpdef void init_zobrist():
    global zobrist_initialized
    if zobrist_initialized:
        return
    srand(<unsigned int>c_time(NULL))
    cdef int r, c, p
    cdef unsigned long long v
    for r in range(19):
        for c in range(19):
            for p in range(2):
                # Build a 64-bit value from four 16-bit rand() calls
                v  = (<unsigned long long>(rand() & 0xFFFF))
                v |= (<unsigned long long>(rand() & 0xFFFF)) << 16
                v |= (<unsigned long long>(rand() & 0xFFFF)) << 32
                v |= (<unsigned long long>(rand() & 0xFFFF)) << 48
                ZOBRIST_TABLE[r][c][p] = v
    zobrist_initialized = True


cdef unsigned long long compute_initial_hash(cnp.int64_t[:, :] board):
    cdef unsigned long long h = 0
    cdef int r, c, val
    for r in range(19):
        for c in range(19):
            val = <int>board[r, c]
            if val == 1:
                h ^= ZOBRIST_TABLE[r][c][0]
            elif val == 2:
                h ^= ZOBRIST_TABLE[r][c][1]
    return h


cdef inline void update_candidates(int[:, :] candidate_board,
                                    int row, int col, int delta):
    cdef int dr, dc, r, c
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            if dr == 0 and dc == 0:
                continue
            r = row + dr
            c = col + dc
            if 0 <= r < 19 and 0 <= c < 19:
                candidate_board[r, c] += delta


# ===========================================================================
# Minimax with alpha-beta, Zobrist transposition table,
# and INCREMENTAL board evaluation
# ===========================================================================
# board_score tracks the exact full-board evaluation for the current position.
# Computing it incrementally (re-score only the 4 lines through the last move)
# reduces leaf-node work from ~400 window evals to ~56 and makes depth-0
# returns O(1) instead of O(N²).

cpdef double minimax(
    cnp.int64_t[:, :] board,
    int depth,
    double alpha,
    double beta,
    bint is_maximizing,
    int player,
    int player1_captures,
    int player2_captures,
    int[:, :] candidate_board,
    tuple last_move,
    unsigned long long current_hash,
    double board_score,          # incremental full-board evaluation
):
    global compteur_heuristique

    cdef double original_alpha = alpha
    cdef double original_beta  = beta
    cdef int opponent = 3 - player
    cdef int current_player = player if is_maximizing else opponent

    # Typed memoryview: direct C pointer arithmetic for board reads/writes,
    # no Python __getitem__/__setitem__ overhead.

    # Transposition-table lookup
    cdef tuple tt_key = (current_hash, player1_captures, player2_captures, is_maximizing)
    if tt_key in transposition_table:
        stored_depth, stored_score, stored_flag = transposition_table[tt_key]
        if stored_depth >= depth:
            if stored_flag == FLAG_EXACT:
                return stored_score
            elif stored_flag == FLAG_ALPHA and stored_score <= alpha:
                return stored_score
            elif stored_flag == FLAG_BETA and stored_score >= beta:
                return stored_score

    # Win detection (checks last placed stone only)
    cdef int lm_r = last_move[0]
    cdef int lm_c = last_move[1]
    if check_win(board, last_move[0], last_move[1], "me",
                 [player1_captures, player2_captures]):
        compteur_heuristique += 1
        winner_color = <int>board_mv[lm_r, lm_c]
        if winner_color == player:
            return 10_000_000.0
        elif winner_color == opponent:
            return -10_000_000.0
        elif winner_color == 3 and current_player == player:
            return 450_000.0
        elif winner_color == 3 and current_player == opponent:
            return -450_000.0
        # Capture-win ambiguity: return the already-correct incremental score
        return board_score

    # ── Leaf node ──────────────────────────────────────────────────────────
    # board_score is the exact full-board evaluation for this position —
    # accumulated via deltas from the root.  No board scan needed. O(1).
    if depth == 0:
        compteur_heuristique += 1
        return board_score

    # ── Candidate moves ────────────────────────────────────────────────────
    cdef bint player_is_1   = (player == 1)

    # Local variable declarations (must be at function scope in Cython)
    cdef double max_score, min_score, score, delta_score
    cdef int p1_cap, p2_cap, captured, r, c, m_r, m_c
    cdef tuple move
    cdef list captured_positions
    cdef unsigned long long next_hash
    cdef int flag

    # Incremental delta variables
    cdef int pre_pl, pre_op, pre_cap_pot, pre_cap_sc
    cdef int post_pl, post_op, post_cap_pot, post_cap_sc

    cdef int dynamic_max
    if depth >= 8:
        dynamic_max = 10  # On explore large en haut de l'arbre
    elif depth >= 3:
        dynamic_max = 5  # On se concentre sur les bonnes pistes au milieu
    else:
        dynamic_max = 3
    candidates = sort_candidates(board, candidate_board, player, dynamic_max)
    if not candidates:
        return board_score

    if is_maximizing:
        max_score = -INF
        for move in candidates:
            m_r = move[0]; m_c = move[1]

            # ── Incremental eval: capture board state BEFORE this move ──
            pre_pl      = _mm_score_4_lines(board, player,   m_r, m_c)
            pre_op      = _mm_score_4_lines(board, opponent, m_r, m_c)
            pre_cap_pot = _mm_capture_score_4_lines(board, player, m_r, m_c)
            if player_is_1:
                pre_cap_sc = _mm_score_captures(player1_captures, player2_captures)
            else:
                pre_cap_sc = _mm_score_captures(player2_captures, player1_captures)

            # ── DO ──────────────────────────────────────────────────────
            board[m_r, m_c] = current_player
            next_hash = current_hash ^ ZOBRIST_TABLE[m_r][m_c][current_player - 1]
            update_candidates(candidate_board, m_r, m_c, 1)

            p1_cap = player1_captures
            p2_cap = player2_captures
            captured_positions = apply_capture(board, move, current_player)
            for cap_r, cap_c in captured_positions:
                next_hash ^= ZOBRIST_TABLE[cap_r][cap_c][opponent - 1]
                update_candidates(candidate_board, cap_r, cap_c, -1)
            captured = len(captured_positions)
            if current_player == 1: p1_cap += captured
            else:                   p2_cap += captured

            # ── Incremental eval: capture board state AFTER this move ───
            post_pl      = _mm_score_4_lines(board, player,   m_r, m_c)
            post_op      = _mm_score_4_lines(board, opponent, m_r, m_c)
            post_cap_pot = _mm_capture_score_4_lines(board, player, m_r, m_c)
            if player_is_1:
                post_cap_sc = _mm_score_captures(p1_cap, p2_cap)
            else:
                post_cap_sc = _mm_score_captures(p2_cap, p1_cap)

            delta_score = <double>(
                (post_pl - post_op + post_cap_pot + post_cap_sc) -
                (pre_pl  - pre_op  + pre_cap_pot  + pre_cap_sc)
            )

            # ── RECURSE ─────────────────────────────────────────────────
            score = minimax(
                board, depth - 1, alpha, beta, False, player,
                p1_cap, p2_cap, candidate_board, move, next_hash,
                board_score + delta_score,
            )

            max_score = max(max_score, score)
            alpha     = max(alpha, score)

            # ── UNDO ────────────────────────────────────────────────────
            board[m_r, m_c] = 0
            update_candidates(candidate_board, m_r, m_c, -1)
            for r, c in captured_positions:
                board[r, c] = opponent
                update_candidates(candidate_board, r, c, 1)

            if beta <= alpha:
                break

        if max_score <= original_alpha: flag = FLAG_ALPHA
        elif max_score >= beta:         flag = FLAG_BETA
        else:                           flag = FLAG_EXACT
        transposition_table[tt_key] = (depth, max_score, flag)
        return max_score

    else:
        min_score = INF
        for move in candidates:
            m_r = move[0]; m_c = move[1]

            # ── Incremental eval: BEFORE ──
            pre_pl      = _mm_score_4_lines(board, player,   m_r, m_c)
            pre_op      = _mm_score_4_lines(board, opponent, m_r, m_c)
            pre_cap_pot = _mm_capture_score_4_lines(board, player, m_r, m_c)
            if player_is_1:
                pre_cap_sc = _mm_score_captures(player1_captures, player2_captures)
            else:
                pre_cap_sc = _mm_score_captures(player2_captures, player1_captures)

            # ── DO ──────────────────────────────────────────────────────
            board[m_r, m_c] = current_player
            next_hash = current_hash ^ ZOBRIST_TABLE[m_r][m_c][current_player - 1]
            update_candidates(candidate_board, m_r, m_c, 1)

            p1_cap = player1_captures
            p2_cap = player2_captures
            captured_positions = apply_capture(board, move, current_player)
            for cap_r, cap_c in captured_positions:
                next_hash ^= ZOBRIST_TABLE[cap_r][cap_c][player - 1]
                update_candidates(candidate_board, cap_r, cap_c, -1)
            captured = len(captured_positions)
            if current_player == 1: p1_cap += captured
            else:                   p2_cap += captured

            # ── Incremental eval: AFTER ──
            post_pl      = _mm_score_4_lines(board, player,   m_r, m_c)
            post_op      = _mm_score_4_lines(board, opponent, m_r, m_c)
            post_cap_pot = _mm_capture_score_4_lines(board, player, m_r, m_c)
            if player_is_1:
                post_cap_sc = _mm_score_captures(p1_cap, p2_cap)
            else:
                post_cap_sc = _mm_score_captures(p2_cap, p1_cap)

            delta_score = <double>(
                (post_pl - post_op + post_cap_pot + post_cap_sc) -
                (pre_pl  - pre_op  + pre_cap_pot  + pre_cap_sc)
            )

            # ── RECURSE ─────────────────────────────────────────────────
            score = minimax(
                board, depth - 1, alpha, beta, True, player,
                p1_cap, p2_cap, candidate_board, move, next_hash,
                board_score + delta_score,
            )

            min_score = min(min_score, score)
            beta      = min(beta, score)

            # ── UNDO ────────────────────────────────────────────────────
            board[m_r, m_c] = 0
            update_candidates(candidate_board, m_r, m_c, -1)
            for r, c in captured_positions:
                board[r, c] = player
                update_candidates(candidate_board, r, c, 1)

            if beta <= alpha:
                break

        if min_score >= original_beta:  flag = FLAG_BETA
        elif min_score <= alpha:        flag = FLAG_ALPHA
        else:                           flag = FLAG_EXACT
        transposition_table[tt_key] = (depth, min_score, flag)
        return min_score


# ===========================================================================
# Search root
# ===========================================================================

cpdef tuple get_best_move(
    cnp.ndarray board,
    int player,
    int player1_captures,
    int player2_captures,
    tuple last_move,
    int depth = 10,
):
    global compteur_heuristique
    compteur_heuristique = 0

    init_zobrist()
    transposition_table.clear()

    # Typed memoryview for all board access in this function
    cdef cnp.int64_t[:, :] board_mv = board

    cdef unsigned long long current_hash = compute_initial_hash(board_mv)
    cdef double start = time.time()
    cdef int opponent = 3 - player
    cdef bint player_is_1 = (player == 1)

    cdef tuple  best_move  = None
    cdef double best_score = -INF
    cdef double alpha = -INF
    cdef double beta  = INF
    cdef int    p1_cap, p2_cap, captured, r, c, m_r, m_c
    cdef double score, delta_score
    cdef tuple  move
    cdef list   captured_positions
    cdef bint   is_empty_board = True
    cdef unsigned long long next_hash

    # Incremental delta variables
    cdef int pre_pl, pre_op, pre_cap_pot, pre_cap_sc
    cdef int post_pl, post_op, post_cap_pot, post_cap_sc

    # Candidate influence matrix in C memory
    cdef cnp.ndarray[cnp.int32_t, ndim=2] cand_np = np.zeros((19, 19), dtype=np.int32)
    cdef int[:, :] candidate_board = cand_np

    # Populate influence matrix from existing stones
    for r in range(19):
        for c in range(19):
            if board_mv[r, c] != 0:
                is_empty_board = False
                update_candidates(candidate_board, r, c, 1)

    if is_empty_board:
        return ((9, 9), 0.0)

    # ── Seed the incremental board_score with a single full-board evaluation ──
    # This is the ONLY full scan; all subsequent evaluations use deltas.
    cdef int p_caps_init = player1_captures if player_is_1 else player2_captures
    cdef int o_caps_init = player2_captures if player_is_1 else player1_captures
    cdef double initial_board_score = evaluate_board_full_mv(
        board_mv, player, p_caps_init, o_caps_init
    )

    sorted_candidates = sort_candidates(board, candidate_board, player)

    for move in sorted_candidates:
        m_r = move[0]; m_c = move[1]

        # ── Incremental eval: BEFORE root move ──
        pre_pl      = _mm_score_4_lines(board_mv, player,   m_r, m_c)
        pre_op      = _mm_score_4_lines(board_mv, opponent, m_r, m_c)
        pre_cap_pot = _mm_capture_score_4_lines(board_mv, player, m_r, m_c)
        if player_is_1:
            pre_cap_sc = _mm_score_captures(player1_captures, player2_captures)
        else:
            pre_cap_sc = _mm_score_captures(player2_captures, player1_captures)

        # ── DO ──────────────────────────────────────────────────────────
        board_mv[m_r, m_c] = player
        next_hash = current_hash ^ ZOBRIST_TABLE[m_r][m_c][player - 1]
        update_candidates(candidate_board, m_r, m_c, 1)

        p1_cap = player1_captures
        p2_cap = player2_captures
        captured_positions = apply_capture(board, move, player)
        for cap_r, cap_c in captured_positions:
            next_hash ^= ZOBRIST_TABLE[cap_r][cap_c][opponent - 1]
            update_candidates(candidate_board, cap_r, cap_c, -1)
        captured = len(captured_positions)
        if player == 1: p1_cap += captured
        else:           p2_cap += captured

        # ── Incremental eval: AFTER root move ──
        post_pl      = _mm_score_4_lines(board_mv, player,   m_r, m_c)
        post_op      = _mm_score_4_lines(board_mv, opponent, m_r, m_c)
        post_cap_pot = _mm_capture_score_4_lines(board_mv, player, m_r, m_c)
        if player_is_1:
            post_cap_sc = _mm_score_captures(p1_cap, p2_cap)
        else:
            post_cap_sc = _mm_score_captures(p2_cap, p1_cap)

        delta_score = <double>(
            (post_pl - post_op + post_cap_pot + post_cap_sc) -
            (pre_pl  - pre_op  + pre_cap_pot  + pre_cap_sc)
        )

        score = minimax(
            board_mv, depth - 1, alpha, beta, False, player,
            p1_cap, p2_cap, candidate_board, move, next_hash,
            initial_board_score + delta_score,
        )

        if score > best_score:
            best_score = score
            best_move  = move

        alpha = max(alpha, best_score)

        # ── UNDO ────────────────────────────────────────────────────────
        for r, c in captured_positions:
            board_mv[r, c] = opponent
            update_candidates(candidate_board, r, c, 1)
        board_mv[m_r, m_c] = 0
        update_candidates(candidate_board, m_r, m_c, -1)

        if best_score >= INF:
            break

    cdef double elapsed = (time.time() - start) * 1000
    print(f"🧠 L'IA a terminé ! Noeuds évalués : {compteur_heuristique}")
    print(f"AI move: {best_move} | score: {best_score} | time: {elapsed:.1f}ms | depth: {depth}")

    return best_move, best_score


cpdef dict get_heatmap_scores(
    cnp.ndarray board,
    int player,
    int player1_captures,
    int player2_captures,
    tuple last_move,
    int depth = 2,
    int max_candidates = 15,
):
    """
    Return a {(row, col): score} map computed with the same root-search
    logic as get_best_move, so heatmap rankings stay aligned with hints.
    """
    init_zobrist()
    transposition_table.clear()

    cdef cnp.int64_t[:, :] board_mv = board
    cdef unsigned long long current_hash = compute_initial_hash(board_mv)
    cdef int opponent = 3 - player
    cdef bint player_is_1 = (player == 1)

    cdef int    p1_cap, p2_cap, captured, r, c, m_r, m_c
    cdef double score, delta_score
    cdef tuple  move
    cdef list   captured_positions
    cdef bint   is_empty_board = True
    cdef unsigned long long next_hash

    # Incremental delta variables
    cdef int pre_pl, pre_op, pre_cap_pot, pre_cap_sc
    cdef int post_pl, post_op, post_cap_pot, post_cap_sc

    cdef cnp.ndarray[cnp.int32_t, ndim=2] cand_np = np.zeros((19, 19), dtype=np.int32)
    cdef int[:, :] candidate_board = cand_np

    cdef dict move_scores = {}

    for r in range(19):
        for c in range(19):
            if board_mv[r, c] != 0:
                is_empty_board = False
                update_candidates(candidate_board, r, c, 1)

    if is_empty_board:
        move_scores[(9, 9)] = 0.0
        return move_scores

    cdef int p_caps_init = player1_captures if player_is_1 else player2_captures
    cdef int o_caps_init = player2_captures if player_is_1 else player1_captures
    cdef double initial_board_score = evaluate_board_full_mv(
        board_mv, player, p_caps_init, o_caps_init
    )

    cdef list sorted_candidates = sort_candidates(board, candidate_board, player, max_candidates)

    for move in sorted_candidates:
        m_r = move[0]; m_c = move[1]

        # BEFORE move
        pre_pl      = _mm_score_4_lines(board_mv, player,   m_r, m_c)
        pre_op      = _mm_score_4_lines(board_mv, opponent, m_r, m_c)
        pre_cap_pot = _mm_capture_score_4_lines(board_mv, player, m_r, m_c)
        if player_is_1:
            pre_cap_sc = _mm_score_captures(player1_captures, player2_captures)
        else:
            pre_cap_sc = _mm_score_captures(player2_captures, player1_captures)

        # DO
        board_mv[m_r, m_c] = player
        next_hash = current_hash ^ ZOBRIST_TABLE[m_r][m_c][player - 1]
        update_candidates(candidate_board, m_r, m_c, 1)

        p1_cap = player1_captures
        p2_cap = player2_captures
        captured_positions = apply_capture(board, move, player)
        for cap_r, cap_c in captured_positions:
            next_hash ^= ZOBRIST_TABLE[cap_r][cap_c][opponent - 1]
            update_candidates(candidate_board, cap_r, cap_c, -1)
        captured = len(captured_positions)
        if player == 1:
            p1_cap += captured
        else:
            p2_cap += captured

        # AFTER move
        post_pl      = _mm_score_4_lines(board_mv, player,   m_r, m_c)
        post_op      = _mm_score_4_lines(board_mv, opponent, m_r, m_c)
        post_cap_pot = _mm_capture_score_4_lines(board_mv, player, m_r, m_c)
        if player_is_1:
            post_cap_sc = _mm_score_captures(p1_cap, p2_cap)
        else:
            post_cap_sc = _mm_score_captures(p2_cap, p1_cap)

        delta_score = <double>(
            (post_pl - post_op + post_cap_pot + post_cap_sc) -
            (pre_pl  - pre_op  + pre_cap_pot  + pre_cap_sc)
        )

        score = minimax(
            board, depth - 1, -INF, INF, False, player,
            p1_cap, p2_cap, candidate_board, move, next_hash,
            initial_board_score + delta_score,
        )
        move_scores[(m_r, m_c)] = score

        # UNDO
        for r, c in captured_positions:
            board_mv[r, c] = opponent
            update_candidates(candidate_board, r, c, 1)
        board_mv[m_r, m_c] = 0
        update_candidates(candidate_board, m_r, m_c, -1)

    return move_scores


# ===========================================================================
# Candidate sorter
# ===========================================================================

cpdef list sort_candidates(cnp.int64_t[:, :] board, int[:, :] candidate_board,
                            int player, int max_count=10):
    cdef int center   = 9
    cdef int opponent = 3 - player

    # Déclarations C strictes
    cdef int r, c, nr, nc, dr, dc, d, step
    cdef int count_ally, count_enemy
    cdef int score_i            
    cdef int distance
    cdef list scored_moves = []
    cdef tuple item
    cdef list final_moves = []
    cdef int count = 0

    cdef int dirs[4][2]
    dirs[0][0] = 0; dirs[0][1] = 1
    dirs[1][0] = 1; dirs[1][1] = 0
    dirs[2][0] = 1; dirs[2][1] = 1
    dirs[3][0] = 1; dirs[3][1] = -1

    for r in range(19):
        for c in range(19):
            if board[r, c] == 0 and candidate_board[r, c] > 0:
                score_i = 0

                for d in range(4):
                    dr = dirs[d][0]; dc = dirs[d][1]

                    # -- Comptage ALLIÉS --
                    count_ally = 0
                    for step in range(1, 5):
                        nr = r + dr * step; nc = c + dc * step
                        if 0 <= nr < 19 and 0 <= nc < 19 and board[nr, nc] == player:
                            count_ally += 1
                        else: break
                        
                    for step in range(1, 5):
                        nr = r - dr * step; nc = c - dc * step
                        if 0 <= nr < 19 and 0 <= nc < 19 and board[nr, nc] == player:
                            count_ally += 1
                        else: break

                    # -- Comptage ENNEMIS --
                    count_enemy = 0
                    for step in range(1, 5):
                        nr = r + dr * step; nc = c + dc * step
                        if 0 <= nr < 19 and 0 <= nc < 19 and board[nr, nc] == opponent:
                            count_enemy += 1
                        else: break
                        
                    for step in range(1, 5):
                        nr = r - dr * step; nc = c - dc * step
                        if 0 <= nr < 19 and 0 <= nc < 19 and board[nr, nc] == opponent:
                            count_enemy += 1
                        else: break

                    # -- SCORING TACTIQUE --
                    if count_ally >= 4:   score_i += 100000
                    elif count_ally == 3: score_i += 1000
                    elif count_ally == 2: score_i += 100
                    elif count_ally == 1: score_i += 10

                    if count_enemy >= 4:   score_i += 80000
                    elif count_enemy == 3: score_i += 800
                    elif count_enemy == 2: score_i += 80
                    elif count_enemy == 1: score_i += 8

                # Spatial tiebreaker: favorise le centre
                distance = abs(r - center) + abs(c - center)
                score_i -= distance

                if not check_double_three(board, r, c, player):
                    scored_moves.append((score_i, (r, c)))

    # Tri du meilleur au pire
    scored_moves.sort(reverse=True)
    
    # === LE GARDE-FOU (SAFEGUARD) ===
    if max_count > 0:
        for item in scored_moves:
            # item[0] = le score, item[1] = les coordonnées (r, c)
            # On prend si on a de la place OU si c'est une urgence vitale (>= 80000)
            if count < max_count or item[0] >= 80000:
                final_moves.append(item[1])
                count += 1
            else:
                break
        return final_moves

    # Si max_count = 0 (désactivé), on renvoie tout
    return [item[1] for item in scored_moves]

cpdef list get_candidates_for_heatmap(cnp.ndarray board, int player):
    """Return all sorted candidates (same policy as minimax, no top-N cutoff)."""
    cdef cnp.int64_t[:, :] board_mv = board
    cdef cnp.ndarray[cnp.int32_t, ndim=2] cand_np = np.zeros((19, 19), dtype=np.int32)
    cdef int[:, :] candidate_board = cand_np
    cdef int r, c
    cdef bint is_empty_board = True

    for r in range(19):
        for c in range(19):
            if board_mv[r, c] != 0:
                is_empty_board = False
                update_candidates(candidate_board, r, c, 1)

    if is_empty_board:
        return [(9, 9)]

    return sort_candidates(board, candidate_board, player, 0)
