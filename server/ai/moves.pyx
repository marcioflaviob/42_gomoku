# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, initializedcheck=False

cimport numpy as cnp
import numpy as np

cnp.import_array()

DEF BOARD_SIZE = 19
DEF EMPTY      = 0

cpdef list apply_capture(cnp.int64_t[:, :] board, tuple move, int player):
    cdef cnp.int64_t[:, :] bv = board
    cdef int row      = move[0]
    cdef int col      = move[1]
    cdef int opponent = 3 - player

    board[row, col] = player

    # 8 unit directions (all 4 axes, both signs)
    cdef int DRS[8]
    cdef int DCS[8]
    DRS[0] =  0; DCS[0] =  1   # right
    DRS[1] =  0; DCS[1] = -1   # left
    DRS[2] =  1; DCS[2] =  0   # down
    DRS[3] = -1; DCS[3] =  0   # up
    DRS[4] =  1; DCS[4] =  1   # down-right
    DRS[5] = -1; DCS[5] = -1   # up-left
    DRS[6] =  1; DCS[6] = -1   # down-left
    DRS[7] = -1; DCS[7] =  1   # up-right

    cdef int d, dr, dc
    cdef int r1, c1, r2, c2, r3, c3
    cdef list captured = []

    for d in range(8):
        dr = DRS[d]; dc = DCS[d]
        r1 = row + dr;     c1 = col + dc
        r2 = row + 2 * dr; c2 = col + 2 * dc
        r3 = row + 3 * dr; c3 = col + 3 * dc
        if (0 <= r3 < BOARD_SIZE and 0 <= c3 < BOARD_SIZE and
                board[r1, c1] == opponent and
                board[r2, c2] == opponent and
                board[r3, c3] == player):
            board[r1, c1] = EMPTY
            board[r2, c2] = EMPTY
            captured.append((r1, c1))
            captured.append((r2, c2))

    return captured

cpdef int check_win(cnp.int64_t[:, :] board, int row, int col, str winner, list captures):
    cdef int color = board[row][col]
    if color == EMPTY:
        return 0
    # Capture-count win
    if captures[color - 1] >= 10:
        return color

    cdef int DRS[4]
    cdef int DCS[4]
    DRS[0] = 0; DCS[0] = 1   # horizontal
    DRS[1] = 1; DCS[1] = 0   # vertical
    DRS[2] = 1; DCS[2] = 1   # diagonal
    DRS[3] = 1; DCS[3] = -1  # anti-diagonal

    cdef int d, dr, dc, i
    cdef int r, c, count

    # win_line stored as flat pairs: [r0,c0, r1,c1, ...]  max 9 stones
    cdef int win_r[9]
    cdef int win_c[9]
    cdef int wlen

    for d in range(4):
        dr = DRS[d]; dc = DCS[d]
        count = 1
        wlen  = 1
        win_r[0] = row; win_c[0] = col

        for i in range(1, 5):
            r = row + dr * i; c = col + dc * i
            if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r, c] == color:
                win_r[wlen] = r; win_c[wlen] = c
                wlen += 1; count += 1
            else:
                break
        for i in range(1, 5):
            r = row - dr * i; c = col - dc * i
            if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r, c] == color:
                win_r[wlen] = r; win_c[wlen] = c
                wlen += 1; count += 1
            else:
                break

        if count >= 5:
            check_no_capture = _win_line_capturable(board, win_r, win_c, wlen, color)
            if not check_no_capture or winner == "oppo":
                return color
            elif check_no_capture:
                return 3
    return 0


cpdef int check_double_three(cnp.int64_t[:, :] board, int row, int col, int color):
    if color == 0:
        return 0

    cdef int opponent = 3 - color

    cdef int DRS[4]
    cdef int DCS[4]
    DRS[0] = 0; DCS[0] = 1
    DRS[1] = 1; DCS[1] = 0
    DRS[2] = 1; DCS[2] = 1
    DRS[3] = 1; DCS[3] = -1

    cdef int free_three_count = 0
    cdef int d, i, j, dr, dc, r, c, v
    cdef int buf[9]
    cdef int wpc, wec
    cdef bint has_opponent, found, left_open, right_open

    for d in range(4):
        dr = DRS[d]; dc = DCS[d]

        for i in range(-4, 5):
            r = row + dr * i
            c = col + dc * i
            if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                buf[i + 4] = color if i == 0 else <int>board[r, c]
            else:
                buf[i + 4] = opponent

        found = False
        for i in range(4):
            wpc = 0; wec = 0; has_opponent = False
            for j in range(6):
                v = buf[i + j]
                if v == color:       wpc += 1
                elif v == EMPTY:     wec += 1
                else:                has_opponent = True

            if has_opponent or wpc != 3 or wec != 3:
                continue

            left_open  = (i == 0)     or (buf[i - 1] == EMPTY)
            right_open = (i + 6 >= 9) or (buf[i + 6] == EMPTY)

            if left_open and right_open:
                found = True
                break

        if found:
            free_three_count += 1

    return 1 if free_three_count >= 2 else 0

cdef bint _win_line_capturable(cnp.int64_t[:, :] bv,
                                int* win_r, int* win_c, int wlen,
                                int color) nogil:
    """
    Returns 1 if any stone in the win line sits in a capture pattern
    (opponent can remove it), 0 if the line is safe.

    Pattern checked (for each cell in the win line, each of 4 directions):
      OXX.  or  .XXO   — where X = color, O = opponent, . = EMPTY
    We look at a 5-cell window centred on the win-line cell.
    """
    cdef int DRS[4]
    cdef int DCS[4]
    DRS[0] = 0; DCS[0] = 1
    DRS[1] = 1; DCS[1] = 0
    DRS[2] = 1; DCS[2] = 1
    DRS[3] = 1; DCS[3] = -1

    cdef int opponent = 3 - color
    cdef int k, d, dr, dc, i
    cdef int r, c
    cdef int seg[5]   # 5-cell window centred on the stone
    cdef int v

    for k in range(wlen):
        r = win_r[k]; c = win_c[k]
        for d in range(4):
            dr = DRS[d]; dc = DCS[d]
            # Fill 5-cell window: positions -2,-1,0,+1,+2
            for i in range(5):
                rr = r + dr * (i - 2)
                cc = c + dc * (i - 2)
                if 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE:
                    seg[i] = <int>bv[rr, cc]
                else:
                    seg[i] = -1   # wall sentinel

            # OXX.  →  seg[0]=opponent, seg[1]=color, seg[2]=color, seg[3]=EMPTY
            if (seg[0] == opponent and seg[1] == color and
                    seg[2] == color and seg[3] == EMPTY):
                return 1
            # .XXO  →  seg[1]=EMPTY, seg[2]=color, seg[3]=color, seg[4]=opponent
            if (seg[1] == EMPTY and seg[2] == color and
                    seg[3] == color and seg[4] == opponent):
                return 1

    return 0
