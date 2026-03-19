cimport numpy as cnp

cpdef list apply_capture(cnp.int64_t[:, :]  board, tuple move, int player)
cpdef int  check_win(cnp.int64_t[:, :]  board, int row, int col, str winner, list captures)
cpdef int  check_double_three(cnp.ndarray board, int row, int col, int color)
