cimport numpy as cnp

cpdef list apply_capture(cnp.ndarray board, tuple move, int player)
cpdef int  check_win(cnp.ndarray board, int row, int col, str winner, list captures)
cpdef int  check_double_three(cnp.ndarray board, int row, int col, int color)
