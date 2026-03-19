cimport numpy as cnp

cpdef double evaluate_board_full_mv(cnp.int64_t[:, :] board, int player,
                                     int p_caps, int o_caps)
cpdef int score_4_lines_for_player(cnp.int64_t[:, :] board, int player,
                                    int row, int col)
cpdef int capture_score_4_lines_for_player(cnp.int64_t[:, :] board, int player,
                                            int row, int col)
cpdef int score_captures_fast(int p_caps, int o_caps)
