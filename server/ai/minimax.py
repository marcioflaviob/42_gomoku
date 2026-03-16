from constants import BOARD_SIZE
from optimizer import get_candidate_moves
from moves import apply_capture
import numpy as np
import time

INF = float('inf')

def minimax(
    board: np.ndarray,
    depth: int,
    alpha: float,
    beta: float,
    is_maximizing: bool,
    player: int,
    player1_captures: int,
    player2_captures: int,
    last_move: tuple[int, int] | None = None
) -> float:
    """
    Returns the heuristic score of the board for `player`.
    Positive = good for player, Negative = bad for player.
    """
    opponent = 2 if player == 1 else 1

    if depth == 0:
        return evaluate_board(board, player, player1_captures, player2_captures)

    candidates = get_candidate_moves(board)
    if not candidates:
        return evaluate_board(board, player, player1_captures, player2_captures)

    current_player = player if is_maximizing else opponent

    if is_maximizing:
        max_score = -INF
        for move in candidates:
            # --- Apply move ---
            board_copy = board.copy()
            p1_cap, p2_cap = player1_captures, player2_captures
            captured = apply_capture(board_copy, move, current_player)
            if current_player == 1:
                p1_cap += captured
            else:
                p2_cap += captured

            score = minimax(
                board_copy, depth - 1, alpha, beta,
                False, player, p1_cap, p2_cap, move
            )
            max_score = max(max_score, score)
            alpha = max(alpha, score)
            if beta <= alpha:
                break  # beta cutoff

        return max_score

    else:
        min_score = INF
        for move in candidates:
            # --- Apply move ---
            board_copy = board.copy()
            p1_cap, p2_cap = player1_captures, player2_captures
            captured = apply_capture(board_copy, move, current_player)
            if current_player == 1:
                p1_cap += captured
            else:
                p2_cap += captured

            score = minimax(
                board_copy, depth - 1, alpha, beta,
                True, player, p1_cap, p2_cap, move
            )
            min_score = min(min_score, score)
            beta = min(beta, score)
            if beta <= alpha:
                break  # alpha cutoff

        return min_score


def get_best_move(
    board: np.ndarray,
    player: int,
    player1_captures: int,
    player2_captures: int,
    depth: int = 10
) -> tuple[tuple[int, int], float]:
    """
    Entry point for the AI. Returns (best_move, score).
    """

    start = time.time()

    candidates = get_candidate_moves(board)
    best_move = None
    best_score = -INF
    alpha = -INF
    beta = INF

    # Sort candidates before searching — critical for alpha-beta efficiency
    candidates = sort_candidates(board, candidates, player)

    for move in candidates:
        board_copy = board.copy()
        p1_cap, p2_cap = player1_captures, player2_captures
        captured = apply_capture(board_copy, move, player)
        if player == 1:
            p1_cap += captured
        else:
            p2_cap += captured

        score = minimax(
            board_copy, depth - 1, alpha, beta,
            False, player, p1_cap, p2_cap, move
        )

        if score > best_score:
            best_score = score
            best_move = move

        alpha = max(alpha, best_score)

        # Immediately return a winning move
        if best_score == INF:
            break

    elapsed = (time.time() - start) * 1000  # ms
    print(f"AI move: {best_move} | score: {best_score} | time: {elapsed:.1f}ms | depth: {depth}")

    return best_move, best_score


def sort_candidates(
    board: np.ndarray,
    candidates: list[tuple[int, int]],
    player: int
) -> list[tuple[int, int]]:
    """
    Orders candidates by a cheap heuristic before full search.
    Better ordering = more alpha-beta cutoffs = faster search.
    Center-weighted distance score for now — replace with
    shallow heuristic once evaluate_board is implemented.
    """
    center = BOARD_SIZE // 2

    def quick_score(move):
        r, c = move
        # Prefer moves closer to center
        distance = abs(r - center) + abs(c - center)
        return -distance  # negative because we sort descending

    return sorted(candidates, key=quick_score, reverse=True)