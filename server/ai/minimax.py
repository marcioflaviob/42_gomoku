from ai.constants import BOARD_SIZE, EMPTY
from ai.optimizer import get_candidate_moves
from ai.moves import apply_capture, check_win
from ai.heuristics import evaluate_board
import numpy as np
import time
import random

INF = float('inf')
compteur_heuristique = 0

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
    global compteur_heuristique
    if depth == 0 or check_win(board,last_move[0],last_move[1],"me", [player1_captures,player2_captures]):
        compteur_heuristique += 1 # +1 évaluation !
        #return evaluate_board(board, player, player1_captures, player2_captures)
        return random.randint(0, 1000)
    candidates = get_candidate_moves(board)
    if not candidates:
        return evaluate_board(board, player, player1_captures, player2_captures)

    current_player = player if is_maximizing else opponent
    candidates = sort_candidates(board, candidates, player)
    if is_maximizing:
        max_score = -INF
        for move in candidates:
            # --- Apply move ---
            board[move[0],move[1]] = player
            p1_cap, p2_cap = player1_captures, player2_captures
            captured = apply_capture(board, move, current_player)
            if current_player == 1:
                p1_cap += captured
            else:
                p2_cap += captured

            score = minimax(
                board, depth - 1, alpha, beta,
                False, player, p1_cap, p2_cap, move
            )
            max_score = max(max_score, score)
            alpha = max(alpha, score)
            if beta <= alpha:
                break  # beta cutoff
            board[move[0],move[1]] = 0
            # for r, c in captured_positions:
            #     board[r][c] = opponent
        return max_score

    else:
        min_score = INF
        for move in candidates:
            # --- Apply move ---
            board[move[0],move[1]] = opponent
            p1_cap, p2_cap = player1_captures, player2_captures
            captured = apply_capture(board, move, current_player)
            if current_player == 1:
                p1_cap += captured
            else:
                p2_cap += captured

            score = minimax(
                board, depth - 1, alpha, beta,
                True, player, p1_cap, p2_cap, move
            )
            min_score = min(min_score, score)
            beta = min(beta, score)
            if beta <= alpha:
                break  # alpha cutoff
            # for r, c in captured_positions:
            #     board[r][c] = player
            board[move[0],move[1]] = 0

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
    p1_cap, p2_cap = player1_captures, player2_captures
    candidates = sort_candidates(board, candidates, player)
    # best_move = get_best_move_parallel(board, candidates, depth, player, p1_cap, p2_cap)
    # print(f"🧠 L'IA a terminé ! Nombre de plateaux évalués : {compteur_heuristique}")
    i = 0
    for move in candidates:
        i = i + 1
        print(move)
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
    print(f"🧠 L'IA a terminé ! Nombre de plateaux évalués : {compteur_heuristique}")
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


import concurrent.futures

def evaluer_un_seul_coup(args):
    """
    Cette fonction sera envoyée sur un cœur de processeur indépendant.
    Elle prend un coup, le joue, et lance un minimax classique.
    """
    board, move, depth, player, p1_cap, p2_cap = args
    
    # On simule le coup (ici on triche un peu pour l'exemple, faites votre DO/UNDO)
    board_local = board.copy() # Ici la copie est permise car c'est la racine !
    row, col = move
    board_local[row][col] = player
    
    # On lance le minimax normal (qui a son propre alpha/beta pour ce sous-arbre)
    score = minimax(
        board_local, depth - 1, float('-inf'), float('inf'),
        False, player, p1_cap, p2_cap, move
    )
    
    return (score, move)


def get_best_move_parallel(board, candidates, depth, player, p1_cap, p2_cap):
    """
    La fonction principale qui distribue le travail aux cœurs du CPU.
    """
    # On prépare les paquets de données pour chaque cœur
    taches = []
    for move in candidates:
        taches.append((board, move, depth, player, p1_cap, p2_cap))
    
    meilleur_score = float('-inf')
    meilleur_coup = None
    
    # On ouvre un "Pool" de processus (1 processus par cœur physique de votre PC)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # map() distribue les tâches et récupère les résultats au fur et à mesure
        resultats = executor.map(evaluer_un_seul_coup, taches)
        
        # On compare les résultats finaux de chaque cœur
        for score, move in resultats:
            if score > meilleur_score:
                meilleur_score = score
                meilleur_coup = move
                
    return meilleur_coup