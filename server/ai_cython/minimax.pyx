from ai_cython.constants import BOARD_SIZE, EMPTY
from ai_cython.optimizer import get_candidate_moves
from ai_cython.moves import apply_capture, check_win
from ai_cython.heuristics import evaluate_board
cimport numpy as cnp
import time
import numpy as np

INF = float('inf')
compteur_heuristique = 0

cnp.import_array()

cpdef set get_empty_neighbors(cnp.ndarray board, int row, int col):
    """
    Renvoie les cases vides dans un rayon de 2.
    Optimisation Cython : On utilise des boucles C pures au lieu de la liste Python NEIGHBOR_OFFSETS_R2.
    """
    cdef set neighbors = set()
    cdef int dr, dc, r, c
    
    # Boucles ultra-rapides en C
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            if dr == 0 and dc == 0:
                continue
            r = row + dr
            c = col + dc
            if 0 <= r < 19 and 0 <= c < 19:
                if board[r, c] == 0:  # 0 = EMPTY
                    neighbors.add((r, c))
    return neighbors


cpdef double minimax(
    cnp.ndarray board,
    int depth,
    double alpha,
    double beta,
    bint is_maximizing,
    int player,
    int player1_captures,
    int player2_captures,
    set current_candidates,
    tuple last_move
):
    global compteur_heuristique
    
    cdef int opponent = 2 if player == 1 else 1
    cdef int current_player = player if is_maximizing else opponent
    
    # Typage des variables de boucle pour une vitesse C
    cdef double max_score, min_score, score
    cdef int p1_cap, p2_cap, captured, r, c, m_r, m_c
    cdef tuple move
    cdef list captured_positions
    cdef set next_candidates, new_neighbors

    cdef int last_mover = opponent if is_maximizing else player

    if check_win(board, last_move[0], last_move[1], "me", [player1_captures, player2_captures]):
        compteur_heuristique += 1
        winner_color = board[last_move[0]][last_move[1]]
        if winner_color == player:
            return 10_000_000
        elif winner_color == opponent:
            return -10_000_000
        # fallthrough: capture win for someone, evaluate normally
        return evaluate_board(board, player, player1_captures, player2_captures)

    if depth == 0:
        compteur_heuristique += 1
        a = evaluate_board(board, player, player1_captures, player2_captures)
        if compteur_heuristique % 1000 == 0:
            print(f"value returned {a}")
        return a
    # Note : sort_candidates doit retourner une liste de tuples
    candidates = sort_candidates(board, list(current_candidates), current_player)
    
    if not candidates:
        score = evaluate_board(board, player, player1_captures, player2_captures)
        print(f"No candidates left, returning score: {score}")
        return score

    if is_maximizing:
        max_score = -INF
        for move in candidates:
            m_r, m_c = move[0], move[1]
            # --- DO (Appliquer le coup) ---
            board[m_r, m_c] = current_player  # 🚨 CORRECTION : Il manquait la pose de la pierre !
            
            p1_cap = player1_captures
            p2_cap = player2_captures
            captured_positions = apply_capture(board, move, current_player)
            captured = len(captured_positions)
            
            if current_player == 1:
                p1_cap += captured
            else:
                p2_cap += captured

            next_candidates = current_candidates.copy()
            next_candidates.remove(move) 
            
            new_neighbors = get_empty_neighbors(board, m_r, m_c)
            next_candidates.update(new_neighbors)
            
            # --- EVALUATE ---
            score = minimax(
                board, depth - 1, alpha, beta, False, player, 
                p1_cap, p2_cap, next_candidates, move
            )
            
            max_score = max(max_score, score)
            alpha = max(alpha, score)
            
            # --- UNDO (Annuler le coup) ---
            board[m_r, m_c] = 0
            for r, c in captured_positions:
                 board[r, c] = opponent
                 
            if beta <= alpha:
                break
        
        #print(f"Maximizing node at depth {depth} returning score: {max_score}")
        return max_score
        
    else:
        min_score = INF
        for move in candidates:
            m_r, m_c = move[0], move[1]
            # --- DO ---
            board[m_r, m_c] = current_player  # 🚨 CORRECTION : Pose de la pierre adverse !
            
            p1_cap = player1_captures
            p2_cap = player2_captures
            captured_positions = apply_capture(board, move, current_player)
            captured = len(captured_positions)
            
            if current_player == 1:
                p1_cap += captured
            else:
                p2_cap += captured
                
            next_candidates = current_candidates.copy()
            next_candidates.remove(move) 
            
            new_neighbors = get_empty_neighbors(board, m_r, m_c)
            next_candidates.update(new_neighbors)
            
            # --- EVALUATE ---
            score = minimax(
                board, depth - 1, alpha, beta, True, player, 
                p1_cap, p2_cap, next_candidates, move
            )
            
            min_score = min(min_score, score)
            beta = min(beta, score)
            
            # --- UNDO ---
            board[m_r, m_c] = 0
            for r, c in captured_positions:
                board[r, c] = player
                
            if beta <= alpha:
                break
                
        # print(f"Minimizing node at depth {depth} returning score: {min_score}")
        return min_score



cpdef tuple get_best_move(
    cnp.ndarray board,
    int player,
    int player1_captures,
    int player2_captures,
    int depth = 10
):
    global compteur_heuristique
    compteur_heuristique = 0  # 🚨 Réinitialisation à chaque nouveau tour !
    
    cdef double start = time.time()
    cdef int opponent = 2 if player == 1 else 1
    
    cdef tuple best_move = None
    cdef double best_score = -INF
    cdef double alpha = -INF
    cdef double beta = INF
    cdef int p1_cap, p2_cap, captured, r, c, m_r, m_c
    cdef double score
    cdef tuple move
    cdef list captured_positions
    cdef bint is_empty_board = True
    cdef set initial_candidates = set()
    
    for r in range(19):
        for c in range(19):
            if board[r, c] != 0:
                is_empty_board = False
                initial_candidates.update(get_empty_neighbors(board, r, c))
                
    if is_empty_board:
        return ((9, 9), 0.0)

    sorted_candidates = sort_candidates(board, list(initial_candidates), player)
    
    score_tab = []
    for move in sorted_candidates:
        m_r, m_c = move[0], move[1]
        
        # --- DO ---
        board[m_r, m_c] = player
        
        p1_cap = player1_captures
        p2_cap = player2_captures
        captured_positions = apply_capture(board, move, player)
        captured = len(captured_positions)
        
        if player == 1:
            p1_cap += captured
        else:
            p2_cap += captured

        print(f"Evaluating move: {move}")

        # --- EVALUATE ---
        root_candidates = initial_candidates - {move}   # exclude the cell AI just occupied
        root_candidates.update(get_empty_neighbors(board, m_r, m_c))  # add new neighbours
        score = minimax(
            board, depth - 1, alpha, beta, False, player, 
            p1_cap, p2_cap, root_candidates, move
        )

        score_tab.append((score, move))

        print(f"ROOT MOVE: {move} -> Score: {score}")
        
        if score > best_score:
            best_score = score
            best_move = move
            
        alpha = max(alpha, best_score)
        
        # --- UNDO (REVERSE ORDER!) ---
        # Must restore board BEFORE undo to test next move on clean board
        board_copy = board.copy()  # Create snapshot BEFORE undo
        board[m_r, m_c] = 0
        for r, c in captured_positions:
            board[r, c] = opponent
        
        if best_score == INF:
            break

    cdef double elapsed = (time.time() - start) * 1000  # ms
    print(f"Score tab for root moves: {score_tab}")
    print(f"🧠 L'IA a terminé ! Nombre de plateaux évalués : {compteur_heuristique}")
    print(f"AI move: {best_move} | score: {best_score} | time: {elapsed:.1f}ms | depth: {depth}")
    return best_move, best_score

cpdef list sort_candidates(cnp.ndarray board, list candidates, int player):
    """
    Trie les coups possibles. 
    Version Cython : Utilise des boucles C pures au lieu du slicing NumPy pour une vitesse maximale.
    """
    cdef int board_size = board.shape[0]
    cdef int center = board_size // 2
    cdef int opponent = 2 if player == 1 else 1
    
    # Déclaration des variables C pour la boucle
    cdef int r, c, dr, dc, nr, nc, val
    cdef int allied_stones, enemy_stones
    cdef float score, distance
    cdef tuple move
    
    # Liste qui contiendra des tuples (score, move)
    cdef list scored_moves = []

    # Parcours de chaque candidat
    for move in candidates:
        r = move[0]
        c = move[1]
        
        allied_stones = 0
        enemy_stones = 0
        
        # 1. LA TACTIQUE (Boucles C ultra-rapides, zéro allocation mémoire)
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr == 0 and dc == 0:
                    continue
                    
                nr = r + dr
                nc = c + dc
                
                # Vérification des limites du plateau
                if 0 <= nr < board_size and 0 <= nc < board_size:
                    val = board[nr, nc]
                    if val == player:
                        allied_stones += 1
                    elif val == opponent:
                        enemy_stones += 1

        # 2. LE CENTRE ET LE SCORE
        # abs() fonctionne très bien et très vite en C
        distance = abs(r - center) + abs(c - center)
        score = (allied_stones * 10) + (enemy_stones * 12) - (distance * 0.1)

        # On stocke le score associé au coup
        scored_moves.append((score, move))

    # 3. LE TRI
    # On trie la liste en fonction du premier élément du tuple (le score)
    scored_moves.sort(reverse=True)

    # 4. LE RETOUR
    cdef tuple item
    return [item[1] for item in scored_moves]
