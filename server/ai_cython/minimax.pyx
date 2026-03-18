from ai_cython.constants import BOARD_SIZE, EMPTY
from ai_cython.optimizer import get_candidate_moves
from ai_cython.moves import apply_capture, check_win
from ai_cython.heuristics import evaluate_board
cimport numpy as cnp
import time
import numpy as np
import random

INF = float('inf')
compteur_heuristique = 0

cnp.import_array()

cdef unsigned long long ZOBRIST_TABLE[19][19][2]
cdef bint zobrist_initialized = False
cdef dict transposition_table = {}

# Flags pour l'Alpha-Beta (Transposition Table)
cdef int FLAG_EXACT = 0
cdef int FLAG_ALPHA = 1  # Borne Supérieure (Cutoff)
cdef int FLAG_BETA = 2   # Borne Inférieure (Cutoff)

cpdef void init_zobrist():
    """ Initialise les nombres aléatoires une seule fois au lancement du serveur. """
    global zobrist_initialized
    if zobrist_initialized:
        return
    cdef int r, c, p
    for r in range(19):
        for c in range(19):
            for p in range(2):
                ZOBRIST_TABLE[r][c][p] = random.getrandbits(64)
    zobrist_initialized = True

cdef unsigned long long compute_initial_hash(cnp.ndarray[cnp.int64_t, ndim=2] board):
    """ Calcule le Hash complet du plateau au tout premier tour. """
    cdef unsigned long long h = 0
    cdef int r, c, val
    for r in range(19):
        for c in range(19):
            val = board[r, c]
            if val == 1:
                h ^= ZOBRIST_TABLE[r][c][0]
            elif val == 2:
                h ^= ZOBRIST_TABLE[r][c][1]
    return h

cdef inline void update_candidates(int[:, :] candidate_board, int row, int col, int delta):
    """
    Ajoute ou retire l'influence d'un pion sur les cases environnantes.
    delta = 1 (quand on pose un pion), delta = -1 (quand on le retire).
    """
    cdef int dr, dc, r, c
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            if dr == 0 and dc == 0:
                continue
            r = row + dr
            c = col + dc
            if 0 <= r < 19 and 0 <= c < 19:
                candidate_board[r, c] += delta

cpdef double minimax(
    cnp.ndarray board,
    int depth,
    double alpha,
    double beta,
    bint is_maximizing,
    int player,
    int player1_captures,
    int player2_captures,
    int[:, :] candidate_board,
    tuple last_move,
    unsigned long long current_hash  # NOUVEAU PARAMÈTRE !
):
    global compteur_heuristique
    
    # 1. SAUVEGARDE DES BORNES ORIGINALES (Pour le Flag)
    cdef double original_alpha = alpha
    cdef double original_beta = beta
    cdef int opponent = 2 if player == 1 else 1
    
    # 2. LECTURE DE LA TRANSPOSITION TABLE
    # La clé est unique pour cette situation exacte.
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
    if check_win(board, last_move[0], last_move[1], "me", [player1_captures, player2_captures]):
        compteur_heuristique += 1
        winner_color = board[last_move[0]][last_move[1]]
        if winner_color == player:
            return 10_000_000
        elif winner_color == opponent:
            return -10_000_000
        # fallthrough: capture win for someone, evaluate normally
        return evaluate_board(board, player, player1_captures, player2_captures)
    # 3. CONDITIONS D'ARRÊT
    if depth == 0:
        compteur_heuristique += 1
        return  evaluate_board(board, player, player1_captures, player2_captures)

    cdef int current_player = player if is_maximizing else opponent
    
    cdef double max_score, min_score, score
    cdef int p1_cap, p2_cap, captured, r, c, m_r, m_c
    cdef tuple move
    cdef list captured_positions
    cdef set next_candidates, new_neighbors
    cdef unsigned long long next_hash
    cdef int flag

    # (Supposons que sort_candidates est bien défini ailleurs et appelé ici)
    candidates = sort_candidates(board, candidate_board, player)
    
    if not candidates:
        return evaluate_board(board, player, player1_captures, player2_captures)

    if is_maximizing:
        max_score = -INF
        for move in candidates:
            m_r, m_c = move[0], move[1]
            
            # --- DO ---
            board[m_r, m_c] = current_player
            # MAGIE XOR : On ajoute la pierre au Hash
            next_hash = current_hash ^ ZOBRIST_TABLE[m_r][m_c][current_player - 1]
            update_candidates(candidate_board, m_r, m_c, 1)
            p1_cap = player1_captures
            p2_cap = player2_captures
            captured_positions = apply_capture(board, move, current_player)
            
            # MAGIE XOR : S'il y a eu capture, il faut effacer ces pierres du Hash !
            for cap_r, cap_c in captured_positions:
                next_hash ^= ZOBRIST_TABLE[cap_r][cap_c][opponent - 1]
                update_candidates(candidate_board, cap_r, cap_c, -1)
            captured = len(captured_positions)
            if current_player == 1: p1_cap += captured
            else: p2_cap += captured
            # --- EVALUATE ---
            score = minimax(
                board, depth - 1, alpha, beta, False, player, 
                p1_cap, p2_cap, candidate_board, move, next_hash # On passe le Hash !
            )
            
            max_score = max(max_score, score)
            alpha = max(alpha, score)
            
            # --- UNDO ---
            board[m_r, m_c] = 0
            update_candidates(candidate_board, m_r, m_c, -1)
            for r, c in captured_positions:
                 board[r, c] = opponent
                 update_candidates(candidate_board, r, c, 1)
            if beta <= alpha:
                break
                
        # 4. ÉCRITURE DANS LA TRANSPOSITION TABLE
        if max_score <= original_alpha: flag = FLAG_ALPHA
        elif max_score >= beta: flag = FLAG_BETA
        else: flag = FLAG_EXACT
        transposition_table[tt_key] = (depth, max_score, flag)
        
        return max_score
        
    else:
        min_score = INF
        for move in candidates:
            m_r, m_c = move[0], move[1]
            
            # --- DO ---
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
            else: p2_cap += captured
    

            # --- EVALUATE ---
            score = minimax(
                board, depth - 1, alpha, beta, True, player, 
                p1_cap, p2_cap, candidate_board, move, next_hash
            )
            
            min_score = min(min_score, score)
            beta = min(beta, score)
            
            # --- UNDO ---
            board[m_r, m_c] = 0
            update_candidates(candidate_board, m_r, m_c, -1)
            for r, c in captured_positions:
                board[r, c] = player
                update_candidates(candidate_board, r, c, 1)
            if beta <= alpha:
                break
                
        # 4. ÉCRITURE DANS LA TRANSPOSITION TABLE
        if min_score >= original_beta: flag = FLAG_BETA
        elif min_score <= alpha: flag = FLAG_ALPHA
        else: flag = FLAG_EXACT
        transposition_table[tt_key] = (depth, min_score, flag)
                
        return min_score


# =============================================================================
# 4. LA RACINE DE L'IA
# =============================================================================
cpdef tuple get_best_move(
    cnp.ndarray board,
    int player,
    int player1_captures,
    int player2_captures,
    tuple last_move,
    int depth = 10
):
    global compteur_heuristique
    compteur_heuristique = 0
    
    # 1. Initialiser le hachage (exécuté qu'une fois)
    init_zobrist()
    
    # 2. Vider le dictionnaire pour éviter que la RAM explose entre deux coups
    transposition_table.clear()
    
    # 3. Calculer l'empreinte de départ de ce tour
    cdef unsigned long long current_hash = compute_initial_hash(board)
    
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
    cdef unsigned long long next_hash
    

   # 🚨 CRÉATION DE LA MATRICE D'INFLUENCE EN MÉMOIRE C
    cdef cnp.ndarray[cnp.int32_t, ndim=2] cand_np = np.zeros((19, 19), dtype=np.int32)
    cdef int[:, :] candidate_board = cand_np
    
    
    # Remplissage initial de l'influence !
    for r in range(19):
        for c in range(19):
            if board[r, c] != 0:
                is_empty_board = False
                update_candidates(candidate_board, r, c, 1)
                
    if is_empty_board:
        return ((9, 9), 0.0)
    sorted_candidates = sort_candidates(board, candidate_board, player)
    for move in sorted_candidates:
        m_r, m_c = move[0], move[1]
        
        # --- DO ---
        board[m_r, m_c] = player
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
        else: p2_cap += captured

        score = minimax(
            board, depth - 1, alpha, beta, False, player, 
            p1_cap, p2_cap, candidate_board, move, next_hash
        )
        if score > best_score:
            best_score = score
            best_move = move
            
        alpha = max(alpha, best_score)
        
        # --- UNDO ---
        for r, c in captured_positions:
            board[r, c] = opponent
            update_candidates(candidate_board, r, c, 1)
        board[m_r, m_c] = 0        
        update_candidates(candidate_board, m_r, m_c, -1)
        if best_score == INF:
            break

    cdef double elapsed = (time.time() - start) * 1000  # ms
    print(f"🧠 L'IA a terminé ! Noeuds évalués : {compteur_heuristique}")
    print(f"AI move: {best_move} | score: {best_score} | time: {elapsed:.1f}ms | depth: {depth}")
    
    return best_move, best_score

cpdef list sort_candidates(cnp.ndarray board, int[:, :] candidate_board, int player):
    cdef int center = 9
    cdef int opponent = 2 if player == 1 else 1
    
    # Déclarations strictes Cython
    cdef int r, c, nr, nc, dr, dc, d, step
    cdef int count_ally, count_enemy
    cdef float score, distance
    cdef list scored_moves = []
    cdef tuple item
    
    # Les 4 axes directionnels : Horizontal (-), Vertical (|), Diagonales (\ et /)
    cdef int dirs[4][2]
    dirs[0][0] = 0; dirs[0][1] = 1
    dirs[1][0] = 1; dirs[1][1] = 0
    dirs[2][0] = 1; dirs[2][1] = 1
    dirs[3][0] = 1; dirs[3][1] = -1

    for r in range(19):
        for c in range(19):
            # 🚨 Le filtre d'influence (on saute les cases inutiles instantanément)
            if board[r, c] == 0 and candidate_board[r, c] > 0:
                score = 0.0
                
                # Scan directionnel sur les 4 axes
                for d in range(4):
                    dr = dirs[d][0]
                    dc = dirs[d][1]
                    
                    # 1. Compter les ALLIÉS consécutifs sur cet axe
                    count_ally = 0
                    for step in range(1, 5): # Direction positive
                        nr = r + dr * step
                        nc = c + dc * step
                        if 0 <= nr < 19 and 0 <= nc < 19 and board[nr, nc] == player:
                            count_ally += 1
                        else: break # La ligne est brisée
                        
                    for step in range(1, 5): # Direction négative
                        nr = r - dr * step
                        nc = c - dc * step
                        if 0 <= nr < 19 and 0 <= nc < 19 and board[nr, nc] == player:
                            count_ally += 1
                        else: break
                        
                    # 2. Compter les ENNEMIS consécutifs sur cet axe
                    count_enemy = 0
                    for step in range(1, 5): # Direction positive
                        nr = r + dr * step
                        nc = c + dc * step
                        if 0 <= nr < 19 and 0 <= nc < 19 and board[nr, nc] == opponent:
                            count_enemy += 1
                        else: break
                        
                    for step in range(1, 5): # Direction négative
                        nr = r - dr * step
                        nc = c - dc * step
                        if 0 <= nr < 19 and 0 <= nc < 19 and board[nr, nc] == opponent:
                            count_enemy += 1
                        else: break
                        
                    # 3. DISTRIBUTION DES POINTS (L'instinct de l'IA)
                    # Opportunités offensives (On gagne)
                    if count_ally >= 4: score += 100000.0
                    elif count_ally == 3: score += 1000.0
                    elif count_ally == 2: score += 100.0
                    elif count_ally == 1: score += 10.0
                    
                    # Opportunités défensives (On bloque une menace vitale)
                    if count_enemy >= 4: score += 80000.0
                    elif count_enemy == 3: score += 800.0
                    elif count_enemy == 2: score += 80.0
                    elif count_enemy == 1: score += 8.0

                # 4. Le départage spatial (On privilégie le centre à score tactique égal)
                distance = abs(r - center) + abs(c - center)
                score -= distance * 0.1
                
                scored_moves.append((score, (r, c)))

    # Tri du meilleur score au pire
    scored_moves.sort(reverse=True)
    
    # On retourne uniquement les 40 meilleurs coups (Beam Search)
    return [item[1] for item in scored_moves[:10]]