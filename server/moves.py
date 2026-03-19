try:
    from ai.moves import check_win
except ImportError as e:
    # Fallback to pure Python modules when Cython extensions are not built yet.
    print("⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️")
    print(f"error: {e}")
    # from ai.moves import play, undo, redo
    # from ai.minimax import get_best_move

def create_snapshot(state):
    return {
        "board": state["board"].copy(), # Indispensable pour copier un array NumPy !
        "last_play": state["last_play"].copy(),
        "captured_white_black": state["captured_white_black"].copy(),
        "player1Score": state["player1Score"],
        "player2Score": state["player2Score"]
    }

def restore_snapshot(state, snapshot):
    state["board"] = snapshot["board"].copy()
    state["last_play"] = snapshot["last_play"].copy()
    state["captured_white_black"] = snapshot["captured_white_black"].copy()
    state["player1Score"] = snapshot["player1Score"]
    state["player2Score"] = snapshot["player2Score"]

def undo(state):
    if not state["history"]:
        return False
    state["future"].append(create_snapshot(state))
    previous_snapshot = state["history"].pop()
    restore_snapshot(state, previous_snapshot)
    return True

def redo(state):
    if not state["future"]:
        return False
    state["history"].append(create_snapshot(state))
    next_snapshot = state["future"].pop()
    restore_snapshot(state, next_snapshot)
    return True

def play(game_state, row, col, color):
    board = game_state["board"]
    captured_white_black = game_state["captured_white_black"]
    last_play = game_state["last_play"]
    if not (0 <= row < 19 and 0 <= col < 19):
        return -1
    if check_double_three(board,row, col, color):
        return -1
    game_state["history"].append(create_snapshot(game_state))
    game_state["future"].clear()
    board[row][col] = color
    check_capture(board,row, col, "remove",captured_white_black)
    check_win_result = check_win(board,row, col,"me",captured_white_black)
    if check_win_result == 1 or check_win_result == 2:
        return check_win_result
    if  last_play != [0, -1] and check_win(board,last_play[0], last_play[1], "oppo",captured_white_black):
        opposite_color = 2 if color == 1 else 1
        return opposite_color
    game_state["last_play"] = [row, col]
    return 0

def check_double_three(board, last_row, last_col, color):
    last_row = int(last_row)
    last_col = int(last_col)

    if color == 0:
        return 0
    directions = [
        (0, 1),  (1, 0),
        (1, 1),  (1, -1) 
    ]
    patterns = [
        "..XXX.",
        ".XXX..",
        ".X.XX.",
        ".XX.X."
    ]
    free_three_count = 0
    for dr, dc in directions:
        line_string = ""
        for i in range(-4, 5):
            r = last_row + (dr * i)
            c = last_col + (dc * i)
            if 0 <= r < 19 and 0 <= c < 19:
                cell = board[r][c]
                if cell == color or i == 0:
                    line_string += "X"
                elif cell == 0:
                    line_string += "."
                else:
                    line_string += "O"
            else:
                line_string += "O" 
                
        is_free_three = False
        for pattern in patterns:
            if pattern in line_string:
                is_free_three = True
                break 
        if is_free_three:
            free_three_count += 1
    if free_three_count >= 2:
        return 1
        
    return 0

def check_capture(board, last_row, last_col, action,captured_white_black):
    color = board[last_row][last_col]
    if color == 0:
        return 0
    opposite_color = 2 if color == 1 else 1
    pieces_captured_this_turn = 0
    directions = [
        (0, 1), (0, -1),   # Droite, Gauche
        (1, 0), (-1, 0),   # Bas, Haut
        (1, 1), (-1, -1),  # Diagonale descendante
        (1, -1), (-1, 1)   # Diagonale ascendante
    ]

    for dr, dc in directions:
        r1, c1 = last_row + dr, last_col + dc
        r2, c2 = last_row + 2 * dr, last_col + 2 * dc
        r3, c3 = last_row + 3 * dr, last_col + 3 * dc
        
        if 0 <= r3 < 19 and 0 <= c3 < 19:
            if (board[r1][c1] == opposite_color and
                board[r2][c2] == opposite_color and
                board[r3][c3] == color):
                if action == "remove":
                    board[r1][c1] = 0
                    board[r2][c2] = 0
                pieces_captured_this_turn += 2
                
    if pieces_captured_this_turn > 0:
        if color == 1:
            captured_white_black[0] += pieces_captured_this_turn
        elif color == 2:
            captured_white_black[1] += pieces_captured_this_turn
    return pieces_captured_this_turn