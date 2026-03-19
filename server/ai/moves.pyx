from ai.constants import BOARD_SIZE, EMPTY
import numpy as np

def get_possible_captures(board: np.ndarray, player: int) -> list[tuple[int, int]]:
    """
    Returns all moves where `player` can capture a pair of opponent stones.
    Pattern: player, opp, opp, EMPTY  →  player plays at EMPTY
         or: EMPTY, opp, opp, player  →  player plays at EMPTY
    """
    opponent = 2 if player == 1 else 1
    captures = []
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board[row][col] != EMPTY:
                continue
            for dr, dc in directions:
                # Check pattern: EMPTY, opp, opp, player
                r1, c1 = row + dr, col + dc
                r2, c2 = row + 2 * dr, col + 2 * dc
                r3, c3 = row + 3 * dr, col + 3 * dc
                if (
                    0 <= r3 < BOARD_SIZE and 0 <= c3 < BOARD_SIZE
                    and board[r1][c1] == opponent
                    and board[r2][c2] == opponent
                    and board[r3][c3] == player
                ):
                    captures.append((row, col))

    return captures


def apply_capture(board: np.ndarray, move: tuple[int, int], player: int) -> list[tuple[int, int]]:
    """
    Places a stone at move for player, removes any captured pairs.
    Returns number of stones captured (0 or 2 per direction, can be multiple).
    """
    row, col = move
    board[row][col] = player
    opponent = 2 if player == 1 else 1
    captured = 0
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    captured_positions = []
    for dr, dc in directions:
        for sign in (1, -1):
            r1, c1 = row + sign * dr, col + sign * dc
            r2, c2 = row + sign * 2 * dr, col + sign * 2 * dc
            r3, c3 = row + sign * 3 * dr, col + sign * 3 * dc
            if (
                0 <= r3 < BOARD_SIZE and 0 <= c3 < BOARD_SIZE
                and board[r1][c1] == opponent
                and board[r2][c2] == opponent
                and board[r3][c3] == player
            ):
                board[r1][c1] = EMPTY
                board[r2][c2] = EMPTY
                captured_positions.append((r1,c1))
                captured_positions.append((r2,c2))
                captured += 2
    return captured_positions

def create_snapshot(state):
    return {
        "board": state["board"].copy(), # Indispensable pour copier un array NumPy !
        "last_play": state["last_play"].copy(),
        "captured_white_black": state["captured_white_black"].copy()
    }

def restore_snapshot(state, snapshot):
    state["board"] = snapshot["board"].copy()
    state["last_play"] = snapshot["last_play"].copy()
    state["captured_white_black"] = snapshot["captured_white_black"].copy()

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
    if check_win_result:
        return check_win_result
    if  last_play != [0, -1] and check_win(board,last_play[0], last_play[1], "oppo",captured_white_black):
        opposite_color = 2 if color == 1 else 1
        return opposite_color
    game_state["last_play"] = [row, col]
    return 0

def check_win(board, last_row, last_col, winner,captured_white_black):
    color = board[last_row][last_col]
    if color == 0:
        return 0
    if (captured_white_black[color - 1] >= 10 ):
        return color
    directions = [
        (0, 1),  (1, 0),
        (1, 1),  (1, -1)
    ]
    
    for dr, dc in directions:
        count = 1
        win_line = [[last_row, last_col]]       
        # Sens direct (+)
        for i in range(1, 5):
            r = last_row + (dr * i)
            c = last_col + (dc * i)
            if 0 <= r < 19 and 0 <= c < 19 and board[r][c] == color:
                count += 1
                win_line.append([r, c])
            else:
                break
        # Sens opposé (-)
        for i in range(1, 5):
            r = last_row - (dr * i)
            c = last_col - (dc * i)
            if 0 <= r < 19 and 0 <= c < 19 and board[r][c] == color:
                count += 1
                win_line.append([r, c])
            else:
                break
        if count >= 5:
            if not check_no_capture_in_win_line(board,win_line, color) or winner == "oppo":
                return color
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
def check_no_capture_in_win_line(board, win_line, color):
    directions = [
        (0, 1),   # Horizontal
        (1, 0),   # Vertical
        (1, 1),   # Diagonale descendante
        (1, -1)   # Diagonale ascendante
    ]
    patterns = ["OXX.", ".XXO"]
    for row, col in win_line:
        for dr, dc in directions:
            line_string = ""                
            for i in range(-2, 3):
                r = row + (dr * i)
                c = col + (dc * i)
                if 0 <= r < 19 and 0 <= c < 19:
                    cell = board[r][c]
                    if cell == color:
                        line_string += "X"
                    elif cell == 0:
                        line_string += "."
                    else:
                        line_string += "O"
                else:
                    line_string += "W"
            for pattern in patterns:
                if pattern in line_string:
                    return 1       
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