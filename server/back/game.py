class Game:
    def __init__(self, is_local=False):
        self.board = [[0 for _ in range(19)] for _ in range(19)]
        self.captured_by_white = 0
        self.captured_by_black = 0
        self.is_local = is_local
        self.last_play = [0,-1]

    def play(self, coord, color):
        col_index = ord(coord[0].upper()) - 65
        row_index = int(coord[1:]) - 1
        position = (19 * row_index) + col_index
        row = position // 19
        col = position % 19
        if (self.check_double_three(row,col,color)):
            return -1
        self.board[row][col] = color
        check_win_result = self.check_win(row, col) 
        if check_win_result:
            return check_win_result            
        self.check_capture(row, col, "remove")
        if self.last_play != [0,-1] and self.check_win(self.last_play[0], self.last_play[1],"oppo"):
            opposite_color = 2 if color == 1 else 1
            return opposite_color
        self.last_play = [row,col]
        return 0

    def check_win(self, last_row, last_col, winner = "me"):
        color = self.board[last_row][last_col]
        if color == 0:
            return 0
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
                if 0 <= r < 19 and 0 <= c < 19 and self.board[r][c] == color:
                    count += 1
                    win_line.append([r, c])
                else:
                    break
            # Sens opposé (-)
            for i in range(1, 5):
                r = last_row - (dr * i)
                c = last_col - (dc * i)
                if 0 <= r < 19 and 0 <= c < 19 and self.board[r][c] == color:
                    count += 1
                    win_line.append([r, c])
                else:
                    break
            if count >= 5:
                if not self.check_no_capture_in_win_line(win_line, color) or winner == "oppo":
                    return color
        return 0

    def check_capture(self, last_row, last_col, action):
        color = self.board[last_row][last_col]
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
                if (self.board[r1][c1] == opposite_color and
                    self.board[r2][c2] == opposite_color and
                    self.board[r3][c3] == color):
                    if action == "remove":
                        self.board[r1][c1] = 0
                        self.board[r2][c2] = 0
                    pieces_captured_this_turn += 2
                    
        if pieces_captured_this_turn > 0:
            if color == 1:
                self.captured_by_black += pieces_captured_this_turn
            elif color == 2:
                self.captured_by_white += pieces_captured_this_turn
        return pieces_captured_this_turn

    def check_no_capture_in_win_line(self, win_line, color):
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
                        cell = self.board[r][c]
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

    def check_double_three(self, last_row, last_col, color):
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
                    cell = self.board[r][c]
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

