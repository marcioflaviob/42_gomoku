
from constants import BOARD_SIZE, EMPTY


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


def apply_capture(board: np.ndarray, move: tuple[int, int], player: int) -> int:
    """
    Places a stone at move for player, removes any captured pairs.
    Returns number of stones captured (0 or 2 per direction, can be multiple).
    """
    row, col = move
    board[row][col] = player
    opponent = 2 if player == 1 else 1
    captured = 0
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

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
                captured += 2

    return captured