import json
import numpy as np
from optimizer import get_candidate_moves, board_from_json

def test_candidate_moves():
    # Read board.json
    with open('board.json', 'r') as f:
        json_board = json.load(f)
    
    # Convert to numpy array
    board = board_from_json(json_board)
    
    # Get candidate moves
    candidates = get_candidate_moves(board, radius=2, player=2)
    
    # Print the result
    print("Candidate moves:", candidates)

if __name__ == "__main__":
    test_candidate_moves()

