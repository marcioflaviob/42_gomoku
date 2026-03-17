import socketio
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ai.moves import play, undo, redo
import numpy as np
from ai.minimax import get_best_move
sio = socketio.AsyncServer(cors_allowed_origins='*', async_mode='asgi')
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Mount Socket.IO ASGI app
app_sio = socketio.ASGIApp(sio, app)

# Store games per client
games = {}

@sio.event
async def connect(sid, environ):
    print(f"🟢 Nouveau client connecté: {sid}")
    games[sid] =  {
    "board": np.zeros((19, 19), dtype=int),
    "last_play": [0, -1],
    "captured_white_black": [0,0],
    "history" : [],
    "future": []  
}

@sio.event
async def disconnect(sid):
    print(f"🔴 Client déconnecté: {sid}")
    games.pop(sid, None)

def build_board_response(game, status="success", winner=0, elapsed=0.0):
    return {
        "status": status,
        "board": game["board"].tolist(),
        "winner": winner,
        "player1Captures": game["captured_white_black"][0],
        "player2Captures": game["captured_white_black"][1],
        "aiResponseTime": elapsed,
        "canUndo": len(game["history"]) > 0,
        "canRedo": len(game["future"]) > 0,
    }


@sio.event
async def update(sid, data):
    current_game = games.get(sid)
    if not current_game:
        await sio.emit('error', {'error': 'Game not found'}, to=sid)
        return

    move = data.get("move")

    if move == "undo":
        undo(current_game)
        await sio.emit('boardUpdate', build_board_response(current_game), to=sid)
        return

    if move == "redo":
        redo(current_game)
        await sio.emit('boardUpdate', build_board_response(current_game), to=sid)
        return

    if move == "reset":
        games[sid] =  {
    "board": np.zeros((19, 19), dtype=int),
    "last_play": [0, -1],
    "captured_white_black": [0,0],
    "history" : [],
    "future": []  
}
        await sio.emit('boardUpdate', build_board_response(games[sid]), to=sid)
        return

    # Default: placePiece
    row = data.get("row")
    col = data.get("col")
    color = data.get("color")
    if row is None or col is None or color is None:
        await sio.emit('error', {"error": "Données invalides"}, to=sid)
        return
    result = play(current_game, row, col, color)

    if result == -1:
        await sio.emit('boardUpdate', {"status": "forbidden", "reason": "Double-Three"}, to=sid)
        return
    response = build_board_response(current_game, winner=result, elapsed=0)
    print(f"🔄 Mise à jour du plateau pour {sid}: {response}")
    await sio.emit('boardUpdate', response, to=sid)

    start_time = time.perf_counter()
    print("1")
    best_move, best_score = get_best_move(current_game["board"],color,current_game["captured_white_black"][0],current_game["captured_white_black"][1],4)
    print("2")
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    result = play(current_game, best_move[0], best_move[1], 2)
    elapsed_time = end_time - start_time
    if result == -1:
        await sio.emit('boardUpdate', {"status": "forbidden", "reason": "Double-Three"}, to=sid)
        return
    response = build_board_response(current_game, winner=result, elapsed=elapsed_time)
    print(f"🔄 Mise à jour du plateau pour {sid}: {response}")
    await sio.emit('boardUpdate', response, to=sid)