import socketio
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from game import Game

# Create Socket.IO server
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
    games[sid] = Game(is_local=False)

@sio.event
async def disconnect(sid):
    print(f"🔴 Client déconnecté: {sid}")
    games.pop(sid, None)

def build_board_response(game, status="success", winner=0, elapsed=0.0):
    return {
        "status": status,
        "board": game.board,
        "winner": winner,
        "player1Captures": game.captured_by_white,
        "player2Captures": game.captured_by_black,
        "aiResponseTime": elapsed,
        "canUndo": len(game.history) > 0,
        "canRedo": len(game.future) > 0,
    }

@sio.event
async def update(sid, data):
    current_game = games.get(sid)
    if not current_game:
        await sio.emit('error', {'error': 'Game not found'}, to=sid)
        return

    move = data.get("move")

    if move == "undo":
        current_game.undo()
        await sio.emit('boardUpdate', build_board_response(current_game), to=sid)
        return

    if move == "redo":
        current_game.redo()
        await sio.emit('boardUpdate', build_board_response(current_game), to=sid)
        return

    if move == "reset":
        games[sid] = Game(is_local=False)
        await sio.emit('boardUpdate', build_board_response(games[sid]), to=sid)
        return

    # Default: placePiece
    row = data.get("row")
    col = data.get("col")
    color = data.get("color")
    if row is None or col is None or color is None:
        await sio.emit('error', {"error": "Données invalides"}, to=sid)
        return
    start_time = time.perf_counter()
    result = current_game.play(row, col, color)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    if result == -1:
        await sio.emit('boardUpdate', {"status": "forbidden", "reason": "Double-Three"}, to=sid)
        return
    response = build_board_response(current_game, winner=result, elapsed=elapsed_time)
    print(f"🔄 Mise à jour du plateau pour {sid}: {response}")
    await sio.emit('boardUpdate', response, to=sid)