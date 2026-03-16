

import socketio
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

@sio.event
async def placePiece(sid, data):
    current_game = games.get(sid)
    if not current_game:
        await sio.emit('error', {'error': 'Game not found'}, to=sid)
        return
    row = data.get("row")
    col = data.get("col")
    color = data.get("color")
    if row is None or col is None or color is None:
        await sio.emit('error', {"error": "Données invalides"}, to=sid)
        return
    result = current_game.play(row, col, color)
    if result == -1:
        await sio.emit('boardUpdate', {"status": "forbidden", "reason": "Double-Three"}, to=sid)
        return
    response = {
        "status": "success",
        "board": current_game.board,
        "winner": result,
        "player1Captures": current_game.captured_by_white,
        "player2Captures": current_game.captured_by_black
    }
    print(f"🔄 Mise à jour du plateau pour {sid}: {response}")
    await sio.emit('boardUpdate', response, to=sid)