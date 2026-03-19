import asyncio
import socketio
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from moves import play, undo, redo

class Status:
    Empty = 0
    Player1 = 1
    Player2 = 2
    Suggested = 3

try:
    # from ai.moves import play, undo, redo, apply_capture
    from ai.minimax import get_best_move, get_heatmap_scores, get_candidates_for_heatmap
    from ai.heuristics import evaluate_board_full_mv
except ImportError as e:
    # Fallback to pure Python modules when Cython extensions are not built yet.
    print("⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️")
    print(f"error: {e}")
    # from ai.moves import play, undo, redo
    # from ai.minimax import get_best_move


def create_new_game_state():
    return {
        "board": np.zeros((19, 19), dtype=int),
        "last_play": [0, -1],
        "captured_white_black": [0, 0],
        "player1Score": 0,
        "player2Score": 0,
        "history": [],
        "future": [],
        "mode": "multiplayer",
    }


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
    games[sid] = create_new_game_state()

@sio.event
async def disconnect(sid):
    print(f"🔴 Client déconnecté: {sid}")
    games.pop(sid, None)


def build_board_response(game, status="success", winner=0, elapsed=0.0, color=2, board=None, heatmap=None):
    return {
        "status": status,
        "board": game["board"].tolist() if board is None else board,
        "heatmap": heatmap,
        "winner": int(winner),
        "player1Captures": game["captured_white_black"][0],
        "player2Captures": game["captured_white_black"][1],
        "aiResponseTime": elapsed,
        "canUndo": len(game["history"]) > 0,
        "canRedo": len(game["future"]) > 0,
        "color": color,
        "player1Score": game.get("player1Score", 0),
        "player2Score": game.get("player2Score", 0)
    }


async def emit_board_update(sid, game, status="success", winner=0, elapsed=0.0, color=2, board=None, heatmap=None):
    response = build_board_response(
        game,
        status=status,
        winner=winner,
        elapsed=elapsed,
        color=color,
        board=board,
        heatmap=heatmap
    )
    await sio.emit('boardUpdate', response, to=sid)


async def emit_forbidden(sid):
    await sio.emit('boardUpdate', {"status": "forbidden", "reason": "Double-Three"}, to=sid)


async def compute_best_move(game, player, last_move, depth):
    best_move, _best_score = await asyncio.to_thread(
        get_best_move,
        game["board"],
        player,
        game["captured_white_black"][0],
        game["captured_white_black"][1],
        last_move,
        depth,
    )
    return best_move


def update_scores(game):
    game["player1Score"] = evaluate_board_full_mv(
        game["board"],
        Status.Player1,
        game["captured_white_black"][0],
        game["captured_white_black"][1]
    )
    game["player2Score"] = evaluate_board_full_mv(
        game["board"],
        Status.Player2,
        game["captured_white_black"][1],
        game["captured_white_black"][0]
    )


def compute_player_heatmap(game, player=Status.Player1, radius=2):
    board = game["board"]
    p1_captures = int(game["captured_white_black"][0])
    p2_captures = int(game["captured_white_black"][1])
    last_move = tuple(game["last_play"])

    move_scores = get_heatmap_scores(
        board,
        player,
        p1_captures,
        p2_captures,
        last_move,
        depth=2,
        max_candidates=0,
    )

    candidates = get_candidates_for_heatmap(board, player)

    raw_scores = [float(move_scores[(row, col)]) for (row, col) in candidates if (row, col) in move_scores]

    unique_scores = sorted(set(raw_scores))
    if len(unique_scores) <= 1:
        score_to_normalized = {unique_scores[0]: 1.0} if unique_scores else {}
    else:
        denom = float(len(unique_scores) - 1)
        score_to_normalized = {
            score_value: (idx / denom)
            for idx, score_value in enumerate(unique_scores)
        }

    heatmap = [[None for _ in range(19)] for _ in range(19)]
    for (row, col) in candidates:
        if board[row, col] != Status.Empty:
            continue
        score = move_scores.get((row, col))
        if score is None:
            continue

        normalized = score_to_normalized.get(float(score), 0.0)
        heatmap[row][col] = normalized

    return heatmap


async def compute_player_heatmap_async(game, player=Status.Player1, radius=2):
    return await asyncio.to_thread(compute_player_heatmap, game, player, radius)


def build_hint_board(game, hint_move):
    board_with_hint = np.copy(game["board"])
    hint_row, hint_col = hint_move
    if board_with_hint[hint_row, hint_col] == Status.Empty:
        board_with_hint[hint_row, hint_col] = Status.Suggested
    return board_with_hint.tolist()


def get_best_move_from_heatmap(heatmap):
    best_move = None
    best_score = None
    for row in range(19):
        for col in range(19):
            score = heatmap[row][col]
            if score is None:
                continue
            if best_score is None or score > best_score:
                best_score = score
                best_move = (row, col)
    return best_move


async def handle_meta_move(sid, move, current_game, mode):
    if move == "undo":
        steps = 2 if mode != "multiplayer" else 1
        while steps > 0 and len(current_game["history"]) > 0:
            undo(current_game)
            steps -= 1
        heatmap = await compute_player_heatmap_async(current_game, player=Status.Player1)
        await emit_board_update(sid, current_game, heatmap=heatmap)
        return True

    if move == "redo":
        steps = 2 if mode != "multiplayer" else 1
        while steps > 0 and len(current_game["future"]) > 0:
            redo(current_game)
            steps -= 1
        heatmap = await compute_player_heatmap_async(current_game, player=Status.Player1)
        await emit_board_update(sid, current_game, heatmap=heatmap)
        return True

    if move == "reset":
        reset_mode = current_game.get("mode", "multiplayer")
        games[sid] = create_new_game_state()
        games[sid]["mode"] = reset_mode
        heatmap = await compute_player_heatmap_async(games[sid], player=Status.Player1)
        await emit_board_update(sid, games[sid], heatmap=heatmap)
        return True

    return False


async def run_ai_battle(sid, game):
    game["mode"] = "ai_battle"
    game["board"] = np.zeros((19, 19), dtype=int)
    game["last_play"] = [0, -1]
    game["captured_white_black"] = [0, 0]
    game["history"] = []
    game["future"] = []
    game["player1Score"] = 0
    game["player2Score"] = 0

    current_player = Status.Player1
    while True:
        last_move = tuple(game["last_play"])
        start_time = time.perf_counter()

        if len(game["history"]) == 0 and current_player == Status.Player1:
            best_move = (9, 9)
        else:
            best_move = await compute_best_move(game, current_player, last_move, depth=10)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        if best_move is None:
            break

        result = play(game, best_move[0], best_move[1], current_player)
        if result == -1:
            break
        if result == 3:
            result = 0

        update_scores(game)

        next_player = Status.Player2 if current_player == Status.Player1 else Status.Player1

        await emit_board_update(
            sid,
            game,
            winner=result,
            elapsed=elapsed_time,
            color=current_player
        )

        if result != 0:
            break

        current_player = next_player

@sio.event
async def update(sid, data):
    current_game = games.get(sid)
    if not current_game:
        await sio.emit('error', {'error': 'Game not found'}, to=sid)
        return

    move = data.get("move")
    mode = data.get("mode")
    if mode in ("multiplayer", "solo", "ai_battle"):
        current_game["mode"] = mode
    effective_mode = current_game.get("mode", "multiplayer")
    show_hints = bool(data.get("showHints", False))

    if move == "start_ai_battle":
        await run_ai_battle(sid, current_game)
        return

    if await handle_meta_move(sid, move, current_game, effective_mode):
        return

    if effective_mode == "ai_battle":
        await sio.emit('error', {'error': 'Manual moves are disabled during AI battle'}, to=sid)
        return

    # Default: placePiece
    row = data.get("row")
    col = data.get("col")
    color = data.get("color")
    if row is None or col is None or color is None:
        await sio.emit('error', {"error": "Données invalides"}, to=sid)
        return

    result = play(current_game, row, col, color)
    if result == 3:
        result = 0
    if result == -1:
        await emit_forbidden(sid)
        return
    
    heatmap = None
    if result == 0:
        heatmap = await compute_player_heatmap_async(current_game, player=Status.Player2 if color == Status.Player1 else Status.Player1)

    await emit_board_update(sid, current_game, winner=result, elapsed=0, color=color, heatmap=heatmap)

    if effective_mode != "multiplayer":
        if result != 0:
            return

        await sio.sleep(0)


        start_time = time.perf_counter()
        best_move = await compute_best_move(current_game, Status.Player2, (row, col), depth=10)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        result = play(current_game, best_move[0], best_move[1], Status.Player2)
        if result == -1:
            await emit_forbidden(sid)
            return
        
        if result == 3:
            result = 0
    
        if result == 0:
            heatmap = await compute_player_heatmap_async(current_game, player=color)

        update_scores(current_game)

        await emit_board_update(
            sid,
            current_game,
            winner=result,
            elapsed=elapsed_time,
            color=Status.Player2
        )

    update_scores(current_game)

    if show_hints:
        hint_move = get_best_move_from_heatmap(heatmap) if heatmap is not None else None
        color_to_compute = Status.Player2 if color == Status.Player1 else Status.Player1
        if hint_move is None:
            last_hint_move = tuple(current_game["last_play"])
            hint_move = await compute_best_move(
                current_game,
                color=color_to_compute if effective_mode == "multiplayer" else color,
                last_move=last_hint_move,
                depth=2
            )
        hint_board = build_hint_board(current_game, hint_move)
        await emit_board_update(
            sid,
            current_game,
            status="hint",
            winner=result,
            elapsed=0,
            color=color,
            board=hint_board,
            heatmap=heatmap,
        )
        await sio.sleep(0)