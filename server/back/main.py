from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from game import Game

app = FastAPI()

@app.websocket("/ws/gomoku")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🟢 Nouveau client connecté")
    current_game = Game(is_local=False)
    try:
        while True:
            data = await websocket.receive_json()
            coord = data.get("coord")
            color = data.get("color")
            if not coord or not color:
                await websocket.send_json({"error": "Données invalides"})
                continue
            row_idx = int(coord[1:]) - 1
            col_idx = ord(coord[0].upper()) - 65
            result = current_game.play(coord, color)
            if result == -1:
                await websocket.send_json({"status": "forbidden", "reason": "Double-Three"})
                continue
            response = {
                "status": "success",
                "board" : current_game.board,
                "winner": result,
                "player1PiecesCaptured": current_game.captured_by_white,
                "player2PiecesCaptured": current_game.captured_by_black
            }
            await websocket.send_json(response)

    except WebSocketDisconnect:
        print("🔴 Client déconnecté")