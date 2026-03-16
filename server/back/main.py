from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from game import Game  # On importe ta classe depuis le fichier game.py

app = FastAPI()

@app.websocket("/ws/gomoku")
async def websocket_endpoint(websocket: WebSocket):
    # 1. On accepte la connexion du client
    await websocket.accept()
    print("🟢 Nouveau client connecté")

    # 2. On crée une NOUVELLE instance de jeu pour cette session WebSocket
    # (Si tu veux que deux joueurs partagent la même partie, il faudra stocker 
    #  cette instance dans un dictionnaire global en dehors de la route)
    current_game = Game(is_local=False)

    try:
        # 3. Boucle infinie pour écouter les messages du client
        while True:
            # On attend un message JSON (ex: {"coord": "H8", "color": 1})
            data = await websocket.receive_json()
            
            coord = data.get("coord")
            color = data.get("color")

            # Sécurité basique
            if not coord or not color:
                await websocket.send_json({"error": "Données invalides"})
                continue

            # On vérifie d'abord le Double-Trois (ton algorithme précédent)
            row_idx = int(coord[1:]) - 1
            col_idx = ord(coord[0].upper()) - 65
            
            # Note : on simule le check sans placer le pion
            if current_game.check_double_three(row_idx, col_idx, color) == 1:
                # Si le coup n'est pas une capture (à affiner selon ta logique)
                await websocket.send_json({"status": "forbidden", "reason": "Double-Three"})
                continue

            # 4. On joue le coup
            result = current_game.play(coord, color)
            if result == -1:
                await websocket.send_json({"status": "forbidden", "reason": "Double-Three"})
                continue
            # 5. On prépare la réponse
            response = {
                "status": "success",
                "board" : current_game.board,
                "coord": coord,
                "color": color,
                "winner": result,  # 0 si rien, 1 ou 2 si quelqu'un a gagné
                "captures_white": current_game.captured_by_white,
                "captures_black": current_game.captured_by_black
            }

            # 6. On renvoie l'état au client
            await websocket.send_json(response)

    except WebSocketDisconnect:
        # Géré automatiquement si le client ferme l'onglet ou perd la connexion
        print("🔴 Client déconnecté")