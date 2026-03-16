export const BOARD_SIZE = 19

export enum Status {
  Empty = 0,
  Player1 = 1,
  Player2 = 2,
  Forbidden = 3,
  Suggested = 4
};

export enum GameStatus {
  InProgress = 'in-progress',
  Player1Win = 'player1-win',
  Player2Win = 'player2-win'
}

export interface Board {
  board: Status[][],
  aiCalculationTime: number,
  player1PiecesCaptured: number,
  player2PiecesCaptured: number,
  gameStatus: GameStatus
  gameMode: 'solo' | 'multiplayer',
}