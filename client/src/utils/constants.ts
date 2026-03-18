export const BOARD_SIZE = 19

export enum Status {
  Empty = 0,
  Player1 = 1,
  Player2 = 2,
  Suggested = 3
};

export enum MoveStatus {
  Success = 'success',
  Forbidden = 'forbidden'
}

export enum GameMode {
  Solo = 'solo',
  Multiplayer = 'multiplayer'
}

export interface Board {
  status: MoveStatus,
  board: Status[][],
  aiResponseTime: number,
  player1Captures: number,
  player2Captures: number,
  color: 1 | 2,
  winner: 1 | 2 | 0,
  gameMode: GameMode,
  canUndo: boolean,
  canRedo: boolean
}