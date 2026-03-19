import { useCallback, useMemo, useState } from 'react';
import type { Board } from '../utils/constants';

type ScoreUpdate = Partial<Pick<Board, 'player1Score' | 'player2Score'>>;

interface UseScoreBarResult {
  player1Score: number;
  player2Score: number;
  player1ScorePct: number;
  player2ScorePct: number;
  updateScores: (update: ScoreUpdate) => void;
}

export const useScoreBar = (balanceCompression = 0.5): UseScoreBarResult => {
  const [player1Score, setPlayer1Score] = useState<number>(0);
  const [player2Score, setPlayer2Score] = useState<number>(0);

  const updateScores = useCallback((update: ScoreUpdate) => {
    if (update.player1Score !== undefined) setPlayer1Score(update.player1Score);
    if (update.player2Score !== undefined) setPlayer2Score(update.player2Score);
  }, []);

  const { player1ScorePct, player2ScorePct } = useMemo(() => {
    // Keep the ratio stable when scores are negative or mixed-sign.
    const minScore = Math.min(player1Score, player2Score);
    const shiftedPlayer1 = player1Score - minScore + 1;
    const shiftedPlayer2 = player2Score - minScore + 1;
    const shiftedTotal = shiftedPlayer1 + shiftedPlayer2;

    const rawPlayer1ScorePct = (shiftedPlayer1 / shiftedTotal) * 100;
    const compressedPlayer1ScorePct = 50 + (rawPlayer1ScorePct - 50) * balanceCompression;

    return {
      player1ScorePct: compressedPlayer1ScorePct,
      player2ScorePct: 100 - compressedPlayer1ScorePct,
    };
  }, [balanceCompression, player1Score, player2Score]);

  return {
    player1Score,
    player2Score,
    player1ScorePct,
    player2ScorePct,
    updateScores,
  };
};