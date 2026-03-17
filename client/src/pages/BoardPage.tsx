import React, { useEffect, useRef, useState } from 'react';
import { BOARD_SIZE, GameMode, MoveStatus, Status, type Board } from '../utils/constants';
import { io, Socket } from 'socket.io-client';
import { useLocation } from 'react-router-dom';
import { useTheme } from '../context/ThemeContext';

const BoardPage: React.FC = () => {
    const location = useLocation();
    const { mode } : { mode: GameMode } = location.state || {};
    const { theme } = useTheme();
    const [currentPlayer, setCurrentPlayer] = useState<number>(1);
    const [loading, setLoading] = useState<boolean>(false);
    const socketRef = useRef<Socket | null>(null);
    const [hoveredCell, setHoveredCell] = useState<{ row: number; col: number } | null>(null);
    const [player1Captures, setPlayer1Captures] = useState<number>(0);
    const [player2Captures, setPlayer2Captures] = useState<number>(0);
    const [aiResponseTime, setAiResponseTime] = useState<number>(0);
    const [winner, setWinner] = useState<Board['winner']>(0);
    const [status, setStatus] = useState<MoveStatus>(MoveStatus.Success);
    const [canUndo, setCanUndo] = useState<boolean>(false);
    const [canRedo, setCanRedo] = useState<boolean>(false);
    const [board, setBoard] = useState<Status[][]>(Array.from({ length: BOARD_SIZE }, () =>
      Array.from({ length: BOARD_SIZE }, () => Status.Empty)
    ));

    const rows = Array.from({ length: BOARD_SIZE }, (_, i) => i);
    const cols = Array.from({ length: BOARD_SIZE }, (_, i) => i);
    const cellSize = 40;
    const statusMessage = winner !== 0
      ? `Player ${winner} wins!`
      : status !== MoveStatus.Success
        ? 'Forbidden move. Try another position.'
        : '';
    const statusClassName = winner !== 0
      ? 'bg-emerald-100 text-emerald-800 border-emerald-300'
      : status !== MoveStatus.Success
        ? 'bg-red-100 text-red-800 border-red-300'
        : 'bg-transparent text-transparent border-transparent';

    const [showHints, setShowHints] = useState<boolean>(false);
    const [showHeatmap, setShowHeatmap] = useState<boolean>(false);

    // Column indicators: 0-18 for 19 columns
    const colIndicators = Array.from({ length: BOARD_SIZE }, (_, i) => i.toString());

    const handlePiecePlacement = (row: number, col: number, piece: Status) => {
      if (piece !== Status.Empty && piece !== Status.Suggested) return;
      if (loading || winner) return;
      socketRef.current?.emit('update', { move: "placePiece", row, col, color: currentPlayer });
      if (mode === GameMode.Multiplayer) setCurrentPlayer((prev) => (prev === 1 ? 2 : 1));
      setLoading(true);
    };

    const handleUndo = () => {
      if (!canUndo || loading) return;
      socketRef.current?.emit('update', { move: 'undo' });
      setLoading(true);
    };

    const handleRedo = () => {
      if (!canRedo || loading) return;
      socketRef.current?.emit('update', { move: 'redo' });
      setLoading(true);
    };

    const handlePlayAgain = () => {
      if (winner === 0 || loading) return;
      socketRef.current?.emit('update', { move: 'reset' });
      setCurrentPlayer(1);
      setLoading(true);
    };

    const getClassNameForPiece = (piece: Status) => {
      switch (piece) {
        case Status.Empty:
          return 'hover:opacity-50 hover:bg-blue-300 cursor-pointer';
        case Status.Player1:
          return 'bg-white border border-gray-400';
        case Status.Player2:
          if (theme === 'dark') return 'bg-blue-500 border border-gray-400';
          return 'bg-black';
        case Status.Suggested:
          return 'hover:opacity-50 bg-green-500 cursor-pointer';
        default:
          return '';
      }
    }

    useEffect(() => {
      socketRef.current = io(import.meta.env.VITE_API_URL);

      socketRef.current.on('boardUpdate', (data: Board) => {
        if (data.board) setBoard(data.board);
        if (data.status) setStatus(data.status);
        if (data.player1PiecesCaptured !== undefined) setPlayer1Captures(data.player1PiecesCaptured);
        if (data.player2PiecesCaptured !== undefined) setPlayer2Captures(data.player2PiecesCaptured);
        if (data.aiResponseTime !== undefined) setAiResponseTime(data.aiResponseTime);
        if (data.canUndo !== undefined) setCanUndo(data.canUndo);
        if (data.canRedo !== undefined) setCanRedo(data.canRedo);
        if (data.winner !== undefined) setWinner(data.winner);
        setLoading(false);
      });

      socketRef.current?.on('error', (err) => {
        console.error('Socket error:', err);
      });

      return () => {
        socketRef.current?.disconnect();
      };
    }, []);

    return (
    <div className={`flex flex-col items-center justify-start gap-6 p-6 h-full ${theme === 'dark' ? 'bg-gray-900' : 'bg-amber-50'}`}>
      <div className="flex items-center gap-4">
        <h1 className={`text-3xl font-bold ${theme === 'dark' ? 'text-white' : 'text-gray-800'}`}>Gomoku Board</h1>
        <button
          onClick={handleUndo}
          disabled={!canUndo || loading}
          className={`px-3 py-1 rounded font-semibold disabled:opacity-40 transition-colors ${
            theme === 'dark' 
              ? 'bg-gray-700 text-white hover:bg-gray-600' 
              : 'bg-gray-200 text-gray-800 hover:bg-gray-300'
          }`}
        >⟲ Undo</button>
        <button
          onClick={handleRedo}
          disabled={!canRedo || loading}
          className={`px-3 py-1 rounded font-semibold disabled:opacity-40 transition-colors ${
            theme === 'dark' 
              ? 'bg-gray-700 text-white hover:bg-gray-600' 
              : 'bg-gray-200 text-gray-800 hover:bg-gray-300'
          }`}
        >⟳ Redo</button>
        {winner !== 0 && (
          <button
            onClick={handlePlayAgain}
            disabled={loading}
            className="px-3 py-1 rounded text-white font-semibold disabled:opacity-40 bg-emerald-600 hover:bg-emerald-700 transition-colors"
          >Play Again</button>
        )}
        {/* Switch buttons for hints and heatmap */}
        <div className="flex items-center gap-2 ml-6">
          <label className="flex items-center cursor-pointer">
            <input
              type="checkbox"
              checked={showHints}
              onChange={() => setShowHints((prev) => !prev)}
              className="accent-blue-500 mr-1"
            />
            <span className={`text-sm font-medium ${theme === 'dark' ? 'text-white' : 'text-gray-800'}`}>Hints</span>
          </label>
          <label className="flex items-center cursor-pointer">
            <input
              type="checkbox"
              checked={showHeatmap}
              onChange={() => setShowHeatmap((prev) => !prev)}
              className="accent-red-500 mr-1"
            />
            <span className={`text-sm font-medium ${theme === 'dark' ? 'text-white' : 'text-gray-800'}`}>Heatmap</span>
          </label>
        </div>
      </div>

      {/* Game stats */}
      <div className="flex gap-8 justify-center w-full">
        <div className={`rounded-lg shadow-md p-4 min-w-max ${theme === 'dark' ? 'bg-gray-800 text-white' : 'bg-white text-gray-900'}`}>
          <h3 className={`text-sm font-semibold mb-2 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>Player 1 Captures</h3>
          <p className="text-2xl font-bold text-black">{player1Captures}</p>
        </div>
        <div className={`rounded-lg shadow-md p-4 min-w-max ${theme === 'dark' ? 'bg-gray-800 text-white' : 'bg-white text-gray-900'}`}>
          <h3 className={`text-sm font-semibold mb-2 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>Player 2 Captures</h3>
          <p className="text-2xl font-bold text-white bg-gray-900 border border-gray-600 w-fit px-2 rounded">{player2Captures}</p>
        </div>
        <div className={`rounded-lg shadow-md p-4 min-w-max ${theme === 'dark' ? 'bg-gray-800 text-white' : 'bg-white text-gray-900'}`}>
          <h3 className={`text-sm font-semibold mb-2 ${theme === 'dark' ? 'text-gray-300' : 'text-gray-700'}`}>AI Response Time</h3>
          <p className="text-2xl font-bold text-blue-600">{aiResponseTime.toFixed(5)}ms</p>
        </div>
      </div>

      <div className="w-full flex justify-center" aria-live="polite">
        <div
          className={`min-h-10 px-4 py-2 rounded-md border font-semibold transition-opacity ${statusMessage ? 'opacity-100' : 'opacity-0'} ${statusClassName}`}
        >
          {statusMessage || 'Status'}
        </div>
      </div>

      <div className="relative inline-block" role="region" aria-label="Gomoku board" style={{
        width: `${(BOARD_SIZE - 1) * cellSize + 60}px`,
        height: `${(BOARD_SIZE - 1) * cellSize + 60}px`,
      }}>
        {/* Column indicators (0-18) */}
        {colIndicators.map((colNum, idx) => (
          <span
            key={`col-indicator-${colNum}`}
            className={`absolute text-xs font-semibold select-none ${hoveredCell && hoveredCell.col === idx ? theme === 'dark' ? 'bg-amber-400 text-gray-900 rounded px-1' : 'bg-blue-300 text-blue-900 rounded px-1' : theme === 'dark' ? 'text-gray-400' : 'text-gray-700'}`}
            style={{
              left: `${20 + idx * cellSize}px`,
              top: '0px',
              transform: 'translate(-50%, 0)',
              width: '24px',
              textAlign: 'center',
            }}
          >{colNum}</span>
        ))}

        {/* Row indicators (0-18) */}
        {rows.map((row) => (
          <span
            key={`row-indicator-${row}`}
            className={`absolute text-xs font-semibold select-none ${hoveredCell && hoveredCell.row === row ? theme === 'dark' ? 'bg-amber-400 text-gray-900 rounded px-1' : 'bg-blue-300 text-blue-900 rounded px-1' : theme === 'dark' ? 'text-gray-400' : 'text-gray-700'}`}
            style={{
              left: '0px',
              top: `${20 + row * cellSize}px`,
              transform: 'translate(0, -50%)',
              width: '24px',
              textAlign: 'center',
            }}
          >{row}</span>
        ))}

        <svg className="absolute inset-0 w-full h-full" style={{
          filter: 'none'
        }}>
          {/* Horizontal lines */}
          {rows.map((row) => (
            <line
              key={`h-${row}`}
              x1="20"
              y1={20 + row * cellSize}
              x2={20 + (BOARD_SIZE - 1) * cellSize}
              y2={20 + row * cellSize}
              stroke={theme === 'dark' ? '#4b5563' : '#8b7355'}
              strokeWidth="1.5"
            />
          ))}
          {/* Vertical lines */}
          {cols.map((col) => (
            <line
              key={`v-${col}`}
              x1={20 + col * cellSize}
              y1="20"
              x2={20 + col * cellSize}
              y2={20 + (BOARD_SIZE - 1) * cellSize}
              stroke={theme === 'dark' ? '#4b5563' : '#8b7355'}
              strokeWidth="1.5"
            />
          ))}

          {[3, 9, 15].map((pos1) =>
            [3, 9, 15].map((pos2) => (
              <circle
                key={`star-${pos1}-${pos2}`}
                cx={20 + pos1 * cellSize}
                cy={20 + pos2 * cellSize}
                r="3"
                fill={theme === 'dark' ? '#9ca3af' : '#8b7355'}
              />
            ))
          )}
        </svg>

        {rows.map((row) =>
          cols.map((col) => {
            const piece = board[row][col];
            return (
              <button
                key={`${row}-${col}`}
                type="button"
                className={`absolute w-6 h-6 rounded-full -translate-x-1/2 -translate-y-1/2 ${getClassNameForPiece(piece)}`}
                aria-label={`Cell ${row}, ${colIndicators[col]}`}
                style={{
                  left: `${20 + col * cellSize}px`,
                  top: `${20 + row * cellSize}px`,
                }}
                onClick={() => handlePiecePlacement(row, col, piece)}
                onMouseEnter={() => setHoveredCell({ row, col })}
                onMouseLeave={() => setHoveredCell(null)}
              />
          )})
        )}
      </div>
    </div>
    );
};

export default BoardPage;
