import React, { useEffect, useRef, useState } from 'react';
import { BOARD_SIZE, Status, type Board } from '../utils/constants';
import { io, Socket } from 'socket.io-client';

const BoardPage: React.FC = () => {
    const socketRef = useRef<Socket | null>(null);
    const [hoveredCell, setHoveredCell] = useState<{ row: number; col: number } | null>(null);
    const [player1Captures, setPlayer1Captures] = useState<number>(0);
    const [player2Captures, setPlayer2Captures] = useState<number>(0);
    const [aiResponseTime, setAiResponseTime] = useState<number>(0);
    const boardRef = useRef<Status[][]>(Array.from({ length: BOARD_SIZE }, () =>
      Array.from({ length: BOARD_SIZE }, () => Status.Empty)
    ));

    // temporary
    if (boardRef.current.length === BOARD_SIZE && boardRef.current[0].length === BOARD_SIZE) {
      boardRef.current[3][3] = Status.Player1;
      boardRef.current[3][4] = Status.Player2;
      boardRef.current[4][3] = Status.Player2;
      boardRef.current[4][4] = Status.Player1;
      boardRef.current[9][9] = Status.Forbidden;
      boardRef.current[10][10] = Status.Suggested;
    }

    const rows = Array.from({ length: boardRef.current.length }, (_, i) => i);
    const cols = Array.from({ length: boardRef.current[0].length }, (_, i) => i);
    const cellSize = 40;

    // Column indicators: A-S for 19 columns
    const colIndicators = Array.from({ length: BOARD_SIZE }, (_, i) => String.fromCharCode(65 + i));

    const handlePiecePlacement = (row: number, col: number, piece: Status) => {
      if (piece !== Status.Empty && piece !== Status.Suggested) return; // Only allow placing on empty or suggested cells
      console.log(`row ${row} col ${col} piece ${piece}`)

      socketRef.current?.emit('placePiece', { row, col });
    }

    const getClassNameForPiece = (piece: Status) => {
      switch (piece) {
        case Status.Empty:
          return 'hover:opacity-50 hover:bg-blue-300 cursor-pointer';
        case Status.Player1:
          return 'bg-black';
        case Status.Player2:
          return 'bg-white border border-gray-400';
        case Status.Forbidden:
          return 'hover:opacity-50 hover:bg-red-500 cursor-not-allowed';
        case Status.Suggested:
          return 'hover:opacity-50 bg-green-500 cursor-pointer';
        default:
          return '';
      }
    }

    useEffect(() => {
      socketRef.current = io(import.meta.env.VITE_API_URL);

      socketRef.current.on('boardUpdate', (data: Board & { player1Captures?: number; player2Captures?: number; aiResponseTime?: number }) => {
        boardRef.current = data.board;
        if (data.player1Captures !== undefined) setPlayer1Captures(data.player1Captures);
        if (data.player2Captures !== undefined) setPlayer2Captures(data.player2Captures);
        if (data.aiResponseTime !== undefined) setAiResponseTime(data.aiResponseTime);
      });

      return () => {
        socketRef.current?.disconnect();
      };
    }, []);

    return (
    <div className="flex flex-col items-center justify-start gap-6 p-6 h-full bg-amber-50">
      <div className="text-center">
        <h1 className="text-3xl font-bold text-gray-800">Gomoku Board</h1>
      </div>

      {/* Game stats */}
      <div className="flex gap-8 justify-center w-full">
        <div className="bg-white rounded-lg shadow-md p-4 min-w-max">
          <h3 className="text-sm font-semibold text-gray-700 mb-2">Player 1 Captures</h3>
          <p className="text-2xl font-bold text-black">{player1Captures}</p>
        </div>
        <div className="bg-white rounded-lg shadow-md p-4 min-w-max">
          <h3 className="text-sm font-semibold text-gray-700 mb-2">Player 2 Captures</h3>
          <p className="text-2xl font-bold text-white bg-gray-900 border border-gray-600 w-fit px-2 rounded">{player2Captures}</p>
        </div>
        <div className="bg-white rounded-lg shadow-md p-4 min-w-max">
          <h3 className="text-sm font-semibold text-gray-700 mb-2">AI Response Time</h3>
          <p className="text-2xl font-bold text-blue-600">{aiResponseTime}ms</p>
        </div>
      </div>

      <div className="relative inline-block" role="region" aria-label="Gomoku board" style={{
        width: `${(BOARD_SIZE - 1) * cellSize + 60}px`,
        height: `${(BOARD_SIZE - 1) * cellSize + 60}px`,
      }}>
        {/* Column indicators (A-S) */}
        {colIndicators.map((letter, idx) => (
          <span
            key={`col-indicator-${letter}`}
            className={`absolute text-xs font-semibold select-none ${hoveredCell && hoveredCell.col === idx ? 'bg-blue-300 text-blue-900 rounded px-1' : 'text-gray-700'}`}
            style={{
              left: `${20 + idx * cellSize}px`,
              top: '0px',
              transform: 'translate(-50%, 0)',
              width: '24px',
              textAlign: 'center',
            }}
          >{letter}</span>
        ))}

        {/* Row indicators (1-19) */}
        {rows.map((row) => (
          <span
            key={`row-indicator-${row}`}
            className={`absolute text-xs font-semibold select-none ${hoveredCell && hoveredCell.row === row ? 'bg-blue-300 text-blue-900 rounded px-1' : 'text-gray-700'}`}
            style={{
              left: '0px',
              top: `${20 + row * cellSize}px`,
              transform: 'translate(0, -50%)',
              width: '24px',
              textAlign: 'center',
            }}
          >{row + 1}</span>
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
              stroke="#8b7355"
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
              stroke="#8b7355"
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
                fill="#8b7355"
              />
            ))
          )}
        </svg>

        {rows.map((row) =>
          cols.map((col) => {
            const piece = boardRef.current[row][col];
            return (
              <button
                key={`${row}-${col}`}
                type="button"
                className={`absolute w-6 h-6 rounded-full -translate-x-1/2 -translate-y-1/2 ${getClassNameForPiece(piece)}`}
                aria-label={`Cell ${row + 1}, ${colIndicators[col]}`}
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
