import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { GameMode } from '../utils/constants';
import { useTheme } from '../context/ThemeContext';

const HomePage: React.FC = () => {
    const { theme } = useTheme();
    const [pro, setPro] = useState<boolean>(false);

    return (
        <div className={`flex flex-col items-center justify-center gap-6 h-full ${theme === 'dark' ? 'bg-gray-900' : 'bg-amber-50'}`}>
            <div className="text-center">
                <h1 className={`text-5xl font-bold mb-2 ${theme === 'dark' ? 'text-white' : 'text-gray-800'}`}>Gomoku</h1>
                <p className={`text-lg ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>A 42 project written by Sebastien Bo and Marcio Brandao</p>
            </div>
            <div className='flex flex-row gap-6'>
                <label className="flex items-center cursor-pointer">
                    <input
                    type="checkbox"
                    checked={pro}
                    onChange={() => setPro((prev) => !prev)}
                    className="accent-blue-500 mr-1"
                    />
                    <span className={`text-sm font-medium ${theme === 'dark' ? 'text-white' : 'text-gray-800'}`}>Pro mode</span>
                </label>
            </div>
            <div className='flex flex-row gap-6'>
                <Link to="/board" state={{ mode: GameMode.Multiplayer, pro: pro }} className={`px-8 py-3 font-semibold rounded-lg transition-colors ${theme === 'dark' ? 'bg-amber-400 text-gray-900 hover:bg-amber-300' : 'bg-gray-800 text-white hover:bg-gray-700'}`}>
                    Play with a friend
                </Link>
                <Link to="/board" state={{ mode: GameMode.Solo, pro: pro }} className={`px-8 py-3 font-semibold rounded-lg transition-colors ${theme === 'dark' ? 'bg-amber-400 text-gray-900 hover:bg-amber-300' : 'bg-gray-800 text-white hover:bg-gray-700'}`}>
                    Play against AI
                </Link>
            </div>
            <div className='flex flex-row gap-6'>
                <Link to="/board" state={{ mode: GameMode.AIBattle, pro: pro }} className={`px-8 py-3 font-semibold rounded-lg transition-colors ${theme === 'dark' ? 'bg-amber-400 text-gray-900 hover:bg-amber-300' : 'bg-gray-800 text-white hover:bg-gray-700'}`}>
                    AI vs AI
                </Link>
            </div>
        </div>
    )
}

export default HomePage;