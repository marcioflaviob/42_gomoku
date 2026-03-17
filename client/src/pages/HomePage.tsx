import React from 'react';
import { Link } from 'react-router-dom';
import { GameMode } from '../utils/constants';
import { useTheme } from '../context/ThemeContext';

const HomePage: React.FC = () => {
    const { theme } = useTheme();

    return (
        <div className={`flex flex-col items-center justify-center gap-6 h-full ${theme === 'dark' ? 'bg-gray-900' : 'bg-amber-50'}`}>
            <div className="text-center">
                <h1 className={`text-5xl font-bold mb-2 ${theme === 'dark' ? 'text-white' : 'text-gray-800'}`}>Gomoku</h1>
                <p className={`text-lg ${theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}`}>A 42 project written by Sebastien Bo and Marcio Brandao</p>
            </div>
            <div className='flex flex-row gap-6'>
                <Link to="/board" state={{ mode: GameMode.Multiplayer }} className={`px-8 py-3 font-semibold rounded-lg transition-colors ${theme === 'dark' ? 'bg-amber-400 text-gray-900 hover:bg-amber-300' : 'bg-gray-800 text-white hover:bg-gray-700'}`}>
                    Play with a friend
                </Link>
                <Link to="/board" state={{ mode: GameMode.Solo }} className={`px-8 py-3 font-semibold rounded-lg transition-colors ${theme === 'dark' ? 'bg-amber-400 text-gray-900 hover:bg-amber-300' : 'bg-gray-800 text-white hover:bg-gray-700'}`}>
                    Play against AI
                </Link>
            </div>
        </div>
    )
}

export default HomePage;