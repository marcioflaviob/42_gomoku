import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useTheme } from '../context/ThemeContext';

const Header: React.FC = () => {
    const navigate = useNavigate();
    const { theme, toggleTheme } = useTheme();

    const handleReturnButton = () => {
        navigate('/');
    }

    return (
        <header className={`p-4 flex justify-between items-center ${theme === 'dark' ? 'bg-gray-800 text-white' : 'bg-white text-gray-800 border-b border-gray-200'}`}>
            <h1 className="text-xl font-bold cursor-pointer hover:opacity-80 transition-opacity" onClick={handleReturnButton}>Gomoku</h1>
            <button
                onClick={toggleTheme}
                className={`px-4 py-2 rounded-lg font-semibold transition-colors ${
                    theme === 'dark'
                        ? 'bg-amber-400 text-gray-900 hover:bg-amber-300'
                        : 'bg-gray-800 text-white hover:bg-gray-700'
                }`}
            >
                {theme === 'dark' ? '☀️ Light' : '🌙 Dark'}
            </button>
        </header>
    )
};

export default Header;