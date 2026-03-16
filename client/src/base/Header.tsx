import React from 'react';
import { useNavigate } from 'react-router-dom';

const Header: React.FC = () => {
    const navigate = useNavigate();
    const handleReturnButton = () => {
        navigate('/');
    }

    return (
        <header className="bg-gray-800 text-white p-4">
            <h1 className="text-xl font-bold cursor-pointer" onClick={handleReturnButton}>Gomoku</h1>
        </header>
    )
};

export default Header;