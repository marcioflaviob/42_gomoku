import React from 'react';
import { Link } from 'react-router-dom';

const HomePage: React.FC = () => {

    return (
        <div className="flex flex-col items-center justify-center gap-6 h-full bg-amber-50">
            <div className="text-center">
                <h1 className="text-5xl font-bold text-gray-800 mb-2">Gomoku</h1>
                <p className="text-gray-600 text-lg">A 42 project written by Sebastien Bo and Marcio Brandao</p>
            </div>
            <div className='flex flex-row gap-6'>
                <Link to="/board" className="px-8 py-3 bg-gray-800 text-white font-semibold rounded-lg hover:bg-gray-700 transition-colors">
                    Play with a friend
                </Link>
                <Link to="/board" className="px-8 py-3 bg-gray-800 text-white font-semibold rounded-lg hover:bg-gray-700 transition-colors">
                    Play against AI
                </Link>
            </div>
        </div>
    )
}

export default HomePage;