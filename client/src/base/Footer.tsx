import React from 'react';
import { useTheme } from '../context/ThemeContext';

const Footer: React.FC = () => {
  const { theme } = useTheme();

  return (
    <footer className={`p-4 text-center ${theme === 'dark' ? 'bg-gray-800 text-white' : 'bg-white text-gray-800 border-t border-gray-200'}`}>
      <span>A 42 project written by Sebastien Bo and Marcio Brandao</span>
    </footer>
  );
};

export default Footer;