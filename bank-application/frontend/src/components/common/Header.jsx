import React from 'react';
import { useAuth } from '../../context/AuthContext';
import './Header.css';

const Header = () => {
  const { user, logout } = useAuth();

  const handleLogout = () => {
    logout();
  };

  return (
    <header className="app-header">
      <div className="header-content">
        <div className="header-left">
          <h1>Bank Management System</h1>
        </div>
        <div className="header-right">
          <span className="user-welcome">Welcome, {user?.full_name}</span>
          <button onClick={handleLogout} className="logout-btn">
            Logout
          </button>
        </div>
      </div>
    </header>
  );
};

export default Header;
