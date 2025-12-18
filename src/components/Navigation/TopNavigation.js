/**
 * TopNavigation.js
 *
 * Top navigation bar component for the Physical AI & Humanoid Robotics textbook.
 * Provides global navigation and user controls.
 */

import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import './TopNavigation.module.css';

const TopNavigation = ({ user, onLogin, onLogout, onTranslate }) => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  return (
    <nav className="top-navigation">
      <div className="nav-container">
        <div className="nav-brand">
          <Link to="/" className="nav-logo">
            <span className="logo-icon">🤖</span>
            <span className="logo-text">Physical AI</span>
          </Link>
        </div>

        <div className={`nav-menu ${isMenuOpen ? 'active' : ''}`}>
          <Link to="/" className="nav-link">
            Home
          </Link>
          <Link to="/modules" className="nav-link">
            Modules
          </Link>
          <Link to="/capstone" className="nav-link">
            Capstone
          </Link>
          <Link to="/glossary" className="nav-link">
            Glossary
          </Link>
        </div>

        <div className="nav-actions">
          <button
            className="nav-action-btn translate-btn"
            onClick={() => onTranslate && onTranslate('ur')}
            title="Translate to Urdu"
          >
            🇵🇰 UR
          </button>

          {user ? (
            <div className="user-menu">
              <button className="user-btn">
                <span className="user-avatar">👤</span>
                <span className="user-name">{user.name}</span>
              </button>
              <button
                className="logout-btn"
                onClick={onLogout}
              >
                Logout
              </button>
            </div>
          ) : (
            <button
              className="login-btn"
              onClick={onLogin}
            >
              Login
            </button>
          )}

          <button
            className="mobile-menu-btn"
            onClick={toggleMenu}
          >
            ☰
          </button>
        </div>
      </div>
    </nav>
  );
};

export default TopNavigation;