/**
 * NavigationSidebar.js
 *
 * Sidebar navigation component for the Physical AI & Humanoid Robotics textbook.
 * Provides navigation between modules and chapters with hardware-aware personalization.
 */

import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import './NavigationSidebar.module.css';

const NavigationSidebar = ({ modules }) => {
  const [activeModule, setActiveModule] = useState(null);
  const [expandedChapters, setExpandedChapters] = useState({});
  const location = useLocation();

  useEffect(() => {
    // Set active module based on current location
    const currentPath = location.pathname;
    const moduleMatch = modules.find(module =>
      currentPath.includes(module.slug) || currentPath.includes(module.path)
    );
    if (moduleMatch) {
      setActiveModule(moduleMatch.id);
      // Expand the current module's chapters
      setExpandedChapters(prev => ({
        ...prev,
        [moduleMatch.id]: true
      }));
    }
  }, [location, modules]);

  const toggleModule = (moduleId) => {
    setExpandedChapters(prev => ({
      ...prev,
      [moduleId]: !prev[moduleId]
    }));
  };

  const getActiveChapter = (module) => {
    const currentPath = location.pathname;
    return module.chapters?.find(chapter =>
      currentPath.includes(chapter.slug) || currentPath === chapter.path
    );
  };

  return (
    <div className="navigation-sidebar">
      <div className="sidebar-header">
        <h3>Physical AI Curriculum</h3>
      </div>

      <nav className="sidebar-nav">
        <ul className="module-list">
          {modules.map((module) => (
            <li key={module.id} className="module-item">
              <button
                className={`module-toggle ${activeModule === module.id ? 'active' : ''}`}
                onClick={() => toggleModule(module.id)}
              >
                <span className="module-number">{module.number}.</span>
                <span className="module-title">{module.title}</span>
                <span className={`expand-icon ${expandedChapters[module.id] ? 'expanded' : ''}`}>
                  ▼
                </span>
              </button>

              {expandedChapters[module.id] && (
                <ul className="chapter-list">
                  {module.chapters?.map((chapter) => (
                    <li key={chapter.id} className="chapter-item">
                      <a
                        href={chapter.path}
                        className={`chapter-link ${getActiveChapter(module)?.id === chapter.id ? 'active' : ''}`}
                      >
                        <span className="chapter-number">{chapter.number}.</span>
                        <span className="chapter-title">{chapter.title}</span>
                      </a>
                    </li>
                  ))}
                </ul>
              )}
            </li>
          ))}
        </ul>
      </nav>

      <div className="sidebar-footer">
        <button className="personalize-btn">
          🎯 Personalize Content
        </button>
      </div>
    </div>
  );
};

export default NavigationSidebar;