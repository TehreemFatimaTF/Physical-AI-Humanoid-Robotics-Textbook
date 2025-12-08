import React, { useEffect, useRef, useState } from 'react';
import clsx from 'clsx';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import styles from './index.module.css';

// Floating Elements Component
const FloatingElements = () => {
  const elements = ['🤖', '🧠', '⚙️', '🔧', '💡', '🚀', '👁️', '🦾'];
  return (
    <div className="floating-elements">
      {elements.map((emoji, index) => (
        <div 
          key={index}
          className="floating-element"
          style={{
            left: `${10 + (index * 10)}%`,
            animationDelay: `${index * 2}s`,
            animationDuration: `${15 + index * 3}s`
          }}
        >
          {emoji}
        </div>
      ))}
    </div>
  );
};

// Enhanced About Section Component
const EnhancedAboutSection = () => {
  const sectionRef = useRef(null);
  const butterflyRef = useRef(null);

  useEffect(() => {
    const createButterfly = () => {
      if (!butterflyRef.current) return;
      
      const butterfly = document.createElement('div');
      butterfly.className = 'butterfly';
      
      const colors = ['#ff6b6b', '#4ecdc4', '#ffe66d', '#ff8e53', '#9b59b6'];
      const sizes = ['small', 'medium', 'large'];
      const color = colors[Math.floor(Math.random() * colors.length)];
      const size = sizes[Math.floor(Math.random() * sizes.length)];
      
      butterfly.style.left = `${Math.random() * 100}%`;
      butterfly.style.top = `${Math.random() * 100}%`;
      butterfly.style.setProperty('--butterfly-color', color);
      butterfly.classList.add(size);
      
      butterfly.innerHTML = `
        <div class="wing left"></div>
        <div class="wing right"></div>
      `;
      
      butterflyRef.current.appendChild(butterfly);
      
      const duration = 15 + Math.random() * 20;
      const xMovement = (Math.random() - 0.5) * 200;
      
      butterfly.animate([
        { transform: 'translate(0, 0) rotate(0deg)' },
        { transform: `translate(${xMovement}px, -100px) rotate(180deg)` }
      ], {
        duration: duration * 1000,
        easing: 'cubic-bezier(0.42, 0, 0.58, 1)',
        iterations: Infinity,
        direction: 'alternate'
      });
      
      setTimeout(() => {
        if (butterfly.parentNode) {
          butterfly.remove();
          createButterfly();
        }
      }, duration * 1000);
    };
    
    for (let i = 0; i < 8; i++) {
      setTimeout(createButterfly, i * 500);
    }
    
  
      
      const rect = sectionRef.current.getBoundingClientRect();
      const isInView = rect.top < window.innerHeight && rect.bottom > 0;
      
   
        
        sectionRef.current.style.transform = `perspective(1000px) rotateX(${rotation}deg) scale(${scale})`;
      }
    };
    };
  }, []);

  
// Feature Cards Component
const FeatureCards = () => {
  const features = [
    {
      title: "AI-First Approach",
      description: "Learn robotics through AI-native methods",
      icon: "🤖",
      gradient: "from-purple-500 to-pink-500"
    },
    {
      title: "Hands-On Projects",
      description: "Build real robots and simulations",
      icon: "🔧",
      gradient: "from-blue-500 to-cyan-500"
    },
    {
      title: "Industry Ready",
      description: "Skills demanded by robotics companies",
      icon: "💼",
      gradient: "from-green-500 to-emerald-500"
    },
    {
      title: "Community Driven",
      description: "Learn with global robotics enthusiasts",
      icon: "🌍",
      gradient: "from-orange-500 to-red-500"
    }
  ];

  return (
    <section className="features-section">
      <h2 className="section-title">Why This Curriculum?</h2>
      <div className="features-grid">
        {features.map((feature, index) => (
          <div key={index} className="feature-card">
            <div className={`feature-icon gradient-${index}`}>
              {feature.icon}
            </div>
            <h3>{feature.title}</h3>
            <p>{feature.description}</p>
          </div>
        ))}
      </div>
    </section>
  );
};

// Module Cards Component
const ModuleCards = () => {
  const modules = [
    {
      title: "Module 1: ROS 2 Foundations",
      description: "Learn ROS 2 — the nervous system of modern robots. Build nodes, topics, services, actions, publishers, subscribers, QoS, and real robot workflows.",
      link: "/docs/ros2-foundations/module-1-ros2",
      icon: "🧭"
    },
    {
      title: "Module 2: Simulation & Digital Twins",
      description: "Master simulation systems: Gazebo, Unity Robotics, Isaac Sim, and digital twin workflows for training and testing robots safely.",
      link: "/docs/simulation/module-2-simulation",
      icon: "🎮"
    },
    {
      title: "Module 3: Hardware Foundations",
      description: "Motors, actuators, torque control, IMUs, sensors, microcontrollers, embedded systems — everything real humanoids need.",
      link: "/docs/hardware-basics/module-3-hardware",
      icon: "⚙️"
    },
    {
      title: "Module 4: VLA — Vision, Language, Action",
      description: "Learn the most advanced robotics architecture: perception models, LLM-driven command systems, action planners, and embodied AI agents.",
      link: "/docs/vla-systems/module-4-vla-foundations",
      icon: "👁️"
    },
    {
      title: "Module 5: Advanced AI & Motion Control",
      description: "Reinforcement learning, motion planning, MPC, trajectory optimization, and how robots think and move intelligently.",
      link: "/docs/advanced-ai-control/module-5-advanced-ai",
      icon: "🧠"
    },
    {
      title: "Module 6: Designing Humanoid Robots",
      description: "Learn end-to-end humanoid creation: mechanical design, kinematics, actuators, morphologies, energy systems, and AI-driven movement.",
      link: "/docs/humanoid-design/module-6-humanoid-design",
      icon: "🦾"
    },
    {
      title: "Appendix",
      description: "Glossary, research papers, references, external resources, and further reading for mastering robotics and AI.",
      link: "/docs/appendix/glossary",
      icon: "📚"
    }
  ];

  return (
    <section className="modules-section">
      <div className="section-header">
        <h2 className="section-title">Explore All Modules</h2>
        <p className="section-subtitle">Master robotics step by step with our comprehensive curriculum</p>
      </div>
      
      <div className="modules-grid">
        {modules.map((module, index) => (
          <div key={index} className="module-card">
            <div className="module-icon">{module.icon}</div>
            <h3>{module.title}</h3>
            <p>{module.description}</p>
            <Link to={module.link} className="module-link">
              Open Module <span className="arrow">→</span>
            </Link>
          </div>
        ))}
      </div>
    </section>
  );
};

// Hero Section with Robot Image
const HeroSection = () => {
  const { siteConfig } = useDocusaurusContext();
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });

  useEffect(() => {
    const handleMouseMove = (e) => {
      const x = (e.clientX / window.innerWidth - 0.5) * 20;
      const y = (e.clientY / window.innerHeight - 0.5) * 20;
      setMousePosition({ x, y });
    };

    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  return (
    <header className="hero-section">
      <div className="hero-background"></div>
      
      <div className="container hero-container">
        <div className="hero-content">
          <h1 className="hero-title">
            <span className="gradient-text">{siteConfig.title}</span>
          </h1>
          <p className="hero-subtitle">{siteConfig.tagline}</p>
          
          <div className="hero-buttons">
            <Link className="btn btn-primary" to="/docs/intro">
              Start Reading
            </Link>
            <Link className="btn btn-secondary" to="/docs/intro">
              View Curriculum
            </Link>
          </div>
          
          <div className="hero-stats">
            <div className="stat">
              <span className="stat-number">7</span>
              <span className="stat-label">Modules</span>
            </div>
            <div className="stat">
              <span className="stat-number">50+</span>
              <span className="stat-label">Projects</span>
            </div>
            <div className="stat">
              <span className="stat-number">∞</span>
              <span className="stat-label">Possibilities</span>
            </div>
          </div>
        </div>
        
        <div 
          className="hero-image"
          style={{
            transform: `translate(${mousePosition.x}px, ${mousePosition.y}px)`
          }}
        >
          <img 
            src="https://png.pngtree.com/png-vector/20240320/ourmid/pngtree-curious-robot-reading-book-png-image_12181635.png" 
            alt="Robot Reading Book"
            className="robot-image"
          />
          <div className="robot-glow"></div>
        </div>
      </div>
      
      <div className="scroll-indicator">
        <div className="mouse">
          <div className="wheel"></div>
        </div>
        <span>Scroll to explore</span>
      </div>
    </header>
  );
};

// Main Home Component
export default function Home() {
  const { siteConfig } = useDocusaurusContext();

  return (
    <Layout
      title={`${siteConfig.title}`}
      description="Complete AI-native engineering curriculum for physical AI, humanoid robots, and embodied intelligence">
      
      <style>
        {`
          @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&family=Montserrat:wght@700;800;900&display=swap');
          
          :root {
            --primary: #ffd700;
            --secondary: #4ecdc4;
            --accent: #ff6b6b;
            --dark: #1a1a2e;
            --light: #f8f9fa;
          }
          
          /* Global Styles */
          body {
            font-family: 'Poppins', sans-serif;
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
            color: white;
            overflow-x: hidden;
          }
          
          /* Hero Section */
          .hero-section {
            min-height: 100vh;
            display: flex;
            align-items: center;
            position: relative;
            overflow: hidden;
            padding: 80px 20px;
          }
          
          .hero-background {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: radial-gradient(circle at 20% 50%, rgba(255, 215, 0, 0.1) 0%, transparent 50%),
                       radial-gradient(circle at 80% 20%, rgba(78, 205, 196, 0.1) 0%, transparent 50%),
                       radial-gradient(circle at 40% 80%, rgba(255, 107, 107, 0.1) 0%, transparent 50%);
            animation: gradientShift 10s ease-in-out infinite alternate;
          }
          
          @keyframes gradientShift {
            0% { opacity: 0.3; }
            100% { opacity: 0.7; }
          }
          
          .hero-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 60px;
            align-items: center;
            max-width: 1400px;
            margin: 0 auto;
            position: relative;
            z-index: 1;
          }
          
          @media (max-width: 992px) {
            .hero-container {
              grid-template-columns: 1fr;
              text-align: center;
            }
          }
          
          .hero-content {
            animation: fadeInUp 1s ease-out;
          }
          
          .hero-title {
            font-family: 'Montserrat', sans-serif;
            font-size: 5rem;
            font-weight: 900;
            margin-bottom: 20px;
            line-height: 1.2;
          }
          
          .gradient-text {
            background: linear-gradient(45deg, #ffd700, #ff6b6b, #4ecdc4);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            background-size: 200% 200%;
            animation: gradient 3s ease infinite;
          }
          
          @keyframes gradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
          }
          
          .hero-subtitle {
            font-size: 1.8rem;
            margin-bottom: 40px;
            color: rgba(255, 255, 255, 0.9);
            max-width: 600px;
          }
          
          .hero-buttons {
            display: flex;
            gap: 20px;
            margin-bottom: 50px;
            flex-wrap: wrap;
          }
          
          .btn {
            padding: 16px 32px;
            border-radius: 50px;
            font-weight: 600;
            font-size: 1.1rem;
            text-decoration: none;
            transition: all 0.3s ease;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 180px;
          }
          
          .btn-primary {
            background: linear-gradient(45deg, #ffd700, #ffb347);
            color: #000;
            box-shadow: 0 10px 30px rgba(255, 215, 0, 0.4);
          }
          
          .btn-primary:hover {
            transform: translateY(-3px);
            box-shadow: 0 15px 40px rgba(255, 215, 0, 0.6);
          }
          
          .btn-secondary {
            background: transparent;
            color: white;
            border: 2px solid rgba(255, 255, 255, 0.3);
            backdrop-filter: blur(10px);
          }
          
          .btn-secondary:hover {
            border-color: #ffd700;
            background: rgba(255, 215, 0, 0.1);
          }
          
          .hero-stats {
            display: flex;
            gap: 40px;
            margin-top: 30px;
          }
          
          .stat {
            display: flex;
            flex-direction: column;
            align-items: flex-start;
          }
          
          .stat-number {
            font-size: 3rem;
            font-weight: 800;
            background: linear-gradient(45deg, #ffd700, #4ecdc4);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
          }
          
          .stat-label {
            font-size: 0.9rem;
            color: rgba(255, 255, 255, 0.7);
            text-transform: uppercase;
            letter-spacing: 1px;
          }
          
          .hero-image {
            position: relative;
            animation: float 6s ease-in-out infinite;
            transition: transform 0.1s ease-out;
          }
          
          @keyframes float {
            0%, 100% { transform: translateY(0px) rotate(0deg); }
            50% { transform: translateY(-20px) rotate(2deg); }
          }
          
          .robot-image {
            width: 100%;
            max-width: 600px;
            filter: drop-shadow(0 20px 40px rgba(0, 0, 0, 0.5));
            animation: robotGlow 2s ease-in-out infinite alternate;
          }
          
          @keyframes robotGlow {
            0% { filter: drop-shadow(0 20px 40px rgba(255, 215, 0, 0.3)); }
            100% { filter: drop-shadow(0 20px 60px rgba(255, 107, 107, 0.5)); }
          }
          
          .robot-glow {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 120%;
            height: 120%;
            background: radial-gradient(circle, rgba(255, 215, 0, 0.2) 0%, transparent 70%);
            border-radius: 50%;
            z-index: -1;
            animation: pulse 4s ease-in-out infinite;
          }
          
          @keyframes pulse {
            0%, 100% { opacity: 0.5; transform: translate(-50%, -50%) scale(1); }
            50% { opacity: 0.8; transform: translate(-50%, -50%) scale(1.1); }
          }
          
          .scroll-indicator {
            position: absolute;
            bottom: 40px;
            left: 50%;
            transform: translateX(-50%);
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 10px;
            color: rgba(255, 255, 255, 0.7);
            animation: fadeIn 2s ease-out;
          }
          
          .mouse {
            width: 30px;
            height: 50px;
            border: 2px solid rgba(255, 255, 255, 0.3);
            border-radius: 20px;
            position: relative;
          }
          
          .wheel {
            width: 4px;
            height: 10px;
            background: #ffd700;
            border-radius: 2px;
            position: absolute;
            top: 10px;
            left: 50%;
            transform: translateX(-50%);
            animation: scroll 2s ease-in-out infinite;
          }
          
          @keyframes scroll {
            0% { transform: translateX(-50%) translateY(0); opacity: 1; }
            100% { transform: translateX(-50%) translateY(20px); opacity: 0; }
          }
          
          /* Features Section */
          .features-section {
            padding: 100px 20px;
            background: rgba(255, 255, 255, 0.02);
            backdrop-filter: blur(10px);
          }
          
          .section-title {
            font-family: 'Montserrat', sans-serif;
            font-size: 3.5rem;
            text-align: center;
            margin-bottom: 60px;
            background: linear-gradient(45deg, #ffd700, #4ecdc4);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
          }
          
          .features-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 30px;
            max-width: 1200px;
            margin: 0 auto;
          }
          
          .feature-card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 20px;
            padding: 40px 30px;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
          }
          
          .feature-card:hover {
            transform: translateY(-10px);
            border-color: #ffd700;
            box-shadow: 0 20px 40px rgba(255, 215, 0, 0.2);
          }
          
          .feature-icon {
            font-size: 3rem;
            margin-bottom: 20px;
            display: inline-block;
            padding: 20px;
            border-radius: 50%;
            background: linear-gradient(45deg, var(--accent), var(--secondary));
          }
          
          .feature-card h3 {
            font-size: 1.5rem;
            margin-bottom: 15px;
            color: #ffd700;
          }
          
          .feature-card p {
            color: rgba(255, 255, 255, 0.8);
            line-height: 1.6;
          }
          
          /* Enhanced About Section */
          .enhanced-about-section {
            padding: 100px 20px;
            max-width: 1200px;
            margin: 80px auto;
            background: linear-gradient(135deg, rgba(255, 215, 0, 0.1) 0%, rgba(78, 205, 196, 0.1) 100%);
            border-radius: 30px;
            position: relative;
            overflow: hidden;
            transform-style: preserve-3d;
            transition: transform 0.5s ease-out;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
          }
          
          .about-title {
            font-family: 'Montserrat', sans-serif;
            font-size: 4rem;
            margin-bottom: 40px;
            text-align: center;
            background: linear-gradient(45deg, #ffd700, #4ecdc4, #ff6b6b);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            position: relative;
            z-index: 1;
          }
          
          .about-content {
            font-family: 'Poppins', sans-serif;
            font-size: 1.6rem;
            line-height: 1.8;
            color: rgba(255, 255, 255, 0.9);
            text-align: center;
            max-width: 900px;
            margin: 0 auto;
            position: relative;
            z-index: 1;
            padding: 40px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 20px;
            backdrop-filter: blur(10px);
          }
          
          .highlight-box {
            display: inline-block;
            padding: 8px 20px;
            background: linear-gradient(45deg, #ff6b6b, #ffd700);
            color: #000;
            border-radius: 50px;
            margin: 0 10px 10px;
            font-weight: 700;
            transform: translateZ(40px);
            animation: pulse 2s infinite;
            font-size: 0.9em;
          }
          
          /* Modules Section */
          .modules-section {
            padding: 100px 20px;
            background: rgba(0, 0, 0, 0.3);
          }
          
          .section-header {
            text-align: center;
            margin-bottom: 60px;
          }
          
          .section-subtitle {
            font-size: 1.4rem;
            color: rgba(255, 255, 255, 0.7);
            max-width: 600px;
            margin: 20px auto 0;
          }
          
          .modules-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 30px;
            max-width: 1400px;
            margin: 0 auto;
          }
          
          .module-card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 20px;
            padding: 30px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
            position: relative;
            overflow: hidden;
          }
          
          .module-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #ffd700, #4ecdc4, #ff6b6b);
          }
          
          .module-card:hover {
            transform: translateY(-10px);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
            border-color: #ffd700;
          }
          
          .module-icon {
            font-size: 2.5rem;
            margin-bottom: 20px;
          }
          
          .module-card h3 {
            font-size: 1.4rem;
            margin-bottom: 15px;
            color: #ffd700;
          }
          
          .module-card p {
            color: rgba(255, 255, 255, 0.8);
            line-height: 1.6;
            margin-bottom: 20px;
          }
          
          .module-link {
            display: inline-flex;
            align-items: center;
            color: #4ecdc4;
            text-decoration: none;
            font-weight: 600;
            transition: all 0.3s ease;
          }
          
          .module-link:hover {
            color: #ffd700;
          }
          
          .arrow {
            margin-left: 8px;
            transition: transform 0.3s ease;
          }
          
          .module-link:hover .arrow {
            transform: translateX(5px);
          }
          
          /* Floating Elements */
          .floating-elements {
            position: absolute;
            width: 100%;
            height: 100%;
            top: 0;
            left: 0;
            pointer-events: none;
            z-index: 0;
          }
          
          .floating-element {
            position: absolute;
            font-size: 2rem;
            opacity: 0.2;
            animation: floatElement 20s linear infinite;
          }
          
          @keyframes floatElement {
            0% { transform: translateY(100vh) rotate(0deg); }
            100% { transform: translateY(-100px) rotate(360deg); }
          }
          
          /* Butterflies */
          .butterflies-container {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 0;
            overflow: hidden;
          }
          
          .butterfly {
            position: absolute;
            width: 40px;
            height: 40px;
            pointer-events: none;
            z-index: 0;
          }
          
          .butterfly .wing {
            position: absolute;
            width: 50%;
            height: 100%;
            background: var(--butterfly-color);
            border-radius: 50% 50% 50% 50% / 60% 60% 40% 40%;
            top: 0;
          }
          
          /* Responsive */
          @media (max-width: 768px) {
            .hero-title {
              font-size: 3rem;
            }
            
            .hero-subtitle {
              font-size: 1.4rem;
            }
            
            .section-title {
              font-size: 2.5rem;
            }
            
            .about-title {
              font-size: 2.5rem;
            }
            
            .about-content {
              font-size: 1.2rem;
              padding: 20px;
            }
            
            .hero-stats {
              flex-direction: column;
              gap: 20px;
              align-items: center;
            }
            
            .stat {
              align-items: center;
            }
            
            .modules-grid {
              grid-template-columns: 1fr;
            }
          }
          
          @keyframes fadeInUp {
            from {
              opacity: 0;
              transform: translateY(30px);
            }
            to {
              opacity: 1;
              transform: translateY(0);
            }
          }
        `}
      </style>
      
      <HeroSection />
      <FeatureCards />
      <EnhancedAboutSection />
      <ModuleCards />
    </Layout>
  );
}
