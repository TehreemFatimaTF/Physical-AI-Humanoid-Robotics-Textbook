import React, { useEffect, useRef, useState } from 'react';
import clsx from 'clsx';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import styles from './index.module.css';

// Book Opening Animation Component
const BookAnimation = () => {
  const [isOpen, setIsOpen] = useState(false);
  
  return (
    <div className="book-container" onClick={() => setIsOpen(!isOpen)}>
      <div className={`book ${isOpen ? 'open' : ''}`}>
        <div className="cover">
          <div className="cover-title">ROBOTICS</div>
          <div className="cover-subtitle">AI-NATIVE CURRICULUM</div>
        </div>
        <div className="pages">
          <div className="page page1">
            <div className="page-content">
              <h3>Module 1</h3>
              <p>ROS 2 Foundations</p>
            </div>
          </div>
          <div className="page page2">
            <div className="page-content">
              <h3>Module 2</h3>
              <p>Digital Twins</p>
            </div>
          </div>
          <div className="page page3">
            <div className="page-content">
              <h3>Module 3</h3>
              <p>Hardware Mastery</p>
            </div>
          </div>
          <div className="page page4">
            <div className="page-content">
              <h3>Module 4</h3>
              <p>VLA Systems</p>
            </div>
          </div>
        </div>
      </div>
      <div className="book-glow"></div>
    </div>
  );
};

// Floating Keywords Component
const FloatingKeywords = () => {
  const keywords = [
    "Humanoid Robots", "ROS 2", "Digital Twins", "VLA", "Embodied AI",
    "Simulation", "Hardware", "Motion Control", "AI Agents", "Robotics"
  ];
  
  return (
    <div className="floating-keywords">
      {keywords.map((keyword, index) => (
        <div 
          key={index}
          className="keyword"
          style={{
            animationDelay: `${index * 0.5}s`,
            left: `${(index * 15) % 100}%`,
            animationDuration: `${15 + (index % 10)}s`
          }}
        >
          {keyword}
        </div>
      ))}
    </div>
  );
};

// New What This Textbook Covers Section
const TextbookCoversSection = () => {
  const cards = [
    {
      icon: "🤖",
      title: "Physical AI Systems",
      description: "End-to-end AI integration in physical robots",
      gradient: "from-yellow-400 to-amber-500",
      features: ["Real-time AI", "Sensor fusion", "Edge computing"]
    },
    {
      icon: "🦾",
      title: "Humanoid Robotics",
      description: "Design and control of human-like robots",
      gradient: "from-amber-400 to-orange-500",
      features: ["Bipedal locomotion", "Human-like dexterity", "Social interaction"]
    },
    {
      icon: "🧠",
      title: "Embodied Intelligence",
      description: "AI that interacts with physical world",
      gradient: "from-yellow-500 to-yellow-600",
      features: ["Spatial reasoning", "Physical interaction", "Real-world learning"]
    },
    {
      icon: "⚡",
      title: "ROS 2 Programming",
      description: "Modern robotics middleware ecosystem",
      gradient: "from-yellow-300 to-yellow-400",
      features: ["Real-time systems", "Distributed computing", "Micro-ROS"]
    },
    {
      icon: "🌐",
      title: "Digital Twin Simulations",
      description: "Virtual replicas for testing and training",
      gradient: "from-amber-300 to-amber-400",
      features: ["Physics simulation", "AI training", "Risk-free testing"]
    },
    {
      icon: "👁️",
      title: "Vision-Language-Action",
      description: "Multi-modal AI perception and execution",
      gradient: "from-yellow-200 to-yellow-300",
      features: ["Visual perception", "Natural language", "Action planning"]
    }
  ];

  return (
    <section className="textbook-covers-section">
      <div className="section-background"></div>
      
      <div className="container">
        <div className="section-header">
          <h2 className="section-title">
            <span className="title-main">What This Textbook</span>
            <span className="title-highlight">Covers</span>
          </h2>
          <p className="section-subtitle">
            A complete AI-native engineering curriculum designed from the ground up
          </p>
        </div>
        
        <div className="book-animation-container">
          <BookAnimation />
        </div>
        
        <FloatingKeywords />
        
        <div className="curriculum-grid">
          {cards.map((card, index) => (
            <div key={index} className={`curriculum-card card-${index}`}>
              <div className="card-icon">
                {card.icon}
              </div>
              <h3>{card.title}</h3>
              <p>{card.description}</p>
              <div className="card-features">
                {card.features.map((feature, idx) => (
                  <span key={idx} className="feature-tag">
                    {feature}
                  </span>
                ))}
              </div>
              <div className="card-glow"></div>
            </div>
          ))}
        </div>
        
        <div className="learning-path">
          <h3>Learning Path</h3>
          <div className="path-steps">
            <div className="step">
              <div className="step-number">01</div>
              <div className="step-content">
                <h4>Foundations</h4>
                <p>ROS 2, Python, Linux, Git</p>
              </div>
            </div>
            <div className="step">
              <div className="step-number">02</div>
              <div className="step-content">
                <h4>Simulation</h4>
                <p>Gazebo, Unity, Digital Twins</p>
              </div>
            </div>
            <div className="step">
              <div className="step-number">03</div>
              <div className="step-content">
                <h4>Hardware</h4>
                <p>Actuators, Sensors, Embedded</p>
              </div>
            </div>
            <div className="step">
              <div className="step-number">04</div>
              <div className="step-content">
                <h4>AI Integration</h4>
                <p>VLA, RL, Motion Planning</p>
              </div>
            </div>
            <div className="step">
              <div className="step-number">05</div>
              <div className="step-content">
                <h4>Humanoid Design</h4>
                <p>Complete robot building</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

// Hero Section
const HeroSection = () => {
  const { siteConfig } = useDocusaurusContext();
  
  return (
    <header className="hero-section">
      <div className="hero-background"></div>
      
      <div className="container hero-container">
        <div className="hero-content">
          <div className="hero-badge">
            <span>🤖 AI-NATIVE CURRICULUM</span>
          </div>
          
          <h1 className="hero-title">
            <span className="hero-title-line">Master</span>
            <span className="hero-title-highlight">Robotics</span>
            <span className="hero-title-line">& AI Engineering</span>
          </h1>
          
          <p className="hero-subtitle">
            Complete hands-on curriculum for humanoid robots, physical AI, 
            and embodied intelligence systems
          </p>
          
          <div className="hero-buttons">
            <Link className="btn btn-primary" to="/docs/intro">
              Start Learning Free
            </Link>
            <Link className="btn btn-secondary" to="/docs/curriculum">
              View Full Curriculum
            </Link>
          </div>
          
          <div className="hero-features">
            <div className="feature">
              <div className="feature-icon">📚</div>
              <div className="feature-text">
                <strong>7 Modules</strong>
                <span>Complete Curriculum</span>
              </div>
            </div>
            <div className="feature">
              <div className="feature-icon">🛠️</div>
              <div className="feature-text">
                <strong>50+ Projects</strong>
                <span>Hands-on Labs</span>
              </div>
            </div>
            <div className="feature">
              <div className="feature-icon">🎓</div>
              <div className="feature-text">
                <strong>Industry Ready</strong>
                <span>Job-focused Skills</span>
              </div>
            </div>
          </div>
        </div>
        
        <div className="hero-image">
          <img 
            src="https://png.pngtree.com/png-vector/20240320/ourmid/pngtree-curious-robot-reading-book-png-image_12181635.png" 
            alt="Robot Reading Book"
            className="robot-image"
          />
          <div className="hero-image-glow"></div>
        </div>
      </div>
    </header>
  );
};

// Module Cards Component
const ModuleCards = () => {
  const modules = [
    {
      title: "Module 1: ROS 2 Foundations",
      description: "Learn ROS 2 — the nervous system of modern robots. Build nodes, topics, services, actions, publishers, subscribers, QoS, and real robot workflows.",
      link: "/docs/ros2-foundations/module-1-ros2",
      icon: "🧭",
      color: "#FFD700"
    },
    {
      title: "Module 2: Simulation & Digital Twins",
      description: "Master simulation systems: Gazebo, Unity Robotics, Isaac Sim, and digital twin workflows for training and testing robots safely.",
      link: "/docs/simulation/module-2-simulation",
      icon: "🎮",
      color: "#FFC107"
    },
    {
      title: "Module 3: Hardware Foundations",
      description: "Motors, actuators, torque control, IMUs, sensors, microcontrollers, embedded systems — everything real humanoids need.",
      link: "/docs/hardware-basics/module-3-hardware",
      icon: "⚙️",
      color: "#FFB300"
    },
    {
      title: "Module 4: VLA — Vision, Language, Action",
      description: "Learn the most advanced robotics architecture: perception models, LLM-driven command systems, action planners, and embodied AI agents.",
      link: "/docs/vla-systems/module-4-vla-foundations",
      icon: "👁️",
      color: "#FFA000"
    },
    {
      title: "Module 5: Advanced AI & Motion Control",
      description: "Reinforcement learning, motion planning, MPC, trajectory optimization, and how robots think and move intelligently.",
      link: "/docs/advanced-ai-control/module-5-advanced-ai",
      icon: "🧠",
      color: "#FF8F00"
    },
    {
      title: "Module 6: Designing Humanoid Robots",
      description: "Learn end-to-end humanoid creation: mechanical design, kinematics, actuators, morphologies, energy systems, and AI-driven movement.",
      link: "/docs/humanoid-design/module-6-humanoid-design",
      icon: "🦾",
      color: "#FF6F00"
    }
  ];

  return (
    <section className="modules-section">
      <div className="container">
        <div className="section-header">
          <h2 className="section-title">Complete Learning Modules</h2>
          <p className="section-subtitle">Step-by-step curriculum from basics to advanced humanoid robotics</p>
        </div>
        
        <div className="modules-grid">
          {modules.map((module, index) => (
            <div key={index} className="module-card">
              <div className="module-icon" style={{ backgroundColor: module.color }}>
                {module.icon}
              </div>
              <h3>{module.title}</h3>
              <p>{module.description}</p>
              <Link to={module.link} className="module-link">
                Explore Module <span className="arrow">→</span>
              </Link>
              <div className="module-progress">
                <div className="progress-bar">
                  <div 
                    className="progress-fill" 
                    style={{ 
                      width: `${(index + 1) * 15}%`,
                      backgroundColor: module.color
                    }}
                  ></div>
                </div>
                <span className="progress-text">Progress: {((index + 1) * 15)}%</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
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
          @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800;900&family=Montserrat:wght@800;900&display=swap');
          
          :root {
            --yellow-50: #FFFDE7;
            --yellow-100: #FFF9C4;
            --yellow-200: #FFF59D;
            --yellow-300: #FFF176;
            --yellow-400: #FFEE58;
            --yellow-500: #FFEB3B;
            --yellow-600: #FDD835;
            --yellow-700: #FBC02D;
            --yellow-800: #F9A825;
            --yellow-900: #F57F17;
            --amber-600: #FFB300;
            --orange-600: #FB8C00;
          }
          
          /* Global Yellow Theme */
          body {
            font-family: 'Poppins', sans-serif;
            background: linear-gradient(135deg, #FFFDE7 0%, #FFF9C4 100%);
            color: #333;
            margin: 0;
            overflow-x: hidden;
          }
          
          /* Hero Section */
          .hero-section {
            min-height: 90vh;
            display: flex;
            align-items: center;
            position: relative;
            padding: 60px 20px;
            background: linear-gradient(135deg, #FFEB3B 0%, #FFD600 100%);
            overflow: hidden;
          }
          
          .hero-background {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: 
              radial-gradient(circle at 20% 50%, rgba(255, 255, 255, 0.3) 0%, transparent 50%),
              radial-gradient(circle at 80% 20%, rgba(255, 235, 59, 0.2) 0%, transparent 50%),
              radial-gradient(circle at 40% 80%, rgba(255, 193, 7, 0.2) 0%, transparent 50%);
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
          
          .hero-badge {
            display: inline-block;
            background: rgba(0, 0, 0, 0.1);
            padding: 10px 20px;
            border-radius: 50px;
            margin-bottom: 30px;
            font-weight: 600;
            font-size: 0.9rem;
            letter-spacing: 1px;
            color: #333;
          }
          
          .hero-title {
            font-family: 'Montserrat', sans-serif;
            font-size: 5rem;
            font-weight: 900;
            margin-bottom: 20px;
            line-height: 1.2;
          }
          
          .hero-title-line {
            display: block;
            color: #333;
          }
          
          .hero-title-highlight {
            display: block;
            background: linear-gradient(45deg, #FF6F00, #FFD600);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            text-shadow: 3px 3px 0px rgba(255, 111, 0, 0.1);
          }
          
          .hero-subtitle {
            font-size: 1.5rem;
            margin-bottom: 40px;
            color: #5D4037;
            max-width: 600px;
            line-height: 1.6;
          }
          
          .hero-buttons {
            display: flex;
            gap: 20px;
            margin-bottom: 50px;
            flex-wrap: wrap;
          }
          
          .btn {
            padding: 18px 36px;
            border-radius: 12px;
            font-weight: 700;
            font-size: 1.1rem;
            text-decoration: none;
            transition: all 0.3s ease;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 200px;
            border: none;
            cursor: pointer;
            position: relative;
            overflow: hidden;
          }
          
          .btn::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
            transition: 0.5s;
          }
          
          .btn:hover::before {
            left: 100%;
          }
          
          .btn-primary {
            background: linear-gradient(45deg, #FFD600, #FFB300);
            color: #000;
            box-shadow: 0 10px 30px rgba(255, 179, 0, 0.4);
          }
          
          .btn-primary:hover {
            transform: translateY(-3px);
            box-shadow: 0 15px 40px rgba(255, 179, 0, 0.6);
          }
          
          .btn-secondary {
            background: transparent;
            color: #333;
            border: 3px solid rgba(0, 0, 0, 0.2);
            backdrop-filter: blur(10px);
          }
          
          .btn-secondary:hover {
            border-color: #FFD600;
            background: rgba(255, 214, 0, 0.1);
          }
          
          .hero-features {
            display: flex;
            gap: 30px;
            margin-top: 40px;
          }
          
          .feature {
            display: flex;
            align-items: center;
            gap: 15px;
          }
          
          .feature-icon {
            font-size: 2rem;
            background: rgba(255, 255, 255, 0.3);
            padding: 15px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
          }
          
          .feature-text {
            display: flex;
            flex-direction: column;
          }
          
          .feature-text strong {
            font-size: 1.5rem;
            color: #000;
          }
          
          .feature-text span {
            font-size: 0.9rem;
            color: #666;
          }
          
          .hero-image {
            position: relative;
            animation: float 6s ease-in-out infinite;
          }
          
          @keyframes float {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-20px); }
          }
          
          .robot-image {
            width: 100%;
            max-width: 600px;
            filter: drop-shadow(0 20px 40px rgba(0, 0, 0, 0.2));
            animation: robotGlow 3s ease-in-out infinite alternate;
          }
          
          @keyframes robotGlow {
            0% { filter: drop-shadow(0 20px 40px rgba(255, 214, 0, 0.3)); }
            100% { filter: drop-shadow(0 20px 60px rgba(255, 111, 0, 0.5)); }
          }
          
          .hero-image-glow {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 120%;
            height: 120%;
            background: radial-gradient(circle, rgba(255, 214, 0, 0.3) 0%, transparent 70%);
            border-radius: 50%;
            z-index: -1;
            animation: glowPulse 4s ease-in-out infinite;
          }
          
          @keyframes glowPulse {
            0%, 100% { opacity: 0.5; transform: translate(-50%, -50%) scale(1); }
            50% { opacity: 0.8; transform: translate(-50%, -50%) scale(1.1); }
          }
          
          /* What This Textbook Covers Section */
          .textbook-covers-section {
            padding: 100px 20px;
            position: relative;
            background: linear-gradient(135deg, #FFF8E1 0%, #FFECB3 100%);
            overflow: hidden;
          }
          
          .section-background {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: 
              url("data:image/svg+xml,%3Csvg width='100' height='100' viewBox='0 0 100 100' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M11 18c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm48 25c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm-43-7c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm63 31c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM34 90c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm56-76c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM12 86c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm28-65c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm23-11c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-6 60c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm29 22c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zM32 63c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm57-13c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-9-21c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM60 91c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM35 41c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM12 60c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2z' fill='%23FFD600' fill-opacity='0.1' fill-rule='evenodd'/%3E%3C/svg%3E");
          }
          
          .section-header {
            text-align: center;
            margin-bottom: 80px;
            position: relative;
            z-index: 2;
          }
          
          .section-title {
            font-family: 'Montserrat', sans-serif;
            font-size: 4.5rem;
            font-weight: 900;
            margin-bottom: 20px;
          }
          
          .title-main {
            color: #333;
            display: block;
          }
          
          .title-highlight {
            background: linear-gradient(45deg, #FFD600, #FF6F00);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            display: block;
            text-shadow: 4px 4px 0px rgba(255, 111, 0, 0.1);
          }
          
          .section-subtitle {
            font-size: 1.4rem;
            color: #666;
            max-width: 600px;
            margin: 0 auto;
            line-height: 1.6;
          }
          
          /* Book Animation */
          .book-animation-container {
            display: flex;
            justify-content: center;
            margin: 60px auto;
            position: relative;
            z-index: 2;
          }
          
          .book-container {
            position: relative;
            cursor: pointer;
            transform-style: preserve-3d;
            perspective: 1000px;
          }
          
          .book {
            width: 300px;
            height: 400px;
            position: relative;
            transform-style: preserve-3d;
            transition: transform 1.5s cubic-bezier(0.68, -0.55, 0.27, 1.55);
          }
          
          .book.open {
            transform: rotateY(-30deg);
          }
          
          .cover {
            position: absolute;
            width: 100%;
            height: 100%;
            background: linear-gradient(45deg, #FFD600, #FFB300);
            border-radius: 10px 20px 20px 10px;
            transform-origin: left;
            transition: transform 1.5s cubic-bezier(0.68, -0.55, 0.27, 1.55);
            box-shadow: 
              -10px 0 30px rgba(0, 0, 0, 0.2),
              inset -5px 0 10px rgba(255, 255, 255, 0.5);
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            padding: 40px;
            z-index: 2;
          }
          
          .book.open .cover {
            transform: rotateY(-140deg);
          }
          
          .cover-title {
            font-family: 'Montserrat', sans-serif;
            font-size: 2.5rem;
            font-weight: 900;
            color: #000;
            margin-bottom: 10px;
          }
          
          .cover-subtitle {
            font-size: 1rem;
            color: #333;
            font-weight: 600;
            letter-spacing: 2px;
          }
          
          .pages {
            position: absolute;
            width: 95%;
            height: 95%;
            top: 2.5%;
            left: 2.5%;
            background: white;
            border-radius: 5px;
            overflow: hidden;
            box-shadow: 
              0 0 20px rgba(0, 0, 0, 0.1),
              inset 0 0 50px rgba(0, 0, 0, 0.05);
          }
          
          .page {
            position: absolute;
            width: 100%;
            height: 100%;
            background: white;
            padding: 30px;
            display: flex;
            align-items: center;
            justify-content: center;
          }
          
          .page-content {
            text-align: center;
          }
          
          .page-content h3 {
            color: #FF6F00;
            margin-bottom: 10px;
            font-size: 1.5rem;
          }
          
          .page-content p {
            color: #666;
            font-size: 1.1rem;
          }
          
          .book-glow {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 150%;
            height: 150%;
            background: radial-gradient(circle, rgba(255, 214, 0, 0.2) 0%, transparent 70%);
            border-radius: 50%;
            z-index: -1;
            animation: bookGlow 3s ease-in-out infinite alternate;
          }
          
          @keyframes bookGlow {
            0% { opacity: 0.3; transform: translate(-50%, -50%) scale(1); }
            100% { opacity: 0.6; transform: translate(-50%, -50%) scale(1.2); }
          }
          
          /* Floating Keywords */
          .floating-keywords {
            position: absolute;
            width: 100%;
            height: 100%;
            top: 0;
            left: 0;
            pointer-events: none;
            z-index: 1;
          }
          
          .keyword {
            position: absolute;
            background: rgba(255, 214, 0, 0.1);
            color: #FF6F00;
            padding: 10px 20px;
            border-radius: 50px;
            font-weight: 600;
            font-size: 0.9rem;
            white-space: nowrap;
            animation: floatKeywords 20s linear infinite;
            border: 2px solid rgba(255, 111, 0, 0.2);
            backdrop-filter: blur(5px);
          }
          
          @keyframes floatKeywords {
            0% { 
              transform: translateY(100vh) rotate(0deg) scale(0.8);
              opacity: 0;
            }
            10% { opacity: 1; }
            90% { opacity: 1; }
            100% { 
              transform: translateY(-100px) rotate(360deg) scale(1.2);
              opacity: 0;
            }
          }
          
          /* Curriculum Grid */
          .curriculum-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 30px;
            margin: 60px 0;
            position: relative;
            z-index: 2;
          }
          
          .curriculum-card {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 20px;
            padding: 30px;
            position: relative;
            overflow: hidden;
            border: 2px solid rgba(255, 214, 0, 0.3);
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
          }
          
          .curriculum-card:hover {
            transform: translateY(-10px);
            box-shadow: 
              0 20px 40px rgba(255, 179, 0, 0.3),
              0 0 0 1px rgba(255, 214, 0, 0.5);
          }
          
          .card-icon {
            font-size: 3rem;
            width: 80px;
            height: 80px;
            background: linear-gradient(45deg, #FFD600, #FFB300);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 20px;
            color: #000;
          }
          
          .curriculum-card h3 {
            font-size: 1.5rem;
            color: #333;
            margin-bottom: 15px;
          }
          
          .curriculum-card p {
            color: #666;
            line-height: 1.6;
            margin-bottom: 20px;
          }
          
          .card-features {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
          }
          
          .feature-tag {
            background: rgba(255, 214, 0, 0.2);
            color: #FF6F00;
            padding: 6px 15px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
          }
          
          .card-glow {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 5px;
            background: linear-gradient(90deg, #FFD600, #FF6F00);
          }
          
          /* Learning Path */
          .learning-path {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 20px;
            padding: 40px;
            margin-top: 80px;
            position: relative;
            z-index: 2;
            border: 2px solid rgba(255, 214, 0, 0.3);
          }
          
          .learning-path h3 {
            font-size: 2rem;
            color: #333;
            margin-bottom: 40px;
            text-align: center;
          }
          
          .path-steps {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 30px;
          }
          
          .step {
            display: flex;
            align-items: flex-start;
            gap: 20px;
            padding: 20px;
            background: rgba(255, 214, 0, 0.1);
            border-radius: 15px;
            transition: all 0.3s ease;
          }
          
          .step:hover {
            background: rgba(255, 214, 0, 0.2);
            transform: translateY(-5px);
          }
          
          .step-number {
            font-family: 'Montserrat', sans-serif;
            font-size: 2.5rem;
            font-weight: 900;
            color: #FFD600;
            line-height: 
