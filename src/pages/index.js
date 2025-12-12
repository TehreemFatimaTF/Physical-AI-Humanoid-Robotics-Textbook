import React, { useEffect, useRef, useState } from 'react';
import clsx from 'clsx';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import styles from './index.module.css';

// Yellow Theme Floating Elements
const FloatingElements = () => {
  const elements = ['🤖', '⚡', '💻', '🔬', '⚙️', '🧠', '🎓', '🚀'];
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

// Feature Cards Component - Yellow Theme
const FeatureCards = () => {
  const features = [
    {
      title: "AI-First Robotics",
      description: "Learn robotics through modern AI-native approaches",
      icon: "🤖",
      gradient: "from-yellow-400 to-amber-500"
    },
    {
      title: "Hands-On Learning",
      description: "Build real robots with practical projects",
      icon: "🔧",
      gradient: "from-amber-500 to-orange-500"
    },
    {
      title: "Industry Standard",
      description: "Master tools used by top robotics companies",
      icon: "💼",
      gradient: "from-orange-500 to-yellow-600"
    },
    {
      title: "Community Support",
      description: "Join global robotics learning community",
      icon: "🌍",
      gradient: "from-yellow-600 to-amber-700"
    }
  ];

  return (
    <section className="features-section">
      <h2 className="section-title">Why Choose This Curriculum?</h2>
      <div className="features-grid">
        {features.map((feature, index) => (
          <div key={index} className="feature-card">
            <div className="feature-icon">
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

// Module Cards Component - Yellow Theme
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
      title: "Appendix & Resources",
      description: "Glossary, research papers, references, external resources, and further reading for mastering robotics and AI.",
      link: "/docs/appendix/glossary",
      icon: "📚"
    }
  ];

  return (
    <section className="modules-section">
      <div className="section-header">
        <h2 className="section-title">Explore All Modules</h2>
        <p className="section-subtitle">Complete step-by-step curriculum to master robotics engineering</p>
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

// Hero Section with Robot Image - Pure Yellow Theme
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
  <header className="hero-section"> <div className="hero-background"></div> <div className="container hero-container"> <div className="hero-content"> <div className="hero-badge"> <span>🤖 AI Robotics Curriculum</span> </div> <h1 className="hero-title"> <span className="gradient-text">{siteConfig.title}</span> </h1> <p className="hero-subtitle">{siteConfig.tagline}</p> <div className="hero-buttons"> <Link className="btn btn-primary" to="/docs/intro"> 🚀 Start Learning Now </Link> <Link className="btn btn-secondary" to="/docs/intro"> 📚 View All Modules </Link> </div> <div className="hero-highlights"> <div className="highlight"> <span className="highlight-icon">✅</span> <span>Complete Curriculum</span> </div> <div className="highlight"> <span className="highlight-icon">⚡</span> <span>Practical Projects</span> </div> <div className="highlight"> <span className="highlight-icon">🎯</span> <span>Industry Relevant</span> </div> </div> </div> <div className="hero-image" style={{ transform: translate(${mousePosition.x}px, ${mousePosition.y}px) }} > <img src="https://png.pngtree.com/png-vector/20240810/ourmid/pngtree-a-robot-is-busy-in-study-png-image_13439400.png" alt="Robot Reading Book" className="robot-image" /> <div className="robot-glow"></div> </div> </div> </header>
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
            --primary-yellow: #FFD700;
            --dark-yellow: #FFC107;
            --light-yellow: #FFF176;
            --accent-yellow: #FFB300;
            --dark-bg: #1A1200;
            --card-bg: #FFF8E1;
          }
          
          /* Global Styles - Pure Yellow Theme */
          body {
            font-family: 'Poppins', sans-serif;
            background: linear-gradient(135deg, #1A1200 0%, #2D1B00 50%, #1A1200 100%);
            color: #FFF8E1;
            overflow-x: hidden;
            margin: 0;
            padding: 0;
          }
          
          /* Hero Section - Yellow Theme */
          .hero-section {
            min-height: 90vh;
            display: flex;
            align-items: center;
            position: relative;
            overflow: hidden;
            padding: 60px 20px;
            background: linear-gradient(135deg, #1A1200 0%, #2D1B00 100%);
          }
          
          .hero-background {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: 
              radial-gradient(circle at 20% 50%, rgba(255, 215, 0, 0.15) 0%, transparent 50%),
              radial-gradient(circle at 80% 20%, rgba(255, 193, 7, 0.1) 0%, transparent 50%),
              radial-gradient(circle at 40% 80%, rgba(255, 179, 0, 0.2) 0%, transparent 50%);
            animation: gradientShift 8s ease-in-out infinite alternate;
          }
          
          @keyframes gradientShift {
            0% { opacity: 0.4; }
            100% { opacity: 0.8; }
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
              gap: 40px;
            }
          }
          
          .hero-content {
            animation: fadeInUp 1s ease-out;
          }
          
          .hero-badge {
            display: inline-block;
            background: linear-gradient(45deg, #FFD700, #FFB300);
            padding: 8px 20px;
            border-radius: 30px;
            margin-bottom: 20px;
            font-weight: 600;
            font-size: 0.9rem;
            color: #1A1200;
            animation: badgePulse 2s infinite;
          }
          
          @keyframes badgePulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
          }
          
          .hero-title {
            font-family: 'Montserrat', sans-serif;
            font-size: 4.5rem;
            font-weight: 900;
            margin-bottom: 20px;
            line-height: 1.2;
          }
          
          .gradient-text {
            background: linear-gradient(45deg, #FFD700, #FFB300, #FFC107);
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
            color: rgba(255, 248, 225, 0.9);
            max-width: 600px;
            line-height: 1.6;
          }
          
          .hero-buttons {
            display: flex;
            gap: 20px;
            margin-bottom: 40px;
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
            min-width: 200px;
            gap: 10px;
          }
          
          .btn-primary {
            background: linear-gradient(45deg, #FFD700, #FFB300);
            color: #1A1200;
            box-shadow: 0 10px 30px rgba(255, 215, 0, 0.5);
            border: none;
          }
          
          .btn-primary:hover {
            transform: translateY(-3px);
            box-shadow: 0 15px 40px rgba(255, 215, 0, 0.7);
            background: linear-gradient(45deg, #FFB300, #FFD700);
          }
          
          .btn-secondary {
            background: transparent;
            color: #FFD700;
            border: 2px solid #FFD700;
            backdrop-filter: blur(10px);
          }
          
          .btn-secondary:hover {
            background: rgba(255, 215, 0, 0.1);
            transform: translateY(-3px);
            box-shadow: 0 10px 30px rgba(255, 215, 0, 0.3);
          }
          
          .hero-highlights {
            display: flex;
            gap: 30px;
            flex-wrap: wrap;
            margin-top: 30px;
          }
          
          .highlight {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 1rem;
            color: #FFD700;
          }
          
          .highlight-icon {
            font-size: 1.2rem;
            animation: bounce 2s infinite;
          }
          
          @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-5px); }
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
            filter: drop-shadow(0 20px 40px rgba(255, 215, 0, 0.4));
            animation: robotGlow 3s ease-in-out infinite alternate;
          }
          
          @keyframes robotGlow {
            0% { filter: drop-shadow(0 20px 40px rgba(255, 215, 0, 0.3)); }
            100% { filter: drop-shadow(0 20px 60px rgba(255, 179, 0, 0.6)); }
          }
          
          .robot-glow {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 120%;
            height: 120%;
            background: radial-gradient(circle, rgba(255, 215, 0, 0.3) 0%, transparent 70%);
            border-radius: 50%;
            z-index: -1;
            animation: pulse 4s ease-in-out infinite;
          }
          
          @keyframes pulse {
            0%, 100% { opacity: 0.4; transform: translate(-50%, -50%) scale(1); }
            50% { opacity: 0.7; transform: translate(-50%, -50%) scale(1.1); }
          }
          
          /* Features Section - Yellow Theme */
          .features-section {
            padding: 80px 20px;
            background: linear-gradient(135deg, #1A1200 0%, #2D1B00 100%);
            position: relative;
            overflow: hidden;
          }
          
          .section-title {
            font-family: 'Montserrat', sans-serif;
            font-size: 3.2rem;
            text-align: center;
            margin-bottom: 60px;
            background: linear-gradient(45deg, #FFD700, #FFB300);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            position: relative;
            z-index: 1;
          }
          
          .features-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 30px;
            max-width: 1200px;
            margin: 0 auto;
            position: relative;
            z-index: 1;
          }
          
          .feature-card {
            background: rgba(255, 215, 0, 0.05);
            border-radius: 20px;
            padding: 40px 30px;
            text-align: center;
            border: 1px solid rgba(255, 215, 0, 0.2);
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
            position: relative;
            overflow: hidden;
          }
          
          .feature-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #FFD700, #FFB300, #FFC107);
          }
          
          .feature-card:hover {
            transform: translateY(-10px);
            border-color: #FFD700;
            box-shadow: 0 20px 40px rgba(255, 215, 0, 0.2);
            background: rgba(255, 215, 0, 0.1);
          }
          
          .feature-icon {
            font-size: 3.5rem;
            margin-bottom: 25px;
            display: inline-block;
            padding: 25px;
            border-radius: 50%;
            background: linear-gradient(45deg, #FFD700, #FFB300);
            color: #1A1200;
            animation: iconFloat 3s ease-in-out infinite;
          }
          
          @keyframes iconFloat {
            0%, 100% { transform: translateY(0) rotate(0deg); }
            50% { transform: translateY(-10px) rotate(5deg); }
          }
          
          .feature-card h3 {
            font-size: 1.6rem;
            margin-bottom: 15px;
            color: #FFD700;
            font-weight: 700;
          }
          
          .feature-card p {
            color: rgba(255, 248, 225, 0.9);
            line-height: 1.6;
            font-size: 1.1rem;
          }
          
          /* Modules Section - Yellow Theme */
          .modules-section {
            padding: 80px 20px;
            background: linear-gradient(135deg, #2D1B00 0%, #1A1200 100%);
          }
          
          .section-header {
            text-align: center;
            margin-bottom: 60px;
          }
          
          .section-subtitle {
            font-size: 1.4rem;
            color: rgba(255, 215, 0, 0.8);
            max-width: 600px;
            margin: 20px auto 0;
            font-weight: 300;
          }
          
          .modules-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 30px;
            max-width: 1400px;
            margin: 0 auto;
          }
          
          .module-card {
            background: rgba(255, 215, 0, 0.05);
            border-radius: 20px;
            padding: 35px 30px;
            border: 1px solid rgba(255, 215, 0, 0.2);
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
            background: linear-gradient(90deg, #FFD700, #FFB300, #FFC107);
          }
          
          .module-card:hover {
            transform: translateY(-10px);
            box-shadow: 0 20px 40px rgba(255, 215, 0, 0.25);
            border-color: #FFD700;
            background: rgba(255, 215, 0, 0.1);
          }
          
          .module-icon {
            font-size: 2.8rem;
            margin-bottom: 20px;
            display: inline-block;
            padding: 15px;
            background: rgba(255, 215, 0, 0.2);
            border-radius: 15px;
            color: #FFD700;
          }
          
          .module-card h3 {
            font-size: 1.5rem;
            margin-bottom: 15px;
            color: #FFD700;
            font-weight: 700;
          }
          
          .module-card p {
            color: rgba(255, 248, 225, 0.9);
            line-height: 1.6;
            margin-bottom: 25px;
            font-size: 1.1rem;
          }
          
          .module-link {
            display: inline-flex;
            align-items: center;
            color: #FFD700;
            text-decoration: none;
            font-weight: 600;
            font-size: 1.1rem;
            transition: all 0.3s ease;
            background: rgba(255, 215, 0, 0.1);
            padding: 10px 20px;
            border-radius: 10px;
          }
          
          .module-link:hover {
            color: #FFB300;
            background: rgba(255, 215, 0, 0.2);
            transform: translateX(5px);
          }
          
          .arrow {
            margin-left: 8px;
            transition: transform 0.3s ease;
          }
          
          .module-link:hover .arrow {
            transform: translateX(5px);
          }
          
          /* Floating Elements - Yellow Theme */
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
            font-size: 2.5rem;
            opacity: 0.1;
            animation: floatElement 25s linear infinite;
            color: #FFD700;
          }
          
          @keyframes floatElement {
            0% { transform: translateY(100vh) rotate(0deg) scale(1); }
            100% { transform: translateY(-100px) rotate(360deg) scale(1.2); }
          }
          
          /* Responsive Design */
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
            
            .hero-buttons {
              flex-direction: column;
              align-items: center;
            }
            
            .btn {
              width: 100%;
              max-width: 300px;
            }
            
            .hero-highlights {
              flex-direction: column;
              gap: 15px;
              align-items: center;
            }
            
            .modules-grid {
              grid-template-columns: 1fr;
            }
            
            .features-grid {
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
          
          /* Additional Yellow Elements */
          .spark {
            position: absolute;
            width: 4px;
            height: 4px;
            background: #FFD700;
            border-radius: 50%;
            animation: sparkle 1.5s infinite;
          }
          
          @keyframes sparkle {
            0%, 100% { opacity: 0; transform: scale(0); }
            50% { opacity: 1; transform: scale(1); }
          }
        `}
      </style>
      
      <HeroSection />
      <FeatureCards />
      <ModuleCards />
      
      {/* Add floating sparks */}
      {[...Array(20)].map((_, i) => (
        <div 
          key={i}
          className="spark"
          style={{
            left: `${Math.random() * 100}%`,
            top: `${Math.random() * 100}%`,
            animationDelay: `${Math.random() * 2}s`
          }}
        />
      ))}
    </Layout>
  );
}
