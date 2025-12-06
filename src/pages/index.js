import React, { useEffect, useRef } from 'react';
import clsx from 'clsx';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import styles from './index.module.css';

// Enhanced About Section Component
const EnhancedAboutSection = () => {
  const sectionRef = useRef(null);
  const butterflyRef = useRef(null);

  useEffect(() => {
    // Create floating butterflies
    const createButterfly = () => {
      if (!butterflyRef.current) return;
      
      const butterfly = document.createElement('div');
      butterfly.className = 'butterfly';
      
      // Random butterfly styles
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
      
      // Animate butterfly
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
      
      // Remove butterfly after some time and create new one
      setTimeout(() => {
        if (butterfly.parentNode) {
          butterfly.remove();
          createButterfly();
        }
      }, duration * 1000);
    };
    
    // Create initial butterflies
    for (let i = 0; i < 8; i++) {
      setTimeout(createButterfly, i * 500);
    }
    
    // Add scroll animation for 3D effect
    const handleScroll = () => {
      if (!sectionRef.current) return;
      
      const rect = sectionRef.current.getBoundingClientRect();
      const isInView = rect.top < window.innerHeight && rect.bottom > 0;
      
      if (isInView) {
        const scrollPercent = 1 - (rect.top / window.innerHeight);
        const rotation = scrollPercent * 5;
        const scale = 0.95 + (scrollPercent * 0.1);
        
        sectionRef.current.style.transform = `perspective(1000px) rotateX(${rotation}deg) scale(${scale})`;
      }
    };
    
    window.addEventListener('scroll', handleScroll);
    return () => {
      window.removeEventListener('scroll', handleScroll);
    };
  }, []);

  return (
    <>
      <style>
        {`
          @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&family=Montserrat:wght@700;800;900&display=swap');
          
          .enhanced-about-section {
            padding: 100px 20px;
            max-width: 12000px;
            margin: 80px auto;
            background: linear-gradient(135deg, #faf615 0%, hsl(56, 100%, 62%) 100%);
            border-radius: 30px;
            position: relative;
            overflow: hidden;
            transform-style: preserve-3d;
            transition: transform 0.5s ease-out;
            box-shadow: 
              0 20px 60px rgba(250, 246, 21, 0.3),
              0 0 0 1px rgba(255, 255, 255, 0.1),
              inset 0 0 30px rgba(255, 255, 255, 0.5);
          }
          
          .enhanced-about-section::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 5px;
            background: linear-gradient(90deg, 
              #ff6b6b, #4ecdc4, #ffe66d, #ff8e53, #9b59b6);
            z-index: 2;
          }
          
          .enhanced-about-section::after {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 1px, transparent 1px);
            background-size: 50px 50px;
            animation: float 20s linear infinite;
            z-index: 0;
          }
          
          @keyframes float {
            0% { transform: translate(0, 0) rotate(0deg); }
            100% { transform: translate(-50px, -50px) rotate(360deg); }
          }
          
          .about-title {
            font-family: 'Montserrat', sans-serif;
            font-size: 4.5rem;
            margin-bottom: 40px;
            text-align: center;
            background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #ffe66d);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            text-shadow: 3px 3px 0px rgba(0, 0, 0, 0.1);
            position: relative;
            z-index: 1;
            letter-spacing: 1px;
            transform: translateZ(50px);
            animation: titleGlow 3s ease-in-out infinite alternate;
          }
          
          @keyframes titleGlow {
            0% { text-shadow: 3px 3px 0px rgba(0, 0, 0, 0.1), 0 0 20px rgba(255, 107, 107, 0.3); }
            100% { text-shadow: 3px 3px 0px rgba(0, 0, 0, 0.1), 0 0 40px rgba(255, 230, 109, 0.7); }
          }
          
          .about-content {
            font-family: 'Poppins', sans-serif;
            font-size: 1.8rem;
            line-height: 1.8;
            color: #333;
            text-align: center;
            max-width: 900px;
            margin: 0 auto;
            position: relative;
            z-index: 1;
            padding: 30px;
            background: rgba(255, 255, 255, 0.85);
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            transform: translateZ(30px);
            border-left: 5px solid #ff6b6b;
            border-right: 5px solid #4ecdc4;
          }
          
          .about-content strong {
            color: #ff6b6b;
            font-weight: 700;
          }
          
          .about-content em {
            color: #4ecdc4;
            font-style: italic;
            font-weight: 600;
          }
          
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
          
          .butterfly.small {
            width: 30px;
            height: 30px;
          }
          
          .butterfly.large {
            width: 50px;
            height: 50px;
          }
          
          .butterfly .wing {
            position: absolute;
            width: 50%;
            height: 100%;
            background: var(--butterfly-color);
            border-radius: 50% 50% 50% 50% / 60% 60% 40% 40%;
            top: 0;
          }
          
          .butterfly .wing.left {
            left: 0;
            transform-origin: right center;
            animation: flapLeft 0.5s ease-in-out infinite alternate;
          }
          
          .butterfly .wing.right {
            right: 0;
            transform-origin: left center;
            animation: flapRight 0.5s ease-in-out infinite alternate;
          }
          
          @keyframes flapLeft {
            0% { transform: rotateY(0deg); }
            100% { transform: rotateY(60deg); }
          }
          
          @keyframes flapRight {
            0% { transform: rotateY(0deg); }
            100% { transform: rotateY(-60deg); }
          }
          
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
            opacity: 0.1;
            animation: floatElement 15s linear infinite;
          }
          
          @keyframes floatElement {
            0% { transform: translateY(0) rotate(0deg); }
            100% { transform: translateY(-1000px) rotate(360deg); }
          }
          
          .highlight-box {
            display: inline-block;
            padding: 10px 25px;
            background: linear-gradient(45deg, #ff6b6b, #ffe66d);
            color: white;
            border-radius: 50px;
            margin: 10px 5px;
            font-weight: 700;
            transform: translateZ(40px);
            box-shadow: 0 5px 15px rgba(255, 107, 107, 0.4);
            animation: pulse 2s infinite;
          }
          
          @keyframes pulse {
            0% { transform: translateZ(40px) scale(1); }
            50% { transform: translateZ(40px) scale(1.05); }
            100% { transform: translateZ(40px) scale(1); }
          }
          
          @media (max-width: 768px) {
            .enhanced-about-section {
              padding: 60px 15px;
              margin: 40px auto;
              border-radius: 20px;
            }
            
            .about-title {
              font-size: 2.8rem;
            }
            
            .about-content {
              font-size: 1.4rem;
              padding: 20px;
            }
            
            .highlight-box {
              padding: 8px 16px;
              font-size: 0.9rem;
            }
          }
        `}
      </style>
      
      <section 
        className="enhanced-about-section"
        ref={sectionRef}
      >
        <div className="butterflies-container" ref={butterflyRef}></div>
        
        <div className="floating-elements">
          {['🤖', '🚀', '💻', '🔬', '⚙️', '🧠', '👁️', '🦾'].map((emoji, index) => (
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
        
        <h2 className="about-title">
          What This Textbook Covers
        </h2>
        
        <div className="about-content">
          This is a <strong>complete AI-native engineering curriculum</strong> designed for 
          <span className="highlight-box">physical AI</span>, 
          <span className="highlight-box">humanoid robots</span>,
          <span className="highlight-box">embodied intelligence</span>, 
          <span className="highlight-box">ROS 2 programming</span>,
          <span className="highlight-box">digital twin simulations</span>, and 
          <span className="highlight-box">Vision-Language-Action (VLA)</span> systems. 
          Each module builds your <em>robotics superpowers</em> step by step.
        </div>
      </section>
    </>
  );
};

// HomepageHeader Component
function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <h1 className="hero__title">{siteConfig.title}</h1>
        <p className="hero__subtitle">{siteConfig.tagline}</p>

         <div >
   <Link className="button button--secondary button--lg" to="/docs/intro">
     Start Reading
   </Link>
 </div>

      </div>
    </header>
  );
}

// Main Home Component
export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`${siteConfig.title}`}
      description="Description will go into a meta tag in <head />">
      <HomepageHeader />
      <main>
        <EnhancedAboutSection />
        {/* Add more sections here if needed */}
      </main>
         {/* MODULE CARDS */}
      <section style={{ padding: "60px 20px", background: "linear-gradient(135deg, #faf615 0%, hsl(56, 100%, 62%) 100%)"}}>
        <h2 style={{ fontSize: "52px", marginBottom: "40px", textAlign: "center" ,color:"black"
        }}>
          Explore All Modules
        </h2>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
            gap: "25px",
            maxWidth: "1200px",
            margin: "0 auto",
          }}
        >
          {/* MODULE 1 */}
          <div style={cardStyle}>
            <h3 style={cardTitle}>Module 1: ROS 2 Foundations</h3>
            <p style={cardText}>
              Learn ROS 2 — the nervous system of modern robots. Build nodes, topics,
              services, actions, publishers, subscribers, QoS, and real robot workflows.
            </p>
            <Link style={cardBtn} to="/docs/ros2-foundations/module-1-ros2">
              Open Module →
            </Link>
          </div>

          {/* MODULE 2 */}
          <div style={cardStyle}>
            <h3 style={cardTitle}>Module 2: Simulation & Digital Twins</h3>
            <p style={cardText}>
              Master simulation systems: Gazebo, Unity Robotics, Isaac Sim, and digital
              twin workflows for training and testing robots safely.
            </p>
            <Link style={cardBtn} to="/docs/simulation/module-2-simulation">
              Open Module →
            </Link>
          </div>

          {/* MODULE 3 */}
          <div style={cardStyle}>
            <h3 style={cardTitle}>Module 3: Hardware Foundations</h3>
            <p style={cardText}>
              Motors, actuators, torque control, IMUs, sensors, microcontrollers,
              embedded systems — everything real humanoids need.
            </p>
            <Link style={cardBtn} to="/docs/hardware-basics/module-3-hardware">
              Open Module →
            </Link>
          </div>

          {/* MODULE 4 */}
          <div style={cardStyle}>
            <h3 style={cardTitle}>Module 4: VLA — Vision, Language, Action</h3>
            <p style={cardText}>
              Learn the most advanced robotics architecture: perception models,
              LLM-driven command systems, action planners, and embodied AI agents.
            </p>
            <Link style={cardBtn} to="/docs/vla-systems/module-4-vla-foundations">
              Open Module →
            </Link>
          </div>

          {/* MODULE 5 */}
          <div style={cardStyle}>
            <h3 style={cardTitle}>Module 5: Advanced AI & Motion Control</h3>
            <p style={cardText}>
              Reinforcement learning, motion planning, MPC, trajectory optimization,
              and how robots think and move intelligently.
            </p>
            <Link style={cardBtn} to="/docs/advanced-ai-control/module-5-advanced-ai">
              Open Module →
            </Link>
          </div>

          {/* MODULE 6 */}
          <div style={cardStyle}>
            <h3 style={cardTitle}>Module 6: Designing Humanoid Robots</h3>
            <p style={cardText}>
              Learn end-to-end humanoid creation: mechanical design, kinematics, actuators,
              morphologies, energy systems, and AI-driven movement.
            </p>
            <Link style={cardBtn} to="/docs/humanoid-design/module-6-humanoid-design">
              Open Module →
            </Link>
          </div>

          {/* APPENDIX */}
          <div style={cardStyle}>
            <h3 style={cardTitle}>Appendix</h3>
            <p style={cardText}>
              Glossary, research papers, references, external resources, and further reading
              for mastering robotics and AI.
            </p>
            <Link style={cardBtn} to="/docs/appendix/glossary">
              Open Appendix →
            </Link>
          </div>
        </div>
      </section>
    </Layout>
  );
}

/* ======== STYLES ======== */
const cardStyle = {
  background: "#fffbe6",
  padding: "22px",
  borderRadius: "14px",
  boxShadow: "0 4px 14px rgba(255, 193, 7, 0.25)",
  border: "1px solid #ffe066",
};

const cardTitle = {
  fontSize: "20px",
  fontWeight: "700",
  color: "#b78900",
  marginBottom: "8px",
};

const cardText = {
  fontSize: "15px",
  color: "#6e5f00",
  marginBottom: "18px",
  lineHeight: "1.6",
};

const cardBtn = {
  textDecoration: "none",
  background: "#ffc107",
  padding: "10px 18px",
  color: "black",
  borderRadius: "10px",
  fontSize: "15px",
  fontWeight: "600",
  boxShadow: "0 3px 8px rgba(0,0,0,0.15)",
  transition: "0.25s",
};

const featureBox = {
  padding: "22px",
  background: "#fff9d6",
  borderRadius: "12px",
  border: "1px solid #ffe08a",
  boxShadow: "0 2px 10px rgba(255, 193, 7, 0.15)",
};
