/**
 * Modules Page
 *
 * This page displays all Physical AI modules in a card-based UI layout,
 * allowing users to navigate to different modules of the curriculum.
 */

import React from 'react';
import clsx from 'clsx';
import Layout from '@theme/Layout';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import ModuleCard from '../components/ModuleCards/ModuleCard';

function ModulesPage() {
  const { siteConfig } = useDocusaurusContext();

  // Module data for the Physical AI curriculum
  const modules = [
    {
      id: 1,
      title: 'The Robotic Nervous System (ROS 2)',
      description: 'Learn about middleware for robot control, including ROS 2 Nodes, Topics, and Services. Understand how to bridge Python AI agents to ROS controllers using rclpy.',
      link: '/docs/module-1-ros2/chapter-1-intro-ros2',
    },
    {
      id: 2,
      title: 'The Digital Twin (Gazebo & Unity)',
      description: 'Master physics simulation and environment building. Simulate physics, gravity, and collisions in Gazebo, with high-fidelity rendering and human-robot interaction in Unity.',
      link: '/docs/module-2-digital-twin/chapter-1-physics-simulation',
    },
    {
      id: 3,
      title: 'The AI-Robot Brain (NVIDIA Isaac™)',
      description: 'Explore advanced perception and training with NVIDIA Isaac Sim for photorealistic simulation and synthetic data generation, and Isaac ROS for hardware-accelerated navigation.',
      link: '/docs/module-3-nvidia-isaac/chapter-1-isaac-sim',
    },
    {
      id: 4,
      title: 'Vision-Language-Action (VLA)',
      description: 'Understand the convergence of LLMs and Robotics. Learn voice-to-action using OpenAI Whisper and cognitive planning using LLMs to translate natural language into ROS 2 actions.',
      link: '/docs/module-4-vla-humanoids/chapter-1-voice-to-action',
    }
  ];

  return (
    <Layout
      title={`Modules - ${siteConfig.title}`}
      description="Physical AI & Humanoid Robotics curriculum modules">
      <main>
        <div className="container margin-vert--lg">
          <div className="row">
            <div className="col col--12">
              <header className="hero hero--primary">
                <div className="container">
                  <h1 className="hero__title">Physical AI Curriculum Modules</h1>
                  <p className="hero__subtitle">
                    Explore the comprehensive Physical AI & Humanoid Robotics curriculum through these four specialized modules
                  </p>
                </div>
              </header>
            </div>
          </div>

          <div className="row margin-vert--lg">
            {modules.map((module) => (
              <ModuleCard
                key={module.id}
                title={module.title}
                description={module.description}
                moduleNumber={module.id}
                link={module.link}
              />
            ))}
          </div>

          <div className="row">
            <div className="col col--12 text--center">
              <p className="text--large">
                Ready to dive deep into Physical AI? Start with any module above or{' '}
                <a href="/docs/intro">read the introduction</a> first.
              </p>
            </div>
          </div>
        </div>
      </main>
    </Layout>
  );
}

export default ModulesPage;