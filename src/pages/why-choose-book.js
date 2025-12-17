/**
 * Why Choose This Book Page
 *
 * This page explains the value and benefits of the Physical AI & Humanoid Robotics textbook,
 * highlighting why it's the best choice for learning about embodied AI systems.
 */

import React from 'react';
import clsx from 'clsx';
import Layout from '@theme/Layout';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Link from '@docusaurus/Link';

function WhyChooseThisBook() {
  const { siteConfig } = useDocusaurusContext();

  return (
    <Layout
      title={`Why Choose This Book? - ${siteConfig.title}`}
      description="Discover why the Physical AI & Humanoid Robotics textbook is the best choice for learning about embodied AI systems">
      <main>
        <div className="container margin-vert--lg">
          <div className="row">
            <div className="col col--12">
              <header className="hero hero--primary">
                <div className="container">
                  <h1 className="hero__title">Why Choose This Book?</h1>
                  <p className="hero__subtitle">
                    The definitive guide to Physical AI & Humanoid Robotics
                  </p>
                </div>
              </header>
            </div>
          </div>

          <div className="row margin-vert--lg">
            <div className="col col--8 col--offset-2">
              <section className="margin-bottom--lg">
                <h2 className="text--center">The Future of AI is Physical</h2>
                <p>
                  The future of AI extends beyond digital spaces into the physical world. This capstone
                  quarter introduces <strong>Physical AI</strong>—AI systems that function in reality
                  and comprehend physical laws. Students learn to design, simulate, and deploy humanoid
                  robots capable of natural human interactions using ROS 2, Gazebo, and NVIDIA Isaac.
                </p>
              </section>

              <section className="margin-bottom--lg">
                <h2 className="text--center">Comprehensive Curriculum</h2>
                <p>
                  Our curriculum covers four essential modules that progressively build your understanding
                  of Physical AI:
                </p>
                <ul>
                  <li><strong>Module 1:</strong> The Robotic Nervous System (ROS 2)</li>
                  <li><strong>Module 2:</strong> The Digital Twin (Gazebo & Unity)</li>
                  <li><strong>Module 3:</strong> The AI-Robot Brain (NVIDIA Isaac™)</li>
                  <li><strong>Module 4:</strong> Vision-Language-Action (VLA)</li>
                </ul>
              </section>

              <section className="margin-bottom--lg">
                <h2 className="text--center">Cutting-Edge Technology</h2>
                <p>
                  Learn with industry-standard tools and frameworks including:
                </p>
                <ul>
                  <li>ROS 2 for robotic middleware and control</li>
                  <li>NVIDIA Isaac for advanced perception and navigation</li>
                  <li>Gazebo and Unity for digital twin simulation</li>
                  <li>Vision-Language-Action systems for human-robot interaction</li>
                </ul>
              </section>

              <section className="margin-bottom--lg">
                <h2 className="text--center">Hands-On Learning</h2>
                <p>
                  This isn't just theoretical knowledge. Our curriculum emphasizes practical application
                  with:
                </p>
                <ul>
                  <li>Real hardware setup guides (Jetson Orin Nano, Intel RealSense, etc.)</li>
                  <li>Sim-to-real transfer techniques</li>
                  <li>Capstone project implementing autonomous humanoid robots</li>
                  <li>Hardware-aware content personalization</li>
                </ul>
              </section>

              <section className="margin-bottom--lg">
                <h2 className="text--center">AI-Powered Learning Support</h2>
                <p>
                  Access our RAG-based AI assistant that answers questions based solely on textbook
                  content, ensuring accurate and relevant responses. Plus, content is available in
                  multiple languages including Urdu for broader accessibility.
                </p>
              </section>

              <div className="text--center margin-vert--lg">
                <Link
                  className="button button--primary button--lg"
                  to="/docs/intro">
                  Start Learning Now
                </Link>
              </div>

              <div className="text--center margin-vert--lg">
                <Link
                  className="button button--secondary button--lg"
                  to="/modules">
                  Browse Modules
                </Link>
              </div>
            </div>
          </div>
        </div>
      </main>
    </Layout>
  );
}

export default WhyChooseThisBook;