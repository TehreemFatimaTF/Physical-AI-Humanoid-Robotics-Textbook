/**
 * Ready to Build the Future Page
 *
 * This page serves as the closing page for the Physical AI & Humanoid Robotics textbook,
 * inspiring students to apply their knowledge and build the future of robotics.
 */

import React from 'react';
import clsx from 'clsx';
import Layout from '@theme/Layout';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Link from '@docusaurus/Link';

function ReadyToBuildTheFuture() {
  const { siteConfig } = useDocusaurusContext();

  return (
    <Layout
      title={`Ready to Build the Future? - ${siteConfig.title}`}
      description="The closing page for the Physical AI & Humanoid Robotics textbook, inspiring students to build the future of robotics">
      <main>
        <div className="container margin-vert--lg">
          <div className="row">
            <div className="col col--12">
              <header className="hero hero--primary">
                <div className="container">
                  <h1 className="hero__title">Ready to Build the Future?</h1>
                  <p className="hero__subtitle">
                    You've learned the fundamentals of Physical AI & Humanoid Robotics
                  </p>
                </div>
              </header>
            </div>
          </div>

          <div className="row margin-vert--lg">
            <div className="col col--8 col--offset-2">
              <section className="margin-bottom--lg">
                <h2 className="text--center">Your Journey Doesn't End Here</h2>
                <p>
                  Congratulations! You've completed the comprehensive Physical AI & Humanoid Robotics curriculum.
                  You now have the knowledge to design, simulate, and deploy humanoid robots capable of natural
                  human interactions using ROS 2, Gazebo, NVIDIA Isaac, and Vision-Language-Action systems.
                </p>
              </section>

              <section className="margin-bottom--lg">
                <h2 className="text--center">What You've Learned</h2>
                <ul>
                  <li>ROS 2 fundamentals: Nodes, Topics, Services, and URDF for humanoid robots</li>
                  <li>Digital twin simulation: Physics, rendering, and sensor simulation in Gazebo and Unity</li>
                  <li>NVIDIA Isaac: Advanced perception, Isaac Sim, Isaac ROS, and hardware-accelerated navigation</li>
                  <li>Vision-Language-Action systems: Voice commands, cognitive planning, and computer vision</li>
                  <li>Capstone project: Implementing autonomous humanoid robots</li>
                </ul>
              </section>

              <section className="margin-bottom--lg">
                <h2 className="text--center">Next Steps</h2>
                <p>
                  Now that you have the theoretical knowledge, it's time to put it into practice:
                </p>
                <ol>
                  <li>Set up your hardware environment (Jetson Orin Nano, sensors, etc.)</li>
                  <li>Start with the capstone project to apply everything you've learned</li>
                  <li>Join our community of Physical AI practitioners</li>
                  <li>Continue learning with advanced topics and research papers</li>
                </ol>
              </section>

              <section className="margin-bottom--lg">
                <h2 className="text--center">The Future is Embodied</h2>
                <p>
                  Physical AI represents the convergence of digital intelligence and the physical world.
                  As you venture into building real robots, remember that you're not just writing code—
                  you're creating systems that interact with and understand the physical world.
                </p>
                <p>
                  The humanoid robots of tomorrow depend on your understanding of the principles you've learned:
                  robust control systems, accurate perception, intelligent decision-making, and seamless
                  human-robot interaction.
                </p>
              </section>

              <div className="text--center margin-vert--lg">
                <Link
                  className="button button--primary button--lg"
                  to="/docs/capstone-project/project-overview">
                  Start Capstone Project
                </Link>
              </div>

              <div className="text--center margin-vert--lg">
                <Link
                  className="button button--secondary button--lg"
                  to="/modules">
                  Review Modules
                </Link>
              </div>

              <div className="text--center margin-vert--lg">
                <Link
                  className="button button--secondary button--lg"
                  to="/docs/intro">
                  Revisit Introduction
                </Link>
              </div>
            </div>
          </div>

          <div className="row">
            <div className="col col--8 col--offset-2">
              <blockquote className="text--center">
                <p>
                  "The future belongs to those who understand the harmony between artificial intelligence
                  and physical systems. You are now equipped to build that future."
                </p>
                <p>
                  <em>— The Physical AI & Humanoid Robotics Team</em>
                </p>
              </blockquote>
            </div>
          </div>
        </div>
      </main>
    </Layout>
  );
}

export default ReadyToBuildTheFuture;