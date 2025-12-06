import React from 'react';
import Layout from '@theme/Layout';
import styles from './modules.module.css'; // Assuming you'll create a CSS module for this page

function ModuleCard({title, description, link}) {
  return (
    <div className={styles.moduleCard}>
      <h3><a href={link}>{title}</a></h3>
      <p>{description}</p>
    </div>
  );
}

export default function Modules() {
  return (
    <Layout title="Modules" description="Overview of all textbook modules.">
      <header className="hero hero--primary">
        <div className="container">
          <h1 className="hero__title">Textbook Modules</h1>
          <p className="hero__subtitle">Explore the different learning paths.</p>
        </div>
      </header>
      <main className="container margin-vert--lg">
        <div className={styles.modulesGrid}>
          <ModuleCard
            title="ROS 2 Fundamentals"
            description="Learn the basics of ROS 2, nodes, topics, services, and URDF."
            link="/docs/02-ros2-foundations/module-1-ros2"
          />
          <ModuleCard
            title="Digital Twin Simulation"
            description="Dive into Gazebo and Unity for robot physics and sensor simulation."
            link="/docs/03-simulation/digital-twins"
          />
          <ModuleCard
            title="NVIDIA Isaac"
            description="Master Isaac Sim, Isaac ROS for VSLAM, and autonomous navigation with Nav2."
            link="/docs/04-hardware-basics/module-3-hardware"
          />
          <ModuleCard
            title="Vision-Language-Action Systems"
            description="Build end-to-end autonomous humanoids using LLMs and multi-modal perception."
            link="/docs/05-vla-systems/module-4-vla-foundations"
          />
          {/* Add more ModuleCards as chapters are created */}
        </div>
      </main>
    </Layout>
  );
}
