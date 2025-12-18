/**
 * ProjectTracker Component
 *
 * A component that tracks the progress of the capstone project for students,
 * allowing them to monitor their completion of various milestones.
 */

import React, { useState } from 'react';
import clsx from 'clsx';
import styles from './ProjectTracker.module.css';

const ProjectTracker = ({ projectTitle = "Autonomous Humanoid Robot" }) => {
  const [milestones, setMilestones] = useState([
    {
      id: 1,
      title: "Project Planning",
      description: "Define project requirements and create implementation plan",
      completed: false,
      dueDate: "2025-01-15"
    },
    {
      id: 2,
      title: "Hardware Setup",
      description: "Configure Jetson Orin Nano and sensors (Intel RealSense, ReSpeaker)",
      completed: false,
      dueDate: "2025-01-22"
    },
    {
      id: 3,
      title: "ROS 2 Integration",
      description: "Implement ROS 2 nodes for robot control and communication",
      completed: false,
      dueDate: "2025-02-05"
    },
    {
      id: 4,
      title: "Voice Command Recognition",
      description: "Implement OpenAI Whisper for voice-to-action functionality",
      completed: false,
      dueDate: "2025-02-12"
    },
    {
      id: 5,
      title: "Path Planning",
      description: "Implement Nav2 for bipedal humanoid movement and navigation",
      completed: false,
      dueDate: "2025-02-19"
    },
    {
      id: 6,
      title: "Computer Vision",
      description: "Implement object identification and manipulation using Isaac ROS",
      completed: false,
      dueDate: "2025-02-26"
    },
    {
      id: 7,
      title: "Integration & Testing",
      description: "Combine all components and test autonomous functionality",
      completed: false,
      dueDate: "2025-03-05"
    },
    {
      id: 8,
      title: "Final Demonstration",
      description: "Demonstrate the complete autonomous humanoid robot project",
      completed: false,
      dueDate: "2025-03-12"
    }
  ]);

  const toggleMilestone = (id) => {
    setMilestones(prev =>
      prev.map(milestone =>
        milestone.id === id
          ? { ...milestone, completed: !milestone.completed }
          : milestone
      )
    );
  };

  const completedCount = milestones.filter(m => m.completed).length;
  const totalCount = milestones.length;
  const progressPercentage = totalCount > 0 ? Math.round((completedCount / totalCount) * 100) : 0;

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric'
    });
  };

  const isOverdue = (dueDate) => {
    const today = new Date();
    const due = new Date(dueDate);
    return due < today && !milestones.find(m => m.id === parseInt(dueDate.split('-')[2]) && m.completed);
  };

  return (
    <div className={styles.projectTracker}>
      <div className={styles.header}>
        <h3 className={styles.title}>{projectTitle}</h3>
        <div className={styles.progressContainer}>
          <div className={styles.progressBar}>
            <div
              className={styles.progressFill}
              style={{ width: `${progressPercentage}%` }}
            ></div>
          </div>
          <div className={styles.progressText}>
            {completedCount} of {totalCount} milestones completed ({progressPercentage}%)
          </div>
        </div>
      </div>

      <div className={styles.milestones}>
        {milestones.map((milestone) => (
          <div
            key={milestone.id}
            className={clsx(
              styles.milestone,
              milestone.completed && styles.completed,
              isOverdue(milestone.dueDate) && !milestone.completed && styles.overdue
            )}
          >
            <div className={styles.milestoneHeader}>
              <label className={styles.milestoneCheckbox}>
                <input
                  type="checkbox"
                  checked={milestone.completed}
                  onChange={() => toggleMilestone(milestone.id)}
                  className={styles.checkbox}
                />
                <span className={styles.checkmark}></span>
                <span className={styles.milestoneTitle}>{milestone.title}</span>
              </label>
              <span className={styles.dueDate}>
                Due: {formatDate(milestone.dueDate)}
              </span>
            </div>

            <div className={styles.milestoneDescription}>
              {milestone.description}
            </div>

            {isOverdue(milestone.dueDate) && !milestone.completed && (
              <div className={styles.overdueNotice}>
                ⚠️ This milestone is overdue
              </div>
            )}
          </div>
        ))}
      </div>

      <div className={styles.summary}>
        <div className={styles.summaryItem}>
          <span className={styles.summaryLabel}>Completed:</span>
          <span className={styles.summaryValue}>{completedCount}</span>
        </div>
        <div className={styles.summaryItem}>
          <span className={styles.summaryLabel}>Remaining:</span>
          <span className={styles.summaryValue}>{totalCount - completedCount}</span>
        </div>
        <div className={styles.summaryItem}>
          <span className={styles.summaryLabel}>Progress:</span>
          <span className={styles.summaryValue}>{progressPercentage}%</span>
        </div>
      </div>

      {progressPercentage === 100 && (
        <div className={styles.completionMessage}>
          🎉 Congratulations! You've completed the capstone project.
        </div>
      )}
    </div>
  );
};

export default ProjectTracker;