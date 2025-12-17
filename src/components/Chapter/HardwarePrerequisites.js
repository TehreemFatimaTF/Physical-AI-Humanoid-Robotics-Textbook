/**
 * HardwarePrerequisites Component
 *
 * A component that displays hardware requirements for a specific chapter,
 * helping users understand what hardware is needed before proceeding.
 */

import React from 'react';
import clsx from 'clsx';
import styles from './HardwarePrerequisites.module.css';

const HardwarePrerequisites = ({
  chapterTitle,
  requiredHardware = [],
  recommendedHardware = [],
  difficultyLevel = 'intermediate'
}) => {
  // Map difficulty levels to display labels
  const difficultyLabels = {
    beginner: { label: 'Beginner', color: 'green' },
    intermediate: { label: 'Intermediate', color: 'orange' },
    advanced: { label: 'Advanced', color: 'red' }
  };

  const difficulty = difficultyLabels[difficultyLevel] || difficultyLabels.intermediate;

  // Function to render hardware list
  const renderHardwareList = (hardwareList, title) => {
    if (!hardwareList || hardwareList.length === 0) return null;

    return (
      <div className={styles.hardwareSection}>
        <h4 className={styles.sectionTitle}>{title}</h4>
        <ul className={styles.hardwareList}>
          {hardwareList.map((item, index) => (
            <li key={index} className={styles.hardwareItem}>
              <span className={styles.hardwareIcon}>✓</span>
              <span className={styles.hardwareText}>{item}</span>
            </li>
          ))}
        </ul>
      </div>
    );
  };

  // Determine if user has required hardware (simplified - in real app this would check user profile)
  const hasRequiredHardware = requiredHardware.length === 0;

  return (
    <div className={styles.prerequisitesContainer}>
      <div className={styles.header}>
        <h3 className={styles.title}>
          Hardware Prerequisites for: {chapterTitle}
        </h3>
        <div className={clsx(
          styles.difficultyBadge,
          styles[difficulty.color]
        )}>
          {difficulty.label}
        </div>
      </div>

      <div className={styles.content}>
        {!hasRequiredHardware && (
          <div className={styles.warning}>
            <span className={styles.warningIcon}>⚠️</span>
            <span className={styles.warningText}>
              Some hardware requirements may not be met with your current setup.
            </span>
          </div>
        )}

        {renderHardwareList(requiredHardware, 'Required Hardware')}
        {renderHardwareList(recommendedHardware, 'Recommended Hardware')}

        {requiredHardware.length === 0 && (
          <div className={styles.noRequirements}>
            <span className={styles.infoIcon}>ℹ️</span>
            <span className={styles.infoText}>
              This chapter has no specific hardware requirements. You can follow along with software simulation only.
            </span>
          </div>
        )}

        <div className={styles.guidance}>
          <h4 className={styles.guidanceTitle}>Getting Started</h4>
          <ul className={styles.guidanceList}>
            {requiredHardware.length > 0 ? (
              <>
                <li>Review the hardware setup guides in the <a href="/docs/setup-guides">Setup Guides</a> section</li>
                <li>Ensure your hardware meets the minimum requirements before proceeding</li>
                <li>Consider starting with simulation if you don't have all required hardware</li>
              </>
            ) : (
              <>
                <li>No special hardware required for this chapter</li>
                <li>Software simulation will be sufficient</li>
                <li>Advanced users can try with real hardware if available</li>
              </>
            )}
          </ul>
        </div>
      </div>
    </div>
  );
};

export default HardwarePrerequisites;