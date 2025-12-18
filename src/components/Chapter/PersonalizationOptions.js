/**
 * PersonalizationOptions Component
 *
 * A component that displays hardware-aware content options for registered users.
 * Allows users to see content tailored to their hardware setup and experience level.
 */

import React, { useState } from 'react';
import clsx from 'clsx';
import styles from './PersonalizationOptions.module.css';

const PersonalizationOptions = ({
  userHardwareProfile,
  onProfileChange,
  contentMetadata
}) => {
  const [showOptions, setShowOptions] = useState(false);
  const [selectedProfile, setSelectedProfile] = useState(userHardwareProfile || {});

  const toggleOptions = () => {
    setShowOptions(!showOptions);
  };

  const handleProfileChange = (newProfile) => {
    setSelectedProfile(newProfile);
    onProfileChange && onProfileChange(newProfile);
  };

  // Check if content is compatible with user's hardware
  const isCompatible = () => {
    if (!contentMetadata || !selectedProfile) return true;

    const requiredHardware = contentMetadata.target_hardware || [];
    if (requiredHardware.length === 0) return true;

    // Simple compatibility check - in a real implementation this would be more sophisticated
    return requiredHardware.some(req =>
      selectedProfile.hardware_type === req ||
      selectedProfile.model === req ||
      selectedProfile.components?.includes(req)
    );
  };

  // Get compatibility message
  const getCompatibilityMessage = () => {
    if (!contentMetadata || !selectedProfile) return null;

    const requiredHardware = contentMetadata.target_hardware || [];
    if (requiredHardware.length === 0) return null;

    if (isCompatible()) {
      return {
        type: 'success',
        message: 'This content is compatible with your hardware setup!'
      };
    } else {
      return {
        type: 'warning',
        message: `This content requires: ${requiredHardware.join(', ')}. Consider reviewing setup guides.`
      };
    }
  };

  const compatibility = getCompatibilityMessage();

  return (
    <div className={styles.personalizationContainer}>
      <div className={styles.header}>
        <h3 className={styles.title}>Personalization Options</h3>
        <button
          className={styles.toggleButton}
          onClick={toggleOptions}
          aria-label={showOptions ? "Hide options" : "Show options"}
        >
          {showOptions ? '▼' : '▶'}
        </button>
      </div>

      {showOptions && (
        <div className={styles.optionsPanel}>
          <div className={styles.profileSelector}>
            <label className={styles.label}>Your Hardware Profile:</label>
            <select
              className={styles.select}
              value={selectedProfile.id || ''}
              onChange={(e) => {
                const profile = JSON.parse(e.target.value);
                handleProfileChange(profile);
              }}
            >
              <option value="">Select your setup</option>
              <option value={JSON.stringify({
                id: 'jetson_orin_nano',
                name: 'Jetson Orin Nano',
                hardware_type: 'jetson',
                model: 'orin_nano_super',
                components: ['realsense_d435i', 'respeaker'],
                software_stack: ['ros2', 'nvidia_isaac']
              })}>
                Jetson Orin Nano + Sensors
              </option>
              <option value={JSON.stringify({
                id: 'cloud_workstation',
                name: 'Cloud Workstation',
                hardware_type: 'cloud',
                model: 'aws_g5_2xlarge',
                components: ['isaac_sim'],
                software_stack: ['nvidia_isaac', 'omniverse']
              })}>
                Cloud Workstation (AWS G5)
              </option>
              <option value={JSON.stringify({
                id: 'basic_ros',
                name: 'Basic ROS Setup',
                hardware_type: 'ros_system',
                model: 'desktop',
                components: ['camera', 'lidar'],
                software_stack: ['ros2']
              })}>
                Basic ROS Setup
              </option>
              <option value={JSON.stringify({
                id: 'custom',
                name: 'Custom Setup',
                hardware_type: 'other',
                components: [],
                software_stack: []
              })}>
                Custom Setup
              </option>
            </select>
          </div>

          {selectedProfile.id && (
            <div className={styles.currentProfile}>
              <h4>Current Profile: {selectedProfile.name}</h4>
              <div className={styles.profileDetails}>
                <p><strong>Type:</strong> {selectedProfile.hardware_type}</p>
                <p><strong>Model:</strong> {selectedProfile.model || 'N/A'}</p>
                <p><strong>Components:</strong> {selectedProfile.components?.join(', ') || 'N/A'}</p>
                <p><strong>Software:</strong> {selectedProfile.software_stack?.join(', ') || 'N/A'}</p>
              </div>
            </div>
          )}

          {compatibility && (
            <div className={clsx(
              styles.compatibilityMessage,
              styles[compatibility.type]
            )}>
              {compatibility.message}
            </div>
          )}

          <div className={styles.suggestedContent}>
            <h4>Suggested Content Based on Your Setup:</h4>
            <ul>
              {selectedProfile.hardware_type === 'jetson' && (
                <li>Hardware-specific examples optimized for Jetson platforms</li>
              )}
              {selectedProfile.software_stack?.includes('ros2') && (
                <li>ROS 2 specific code examples and tutorials</li>
              )}
              {selectedProfile.software_stack?.includes('nvidia_isaac') && (
                <li>NVIDIA Isaac specific implementation details</li>
              )}
              <li>Examples matching your experience level</li>
            </ul>
          </div>
        </div>
      )}

      {compatibility && compatibility.type === 'warning' && (
        <div className={clsx(
          styles.inlineCompatibility,
          styles[compatibility.type]
        )}>
          ⚠️ {compatibility.message}
        </div>
      )}
    </div>
  );
};

export default PersonalizationOptions;