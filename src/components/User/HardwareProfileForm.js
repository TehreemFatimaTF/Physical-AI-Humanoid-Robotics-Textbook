/**
 * HardwareProfileForm Component
 *
 * A form component that collects users' hardware background information
 * during registration or profile updates for personalized content delivery.
 */

import React, { useState } from 'react';
import clsx from 'clsx';
import styles from './HardwareProfileForm.module.css';

const HardwareProfileForm = ({
  initialData = {},
  onSubmit,
  onCancel,
  submitButtonText = "Save Profile",
  isSubmitting = false
}) => {
  const [formData, setFormData] = useState({
    hardware_type: initialData.hardware_type || '',
    model: initialData.model || '',
    components: initialData.components || [],
    software_stack: initialData.software_stack || [],
    experience_level: initialData.experience_level || 'intermediate',
    jetson_model: initialData.jetson_model || '',
    sensors: initialData.sensors || [],
    ...initialData
  });

  const [errors, setErrors] = useState({});

  const hardwareTypes = [
    { value: 'jetson', label: 'NVIDIA Jetson' },
    { value: 'ros_system', label: 'ROS System' },
    { value: 'nvidia_platform', label: 'NVIDIA Platform' },
    { value: 'cloud', label: 'Cloud Workstation' },
    { value: 'other', label: 'Other' }
  ];

  const jetsonModels = [
    { value: 'orin_nano', label: 'Jetson Orin Nano' },
    { value: 'orin_nano_super', label: 'Jetson Orin Nano Super' },
    { value: 'agx_orin', label: 'Jetson AGX Orin' },
    { value: 'xavier', label: 'Jetson Xavier' },
    { value: 'nano', label: 'Jetson Nano' }
  ];

  const commonComponents = [
    { value: 'realsense_d435i', label: 'Intel RealSense D435i' },
    { value: 'realsense_d435', label: 'Intel RealSense D435' },
    { value: 'respeaker', label: 'ReSpeaker Microphone Array' },
    { value: 'camera', label: 'Camera' },
    { value: 'lidar', label: 'LiDAR Sensor' },
    { value: 'imu', label: 'IMU (Inertial Measurement Unit)' }
  ];

  const softwareStacks = [
    { value: 'ros2', label: 'ROS 2' },
    { value: 'nvidia_isaac', label: 'NVIDIA Isaac' },
    { value: 'isaac_sim', label: 'Isaac Sim' },
    { value: 'isaac_ros', label: 'Isaac ROS' },
    { value: 'omniverse', label: 'NVIDIA Omniverse' },
    { value: 'python', label: 'Python' },
    { value: 'c++', label: 'C++' }
  ];

  const experienceLevels = [
    { value: 'beginner', label: 'Beginner' },
    { value: 'intermediate', label: 'Intermediate' },
    { value: 'advanced', label: 'Advanced' }
  ];

  const handleInputChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));

    // Clear error when user starts typing
    if (errors[field]) {
      setErrors(prev => ({
        ...prev,
        [field]: ''
      }));
    }
  };

  const handleComponentToggle = (component) => {
    setFormData(prev => {
      const currentComponents = prev.components || [];
      const newComponents = currentComponents.includes(component)
        ? currentComponents.filter(c => c !== component)
        : [...currentComponents, component];

      return {
        ...prev,
        components: newComponents
      };
    });
  };

  const handleSoftwareToggle = (software) => {
    setFormData(prev => {
      const currentSoftware = prev.software_stack || [];
      const newSoftware = currentSoftware.includes(software)
        ? currentSoftware.filter(s => s !== software)
        : [...currentSoftware, software];

      return {
        ...prev,
        software_stack: newSoftware
      };
    });
  };

  const validateForm = () => {
    const newErrors = {};

    if (!formData.hardware_type) {
      newErrors.hardware_type = 'Hardware type is required';
    }

    if (formData.hardware_type === 'jetson' && !formData.jetson_model) {
      newErrors.jetson_model = 'Jetson model is required for Jetson hardware type';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e) => {
    e.preventDefault();

    if (validateForm()) {
      onSubmit({
        ...formData,
        jetson_model: formData.hardware_type === 'jetson' ? formData.jetson_model : undefined
      });
    }
  };

  return (
    <div className={styles.formContainer}>
      <h3 className={styles.formTitle}>Hardware Profile Information</h3>
      <p className={styles.formSubtitle}>
        Help us personalize your learning experience by sharing your hardware setup
      </p>

      <form onSubmit={handleSubmit} className={styles.form}>
        <div className={styles.formGroup}>
          <label className={styles.label}>
            Hardware Type *
          </label>
          <select
            value={formData.hardware_type}
            onChange={(e) => handleInputChange('hardware_type', e.target.value)}
            className={clsx(styles.select, errors.hardware_type && styles.selectError)}
          >
            <option value="">Select your hardware type</option>
            {hardwareTypes.map(type => (
              <option key={type.value} value={type.value}>
                {type.label}
              </option>
            ))}
          </select>
          {errors.hardware_type && (
            <div className={styles.errorText}>{errors.hardware_type}</div>
          )}
        </div>

        {formData.hardware_type === 'jetson' && (
          <div className={styles.formGroup}>
            <label className={styles.label}>
              Jetson Model *
            </label>
            <select
              value={formData.jetson_model}
              onChange={(e) => handleInputChange('jetson_model', e.target.value)}
              className={clsx(styles.select, errors.jetson_model && styles.selectError)}
            >
              <option value="">Select your Jetson model</option>
              {jetsonModels.map(model => (
                <option key={model.value} value={model.value}>
                  {model.label}
                </option>
              ))}
            </select>
            {errors.jetson_model && (
              <div className={styles.errorText}>{errors.jetson_model}</div>
            )}
          </div>
        )}

        <div className={styles.formGroup}>
          <label className={styles.label}>
            Components & Sensors
          </label>
          <div className={styles.checkboxGroup}>
            {commonComponents.map(component => (
              <label key={component.value} className={styles.checkboxLabel}>
                <input
                  type="checkbox"
                  checked={(formData.components || []).includes(component.value)}
                  onChange={() => handleComponentToggle(component.value)}
                  className={styles.checkbox}
                />
                <span className={styles.checkboxText}>{component.label}</span>
              </label>
            ))}
          </div>
        </div>

        <div className={styles.formGroup}>
          <label className={styles.label}>
            Software Stack
          </label>
          <div className={styles.checkboxGroup}>
            {softwareStacks.map(software => (
              <label key={software.value} className={styles.checkboxLabel}>
                <input
                  type="checkbox"
                  checked={(formData.software_stack || []).includes(software.value)}
                  onChange={() => handleSoftwareToggle(software.value)}
                  className={styles.checkbox}
                />
                <span className={styles.checkboxText}>{software.label}</span>
              </label>
            ))}
          </div>
        </div>

        <div className={styles.formGroup}>
          <label className={styles.label}>
            Experience Level *
          </label>
          <div className={styles.radioGroup}>
            {experienceLevels.map(level => (
              <label key={level.value} className={styles.radioLabel}>
                <input
                  type="radio"
                  name="experience_level"
                  value={level.value}
                  checked={formData.experience_level === level.value}
                  onChange={(e) => handleInputChange('experience_level', e.target.value)}
                  className={styles.radio}
                />
                <span className={styles.radioText}>{level.label}</span>
              </label>
            ))}
          </div>
        </div>

        <div className={styles.buttonGroup}>
          <button
            type="submit"
            disabled={isSubmitting}
            className={clsx(styles.submitButton, isSubmitting && styles.submitButtonDisabled)}
          >
            {isSubmitting ? 'Saving...' : submitButtonText}
          </button>
          {onCancel && (
            <button
              type="button"
              onClick={onCancel}
              className={styles.cancelButton}
            >
              Cancel
            </button>
          )}
        </div>
      </form>
    </div>
  );
};

export default HardwareProfileForm;