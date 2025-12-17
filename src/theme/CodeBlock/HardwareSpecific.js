/**
 * HardwareSpecific Component
 *
 * A custom code block component that shows hardware-specific code examples
 * based on the user's hardware profile or the chapter's hardware requirements.
 */

import React, { useState } from 'react';
import clsx from 'clsx';
import styles from './HardwareSpecific.module.css';
import CodeBlock from '@theme/CodeBlock';

const HardwareSpecific = ({
  children,
  hardwareType = 'all',
  title = 'Hardware-Specific Code Example',
  showSelector = true
}) => {
  const [selectedHardware, setSelectedHardware] = useState(hardwareType);

  const hardwareOptions = [
    { value: 'all', label: 'All Hardware', description: 'Show code for all hardware types' },
    { value: 'jetson', label: 'NVIDIA Jetson', description: 'Code optimized for Jetson platforms' },
    { value: 'ros', label: 'ROS/ROS2', description: 'Generic ROS/ROS2 code' },
    { value: 'nvidia-isaac', label: 'NVIDIA Isaac', description: 'Isaac-specific code' },
    { value: 'simulation', label: 'Simulation', description: 'Simulation-only code' }
  ];

  const getHardwareSpecificCode = () => {
    if (typeof children === 'string') {
      // If children is a string, return it as is
      return children;
    }

    // If children is an object with different hardware implementations
    // In a real implementation, this would parse the children and return
    // the appropriate code block based on the selected hardware
    return children[selectedHardware] || children['all'] || children;
  };

  const currentCode = getHardwareSpecificCode();

  return (
    <div className={styles.hardwareSpecificContainer}>
      <div className={styles.header}>
        <h4 className={styles.title}>{title}</h4>
        {showSelector && (
          <div className={styles.selector}>
            <label htmlFor="hardware-select" className={styles.selectorLabel}>
              Hardware Type:
            </label>
            <select
              id="hardware-select"
              value={selectedHardware}
              onChange={(e) => setSelectedHardware(e.target.value)}
              className={styles.selectorSelect}
            >
              {hardwareOptions.map(option => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>
        )}
      </div>

      <div className={styles.codeContainer}>
        <CodeBlock language="python" title={`${title} - ${hardwareOptions.find(opt => opt.value === selectedHardware)?.label || selectedHardware}`}>
          {currentCode}
        </CodeBlock>
      </div>

      <div className={styles.infoPanel}>
        <div className={styles.infoItem}>
          <span className={styles.infoLabel}>Target Hardware:</span>
          <span className={styles.infoValue}>
            {hardwareOptions.find(opt => opt.value === selectedHardware)?.label || selectedHardware}
          </span>
        </div>
        <div className={styles.infoItem}>
          <span className={styles.infoLabel}>Compatibility:</span>
          <span className={styles.infoValue}>
            {selectedHardware === 'jetson' ? 'Jetson Orin Nano and above' :
             selectedHardware === 'nvidia-isaac' ? 'NVIDIA Isaac Platform' :
             selectedHardware === 'simulation' ? 'Simulation Only' :
             'All Platforms'}
          </span>
        </div>
      </div>
    </div>
  );
};

// Example usage component for demonstration
const HardwareSpecificExample = () => {
  const rosExample = `# Basic ROS 2 Publisher Example
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()`;

  const jetsonExample = `# Jetson-Specific Example with Hardware Acceleration
import jetson.inference
import jetson.utils
import cv2

# Initialize camera
camera = jetson.utils.gstCamera(1280, 720, '/dev/video0')
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

while True:
    img, width, height = camera.CaptureRGBA()
    detections = net.Detect(img, width, height)

    # Process detections on Jetson's GPU
    for detection in detections:
        print(f"Detected: {detection.ClassID} with {detection.Confidence:.2f} confidence")

    # Display result
    jetson.utils.cudaDeviceSynchronize()`;

  const isaacExample = `# NVIDIA Isaac ROS Example
from geometry_msgs.msg import Twist
import rclpy
from rclpy.node import Node

class IsaacNavigationNode(Node):
    def __init__(self):
        super().__init__('isaac_navigation')
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # Isaac-specific navigation parameters
        self.linear_speed = 0.5  # m/s
        self.angular_speed = 0.3  # rad/s

    def move_robot(self, linear, angular):
        twist = Twist()
        twist.linear.x = linear
        twist.angular.z = angular
        self.cmd_vel_pub.publish(twist)`;

  return (
    <div>
      <HardwareSpecific
        hardwareType="ros"
        title="ROS 2 Publisher Example"
        showSelector={true}
      >
        {rosExample}
      </HardwareSpecific>

      <HardwareSpecific
        hardwareType="jetson"
        title="Jetson Hardware Acceleration Example"
        showSelector={true}
      >
        {jetsonExample}
      </HardwareSpecific>

      <HardwareSpecific
        hardwareType="nvidia-isaac"
        title="Isaac Navigation Example"
        showSelector={true}
      >
        {isaacExample}
      </HardwareSpecific>
    </div>
  );
};

HardwareSpecific.Example = HardwareSpecificExample;

export default HardwareSpecific;