---
sidebar_label: 'Chapter 6: Bridging Python Agents to ROS Controllers'
title: 'Chapter 6: Bridging Python Agents to ROS Controllers'
description: 'Connecting Python AI agents to ROS 2 controllers for humanoid robotics applications'
slug: '/module-1-ros2/chapter-6-rclpy-bridge'
difficulty: 'advanced'
requiredHardware: ['ros_system']
recommendedHardware: ['computer', 'nvidia_isaac']
---

# Chapter 6: Bridging Python Agents to ROS Controllers

In this final chapter of Module 1, we'll explore how to connect Python-based AI agents to ROS 2 controllers, enabling humanoid robots to leverage advanced AI capabilities. This bridge is crucial for creating intelligent robots that can make decisions, learn from experience, and adapt to their environment.

## The AI-ROS Bridge Concept

The bridge between Python AI agents and ROS 2 controllers enables:

- **Perception Integration**: AI agents process sensor data from ROS topics
- **Action Selection**: AI agents determine appropriate actions based on observations
- **Control Execution**: Actions are translated to ROS commands for execution
- **Learning Loops**: Feedback from the environment enables learning and adaptation

## Architecture of the Bridge

The bridge typically follows this architecture:

```
AI Agent (Python)
    ↓ (Actions)
ROS Controller Node
    ↓ (ROS Commands)
Robot Hardware/Physics Engine
    ↓ (Sensor Data)
ROS Sensor Nodes
    ↓ (Observations)
AI Agent (Python)
```

## Basic Bridge Implementation

Let's start with a simple example that connects a Python agent to ROS 2:

### ROS 2 Bridge Node

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import String
import threading
import queue
import time

class AgentBridgeNode(Node):
    def __init__(self):
        super().__init__('agent_bridge_node')

        # Publishers for robot commands
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.joint_cmd_publisher = self.create_publisher(Float64MultiArray, 'joint_commands', 10)

        # Subscribers for sensor data
        self.joint_state_subscriber = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10)
        self.imu_subscriber = self.create_subscription(
            String, 'imu_data', self.imu_callback, 10)  # Simplified for example

        # Queues for communication with AI agent
        self.sensor_data_queue = queue.Queue()
        self.action_queue = queue.Queue()

        # Start AI agent in separate thread
        self.ai_agent_thread = threading.Thread(target=self.run_ai_agent)
        self.ai_agent_thread.start()

        # Timer to process actions from AI agent
        self.timer = self.create_timer(0.02, self.process_actions)  # 50Hz

        self.get_logger().info('Agent Bridge Node initialized')

    def joint_state_callback(self, msg):
        """Process joint state messages and put them in the queue for the AI agent"""
        sensor_data = {
            'joint_positions': list(msg.position),
            'joint_velocities': list(msg.velocity),
            'joint_efforts': list(msg.effort),
            'timestamp': self.get_clock().now().nanoseconds
        }
        self.sensor_data_queue.put(('joint_states', sensor_data))

    def imu_callback(self, msg):
        """Process IMU data and put it in the queue for the AI agent"""
        # In a real implementation, this would be a sensor_msgs/Imu message
        imu_data = {
            'orientation': [0.0, 0.0, 0.0, 1.0],  # w, x, y, z quaternion
            'angular_velocity': [0.0, 0.0, 0.0],
            'linear_acceleration': [0.0, 0.0, 9.81]  # Gravity
        }
        self.sensor_data_queue.put(('imu', imu_data))

    def run_ai_agent(self):
        """Run the AI agent in a separate thread"""
        # Initialize your AI agent here
        # This could be a reinforcement learning agent, a neural network, etc.
        ai_agent = SimpleNavigationAgent()

        while rclpy.ok():
            # Get sensor data from queue
            sensor_data = {}
            while not self.sensor_data_queue.empty():
                data_type, data = self.sensor_data_queue.get()
                sensor_data[data_type] = data

            if sensor_data:
                # Process sensor data and get action from AI agent
                action = ai_agent.get_action(sensor_data)
                if action is not None:
                    self.action_queue.put(action)

            time.sleep(0.01)  # Small delay to prevent busy waiting

    def process_actions(self):
        """Process actions from the AI agent queue and send to robot"""
        while not self.action_queue.empty():
            action = self.action_queue.get()
            self.execute_action(action)

    def execute_action(self, action):
        """Execute an action by sending appropriate ROS messages"""
        action_type = action.get('type', 'velocity')

        if action_type == 'velocity':
            # Send velocity command
            cmd_msg = Twist()
            cmd_msg.linear.x = action.get('linear_x', 0.0)
            cmd_msg.linear.y = action.get('linear_y', 0.0)
            cmd_msg.linear.z = action.get('linear_z', 0.0)
            cmd_msg.angular.x = action.get('angular_x', 0.0)
            cmd_msg.angular.y = action.get('angular_y', 0.0)
            cmd_msg.angular.z = action.get('angular_z', 0.0)

            self.cmd_vel_publisher.publish(cmd_msg)

        elif action_type == 'joint_position':
            # Send joint position command
            cmd_msg = Float64MultiArray()
            cmd_msg.data = action.get('positions', [])
            self.joint_cmd_publisher.publish(cmd_msg)

class SimpleNavigationAgent:
    """A simple example agent for demonstration purposes"""

    def __init__(self):
        self.state = 'exploring'  # Current behavior state
        self.target = [1.0, 1.0]  # Target position
        self.position = [0.0, 0.0]  # Current position (estimated)

    def get_action(self, sensor_data):
        """Process sensor data and return an action"""
        # Simple navigation to target
        if 'joint_states' in sensor_data:
            # Extract position from joint states (simplified)
            joints = sensor_data['joint_states']
            # In a real implementation, you'd integrate encoder data
            # or use a more sophisticated localization method

        # Simple proportional controller toward target
        dx = self.target[0] - self.position[0]
        dy = self.target[1] - self.position[1]

        # Calculate desired velocity
        linear_x = min(0.5, max(-0.5, dx * 0.5))  # Limit to 0.5 m/s
        angular_z = min(0.5, max(-0.5, dy * 0.2))  # Turn toward target

        return {
            'type': 'velocity',
            'linear_x': linear_x,
            'angular_z': angular_z
        }

def main(args=None):
    rclpy.init(args=args)
    agent_bridge_node = AgentBridgeNode()

    try:
        rclpy.spin(agent_bridge_node)
    except KeyboardInterrupt:
        pass
    finally:
        agent_bridge_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Advanced AI Integration with Libraries

### Using TensorFlow/PyTorch with ROS 2

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import torch
import numpy as np
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

class DeepLearningBridgeNode(Node):
    def __init__(self):
        super().__init__('deep_learning_bridge_node')

        # Initialize PyTorch model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model()  # Load your trained model
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

        # CV Bridge for image processing
        self.cv_bridge = CvBridge()

        # Publishers and subscribers
        self.image_subscriber = self.create_subscription(
            Image, 'camera/image_raw', self.image_callback, 10)
        self.joint_subscriber = self.create_subscription(
            JointState, 'joint_states', self.joint_callback, 10)
        self.cmd_publisher = self.create_publisher(Twist, 'cmd_vel', 10)

        # Store recent sensor data
        self.recent_joint_states = None
        self.recent_image = None

        self.get_logger().info('Deep Learning Bridge Node initialized')

    def load_model(self):
        """Load your pre-trained model"""
        # Example: Load a model trained for navigation
        # model = YourNavigationModel()
        # model.load_state_dict(torch.load('navigation_model.pth'))
        # return model
        pass

    def image_callback(self, msg):
        """Process camera images for AI perception"""
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # Convert to tensor and normalize
            tensor_image = torch.from_numpy(cv_image).float().permute(2, 0, 1).unsqueeze(0)
            tensor_image = tensor_image / 255.0  # Normalize to [0, 1]
            tensor_image = tensor_image.to(self.device)

            self.recent_image = tensor_image
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def joint_callback(self, msg):
        """Store recent joint states"""
        self.recent_joint_states = {
            'position': np.array(msg.position),
            'velocity': np.array(msg.velocity),
            'effort': np.array(msg.effort)
        }

    def predict_action(self):
        """Use the AI model to predict an action"""
        if self.recent_image is not None and self.recent_joint_states is not None:
            with torch.no_grad():
                # Combine image and joint data
                observation = {
                    'image': self.recent_image,
                    'joints': torch.tensor(self.recent_joint_states['position']).float().to(self.device)
                }

                # Run inference
                action = self.model(observation)

                # Convert action to ROS message
                cmd_msg = Twist()
                cmd_msg.linear.x = float(action[0])
                cmd_msg.angular.z = float(action[1])

                return cmd_msg
        return None

def main(args=None):
    rclpy.init(args=args)
    dl_bridge_node = DeepLearningBridgeNode()

    # Timer to run inference periodically
    def inference_callback():
        cmd_msg = dl_bridge_node.predict_action()
        if cmd_msg is not None:
            dl_bridge_node.cmd_publisher.publish(cmd_msg)

    timer = dl_bridge_node.create_timer(0.1, inference_callback)  # 10Hz inference

    try:
        rclpy.spin(dl_bridge_node)
    except KeyboardInterrupt:
        pass
    finally:
        dl_bridge_node.destroy_node()
        rclpy.shutdown()
```

## Reinforcement Learning Integration

### Using Stable Baselines3 with ROS 2

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from stable_baselines3 import PPO
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray

class RLLearningBridgeNode(Node):
    def __init__(self):
        super().__init__('rl_learning_bridge_node')

        # Load pre-trained RL model
        self.rl_model = PPO.load("humanoid_navigation_model.zip")

        # Publishers and subscribers
        self.joint_subscriber = self.create_subscription(
            JointState, 'joint_states', self.joint_callback, 10)
        self.imu_subscriber = self.create_subscription(
            Float64MultiArray, 'imu_data', self.imu_callback, 10)
        self.cmd_publisher = self.create_publisher(Twist, 'cmd_vel', 10)

        # State storage
        self.current_joint_state = None
        self.current_imu_data = None

        # Action execution timer
        self.timer = self.create_timer(0.05, self.execute_rl_action)  # 20Hz

    def joint_callback(self, msg):
        """Store joint state for RL observation"""
        self.current_joint_state = np.array(msg.position + msg.velocity)

    def imu_callback(self, msg):
        """Store IMU data for RL observation"""
        self.current_imu_data = np.array(msg.data)

    def get_observation(self):
        """Assemble observation for RL model"""
        if self.current_joint_state is not None and self.current_imu_data is not None:
            # Concatenate all sensor data into a single observation vector
            observation = np.concatenate([
                self.current_joint_state,
                self.current_imu_data,
                # Add other sensor data as needed
            ])
            return observation
        return None

    def execute_rl_action(self):
        """Get action from RL model and execute it"""
        observation = self.get_observation()
        if observation is not None:
            # Get action from RL model
            action, _states = self.rl_model.predict(observation, deterministic=True)

            # Convert RL action to ROS command
            cmd_msg = Twist()
            cmd_msg.linear.x = float(action[0])
            cmd_msg.angular.z = float(action[1])

            # Publish command
            self.cmd_publisher.publish(cmd_msg)

def main(args=None):
    rclpy.init(args=args)
    rl_bridge_node = RLLearningBridgeNode()

    try:
        rclpy.spin(rl_bridge_node)
    except KeyboardInterrupt:
        pass
    finally:
        rl_bridge_node.destroy_node()
        rclpy.shutdown()
```

## Real-World Considerations

### Latency and Timing

```python
class TimedAgentBridgeNode(Node):
    def __init__(self):
        super().__init__('timed_agent_bridge_node')

        # Store timestamps for latency calculation
        self.sensor_timestamps = {}
        self.action_timestamps = []

        # Control loop timer
        self.control_period = 0.02  # 50Hz
        self.timer = self.create_timer(self.control_period, self.control_loop)

        # Maximum allowed delay
        self.max_delay = 0.1  # 100ms

    def control_loop(self):
        """Main control loop with timing considerations"""
        current_time = self.get_clock().now()

        # Check if sensor data is too old
        if hasattr(self, 'last_sensor_time'):
            delay = (current_time - self.last_sensor_time).nanoseconds / 1e9
            if delay > self.max_delay:
                self.get_logger().warn(f'Sensor data too old: {delay:.3f}s')
                # Use last known good data or apply safe behavior
                return

        # Process AI inference
        action = self.infer_action()

        # Check inference time
        inference_time = (self.get_clock().now() - current_time).nanoseconds / 1e9
        if inference_time > 0.01:  # More than 10ms for inference
            self.get_logger().warn(f'Inference took too long: {inference_time:.3f}s')

        if action is not None:
            self.publish_action(action)
```

### Safety and Validation

```python
def validate_action(self, action):
    """Validate AI actions before execution"""
    # Check for NaN or Inf values
    if np.any(np.isnan(action)) or np.any(np.isinf(action)):
        self.get_logger().error('Invalid action values detected')
        return False

    # Check action bounds
    max_linear_vel = 1.0  # m/s
    max_angular_vel = 1.0  # rad/s

    if abs(action.linear.x) > max_linear_vel:
        action.linear.x = np.clip(action.linear.x, -max_linear_vel, max_linear_vel)
        self.get_logger().warn(f'Linear velocity clipped: {action.linear.x}')

    if abs(action.angular.z) > max_angular_vel:
        action.angular.z = np.clip(action.angular.z, -max_angular_vel, max_angular_vel)
        self.get_logger().warn(f'Angular velocity clipped: {action.angular.z}')

    return True
```

## Performance Optimization

### Using ROS 2 Components (Composition)

For better performance, consider using component-based architecture:

```python
# In C++ component
#include "rclcpp/rclcpp.hpp"
#include "pluginlib/class_list_macros.hpp"

class AIBridgeComponent : public rclcpp::Node
{
public:
  explicit AIBridgeComponent(const rclcpp::NodeOptions & options)
  : Node("ai_bridge_component", options)
  {
    // Initialize with lower latency parameters
    this->declare_parameter("qos_history", 1);
    this->declare_parameter("qos_depth", 1);
    this->declare_parameter("qos_reliability", "best_effort");
  }
};
```

## Best Practices for AI-ROS Integration

1. **Modular Design**: Keep AI logic separate from ROS communication
2. **Error Handling**: Implement robust error handling for both AI and ROS components
3. **Timing**: Consider real-time requirements for your application
4. **Safety**: Always validate AI outputs before sending to hardware
5. **Monitoring**: Log AI decisions and robot states for debugging
6. **Scalability**: Design for multiple agents if needed
7. **Testing**: Test with simulation before hardware deployment

## Summary

In this chapter, you learned:
- How to create bridges between Python AI agents and ROS 2 controllers
- Different approaches for integrating deep learning and reinforcement learning with ROS
- Important considerations for real-world deployment (latency, safety, validation)
- Performance optimization techniques for AI-ROS integration
- Best practices for creating robust AI-ROS bridges
- How to handle timing and synchronization between AI and control systems

This concludes Module 1: The Robotic Nervous System. You now have the foundation to create intelligent humanoid robots that can perceive, reason, and act in the physical world.

---
**Continue to [Module 2: The Digital Twin](/docs/module-2-digital-twin/chapter-1-physics-simulation)**