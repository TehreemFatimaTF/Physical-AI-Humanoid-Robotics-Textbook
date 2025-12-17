---
sidebar_label: 'Chapter 2: ROS 2 Nodes and Topics'
title: 'Chapter 2: ROS 2 Nodes and Topics'
description: 'Understanding ROS 2 nodes, topics, and the publisher-subscriber communication model for humanoid robotics'
slug: '/module-1-ros2/chapter-2-nodes-topics'
difficulty: 'intermediate'
requiredHardware: ['ros_system']
recommendedHardware: ['computer']
---

# Chapter 2: ROS 2 Nodes and Topics

In this chapter, we'll dive deep into the fundamental communication mechanisms in ROS 2: nodes and topics. These concepts are essential for creating distributed robotic systems where different components need to exchange information.

## Understanding Nodes

A **node** is an executable that uses ROS 2 to communicate with other nodes. Nodes are organized into packages, which are collections of related functionality. In a humanoid robot, you might have nodes for:

- Sensor data processing
- Motion control
- Path planning
- Computer vision
- Human-robot interaction

### Creating a Simple Node

Let's create a simple ROS 2 node in Python that publishes a message:

```python
#!/usr/bin/env python3
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
    main()
```

## Understanding Topics

A **topic** is a named bus over which nodes exchange messages. Topics implement a publish/subscribe communication model where publishers send messages to a topic and subscribers receive messages from a topic.

### Publisher-Subscriber Model

The publisher-subscriber model has several key characteristics:

- **Loose coupling**: Publishers and subscribers don't need to know about each other
- **Asynchronous**: Publishers and subscribers can run at different rates
- **Many-to-many**: Multiple publishers can publish to the same topic, and multiple subscribers can subscribe to the same topic

## Quality of Service (QoS)

ROS 2 introduces Quality of Service (QoS) settings that allow you to configure the behavior of your publishers and subscribers:

- **Reliability**: Best effort vs. reliable delivery
- **Durability**: Volatile vs. transient local (keep last message for new subscribers)
- **History**: Keep all messages vs. keep last N messages
- **Depth**: Size of the message queue

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# Create a QoS profile for real-time communication
qos_profile = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST
)
```

## Practical Example: Sensor Data Pipeline

Let's look at how nodes and topics work in a practical humanoid robot scenario:

### Sensor Node
```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import random

class SensorNode(Node):
    def __init__(self):
        super().__init__('sensor_node')
        self.publisher_ = self.create_publisher(JointState, 'joint_states', 10)
        timer_period = 0.1  # 10Hz
        self.timer = self.create_timer(timer_period, self.publish_sensor_data)

    def publish_sensor_data(self):
        msg = JointState()
        msg.name = ['hip_joint', 'knee_joint', 'ankle_joint']
        msg.position = [random.uniform(-1.0, 1.0) for _ in range(3)]
        msg.velocity = [random.uniform(-0.5, 0.5) for _ in range(3)]
        msg.effort = [random.uniform(0, 10) for _ in range(3)]

        self.publisher_.publish(msg)
        self.get_logger().info(f'Published joint states: {msg.position}')

def main(args=None):
    rclpy.init(args=args)
    sensor_node = SensorNode()
    rclpy.spin(sensor_node)
    sensor_node.destroy_node()
    rclpy.shutdown()
```

### Controller Node
```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

class ControllerNode(Node):
    def __init__(self):
        super().__init__('controller_node')
        self.subscription = self.create_subscription(
            JointState,
            'joint_states',
            self.listener_callback,
            10)
        self.publisher_ = self.create_publisher(Float64MultiArray, 'joint_commands', 10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        # Simple PD controller example
        commands = Float64MultiArray()
        target_positions = [0.0, 0.0, 0.0]  # Desired positions

        # Calculate simple control commands
        control_signals = []
        for i, current_pos in enumerate(msg.position):
            error = target_positions[i] - current_pos
            control_signals.append(error * 2.0)  # Simple proportional control

        commands.data = control_signals
        self.publisher_.publish(commands)
        self.get_logger().info(f'Published control commands: {control_signals}')

def main(args=None):
    rclpy.init(args=args)
    controller_node = ControllerNode()
    rclpy.spin(controller_node)
    controller_node.destroy_node()
    rclpy.shutdown()
```

## Launch Files

For managing multiple nodes, ROS 2 uses launch files. Here's an example launch file that starts both nodes:

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='robot_control',
            executable='sensor_node',
            name='sensor_node'
        ),
        Node(
            package='robot_control',
            executable='controller_node',
            name='controller_node'
        )
    ])
```

## Best Practices

1. **Node Design**: Keep nodes focused on a single responsibility
2. **Topic Naming**: Use descriptive, consistent naming conventions
3. **QoS Settings**: Choose appropriate QoS settings based on your application's requirements
4. **Error Handling**: Implement proper error handling in your nodes
5. **Logging**: Use ROS 2's logging system for debugging and monitoring

## Summary

In this chapter, you learned:
- How nodes and topics work in ROS 2
- The publisher-subscriber communication model
- How to create simple publisher and subscriber nodes
- Quality of Service settings and their importance
- Practical example of a sensor-control pipeline
- Best practices for node design

---
**Continue to [Chapter 3: Services, Actions, and Parameters](/docs/module-1-ros2/chapter-3-services-actions)**