---
sidebar_label: 'Chapter 3: Services, Actions, and Parameters'
title: 'Chapter 3: Services, Actions, and Parameters'
description: 'Understanding ROS 2 services, actions, and parameters for humanoid robotics applications'
slug: '/module-1-ros2/chapter-3-services-actions'
difficulty: 'intermediate'
requiredHardware: ['ros_system']
recommendedHardware: ['computer']
---

# Chapter 3: Services, Actions, and Parameters

In the previous chapter, we explored nodes and topics, which implement an asynchronous, many-to-many communication model. In this chapter, we'll cover the remaining communication patterns in ROS 2: services, actions, and parameters. These patterns are essential for building robust humanoid robotics applications.

## Services: Request-Reply Communication

Services implement a synchronous, request-reply communication model. A service client sends a request to a service server, which processes the request and returns a response. This is ideal for operations that need a guaranteed response, such as:

- Calibration procedures
- Configuration changes
- Complex computations
- Diagnostic queries

### Creating a Service

First, let's define a service interface. Create a file called `SetPosition.srv`:

```
# Request
float64 x
float64 y
float64 z
---
# Response
bool success
string message
```

### Service Server

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from example_interfaces.srv import SetBool  # Using built-in service for example

class PositionService(Node):
    def __init__(self):
        super().__init__('position_service')
        self.srv = self.create_service(
            SetBool,  # In practice, use your custom service type
            'set_target_position',
            self.set_position_callback
        )

    def set_position_callback(self, request, response):
        self.get_logger().info(f'Setting position to: {request.data}')

        # Simulate position setting (replace with actual robot control)
        success = True
        message = f'Position set successfully to {request.data}'

        response.success = success
        response.message = message
        return response

def main(args=None):
    rclpy.init(args=args)
    position_service = PositionService()
    rclpy.spin(position_service)
    position_service.destroy_node()
    rclpy.shutdown()
```

### Service Client

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from example_interfaces.srv import SetBool  # Using built-in service for example

class PositionClient(Node):
    def __init__(self):
        super().__init__('position_client')
        self.cli = self.create_client(SetBool, 'set_target_position')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.req = SetBool.Request()

    def send_request(self, data):
        self.req.data = data
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main(args=None):
    rclpy.init(args=args)
    position_client = PositionClient()

    response = position_client.send_request(True)
    position_client.get_logger().info(f'Response: {response.success}, {response.message}')

    position_client.destroy_node()
    rclpy.shutdown()
```

## Actions: Long-Running Tasks with Feedback

Actions are designed for long-running tasks that require feedback and the ability to cancel. They're perfect for:

- Navigation to a goal
- Complex manipulation tasks
- Calibration processes
- Learning algorithms

An action has three parts:
- **Goal**: What the action should do
- **Feedback**: Progress updates during execution
- **Result**: Final outcome of the action

### Creating an Action Server

```python
#!/usr/bin/env python3
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from example_interfaces.action import Fibonacci  # Using built-in action for example

class FibonacciActionServer(Node):
    def __init__(self):
        super().__init__('fibonacci_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            self.execute_callback
        )

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')

        feedback_msg = Fibonacci.Feedback()
        feedback_msg.sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return Fibonacci.Result()

            feedback_msg.sequence.append(
                feedback_msg.sequence[i] + feedback_msg.sequence[i-1]
            )

            goal_handle.publish_feedback(feedback_msg)
            self.get_logger().info(f'Feedback: {feedback_msg.sequence}')

        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.sequence
        return result

def main(args=None):
    rclpy.init(args=args)
    fibonacci_action_server = FibonacciActionServer()
    rclpy.spin(fibonacci_action_server)
    fibonacci_action_server.destroy_node()
    rclpy.shutdown()
```

### Action Client

```python
#!/usr/bin/env python3
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from example_interfaces.action import Fibonacci  # Using built-in action for example

class FibonacciActionClient(Node):
    def __init__(self):
        super().__init__('fibonacci_action_client')
        self._action_client = ActionClient(
            self,
            Fibonacci,
            'fibonacci'
        )

    def send_goal(self, order):
        goal_msg = Fibonacci.Goal()
        goal_msg.order = order

        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )

        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Received feedback: {feedback.sequence}')

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result: {result.sequence}')

def main(args=None):
    rclpy.init(args=args)
    action_client = FibonacciActionClient()
    action_client.send_goal(10)

    rclpy.spin(action_client)
    action_client.destroy_node()
```

## Parameters: Configuration Management

Parameters allow nodes to be configured at runtime. They're perfect for:

- Tuning control gains
- Setting operational modes
- Configuring hardware-specific settings
- Adjusting algorithm parameters

### Using Parameters in a Node

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

class ParameterNode(Node):
    def __init__(self):
        super().__init__('parameter_node')

        # Declare parameters with default values
        self.declare_parameter('max_velocity', 1.0)
        self.declare_parameter('control_frequency', 50)
        self.declare_parameter('robot_name', 'humanoid_robot')

        # Get parameter values
        self.max_velocity = self.get_parameter('max_velocity').value
        self.control_frequency = self.get_parameter('control_frequency').value
        self.robot_name = self.get_parameter('robot_name').value

        self.get_logger().info(f'Robot: {self.robot_name}, Max velocity: {self.max_velocity}')

        # Set up parameter callback for runtime changes
        self.add_on_set_parameters_callback(self.parameter_callback)

    def parameter_callback(self, params):
        for param in params:
            if param.name == 'max_velocity' and param.type_ == Parameter.Type.DOUBLE:
                self.max_velocity = param.value
                self.get_logger().info(f'Updated max velocity to: {self.max_velocity}')
        return SetParametersResult(successful=True)

def main(args=None):
    rclpy.init(args=args)
    parameter_node = ParameterNode()
    rclpy.spin(parameter_node)
    parameter_node.destroy_node()
    rclpy.shutdown()
```

## Practical Example: Navigation System

Let's see how these communication patterns work together in a humanoid robot navigation system:

### Navigation Service (for immediate goals)
```python
# Service to set a navigation goal immediately
# Request: target coordinates
# Response: success/failure
```

### Navigation Action (for complex navigation tasks)
```python
# Action to navigate to a goal with feedback
# Goal: target coordinates
# Feedback: current progress, distance remaining
# Result: success/failure with final status
```

### Parameters (for navigation configuration)
```python
# Parameters for navigation:
# - maximum speed
# - safety distance
# - obstacle detection threshold
# - planning frequency
```

## Comparison of Communication Patterns

| Pattern | Type | Use Case | Characteristics |
|---------|------|----------|-----------------|
| Topics | Async Pub/Sub | Continuous data streams | Many-to-many, loose coupling |
| Services | Sync Request/Reply | One-time operations | One-to-one, guaranteed response |
| Actions | Async with Feedback | Long-running tasks | Goal-feedback-result, cancellable |
| Parameters | Configuration | Runtime configuration | Key-value pairs, type-safe |

## Best Practices

1. **Choose the right pattern**: Use topics for continuous data, services for quick operations, actions for long tasks
2. **Service design**: Keep services fast; if an operation takes more than a few seconds, consider using actions
3. **Action design**: Design clear feedback messages that provide meaningful progress updates
4. **Parameter validation**: Always validate parameter values to ensure safe operation
5. **Error handling**: Implement proper error handling for all communication patterns

## Summary

In this chapter, you learned:
- How services work for request-reply communication
- How actions handle long-running tasks with feedback
- How parameters manage runtime configuration
- When to use each communication pattern
- Practical examples of combining communication patterns
- Best practices for effective communication design

---
**Continue to [Chapter 4: URDF Robot Modeling](/docs/module-1-ros2/chapter-4-urdf-modeling)**