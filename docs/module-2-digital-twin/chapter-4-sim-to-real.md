---
sidebar_label: 'Chapter 4: Sim-to-Real Transfer Techniques'
title: 'Chapter 4: Sim-to-Real Transfer Techniques'
description: 'Understanding sim-to-real transfer techniques for humanoid robotics applications'
slug: '/module-2-digital-twin/chapter-4-sim-to-real'
difficulty: 'advanced'
requiredHardware: ['computer', 'robot_hardware']
recommendedHardware: ['nvidia_isaac', 'ros_system']
---

# Chapter 4: Sim-to-Real Transfer Techniques

Sim-to-real transfer, also known as domain randomization or domain adaptation, is the process of transferring knowledge, models, or behaviors learned in simulation to real-world robotic systems. This is a critical challenge in humanoid robotics, where the complexity of the robot and environment make real-world training expensive, time-consuming, and potentially dangerous. This chapter explores various techniques to bridge the gap between simulation and reality.

## Understanding the Reality Gap

The "reality gap" refers to the differences between simulated and real environments that can cause policies or models trained in simulation to fail when deployed on real robots:

### Sources of the Reality Gap

1. **Visual Differences**: Lighting, textures, colors, and visual artifacts
2. **Physical Differences**: Mass, friction, damping, and material properties
3. **Sensor Differences**: Noise patterns, calibration errors, and sensor limitations
4. **Actuator Differences**: Motor dynamics, delays, and control precision
5. **Environmental Differences**: Surface properties, obstacles, and external disturbances

### Example Reality Gap Scenarios

```python
# Example: Simple grasping task showing reality gap
# In Simulation: Perfect grasp success rate = 95%
# In Reality: Grasp success rate = 30%

# Simulation parameters (perfect)
SIM_FRICTION = 1.0
SIM_NOISE = 0.0
SIM_CAMERA_CALIBRATION = "perfect"

# Reality parameters (imperfect)
REAL_FRICTION = 0.6  # Lower due to surface conditions
REAL_NOISE = 0.05    # Sensor noise
REAL_CAMERA_CALIBRATION = "with_errors"  # Calibration errors
```

## Domain Randomization

Domain randomization is a technique that randomizes simulation parameters during training to make models more robust to domain shift:

### Visual Domain Randomization

```python
# Visual Domain Randomization Example
import numpy as np
import cv2
import random

class VisualDomainRandomizer:
    def __init__(self):
        self.lighting_range = (0.5, 2.0)  # Lighting intensity range
        self.color_range = (0.8, 1.2)     # Color variation range
        self.texture_range = (0.1, 0.9)   # Texture variation range
        self.blur_range = (0, 3)          # Blur kernel size range
        self.noise_range = (0, 0.05)      # Noise intensity range

    def randomize_image(self, image):
        """Apply random visual transformations to image"""
        # Random lighting adjustment
        lighting_factor = random.uniform(*self.lighting_range)
        image = np.clip(image * lighting_factor, 0, 255).astype(np.uint8)

        # Random color variation
        color_factor = np.random.uniform(*self.color_range, size=3)
        image = np.clip(image * color_factor, 0, 255).astype(np.uint8)

        # Random blur
        blur_size = random.randint(*self.blur_range)
        if blur_size > 0:
            image = cv2.GaussianBlur(image, (blur_size*2+1, blur_size*2+1), 0)

        # Random noise
        noise_intensity = random.uniform(*self.noise_range)
        noise = np.random.normal(0, noise_intensity, image.shape) * 255
        image = np.clip(image + noise, 0, 255).astype(np.uint8)

        return image

# Usage in simulation training
visual_randomizer = VisualDomainRandomizer()

def simulate_training_step():
    # Get image from simulation
    sim_image = get_simulation_image()

    # Apply randomization
    randomized_image = visual_randomizer.randomize_image(sim_image)

    # Train model on randomized image
    train_model(randomized_image)
```

### Physical Domain Randomization

```python
# Physical Domain Randomization Example
import numpy as np
import random

class PhysicalDomainRandomizer:
    def __init__(self):
        # Robot properties
        self.mass_range = (0.8, 1.2)           # Mass variation (±20%)
        self.friction_range = (0.5, 1.5)       # Friction variation
        self.damping_range = (0.8, 1.2)        # Damping variation
        self.joint_friction_range = (0.0, 0.1) # Joint friction range

        # Environmental properties
        self.gravity_range = (9.7, 9.9)        # Gravity variation
        self.surface_friction_range = (0.3, 1.0) # Surface friction

        # Actuator properties
        self.motor_delay_range = (0.0, 0.02)   # Motor command delay (0-20ms)
        self.motor_noise_range = (0.0, 0.01)   # Motor command noise

    def randomize_robot_properties(self, robot):
        """Randomize robot physical properties"""
        # Randomize link masses
        for link in robot.links:
            original_mass = link.mass
            random_factor = random.uniform(*self.mass_range)
            link.mass = original_mass * random_factor

        # Randomize friction coefficients
        for joint in robot.joints:
            original_friction = joint.friction
            random_factor = random.uniform(*self.friction_range)
            joint.friction = original_friction * random_factor

            # Add joint-specific friction
            joint.joint_friction = random.uniform(*self.joint_friction_range)

    def get_random_environment_params(self):
        """Get randomized environment parameters"""
        return {
            'gravity': random.uniform(*self.gravity_range),
            'surface_friction': random.uniform(*self.surface_friction_range),
            'motor_delay': random.uniform(*self.motor_delay_range),
            'motor_noise': random.uniform(*self.motor_noise_range)
        }

# Usage in physics simulation
physical_randomizer = PhysicalDomainRandomizer()

def simulate_with_randomization():
    # Randomize robot properties
    physical_randomizer.randomize_robot_properties(robot)

    # Get randomized environment parameters
    env_params = physical_randomizer.get_random_environment_params()

    # Update simulation with random parameters
    update_physics_engine(
        gravity=env_params['gravity'],
        surface_friction=env_params['surface_friction']
    )

    # Simulate with motor delays and noise
    apply_motor_commands_with_delay_and_noise(
        commands=robot_commands,
        delay=env_params['motor_delay'],
        noise=env_params['motor_noise']
    )
```

## System Identification and Parameter Estimation

Accurate system identification helps reduce the reality gap by estimating real-world parameters:

### Parameter Estimation Example

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Header
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp

class SystemIdentificationNode(Node):
    def __init__(self):
        super().__init__('system_identification_node')

        # Subscribers
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10)

        # Publishers
        self.excitation_pub = self.create_publisher(
            Float64MultiArray, '/excitation_commands', 10)

        # Data storage
        self.joint_positions = []
        self.joint_velocities = []
        self.joint_efforts = []
        self.excitation_commands = []

        # Identification parameters
        self.excitation_signal = None
        self.identification_active = False

        # Timer for excitation
        self.timer = self.create_timer(0.01, self.excitation_timer)

        self.get_logger().info('System Identification Node initialized')

    def joint_callback(self, msg):
        """Store joint state data for identification"""
        if self.identification_active:
            self.joint_positions.append(list(msg.position))
            self.joint_velocities.append(list(msg.velocity))
            self.joint_efforts.append(list(msg.effort))

    def excitation_timer(self):
        """Generate and publish excitation signals"""
        if not self.identification_active:
            return

        # Generate random excitation signal
        excitation = Float64MultiArray()
        excitation.data = [np.random.uniform(-1.0, 1.0) for _ in range(12)]  # 12 joints example
        excitation.header = Header()
        excitation.header.stamp = self.get_clock().now().to_msg()

        self.excitation_commands.append(excitation.data)
        self.excitation_pub.publish(excitation)

    def estimate_parameters(self):
        """Estimate robot parameters using collected data"""
        # Convert stored data to numpy arrays
        pos_data = np.array(self.joint_positions)
        vel_data = np.array(self.joint_velocities)
        eff_data = np.array(self.joint_efforts)
        cmd_data = np.array(self.excitation_commands)

        # Define objective function for parameter estimation
        def objective_function(params):
            # params: [mass1, mass2, ..., friction1, friction2, ...]
            total_error = 0.0

            for i in range(len(pos_data)-1):
                # Simulate using current parameters
                simulated_effort = self.simulate_dynamics(
                    pos_data[i], vel_data[i], cmd_data[i], params)

                # Calculate error
                error = np.sum((eff_data[i] - simulated_effort)**2)
                total_error += error

            return total_error

        # Initial parameter guess
        initial_params = np.ones(24)  # Example: 12 masses + 12 frictions

        # Optimize parameters
        result = minimize(objective_function, initial_params, method='BFGS')

        return result.x

    def simulate_dynamics(self, positions, velocities, commands, params):
        """Simulate robot dynamics with given parameters"""
        # Simplified dynamics model
        # In practice, this would be a more complex physics simulation

        masses = params[:12]  # First 12 parameters are masses
        frictions = params[12:]  # Next 12 parameters are frictions

        # Calculate torques based on simplified model
        torques = commands - frictions * velocities

        # Calculate accelerations
        accelerations = torques / masses

        return torques

def main(args=None):
    rclpy.init(args=args)
    ident_node = SystemIdentificationNode()

    # Start identification process after some time
    def start_identification():
        ident_node.identification_active = True
        ident_node.get_logger().info('Starting system identification...')

    ident_node.create_timer(5.0, start_identification)  # Start after 5 seconds

    try:
        rclpy.spin(ident_node)
    except KeyboardInterrupt:
        # Perform parameter estimation
        params = ident_node.estimate_parameters()
        ident_node.get_logger().info(f'Estimated parameters: {params}')
    finally:
        ident_node.destroy_node()
        rclpy.shutdown()
```

## Sim-to-Real Transfer for Deep Learning

### Domain Adaptation Techniques

```python
# Domain Adaptation for Deep Learning Models
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

class DomainAdaptationNetwork(nn.Module):
    def __init__(self, num_classes=10):
        super(DomainAdaptationNetwork, self).__init__()

        # Feature extractor (shared between domains)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU()
        )

        # Label classifier (for source domain)
        self.label_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

        # Domain classifier (to distinguish source from target)
        self.domain_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),  # 2 domains: source and target
            nn.Softmax(dim=1)
        )

    def forward(self, x, lambda_val=0.0):
        features = self.feature_extractor(x)

        # Label prediction
        label_output = self.label_classifier(features)

        # Domain prediction (with gradient reversal)
        reversed_features = GradientReversal.apply(features, lambda_val)
        domain_output = self.domain_classifier(reversed_features)

        return label_output, domain_output

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, lambda_val):
        ctx.lambda_val = lambda_val
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_val, None

# Training loop with domain adaptation
def train_with_domain_adaptation(model, source_loader, target_loader, epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for (source_data, source_labels), (target_data, _) in zip(source_loader, target_loader):
            # Set lambda for gradient reversal
            lambda_val = 1.0 - epoch / epochs  # Decrease lambda over time

            # Source domain training
            source_labels_pred, source_domain_pred = model(source_data, lambda_val)
            source_label_loss = criterion(source_labels_pred, source_labels)
            source_domain_loss = criterion(source_domain_pred, torch.zeros(len(source_data)).long())

            # Target domain training
            _, target_domain_pred = model(target_data, lambda_val)
            target_domain_loss = criterion(target_domain_pred, torch.ones(len(target_data)).long())

            # Total loss
            total_loss = source_label_loss + source_domain_loss + target_domain_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
```

## Robust Control for Sim-to-Real Transfer

### Robust Control Techniques

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray
import numpy as np
from scipy.linalg import solve_continuous_are
from control import lqr

class RobustControlNode(Node):
    def __init__(self):
        super().__init__('robust_control_node')

        # Subscribers
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10)

        # Publishers
        self.cmd_pub = self.create_publisher(
            Float64MultiArray, '/joint_commands', 10)

        # Control parameters
        self.joint_positions = np.zeros(12)
        self.joint_velocities = np.zeros(12)
        self.target_positions = np.zeros(12)

        # Robust control matrices
        self.A = None  # System matrix
        self.B = None  # Input matrix
        self.Q = None  # State cost matrix
        self.R = None  # Input cost matrix

        # Initialize control matrices
        self.initialize_control_matrices()

        # Timer for control loop
        self.control_timer = self.create_timer(0.01, self.control_loop)

        self.get_logger().info('Robust Control Node initialized')

    def initialize_control_matrices(self):
        """Initialize control matrices for LQR design"""
        n = 24  # 12 positions + 12 velocities
        m = 12  # 12 joint torques

        # System matrices (simplified linearized model)
        self.A = np.zeros((n, n))
        self.B = np.zeros((n, m))

        # State: [positions, velocities]
        # A matrix: [0 I; -M^(-1)K -M^(-1)D]
        # Where M is mass matrix, K is stiffness, D is damping

        # For each joint: position and velocity
        for i in range(12):
            # Position equation: d/dt(pos) = vel
            self.A[i, i + 12] = 1.0  # Position derivative = velocity

            # Velocity equation: d/dt(vel) = -K/M * pos - D/M * vel + 1/M * tau
            stiffness = 100.0  # Spring constant
            damping = 10.0     # Damping coefficient
            mass = 1.0         # Mass (simplified)

            self.A[i + 12, i] = -stiffness / mass  # Spring term
            self.A[i + 12, i + 12] = -damping / mass  # Damping term

            # Input matrix: tau affects velocity directly
            self.B[i + 12, i] = 1.0 / mass

        # Cost matrices
        self.Q = np.eye(n) * 10.0  # High cost for position/velocity errors
        self.R = np.eye(m) * 0.1   # Low cost for control effort

    def joint_callback(self, msg):
        """Update joint state"""
        self.joint_positions = np.array(msg.position)
        self.joint_velocities = np.array(msg.velocity)

    def compute_lqr_control(self):
        """Compute LQR control law"""
        try:
            # Solve Riccati equation for LQR
            P = solve_continuous_are(self.A, self.B, self.Q, self.R)

            # Compute gain matrix: K = R^(-1) * B^T * P
            K = np.linalg.solve(self.R, self.B.T @ P)

            return K
        except np.linalg.LinAlgError:
            self.get_logger().warn('LQR failed, using PD control instead')
            return None

    def robust_control_loop(self):
        """Main control loop with robust control"""
        # Current state: [positions, velocities]
        x = np.concatenate([self.joint_positions, self.joint_velocities])

        # Target state: [target_positions, zeros]
        x_target = np.concatenate([self.target_positions, np.zeros(12)])

        # Error state
        x_error = x_target - x

        # Compute LQR gain
        K = self.compute_lqr_control()

        if K is not None:
            # Apply LQR control: u = K * (x_target - x)
            torques = K @ x_error
        else:
            # Fallback to PD control
            kp = 100.0
            kd = 10.0
            pos_error = self.target_positions - self.joint_positions
            vel_error = -self.joint_velocities
            torques = kp * pos_error + kd * vel_error

        # Add robustness: saturation and filtering
        torques = np.clip(torques, -100.0, 100.0)  # Torque limits

        # Publish commands
        cmd_msg = Float64MultiArray()
        cmd_msg.data = torques.tolist()
        cmd_msg.header.stamp = self.get_clock().now().to_msg()

        self.cmd_pub.publish(cmd_msg)

    def control_loop(self):
        """Main control loop"""
        self.robust_control_loop()

def main(args=None):
    rclpy.init(args=args)
    control_node = RobustControlNode()

    try:
        rclpy.spin(control_node)
    except KeyboardInterrupt:
        pass
    finally:
        control_node.destroy_node()
        rclpy.shutdown()
```

## Sensor Fusion for Robust Perception

### Multi-Sensor Integration

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, LaserScan, Image, PointCloud2
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import numpy as np
from scipy.spatial.transform import Rotation as R
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise

class SensorFusionEKF:
    def __init__(self):
        # State: [x, y, z, vx, vy, vz, qx, qy, qz, qw]
        self.dt = 0.01  # Time step
        self.ekf = ExtendedKalmanFilter(dim_x=10, dim_z=7)  # 10 state vars, 7 measurement vars

        # Initialize state and covariance
        self.ekf.x = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])  # [pos, vel, quat]
        self.ekf.P = np.eye(10) * 100.0  # High initial uncertainty

        # Process noise
        self.ekf.Q = np.eye(10) * 0.1

        # Measurement noise
        self.ekf.R = np.eye(7) * 0.1  # Position + orientation

    def state_transition_function(self, x, dt):
        """State transition function for prediction"""
        # Extract state variables
        pos = x[0:3]
        vel = x[3:6]
        quat = x[6:10] / np.linalg.norm(x[6:10])  # Normalize quaternion

        # Predict new position (constant velocity model)
        new_pos = pos + vel * dt

        # Predict new velocity (with some decay)
        new_vel = vel * 0.99  # Simple damping

        # For quaternion, assume no rotation for simplicity
        # In practice, you'd integrate angular velocity
        new_quat = quat

        return np.concatenate([new_pos, new_vel, new_quat])

    def measurement_function(self, x):
        """Measurement function - extract position and orientation"""
        # Return [x, y, z, qx, qy, qz, qw]
        return x[[0, 1, 2, 6, 7, 8, 9]]

    def update(self, measurement):
        """Update the filter with a measurement"""
        # Prediction
        self.ekf.predict()

        # Update
        self.ekf.update(measurement, hx=self.measurement_function,
                       fx=self.state_transition_function)

class MultiSensorFusionNode(Node):
    def __init__(self):
        super().__init__('multi_sensor_fusion_node')

        # Subscribers
        self.imu_sub = self.create_subscription(Imu, '/imu/data', self.imu_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # Publisher
        self.fused_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/fused_pose', 10)

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Sensor fusion EKF
        self.fusion_ekf = SensorFusionEKF()

        # Data storage
        self.imu_data = None
        self.odom_data = None
        self.last_update_time = self.get_clock().now()

        # Timer for publishing fused data
        self.timer = self.create_timer(0.02, self.publish_fused_data)  # 50Hz

        self.get_logger().info('Multi-Sensor Fusion Node initialized')

    def imu_callback(self, msg):
        """Process IMU data"""
        self.imu_data = msg

    def scan_callback(self, msg):
        """Process laser scan for position estimation"""
        # This is a simplified example
        # In practice, you'd use more sophisticated scan matching
        pass

    def odom_callback(self, msg):
        """Process odometry data"""
        self.odom_data = msg

    def publish_fused_data(self):
        """Publish fused sensor data"""
        if self.odom_data is not None and self.imu_data is not None:
            # Create measurement vector [x, y, z, qx, qy, qz, qw]
            pos = np.array([
                self.odom_data.pose.pose.position.x,
                self.odom_data.pose.pose.position.y,
                self.odom_data.pose.pose.position.z
            ])

            quat = np.array([
                self.imu_data.orientation.x,
                self.imu_data.orientation.y,
                self.imu_data.orientation.z,
                self.imu_data.orientation.w
            ])

            # Normalize quaternion
            quat = quat / np.linalg.norm(quat)

            measurement = np.concatenate([pos, quat])

            # Update EKF
            self.fusion_ekf.update(measurement)

            # Publish fused pose
            pose_msg = PoseWithCovarianceStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = 'map'

            # Set position
            pose_msg.pose.pose.position.x = float(self.fusion_ekf.ekf.x[0])
            pose_msg.pose.pose.position.y = float(self.fusion_ekf.ekf.x[1])
            pose_msg.pose.pose.position.z = float(self.fusion_ekf.ekf.x[2])

            # Set orientation
            pose_msg.pose.pose.orientation.x = float(self.fusion_ekf.ekf.x[6])
            pose_msg.pose.pose.orientation.y = float(self.fusion_ekf.ekf.x[7])
            pose_msg.pose.pose.orientation.z = float(self.fusion_ekf.ekf.x[8])
            pose_msg.pose.pose.orientation.w = float(self.fusion_ekf.ekf.x[9])

            # Set covariance
            pose_msg.pose.covariance = self.fusion_ekf.ekf.P.flatten().tolist()

            self.fused_pose_pub.publish(pose_msg)

            # Broadcast transform
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = 'map'
            t.child_frame_id = 'base_link'

            t.transform.translation.x = float(self.fusion_ekf.ekf.x[0])
            t.transform.translation.y = float(self.fusion_ekf.ekf.x[1])
            t.transform.translation.z = float(self.fusion_ekf.ekf.x[2])

            t.transform.rotation.x = float(self.fusion_ekf.ekf.x[6])
            t.transform.rotation.y = float(self.fusion_ekf.ekf.x[7])
            t.transform.rotation.z = float(self.fusion_ekf.ekf.x[8])
            t.transform.rotation.w = float(self.fusion_ekf.ekf.x[9])

            self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    fusion_node = MultiSensorFusionNode()

    try:
        rclpy.spin(fusion_node)
    except KeyboardInterrupt:
        pass
    finally:
        fusion_node.destroy_node()
        rclpy.shutdown()
```

## Transfer Learning Techniques

### Fine-tuning Pre-trained Models

```python
# Transfer Learning for Robot Perception
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

class RobotPerceptionTransfer(nn.Module):
    def __init__(self, num_robot_tasks=5, pretrained=True):
        super(RobotPerceptionTransfer, self).__init__()

        # Load pre-trained ResNet
        self.backbone = models.resnet50(pretrained=pretrained)

        # Remove the final classification layer
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Task-specific heads for different robot tasks
        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.feature_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, task_size)
            ) for task_size in [4, 10, 6, 3, 2]  # Example task sizes
        ])

        # Domain adaptation layer
        self.domain_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # Source vs Target
        )

    def forward(self, x, task_id=None, domain_adapt=False):
        # Extract features
        features = self.backbone(x)

        if domain_adapt:
            # For domain adaptation training
            domain_output = self.domain_classifier(features)
            return features, domain_output

        if task_id is not None:
            # Task-specific output
            task_output = self.task_heads[task_id](features)
            return task_output
        else:
            # Return features for general use
            return features

# Training with domain adaptation
def train_with_transfer_learning(model, source_loader, target_loader, epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Separate optimizers for different parts
    backbone_optimizer = torch.optim.Adam(model.backbone.parameters(), lr=0.0001)
    task_head_optimizer = torch.optim.Adam(model.task_heads.parameters(), lr=0.001)
    domain_optimizer = torch.optim.Adam(model.domain_classifier.parameters(), lr=0.001)

    criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for (source_data, source_labels), (target_data, _) in zip(source_loader, target_loader):
            source_data, source_labels = source_data.to(device), source_labels.to(device)
            target_data = target_data.to(device)

            # Source domain training
            source_features, source_domain_pred = model(source_data, domain_adapt=True)
            source_task_pred = model(source_data, task_id=0)  # Example task

            source_task_loss = criterion(source_task_pred, source_labels)
            source_domain_loss = domain_criterion(
                source_domain_pred, torch.zeros(len(source_data)).long().to(device))

            # Target domain training (domain adaptation)
            target_features, target_domain_pred = model(target_data, domain_adapt=True)
            target_domain_loss = domain_criterion(
                target_domain_pred, torch.ones(len(target_data)).long().to(device))

            # Domain confusion loss (make domains indistinguishable)
            domain_confusion_loss = -source_domain_loss - target_domain_loss

            # Total losses
            total_task_loss = source_task_loss
            total_domain_loss = source_domain_loss + target_domain_loss

            # Update backbone with domain confusion (reverse gradient)
            backbone_loss = total_task_loss + 0.1 * domain_confusion_loss
            backbone_optimizer.zero_grad()
            backbone_loss.backward(retain_graph=True)
            backbone_optimizer.step()

            # Update task heads
            task_head_optimizer.zero_grad()
            total_task_loss.backward(retain_graph=True)
            task_head_optimizer.step()

            # Update domain classifier
            domain_optimizer.zero_grad()
            total_domain_loss.backward()
            domain_optimizer.step()
```

## Validation and Testing Strategies

### Sim-to-Real Validation Framework

```python
# Sim-to-Real Validation Framework
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from sklearn.metrics import accuracy_score, precision_score, recall_score

class SimRealValidator:
    def __init__(self):
        self.sim_metrics = []
        self.real_metrics = []
        self.transfer_gaps = []

    def validate_policy_transfer(self, sim_policy, real_robot, num_trials=10):
        """Validate policy transfer from simulation to reality"""
        sim_success_rates = []
        real_success_rates = []

        for trial in range(num_trials):
            # Test in simulation
            sim_success = self.test_policy_in_simulation(sim_policy)
            sim_success_rates.append(sim_success)

            # Test on real robot
            real_success = self.test_policy_on_real_robot(real_robot)
            real_success_rates.append(real_success)

        # Calculate transfer gap
        avg_sim_success = np.mean(sim_success_rates)
        avg_real_success = np.mean(real_success_rates)
        transfer_gap = avg_sim_success - avg_real_success

        return {
            'sim_success_rate': avg_sim_success,
            'real_success_rate': avg_real_success,
            'transfer_gap': transfer_gap,
            'sim_std': np.std(sim_success_rates),
            'real_std': np.std(real_success_rates)
        }

    def validate_perception_transfer(self, sim_model, real_model, test_data):
        """Validate perception model transfer"""
        # Test on simulation-like data
        sim_predictions = sim_model.predict(test_data['sim'])
        sim_accuracy = accuracy_score(test_data['sim_labels'], sim_predictions)

        # Test on real-world data
        real_predictions = real_model.predict(test_data['real'])
        real_accuracy = accuracy_score(test_data['real_labels'], real_predictions)

        return {
            'sim_accuracy': sim_accuracy,
            'real_accuracy': real_accuracy,
            'accuracy_gap': sim_accuracy - real_accuracy
        }

    def validate_control_transfer(self, sim_controller, real_controller, trajectories):
        """Validate control policy transfer"""
        sim_errors = []
        real_errors = []

        for trajectory in trajectories:
            # Simulate trajectory
            sim_pos = self.execute_trajectory(sim_controller, trajectory['waypoints'])
            sim_error = self.calculate_tracking_error(
                trajectory['reference'], sim_pos)
            sim_errors.append(sim_error)

            # Execute on real robot
            real_pos = self.execute_trajectory(real_controller, trajectory['waypoints'])
            real_error = self.calculate_tracking_error(
                trajectory['reference'], real_pos)
            real_errors.append(real_error)

        return {
            'sim_error_mean': np.mean(sim_errors),
            'real_error_mean': np.mean(real_errors),
            'error_gap': np.mean(real_errors) - np.mean(sim_errors)
        }

    def calculate_tracking_error(self, reference, actual):
        """Calculate tracking error between reference and actual trajectories"""
        if len(reference) != len(actual):
            # Interpolate to same length
            min_len = min(len(reference), len(actual))
            ref_interp = np.array(reference[:min_len])
            act_interp = np.array(actual[:min_len])
        else:
            ref_interp = np.array(reference)
            act_interp = np.array(actual)

        errors = [euclidean(ref_interp[i], act_interp[i]) for i in range(len(ref_interp))]
        return np.mean(errors)

# Usage example
validator = SimRealValidator()

def run_validation_pipeline():
    """Run complete validation pipeline"""
    results = {}

    # Validate policy transfer
    policy_results = validator.validate_policy_transfer(
        sim_policy=None,  # Your simulation policy
        real_robot=None,  # Your real robot interface
        num_trials=10
    )
    results['policy'] = policy_results

    # Validate perception transfer
    perception_results = validator.validate_perception_transfer(
        sim_model=None,  # Your simulation model
        real_model=None,  # Your real-world model
        test_data=None   # Your test data
    )
    results['perception'] = perception_results

    # Validate control transfer
    control_results = validator.validate_control_transfer(
        sim_controller=None,  # Your simulation controller
        real_controller=None,  # Your real controller
        trajectories=None     # Your test trajectories
    )
    results['control'] = control_results

    # Print summary
    print("Sim-to-Real Transfer Validation Results:")
    print(f"Policy Success - Sim: {results['policy']['sim_success_rate']:.3f}, "
          f"Real: {results['policy']['real_success_rate']:.3f}, "
          f"Gap: {results['policy']['transfer_gap']:.3f}")

    print(f"Perception Accuracy - Sim: {results['perception']['sim_accuracy']:.3f}, "
          f"Real: {results['perception']['real_accuracy']:.3f}, "
          f"Gap: {results['perception']['accuracy_gap']:.3f}")

    print(f"Control Error - Sim: {results['control']['sim_error_mean']:.3f}, "
          f"Real: {results['control']['real_error_mean']:.3f}, "
          f"Gap: {results['control']['error_gap']:.3f}")

    return results
```

## Best Practices for Sim-to-Real Transfer

1. **Gradual Complexity**: Start with simple tasks and gradually increase complexity
2. **Systematic Randomization**: Randomize parameters systematically rather than randomly
3. **Real Data Integration**: Include some real data in training when possible
4. **Validation Framework**: Implement comprehensive validation to measure transfer success
5. **Iterative Refinement**: Continuously refine simulation based on real-world performance
6. **Modular Design**: Keep simulation and real components modular for easy switching
7. **Safety First**: Always implement safety measures when transferring to real robots

## Troubleshooting Transfer Issues

Common issues and solutions:

1. **Large Performance Drop**: Increase domain randomization or collect more real data
2. **Unstable Behavior**: Implement more conservative control strategies
3. **Sensor Mismatch**: Calibrate sensors and validate data quality
4. **Actuator Differences**: Implement actuator models that match real hardware
5. **Timing Issues**: Ensure consistent timing between simulation and reality

## Summary

In this chapter, you learned:
- The challenges of the reality gap in sim-to-real transfer
- Domain randomization techniques to improve robustness
- System identification methods for accurate modeling
- Deep learning approaches for domain adaptation
- Robust control techniques for reliable performance
- Sensor fusion strategies for better perception
- Validation frameworks to measure transfer success
- Best practices for successful sim-to-real transfer

Sim-to-real transfer is crucial for making simulation-based development practical for real humanoid robots. By understanding and applying these techniques, you can bridge the gap between virtual and physical worlds effectively.

---
**Continue to [Module 3: NVIDIA Isaac](/docs/module-3-nvidia-isaac/chapter-1-isaac-sim)**