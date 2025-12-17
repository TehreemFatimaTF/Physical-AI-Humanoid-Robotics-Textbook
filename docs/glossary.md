# Glossary

This glossary contains definitions for key terms used throughout the Physical AI & Humanoid Robotics textbook.

## A

**Actuator**: A mechanical device that converts energy (typically electrical) into physical motion. In robotics, actuators control joint movement and robot locomotion.

**Artificial Intelligence (AI)**: The simulation of human intelligence processes by machines, especially computer systems. In robotics, AI enables perception, decision-making, and learning capabilities.

## B

**Behavior Tree**: A hierarchical structure used in robotics and AI to define complex behaviors by composing simpler tasks in a tree structure.

## C

**Camera Calibration**: The process of determining the intrinsic and extrinsic parameters of a camera to accurately map 3D points to 2D image coordinates.

**Cartesian Space**: A three-dimensional coordinate system used to describe positions and orientations in 3D space, typically using X, Y, Z coordinates.

**Computer Vision**: A field of artificial intelligence that trains computers to interpret and understand the visual world, enabling robots to identify and analyze visual inputs.

**Control Loop**: A feedback mechanism in which a system continuously measures its output, compares it to a desired reference, and adjusts its input to minimize the error.

**Convolutional Neural Network (CNN)**: A class of deep neural networks commonly used in computer vision applications, designed to process pixel data.

## D

**Deep Learning**: A subset of machine learning based on artificial neural networks with representation learning, used in robotics for perception and control tasks.

**Depth Sensor**: A device that measures the distance to objects in a scene, providing 3D spatial information for navigation and manipulation.

**Digital Twin**: A virtual representation of a physical object or system that enables real-time monitoring, analysis, and simulation of the physical counterpart.

**Domain Adaptation**: Techniques used in machine learning to adapt models trained on one domain (e.g., simulation) to perform well on a different but related domain (e.g., real world).

**Domain Randomization**: A technique in robotics and computer vision that randomizes simulation parameters during training to improve the robustness of policies when transferred to the real world.

**Dynamics**: The study of forces and torques and their effect on motion, crucial for understanding robot movement and control.

## E

**Embodiment**: The physical instantiation of an AI system, giving it the ability to interact with the physical world through sensors and actuators.

**Encoder**: A sensor that measures the position or speed of a rotating shaft, commonly used in robot joints to provide feedback on joint angles.

**End Effector**: The device at the end of a robotic arm designed to interact with the environment, such as a gripper or tool.

**Environment Mapping**: The process of creating a representation of the robot's surroundings for navigation and planning purposes.

## F

**Field of View (FOV)**: The extent of the observable world that can be seen by a camera or sensor at any given moment.

**Force/Torque Sensor**: A device that measures the forces and torques applied to a robot's end effector or joints, enabling precise manipulation and interaction.

**Forward Kinematics**: The process of determining the position and orientation of the end effector based on the joint angles of a robot.

## G

**Gazebo**: A 3D simulation environment that provides realistic physics simulation and sensor models for robotics development.

**Gaussian Noise**: Statistical noise that follows a normal distribution, commonly used to model sensor noise in robotics simulations.

**Generalization**: The ability of a machine learning model to perform well on new, unseen data rather than just the training data.

**Global Navigation**: The process of planning a path from a start location to a goal location in a known or partially known environment.

## H

**Hardware-in-the-Loop (HIL)**: A testing methodology that involves connecting real hardware components to a simulation environment for validation and testing.

**Humanoid Robot**: A robot with a physical structure that resembles the human body, typically featuring a head, torso, two arms, and two legs.

## I

**Inverse Kinematics**: The process of determining the joint angles required to achieve a desired position and orientation of the end effector.

**Inertial Measurement Unit (IMU)**: A device that measures and reports a body's specific force, angular rate, and sometimes the magnetic field surrounding the body.

**Iterative Learning Control (ILC)**: A control scheme that improves system performance by learning from repeated executions of the same task.

## J

**Joint Space**: The space defined by the joint angles of a robot, as opposed to Cartesian space which is defined by position and orientation.

## K

**Kinematics**: The study of motion without considering the forces that cause it, crucial for understanding robot movement and control.

**Kalman Filter**: An algorithm that uses a series of measurements observed over time to estimate unknown variables, commonly used in sensor fusion.

## L

**LiDAR**: Light Detection and Ranging, a remote sensing method that uses light in the form of a pulsed laser to measure distances.

**Localization**: The process of determining the robot's position and orientation within a known or unknown environment.

**LQR (Linear Quadratic Regulator)**: An optimal control technique that computes control inputs to minimize a quadratic cost function.

## M

**Machine Learning**: A type of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed.

**Manipulation**: The ability of a robot to interact with objects in its environment using its end effectors.

**Mapping**: The process of creating a representation of the environment based on sensor data for navigation and planning.

**Monte Carlo Localization**: A probabilistic algorithm for robot localization that uses particle filters to represent the robot's belief about its position.

## N

**Navigation2 (Nav2)**: The next-generation navigation system for ROS 2, providing advanced path planning, obstacle avoidance, and navigation capabilities for mobile robots.

**NVIDIA Isaac**: NVIDIA's robotics platform that provides simulation, AI frameworks, and hardware acceleration for robotics development.

**NVIDIA Jetson**: A series of embedded computing platforms designed for AI and robotics applications at the edge, featuring GPU acceleration.

**NVIDIA Isaac Sim**: A robotics simulation application built on NVIDIA Omniverse, providing high-fidelity physics simulation, photorealistic rendering, and synthetic data generation capabilities.

**NVIDIA Isaac ROS**: A collection of hardware-accelerated ROS 2 packages that leverage the power of NVIDIA GPUs for robotics perception and navigation.

## O

**Occupancy Grid**: A probabilistic representation of the environment used in robotics for mapping and path planning.

**Odometry**: The use of data from motion sensors to estimate change in position over time, commonly used for robot navigation.

**OpenCV**: An open-source computer vision and machine learning software library used in robotics applications.

**NVIDIA Isaac**: NVIDIA's robotics platform that provides simulation, AI frameworks, and hardware acceleration for robotics development.

## O

**Occupancy Grid**: A probabilistic representation of the environment used in robotics for mapping and path planning.

**Odometry**: The use of data from motion sensors to estimate change in position over time, commonly used for robot navigation.

**OpenCV**: An open-source computer vision and machine learning software library used in robotics applications.

## P

**Path Planning**: The computational problem of finding a sequence of valid configurations that moves an object from a source to a destination.

**Perception**: The ability of a robot to interpret sensory information from its environment to understand its state and surroundings.

**PID Controller**: A control loop feedback mechanism widely used in robotics to maintain desired system behavior.

**Point Cloud**: A set of data points in space, typically representing the external surface of an object, often generated by 3D scanners or LiDAR.

**Proprioception**: The sense of the relative position of one's own parts of the body and strength of effort being employed in movement, particularly relevant in robotics for self-awareness.

## R

**Reinforcement Learning**: A type of machine learning where an agent learns to make decisions by performing actions and receiving rewards or penalties.

**Robot Operating System (ROS)**: A flexible framework for writing robot software that provides services designed for a heterogeneous computer cluster.

**ROS 2**: The second generation of the Robot Operating System with improved features for real-world deployment.

**Rigid Body Dynamics**: The simulation of the motion of solid objects under the influence of forces and torques.

**RGB-D Camera**: A camera that captures both color (RGB) and depth (D) information simultaneously.

**ROS Bridge**: A connection mechanism that allows communication between ROS-based systems and other software components.

## S

**Sensor Fusion**: The process of combining data from multiple sensors to achieve better accuracy and reliability than could be achieved by using a single sensor.

**Sim-to-Real Transfer**: The process of transferring knowledge, models, or behaviors learned in simulation to real-world robotic systems.

**Simulation**: The imitation of the operation of a real-world process or system over time, particularly important in robotics development.

**SLAM (Simultaneous Localization and Mapping)**: The computational problem of constructing or updating a map of an unknown environment while simultaneously keeping track of an agent's location within it.

**State Estimation**: The process of computing the state of a system from noisy and incomplete measurements.

**Stereo Vision**: A technique that extracts 3D information from 2D images using two or more cameras positioned at different angles.

**System Identification**: The process of developing mathematical models of dynamic systems from measured input-output data.

## T

**Teleoperation**: The remote operation of a robot by a human operator, often used for tasks in dangerous or inaccessible environments.

**Trajectory Planning**: The process of creating a path that a robot will follow with specified timing information.

**Twist Message**: A ROS message type that represents the velocity of a rigid body in 3D space, containing both linear and angular components.

## U

**URDF (Unified Robot Description Format)**: An XML format used in ROS to describe a robot's physical and visual properties.

**Unity**: A real-time 3D development platform that can be used for high-fidelity robotics simulation and visualization.

**Unity Robotics Simulation**: Using the Unity engine for creating realistic physics simulations and visualizations for robotics applications.

**Unstructured Environment**: An environment that lacks predetermined structure or organization, presenting challenges for robot navigation and manipulation.

## V

**VSLAM (Visual Simultaneous Localization and Mapping)**: SLAM using visual sensors such as cameras instead of LiDAR.

**Vision-Language-Action (VLA)**: Systems that integrate visual perception, natural language understanding, and robotic action capabilities.

**Virtual Reality (VR)**: A simulated experience that can be similar to or completely different from the real world, used for robotics training and visualization.

**Vision-Language-Action Systems**: Integrated systems that combine computer vision, natural language processing, and robotic action planning to enable robots to understand and execute complex tasks based on visual and linguistic input.

**Visual Grounding**: The task of connecting natural language expressions to their referents in images, enabling robots to identify objects or regions based on textual descriptions.

**Vision Transformers (ViT)**: Transformer architectures applied to computer vision tasks, using self-attention mechanisms to process visual data.

## W

**Whisper**: OpenAI's automatic speech recognition (ASR) system that converts spoken language into text, commonly used in robotics for voice command processing.

**Wheel Odometry**: The use of wheel encoders to estimate the distance traveled by a mobile robot, commonly used for navigation.

**World Model**: A representation of the environment that a robot uses for planning, navigation, and interaction, often incorporating spatial, semantic, and dynamic information.

**Workspace**: The space within which a robot can operate, defined by the physical constraints of the robot's kinematics.