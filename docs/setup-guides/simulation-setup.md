---
sidebar_label: 'Simulation Setup Guide'
title: 'Simulation Setup Guide'
description: 'Comprehensive guide to setting up Gazebo and Unity simulation environments for humanoid robotics'
slug: '/setup-guides/simulation-setup'
---

# Simulation Setup Guide

This guide provides comprehensive instructions for setting up simulation environments for humanoid robotics development, including both Gazebo and Unity setups.

## Gazebo Simulation Setup

Gazebo is a 3D simulation environment that provides realistic physics simulation and sensor models for robotics development.

### Prerequisites

Before installing Gazebo, ensure your system meets the following requirements:

- Ubuntu 20.04 or 22.04 (recommended for ROS 2 compatibility)
- At least 8GB RAM (16GB recommended for complex humanoid models)
- Dedicated GPU with OpenGL 3.3+ support
- At least 20GB free disk space
- ROS 2 Humble Hawksbill or later

### Installing Gazebo

For the latest version of Gazebo Garden (recommended for robotics research):

```bash
# Add Gazebo repository
sudo apt update && sudo apt install wget
wget https://packages.osrfoundation.org/gazebo.gpg -O /tmp/gazebo.gpg
sudo cp /tmp/gazebo.gpg /usr/share/keyrings/
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/gazebo.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null

# Install Gazebo Garden
sudo apt update
sudo apt install gz-garden
```

For ROS 2 integration, install the ROS 2 Gazebo packages:

```bash
# Source ROS 2
source /opt/ros/humble/setup.bash

# Install Gazebo ROS packages
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros2-control ros-humble-ros2-control ros-humble-ros2-controllers
```

### Setting up a Basic Gazebo World

Create a simple world file to test your installation:

```xml
<!-- ~/.gazebo/worlds/basic_humanoid.world -->
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="basic_humanoid">
    <!-- Include the sun -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Include the ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Physics engine configuration -->
    <physics name="1ms" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000.0</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
    </physics>

    <!-- Add your humanoid robot model here -->
    <!-- We'll cover robot models in the next section -->

  </world>
</sdf>
```

### Running Your First Simulation

```bash
# Launch Gazebo with the basic world
gz sim -r basic_humanoid.world
```

Or using ROS 2 launch:

```bash
# Create a launch file for simulation
# launch/simulation.launch.py

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    world = LaunchConfiguration('world')

    declare_world_cmd = DeclareLaunchArgument(
        'world',
        default_value='basic_humanoid.world',
        description='Choose one of the world files from `/path/to/worlds`'
    )

    # Gazebo server
    gzserver_cmd = Node(
        package='ros_gz_sim',
        executable='gzserver',
        arguments=[PathJoinSubstitution([FindPackageShare('your_robot_description'), 'worlds', world]), '-r'],
        output='screen'
    )

    # Gazebo client
    gzclient_cmd = Node(
        package='ros_gz_sim',
        executable='gzclient',
        output='screen'
    )

    ld = LaunchDescription()
    ld.add_action(declare_world_cmd)
    ld.add_action(gzserver_cmd)
    ld.add_action(gzclient_cmd)

    return ld
```

## Unity Robotics Simulation Setup

Unity provides high-fidelity rendering capabilities and is excellent for creating photorealistic simulation environments.

### Installing Unity Hub and Unity Editor

1. **Download Unity Hub**:
   - Visit https://unity.com/download
   - Download and install Unity Hub
   - Unity Hub manages multiple Unity versions and projects

2. **Install Unity Editor**:
   - Open Unity Hub
   - Go to "Installs" tab
   - Click "Add" to install a Unity version
   - Select version 2021.3 LTS or later for best robotics package support
   - Select modules:
     - Linux Build Support (if you plan to build for Linux)
     - Visual Studio Community (or your preferred IDE)

### Installing Unity Robotics Packages

Unity provides several packages specifically for robotics simulation:

1. **Open your Unity project**
2. **Access Package Manager**:
   - Go to Window → Package Manager
   - Click the "+" button → Add package from git URL
   - Add the following packages:

```bash
# ROS TCP Connector for Unity
com.unity.robotics.ros-tcp-connector

# URDF Importer for Unity
com.unity.robotics.urdf-importer

# Metrics Recorder (for data collection)
com.unity.perception
```

3. **Install NVIDIA Omniverse if using Isaac Sim**:
   - For advanced physics simulation and NVIDIA hardware acceleration

### Setting up a Basic Unity Robotics Scene

1. **Create a new 3D project**

2. **Import your robot model**:
   - If you have a URDF file, use the URDF Importer
   - File → Import Robot from URDF
   - Select your robot's URDF file
   - Follow the import wizard

3. **Configure the ROS TCP Connector**:
   - Add the ROS TCP Connector prefab to your scene
   - Configure the IP address and port
   - Set up publishers and subscribers

```csharp
// Example: Unity ROS Connection Setup
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;

public class RobotControl : MonoBehaviour
{
    ROSConnection ros;
    string rosIP = "127.0.0.1";
    int rosPort = 10000;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.Initialize(rosIP, rosPort);
    }

    void SendRobotCommand()
    {
        // Example: Send joint commands
        var jointCommand = new JointCommand()
        {
            joint_names = new string[] { "joint1", "joint2", "joint3" },
            positions = new double[] { 0.1, 0.2, 0.3 }
        };

        ros.Publish("/joint_commands", jointCommand);
    }
}
```

### Unity-High Level Setup for Humanoid Robotics

1. **Physics Configuration**:
   - Go to Edit → Project Settings → Physics
   - Set gravity to -9.81 m/s² (Earth gravity)
   - Adjust default material properties for realistic friction

2. **Rendering Setup**:
   - Go to Edit → Render Pipeline → Create New Settings
   - Select Universal Render Pipeline for performance
   - Or High Definition Render Pipeline for quality
   - Configure lighting settings for realistic rendering

3. **Robot Configuration**:
   - Ensure all robot joints are configured with appropriate physics properties
   - Add colliders to robot parts for collision detection
   - Configure joint limits and motor properties to match real hardware

## Advanced Simulation Configurations

### Multi-Robot Simulation

For simulating multiple humanoid robots:

```xml
<!-- multi_humanoid.world -->
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="multi_humanoid">
    <!-- Physics -->
    <physics name="1ms" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <gravity>0 0 -9.8</gravity>
    </physics>

    <!-- Robot 1 -->
    <include>
      <name>humanoid_robot_1</name>
      <uri>model://humanoid_robot</uri>
      <pose>0 0 1 0 0 0</pose>
    </include>

    <!-- Robot 2 -->
    <include>
      <name>humanoid_robot_2</name>
      <uri>model://humanoid_robot</uri>
      <pose>2 0 1 0 0 0</pose>
    </include>

    <!-- Environment -->
    <include>
      <uri>model://ground_plane</uri>
    </include>
    <include>
      <uri>model://sun</uri>
    </include>
  </world>
</sdf>
```

### Sensor Integration

Add realistic sensors to your simulated humanoid robot:

```xml
<!-- Add to your robot's URDF/Xacro -->
<gazebo reference="camera_link">
  <sensor name="camera" type="camera">
    <update_rate>30</update_rate>
    <camera name="head_camera">
      <horizontal_fov>1.047</horizontal_fov>
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>10.0</far>
      </clip>
      <noise>
        <type>gaussian</type>
        <mean>0.0</mean>
        <stddev>0.05</stddev>
      </noise>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <frame_name>camera_optical_frame</frame_name>
    </plugin>
  </sensor>
</gazebo>
```

## Performance Optimization

### Gazebo Performance Tips

1. **Reduce Update Rates**: Lower sensor update rates if high-frequency data isn't needed
2. **Simplify Models**: Use simplified collision geometry during simulation
3. **Adjust Physics Parameters**: Use appropriate step sizes and solver iterations
4. **Disable Visualization**: Run headless for faster simulation when visualization isn't needed

```bash
# Run Gazebo headless
gz sim -s -r your_world.sdf
```

### Unity Performance Tips

1. **Level of Detail (LOD)**: Implement LOD for complex robot models
2. **Occlusion Culling**: Enable for large environments
3. **Optimize Materials**: Use efficient shaders for real-time performance
4. **Fixed Timestep**: Configure physics timestep appropriately

```csharp
// Physics configuration in Unity
Time.fixedDeltaTime = 0.01f;  // 100 Hz physics update
Physics.defaultSolverIterations = 8;
Physics.defaultSolverVelocityIterations = 1;
```

## Troubleshooting Common Issues

### Gazebo Issues

1. **Black Screen/No Rendering**:
   - Check graphics drivers and OpenGL support
   - Try running with software rendering: `export LIBGL_ALWAYS_SOFTWARE=1`

2. **Physics Instability**:
   - Reduce time step size
   - Increase solver iterations
   - Check mass and inertia properties

3. **High CPU Usage**:
   - Reduce update rates
   - Simplify collision geometry
   - Close unnecessary GUI components

### Unity Issues

1. **Model Import Problems**:
   - Check that URDF files are properly formatted
   - Verify all referenced mesh files exist
   - Check for invalid characters in joint names

2. **Connection Issues**:
   - Verify IP addresses and ports match between Unity and ROS
   - Check firewall settings
   - Ensure ROS bridge is running

3. **Performance Issues**:
   - Reduce polygon count of meshes
   - Optimize materials and shaders
   - Use appropriate texture resolutions

## Best Practices

1. **Start Simple**: Begin with basic models and gradually add complexity
2. **Validate Against Reality**: Regularly compare simulation behavior with real hardware
3. **Document Parameters**: Keep track of all simulation parameters for reproducibility
4. **Modular Design**: Structure your simulation for easy component swapping
5. **Version Control**: Use version control for simulation assets and configurations
6. **Validation Testing**: Create test scenarios to validate simulation accuracy

## Summary

This guide covered the setup of both Gazebo and Unity simulation environments for humanoid robotics applications. Proper simulation setup is crucial for effective sim-to-real transfer and robotics development. The key aspects include:

- Installing and configuring Gazebo for physics simulation
- Setting up Unity for high-fidelity rendering
- Integrating sensors and actuators in simulation
- Optimizing performance for real-time applications
- Troubleshooting common issues

With these simulation environments properly set up, you can develop and test humanoid robotics applications before deploying to real hardware.