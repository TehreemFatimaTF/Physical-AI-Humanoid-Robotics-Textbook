---
sidebar_label: 'Chapter 4: URDF Robot Modeling'
title: 'Chapter 4: URDF Robot Modeling'
description: 'Understanding URDF (Unified Robot Description Format) for humanoid robot modeling in ROS 2'
slug: '/module-1-ros2/chapter-4-urdf-modeling'
difficulty: 'intermediate'
requiredHardware: ['ros_system']
recommendedHardware: ['computer']
---

# Chapter 4: URDF Robot Modeling

URDF (Unified Robot Description Format) is an XML format used in ROS for representing a robot model. It's crucial for humanoid robots as it defines the robot's physical structure, which is essential for simulation, control, and visualization. In this chapter, we'll explore how to create and work with URDF models for humanoid robots.

## What is URDF?

URDF is an XML format that describes a robot's physical properties, including:

- **Links**: Rigid parts of the robot (e.g., torso, limbs, head)
- **Joints**: Connections between links (e.g., hinges, prismatic joints)
- **Visual properties**: How the robot appears in simulation and visualization
- **Collision properties**: How the robot interacts with the environment in simulation
- **Inertial properties**: Mass, center of mass, and inertia for physics simulation

## Basic URDF Structure

A basic URDF file has the following structure:

```xml
<?xml version="1.0"?>
<robot name="my_robot">
  <!-- Links -->
  <link name="base_link">
    <!-- Link properties -->
  </link>

  <!-- Joints -->
  <joint name="joint_name" type="joint_type">
    <!-- Joint properties -->
  </joint>
</robot>
```

## Links: The Building Blocks

Links represent rigid parts of the robot. Each link can have:

- **Visual**: How the link appears in simulation
- **Collision**: How the link interacts with the environment
- **Inertial**: Physical properties for simulation

### Example Link Definition

```xml
<link name="torso">
  <!-- Visual properties -->
  <visual>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <cylinder length="0.5" radius="0.1"/>
    </geometry>
    <material name="blue">
      <color rgba="0 0 1 1"/>
    </material>
  </visual>

  <!-- Collision properties -->
  <collision>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <cylinder length="0.5" radius="0.1"/>
    </geometry>
  </collision>

  <!-- Inertial properties -->
  <inertial>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <mass value="5.0"/>
    <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
  </inertial>
</link>
```

## Joints: Connecting the Links

Joints define how links connect and move relative to each other. Common joint types include:

- **revolute**: Rotational joint with limited range
- **continuous**: Rotational joint without limits
- **prismatic**: Linear sliding joint
- **fixed**: No movement between links
- **floating**: 6-DOF movement
- **planar**: Movement in a plane

### Example Joint Definition

```xml
<joint name="hip_joint" type="revolute">
  <parent link="torso"/>
  <child link="thigh"/>
  <origin xyz="0 0 -0.25" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
</joint>
```

## Complete Humanoid Robot URDF Example

Here's a simplified example of a humanoid robot URDF:

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid">
  <!-- Base link -->
  <link name="base_link">
    <inertial>
      <mass value="10.0"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="1.0" ixy="0" ixz="0" iyy="1.0" iyz="0" izz="1.0"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.5"/>
      <geometry>
        <box size="0.3 0.3 1.0"/>
      </geometry>
      <material name="grey">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0.5"/>
      <geometry>
        <box size="0.3 0.3 1.0"/>
      </geometry>
    </collision>
  </link>

  <!-- Head -->
  <link name="head">
    <inertial>
      <mass value="2.0"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0"/>
      <geometry>
        <sphere radius="0.15"/>
      </geometry>
      <material name="white">
        <color rgba="1 1 1 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0"/>
      <geometry>
        <sphere radius="0.15"/>
      </geometry>
    </collision>
  </link>

  <!-- Neck joint -->
  <joint name="neck_joint" type="revolute">
    <parent link="base_link"/>
    <child link="head"/>
    <origin xyz="0 0 1.0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="10" velocity="1"/>
  </joint>

  <!-- Left arm (simplified) -->
  <link name="left_upper_arm">
    <inertial>
      <mass value="1.5"/>
      <origin xyz="0 0 -0.1"/>
      <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.01"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.1"/>
      <geometry>
        <cylinder length="0.2" radius="0.05"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 -0.1"/>
      <geometry>
        <cylinder length="0.2" radius="0.05"/>
      </geometry>
    </collision>
  </link>

  <!-- Shoulder joint -->
  <joint name="left_shoulder_joint" type="revolute">
    <parent link="base_link"/>
    <child link="left_upper_arm"/>
    <origin xyz="0.15 0 0.5"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.57" upper="1.57" effort="20" velocity="1"/>
  </joint>
</robot>
```

## URDF Tools and Visualization

### Checking URDF Files

You can validate your URDF files using the `check_urdf` command:

```bash
check_urdf /path/to/your/robot.urdf
```

### Visualizing URDF

Use RViz to visualize your robot model:

```bash
# Launch robot state publisher
ros2 run robot_state_publisher robot_state_publisher --ros-args -p robot_description:='$(cat your_robot.urdf)'

# Then launch RViz
rviz2
```

### Joint State Publisher GUI

For testing joint movements:

```bash
ros2 run joint_state_publisher_gui joint_state_publisher_gui
```

## Xacro: URDF Macros

Xacro is an XML macro language that makes URDF files more readable and maintainable by allowing:

- Variables
- Math expressions
- Macros
- File inclusion

### Xacro Example

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid_with_xacro">

  <!-- Properties -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="torso_length" value="0.5" />
  <xacro:property name="torso_radius" value="0.1" />

  <!-- Macro for leg -->
  <xacro:macro name="leg" params="side">
    <link name="${side}_thigh">
      <visual>
        <geometry>
          <cylinder length="0.4" radius="0.05"/>
        </geometry>
        <material name="grey">
          <color rgba="0.5 0.5 0.5 1"/>
        </material>
      </visual>
    </link>
  </xacro:macro>

  <!-- Use the macro -->
  <xacro:leg side="left"/>
  <xacro:leg side="right"/>
</robot>
```

## URDF in Simulation

URDF models are essential for simulation in Gazebo and other physics engines. They define:

- Collision shapes for physics simulation
- Inertial properties for realistic movement
- Visual models for rendering
- Joint limits and properties for realistic constraints

## Best Practices for URDF

1. **Start Simple**: Begin with a basic model and add complexity gradually
2. **Use Consistent Units**: Stick to meters for length, kilograms for mass
3. **Realistic Inertial Properties**: Use proper mass distribution for stable simulation
4. **Appropriate Collision Models**: Use simplified shapes for collision to improve performance
5. **Joint Limits**: Always specify realistic joint limits
6. **Use Xacro**: For complex robots, use Xacro to manage complexity
7. **Validate Regularly**: Check your URDF frequently during development

## Integration with ROS 2

URDF integrates with ROS 2 through:

- **Robot State Publisher**: Publishes joint states to TF tree
- **Joint State Publisher**: Provides joint state messages for visualization
- **Controllers**: Use URDF joint names for control interfaces
- **Perception**: URDF provides geometric information for perception algorithms

## Summary

In this chapter, you learned:
- What URDF is and why it's important for humanoid robots
- How to define links and joints in URDF
- The structure of a complete humanoid robot URDF
- How to visualize and validate URDF models
- The benefits of using Xacro for complex models
- Best practices for creating effective URDF models
- How URDF integrates with ROS 2 systems

---
**Continue to [Chapter 5: Launch Files and Package Management](/docs/module-1-ros2/chapter-5-launch-files)**