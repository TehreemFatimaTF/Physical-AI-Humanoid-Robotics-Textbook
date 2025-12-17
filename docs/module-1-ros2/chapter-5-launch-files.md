---
sidebar_label: 'Chapter 5: Launch Files and Package Management'
title: 'Chapter 5: Launch Files and Package Management'
description: 'Understanding ROS 2 launch files and package management for humanoid robotics applications'
slug: '/module-1-ros2/chapter-5-launch-files'
difficulty: 'intermediate'
requiredHardware: ['ros_system']
recommendedHardware: ['computer']
---

# Chapter 5: Launch Files and Package Management

In this chapter, we'll explore ROS 2 launch files and package management, which are crucial for organizing and running complex humanoid robotics applications. Launch files allow you to start multiple nodes with specific configurations, while proper package management ensures your code is organized, reusable, and maintainable.

## Launch Files: Coordinating Complex Systems

Launch files are Python scripts that define how to launch multiple nodes with specific configurations. They're essential for humanoid robots, which typically require many coordinated nodes for perception, control, planning, and communication.

### Basic Launch File Structure

A simple launch file looks like this:

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='package_name',
            executable='executable_name',
            name='node_name',
            parameters=[
                {'param_name': 'param_value'}
            ],
            remappings=[
                ('original_topic', 'new_topic')
            ]
        )
    ])
```

### Advanced Launch File Features

Launch files support many advanced features:

```python
from launch import LaunchDescription, LaunchContext
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    robot_name = LaunchConfiguration('robot_name')

    # Include another launch file
    other_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('other_package'),
            '/launch/other_launch.py'
        ])
    )

    # Robot controller node
    controller_node = Node(
        package='robot_control',
        executable='controller_node',
        name='controller_node',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'robot_name': robot_name},
            os.path.join(get_package_share_directory('robot_control'), 'config', 'params.yaml')
        ],
        remappings=[
            ('/joint_states', '/robot/joint_states'),
            ('/cmd_vel', '/robot/cmd_vel')
        ],
        respawn=True,  # Restart if the node dies
        respawn_delay=2.0  # Wait 2 seconds before restarting
    )

    # Launch with a delay
    delayed_node = TimerAction(
        period=5.0,  # Wait 5 seconds
        actions=[Node(
            package='diagnostic_aggregator',
            executable='aggregator_node',
            name='diagnostic_aggregator'
        )]
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time'
        ),
        DeclareLaunchArgument(
            'robot_name',
            default_value='humanoid_robot',
            description='Name of the robot'
        ),
        controller_node,
        delayed_node,
        other_launch
    ])
```

## Package Structure for Humanoid Robots

A well-organized ROS 2 package for humanoid robotics typically follows this structure:

```
robot_name_bringup/
├── launch/
│   ├── robot.launch.py
│   ├── simulation.launch.py
│   └── hardware.launch.py
├── config/
│   ├── controllers.yaml
│   ├── robot_params.yaml
│   └── sensors.yaml
├── rviz/
│   └── robot_view.rviz
├── urdf/
│   ├── robot.urdf.xacro
│   └── materials.xacro
├── src/
│   └── (source files)
├── include/
│   └── (header files)
├── CMakeLists.txt
└── package.xml
```

### Package.xml: Package Manifest

The `package.xml` file describes your package:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>humanoid_bringup</name>
  <version>0.0.1</version>
  <description>Launch files and configurations for the humanoid robot</description>
  <maintainer email="maintainer@todo.todo">maintainer</maintainer>
  <license>TODO: License declaration</license>

  <exec_depend>launch</exec_depend>
  <exec_depend>launch_ros</exec_depend>
  <exec_depend>robot_state_publisher</exec_depend>
  <exec_depend>joint_state_publisher</exec_depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

### CMakeLists.txt: Build Configuration

For Python packages:

```cmake
cmake_minimum_required(VERSION 3.8)
project(humanoid_bringup)

find_package(ament_cmake REQUIRED)

# Install launch files
install(DIRECTORY
  launch
  config
  rviz
  urdf
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
```

## Practical Launch File Examples

### Complete Humanoid Robot Launch

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
import os

def generate_launch_description():
    # Launch configuration variables
    use_sim_time = LaunchConfiguration('use_sim_time')
    robot_description_path = os.path.join(
        get_package_share_directory('humanoid_description'),
        'urdf',
        'humanoid.urdf.xacro'
    )

    # Declare launch arguments
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time if true'
    )

    # Robot state publisher
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'robot_description': open(robot_description_path).read()}
        ]
    )

    # Joint state publisher (for simulation)
    joint_state_publisher_node = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # Joint state publisher GUI (for manual testing)
    joint_state_publisher_gui_node = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(LaunchConfiguration('gui'))
    )

    # Declare gui argument
    declare_gui_cmd = DeclareLaunchArgument(
        'gui',
        default_value='false',
        description='Enable joint_state_publisher_gui'
    )

    # Controller manager
    controller_manager_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[
            robot_description_path,
            os.path.join(
                get_package_share_directory('humanoid_control'),
                'config',
                'controllers.yaml'
            )
        ],
        remappings=[
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static')
        ]
    )

    # Launch the actual nodes
    return LaunchDescription([
        declare_use_sim_time_cmd,
        declare_gui_cmd,
        robot_state_publisher_node,
        joint_state_publisher_node,
        joint_state_publisher_gui_node,
        controller_manager_node,
    ])
```

### Simulation Launch with Gazebo

```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch configuration
    world = LaunchConfiguration('world')

    # Declare launch arguments
    declare_world_cmd = DeclareLaunchArgument(
        'world',
        default_value='empty.sdf',
        description='Choose one of the world files from `/gazebo_ros_pkgs/gazebo_ros/worlds`'
    )

    # Launch Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={
            'world': world
        }.items()
    )

    # Robot spawn node
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'humanoid_robot'
        ],
        output='screen'
    )

    return LaunchDescription([
        declare_world_cmd,
        gazebo,
        spawn_entity
    ])
```

## Parameter Management

Launch files can load parameters from YAML files:

```yaml
# config/controllers.yaml
controller_manager:
  ros__parameters:
    update_rate: 100  # Hz
    use_sim_time: true

joint_state_broadcaster:
  type: joint_state_broadcaster/JointStateBroadcaster

position_trajectory_controller:
  type: position_controllers/JointTrajectoryController

  ros__parameters:
    joints:
      - hip_joint
      - knee_joint
      - ankle_joint

    command_interfaces:
      - position

    state_interfaces:
      - position
      - velocity
```

Load parameters in launch file:

```python
Node(
    package='controller_manager',
    executable='ros2_control_node',
    parameters=[
        robot_description_path,
        os.path.join(get_package_share_directory('robot_control'), 'config', 'controllers.yaml')
    ]
)
```

## Launch File Best Practices

1. **Modular Design**: Create separate launch files for different configurations (simulation, hardware, testing)
2. **Parameter Flexibility**: Use launch arguments to make launch files configurable
3. **Error Handling**: Use event handlers for robust node management
4. **Documentation**: Comment your launch files to explain the purpose of each node
5. **Resource Management**: Use conditions to start nodes only when needed
6. **Consistent Naming**: Use consistent naming conventions for nodes and parameters

## Debugging Launch Files

Common debugging techniques:

1. **Verbose Output**: Run with `--debug` flag
2. **Launch Arguments**: Use `--show-args` to see available arguments
3. **Process Monitoring**: Use `--wait` to pause before starting nodes
4. **Logging**: Check log files in `~/.ros/log/`

```bash
# Debug a launch file
ros2 launch my_package my_launch.py --debug

# Show available arguments
ros2 launch my_package my_launch.py --show-args
```

## Summary

In this chapter, you learned:
- How to create and structure launch files for complex robotic systems
- The importance of proper package organization for humanoid robots
- Advanced launch file features like conditions, event handlers, and parameters
- How to integrate launch files with simulation environments
- Best practices for creating maintainable launch configurations
- Debugging techniques for launch files

---
**Continue to [Chapter 6: Bridging Python Agents to ROS Controllers](/docs/module-1-ros2/chapter-6-rclpy-bridge)**