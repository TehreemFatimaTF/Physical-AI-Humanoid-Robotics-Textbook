---
sidebar_label: 'Chapter 3: Nav2 Path Planning'
title: 'Chapter 3: Nav2 Path Planning'
description: 'Understanding Nav2 for navigation and path planning in humanoid robotics'
slug: '/module-3-nvidia-isaac/chapter-3-nav2-path-planning'
difficulty: 'advanced'
requiredHardware: ['computer', 'nvidia_gpu', 'lidar', 'camera']
recommendedHardware: ['jetson_orin', 'realsense_camera']
---

# Chapter 3: Nav2 Path Planning

Navigation2 (Nav2) is the next-generation navigation system for ROS 2, providing advanced path planning, obstacle avoidance, and navigation capabilities for mobile robots, including humanoid robots. Built on the Navigation Stack 2 architecture, Nav2 offers improved performance, flexibility, and robustness compared to its predecessor. This chapter explores Nav2's components, configuration, and application to humanoid robotics navigation.

## Introduction to Nav2

Nav2 is a comprehensive navigation framework that includes:

- **Global Planner**: Creates optimal paths from start to goal
- **Local Planner**: Handles real-time obstacle avoidance and path following
- **Controller**: Executes low-level motion commands
- **Behavior Trees**: Provides flexible navigation behavior management
- **Recovery Behaviors**: Handles navigation failures and recovery
- **Lifecycle Management**: Manages system state and component lifecycle

### Nav2 Architecture

The Nav2 system is organized into several key components:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Navigation    │    │   Behavior      │    │   Recovery      │
│   Server        │    │   Tree          │    │   Server        │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          ▼                      ▼                      ▼
    ┌─────────────┐        ┌─────────────┐        ┌─────────────┐
    │ Global      │        │ Action      │        │ Recovery    │
    │ Planner     │        │ Server      │        │ Behaviors   │
    └─────────────┘        └─────────────┘        └─────────────┘
          │                      │                      │
          ▼                      ▼                      ▼
    ┌─────────────┐        ┌─────────────┐        ┌─────────────┐
    │ Costmap 2D  │        │ Controller  │        │ Costmap 2D  │
    │ (Global)    │        │ Server      │        │ (Local)     │
    └─────────────┘        └─────────────┘        └─────────────┘
```

## Installing Nav2

### Prerequisites

Before installing Nav2, ensure your system meets the requirements:

- **ROS 2**: Humble Hawksbill or later
- **Ubuntu**: 20.04 LTS or 22.04 LTS
- **C++17**: Compiler support
- **Python 3.8+**: For behavior trees and other components

### Installation Methods

#### Method 1: Debian Package Installation

```bash
# Add ROS 2 repository (if not already added)
sudo apt update && sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update
sudo apt install curl gnupg lsb-release
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update

# Install Nav2 packages
sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup ros-humble-nav2-gui ros-humble-nav2-rviz-plugins
```

#### Method 2: Build from Source

```bash
# Create workspace
mkdir -p ~/nav2_ws/src
cd ~/nav2_ws

# Clone Nav2 repositories
git clone -b humble https://github.com/ros-planning/navigation2.git src/navigation2
git clone -b humble https://github.com/ros-planning/navigation_msgs.git src/navigation_msgs

# Install dependencies
rosdep install --from-paths src --ignore-src -r -y

# Build Nav2
colcon build --symlink-install --packages-select \
  nav2_common \
  nav2_util \
  nav2_msgs \
  nav2_map_server \
  nav2_amcl \
  nav2_dwb_controller \
  nav2_smoother \
  nav2_planner \
  nav2_navfn_planner \
  nav2_behavior_tree \
  nav2_bt_navigator \
  nav2_recovery \
  nav2_core \
  nav2_controller \
  nav2_lifecycle_manager \
  nav2_world_model \
  nav2_rviz_plugins \
  nav2_bt_navigator \
  nav2_dwb_controller \
  nav2_smac_planner \
  nav2_spline_smoother
```

## Nav2 Configuration for Humanoid Robots

### Basic Configuration Files

Create a configuration directory for your humanoid robot:

```bash
mkdir -p ~/nav2_ws/src/your_robot_bringup/config/nav2
```

#### Main Nav2 Configuration (nav2_params.yaml)

```yaml
amcl:
  ros__parameters:
    use_sim_time: True
    alpha1: 0.2
    alpha2: 0.2
    alpha3: 0.2
    alpha4: 0.2
    alpha5: 0.2
    base_frame_id: "base_footprint"
    beam_skip_distance: 0.5
    beam_skip_error_threshold: 0.9
    beam_skip_threshold: 0.3
    do_beamskip: false
    global_frame_id: "map"
    lambda_short: 0.1
    laser_likelihood_max_dist: 2.0
    laser_max_range: 100.0
    laser_min_range: -1.0
    laser_model_type: "likelihood_field"
    max_beams: 60
    max_particles: 2000
    min_particles: 500
    odom_frame_id: "odom"
    pf_err: 0.05
    pf_z: 0.99
    recovery_alpha_fast: 0.0
    recovery_alpha_slow: 0.0
    resample_interval: 1
    robot_model_type: "nav2_amcl::DifferentialMotionModel"
    save_pose_rate: 0.5
    sigma_hit: 0.2
    tf_broadcast: true
    transform_tolerance: 1.0
    update_min_a: 0.2
    update_min_d: 0.25
    z_hit: 0.5
    z_max: 0.05
    z_rand: 0.5
    z_short: 0.05
    scan_topic: scan

amcl_map_client:
  ros__parameters:
    use_sim_time: True

amcl_rclcpp_node:
  ros__parameters:
    use_sim_time: True

bt_navigator:
  ros__parameters:
    use_sim_time: True
    global_frame: map
    robot_base_frame: base_link
    odom_topic: /odom
    bt_loop_duration: 10
    default_server_timeout: 20
    enable_groot_monitoring: True
    groot_zmq_publisher_port: 1666
    groot_zmq_server_port: 1667
    # Specify the path to the Behavior Tree XML file
    default_nav_to_pose_bt_xml: /path/to/your/nav_to_pose_bt.xml
    default_nav_through_poses_bt_xml: /path/to/your/nav_through_poses_bt.xml

bt_navigator_rclcpp_node:
  ros__parameters:
    use_sim_time: True

controller_server:
  ros__parameters:
    use_sim_time: True
    controller_frequency: 20.0
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.5
    min_theta_velocity_threshold: 0.001
    progress_checker_plugin: "progress_checker"
    goal_checker_plugin: "goal_checker"
    controller_plugins: ["FollowPath"]

    # Progress checker parameters
    progress_checker:
      plugin: "nav2_controller::SimpleProgressChecker"
      required_movement_radius: 0.5
      movement_time_allowance: 10.0

    # Goal checker parameters
    goal_checker:
      plugin: "nav2_controller::SimpleGoalChecker"
      xy_goal_tolerance: 0.25
      yaw_goal_tolerance: 0.25
      stateful: True

    # DWB Controller parameters
    FollowPath:
      plugin: "nav2_mppi_controller::MPPIController"
      time_steps: 50
      model_dt: 0.05
      batch_size: 1000
      vx_std: 0.2
      vy_std: 0.1
      wz_std: 0.3
      vx_max: 0.5
      vx_min: -0.15
      vy_max: 0.5
      wz_max: 1.0
      xy_goal_tolerance: 0.1
      yaw_goal_tolerance: 0.15
      simulation_time: 1.3
      control_horizon: 10
      trajectory_generator_plugin: "nav2_mppi_controller::OmnidirectionalModel"
      critic_names: ["BaseObstacleCritic", "GoalCritic", "PathAlignCritic", "PathFollowCritic", "PathOrientCritic", "PreferForwardCritic"]
      BaseObstacleCritic:
        enabled: true
        max_scaling_factor: 3.0
        scaling_speed: 0.5
        obstacle_cost_mult: 3.0
        inflation_cost_mult: 2.0
      GoalCritic:
        enabled: true
        goal_dist_gain: 24.0
      PathAlignCritic:
        enabled: true
        cost_power: 2
        forward_point_distance: 0.1
        path_align_cost_gain: 16.0
        stop_on_failure: false
      PathFollowCritic:
        enabled: true
        cost_power: 2
        forward_point_distance: 0.1
        path_follow_cost_gain: 12.0
        target_speed: 0.3
        velocity_deadband: 0.05
      PathOrientCritic:
        enabled: true
        cost_power: 2
        forward_point_distance: 0.1
        path_orient_cost_gain: 12.0
        stop_on_failure: false
      PreferForwardCritic:
        enabled: true
        cost_power: 2
        prefer_forward_cost_gain: 8.0

local_costmap:
  local_costmap:
    ros__parameters:
      update_frequency: 5.0
      publish_frequency: 2.0
      global_frame: odom
      robot_base_frame: base_link
      use_sim_time: True
      rolling_window: true
      width: 3
      height: 3
      resolution: 0.05
      robot_radius: 0.3
      plugins: ["voxel_layer", "inflation_layer"]
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
      voxel_layer:
        plugin: "nav2_costmap_2d::VoxelLayer"
        enabled: True
        publish_voxel_map: True
        origin_z: 0.0
        z_resolution: 0.05
        z_voxels: 16
        max_obstacle_height: 2.0
        mark_threshold: 0
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0
      always_send_full_costmap: True
  local_costmap_client:
    ros__parameters:
      use_sim_time: True
  local_costmap_rclcpp_node:
    ros__parameters:
      use_sim_time: True

global_costmap:
  global_costmap:
    ros__parameters:
      update_frequency: 1.0
      publish_frequency: 1.0
      global_frame: map
      robot_base_frame: base_link
      use_sim_time: True
      robot_radius: 0.3
      resolution: 0.05
      track_unknown_space: true
      plugins: ["static_layer", "obstacle_layer", "inflation_layer"]
      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        enabled: True
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0
      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
        map_subscribe_transient_local: True
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
      always_send_full_costmap: True
  global_costmap_client:
    ros__parameters:
      use_sim_time: True
  global_costmap_rclcpp_node:
    ros__parameters:
      use_sim_time: True

map_server:
  ros__parameters:
    use_sim_time: True
    yaml_filename: "turtlebot3_world.yaml"

map_saver:
  ros__parameters:
    use_sim_time: True
    save_map_timeout: 5.0
    free_thresh_default: 0.25
    occupied_thresh_default: 0.65

planner_server:
  ros__parameters:
    expected_planner_frequency: 20.0
    use_sim_time: True
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_navfn_planner::NavfnPlanner"
      tolerance: 0.5
      use_astar: false
      allow_unknown: true

smoother_server:
  ros__parameters:
    use_sim_time: True
    smoother_plugins: ["simple_smoother"]
    simple_smoother:
      plugin: "nav2_smoother::SimpleSmoother"
      tolerance: 0.01
      max_its: 1000
      do_refinement: True

behavior_server:
  ros__parameters:
    costmap_topic: local_costmap/costmap_raw
    footprint_topic: local_costmap/published_footprint
    cycle_frequency: 10.0
    behavior_plugins: ["spin", "backup", "drive_on_heading", "wait"]
    spin:
      plugin: "nav2_behaviors::Spin"
      spin_dist: 1.57
    backup:
      plugin: "nav2_behaviors::BackUp"
      backup_dist: 0.15
      backup_speed: 0.025
    drive_on_heading:
      plugin: "nav2_behaviors::DriveOnHeading"
      drive_on_heading_dist: 1.0
      drive_on_heading_angle_tolerance: 0.785
    wait:
      plugin: "nav2_behaviors::Wait"
      wait_duration: 1.0

waypoint_follower:
  ros__parameters:
    loop_rate: 20
    stop_on_failure: false
    waypoint_task_executor_plugin: "wait_at_waypoint"
    wait_at_waypoint:
      plugin: "nav2_waypoint_follower::WaitAtWaypoint"
      enabled: true
      waypoint_pause_duration: 200
```

## Nav2 Behavior Trees

Behavior trees provide a flexible way to define navigation behaviors:

### Nav2 Behavior Tree Example

```xml
<!-- nav_to_pose_bt.xml -->
<root main_tree_to_execute="MainTree">
  <BehaviorTree ID="MainTree">
    <Sequence name="NavigateToPose">
      <GoalReached goal_reached_topic="goal_reached"/>
      <ComputePathToPose goal="{goal}" path="{path}" planner_id="GridBased"/>
      <SmoothPath path="{path}" smooth_path="{smoothed_path}" smoother_id="simple_smoother"/>
      <FollowPath path="{smoothed_path}" controller_id="FollowPath"/>
    </Sequence>
  </BehaviorTree>
</root>
```

### Custom Behavior Tree for Humanoid Navigation

```xml
<!-- humanoid_nav_to_pose_bt.xml -->
<root main_tree_to_execute="MainTree">
  <BehaviorTree ID="MainTree">
    <ReactiveSequence name="NavigateToPose">
      <!-- Check if goal is valid -->
      <GoalUpdated current_goal="{goal}" new_goal="{new_goal}"/>

      <!-- Compute path to goal -->
      <ComputePathToPose goal="{goal}" path="{path}" planner_id="GridBased"/>

      <!-- Smooth the path for humanoid gait -->
      <SmoothPath path="{path}" smooth_path="{smoothed_path}" smoother_id="simple_smoother"/>

      <!-- Follow the path with humanoid-specific controller -->
      <FollowPath path="{smoothed_path}" controller_id="HumanoidController"/>

      <!-- Check if goal reached -->
      <GoalReached goal="{goal}" path="{path}"/>
    </ReactiveSequence>

    <!-- Recovery behaviors -->
    <RecoveryNode number_of_retries="6" name="NavigateRecovery">
      <PipelineSequence name="RecoveryFallback">
        <ReactiveFallback name="RecoveryMethod">
          <ClearEntireCostmap name="ClearLocalCostmap" service_name="local_costmap/clear_entirely_local_costmap"/>
          <ClearEntireCostmap name="ClearGlobalCostmap" service_name="global_costmap/clear_entirely_global_costmap"/>
          <BackUp backup_dist="0.15" backup_speed="0.025"/>
          <Spin spin_dist="1.57"/>
          <Wait wait_duration="1.0"/>
        </ReactiveFallback>
      </PipelineSequence>
      <ReactiveSequence name="NavigateWRecovery">
        <GoalUpdated current_goal="{goal}" new_goal="{new_goal}"/>
        <ComputePathToPose goal="{goal}" path="{path}" planner_id="GridBased"/>
        <SmoothPath path="{path}" smooth_path="{smoothed_path}" smoother_id="simple_smoother"/>
        <FollowPath path="{smoothed_path}" controller_id="HumanoidController"/>
        <GoalReached goal="{goal}" path="{path}"/>
      </ReactiveSequence>
    </RecoveryNode>
  </BehaviorTree>
</root>
```

## Nav2 Planners for Humanoid Robots

### Custom Humanoid Path Planner

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav2_msgs.action import ComputePathToPose
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import Pose, Point
from builtin_interfaces.msg import Duration
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import numpy as np
from scipy.spatial.distance import euclidean
import math

class HumanoidPathPlanner(Node):
    def __init__(self):
        super().__init__('humanoid_path_planner')

        # Create action server for path planning
        self._action_server = rclpy.action.ActionServer(
            self,
            ComputePathToPose,
            'compute_path_to_pose',
            self.execute_callback
        )

        # TF buffer for coordinate transformations
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Costmap subscription
        self.costmap_sub = self.create_subscription(
            OccupancyGrid,
            '/global_costmap/costmap',
            self.costmap_callback,
            1
        )

        # Internal state
        self.costmap = None
        self.costmap_resolution = 0.05
        self.costmap_origin = [0, 0]

        self.get_logger().info('Humanoid Path Planner initialized')

    def costmap_callback(self, msg):
        """Update internal costmap representation"""
        self.costmap = np.array(msg.data).reshape(msg.info.height, msg.info.width)
        self.costmap_resolution = msg.info.resolution
        self.costmap_origin = [msg.info.origin.position.x, msg.info.origin.position.y]

    def execute_callback(self, goal_handle):
        """Execute path planning request"""
        self.get_logger().info('Received path planning request')

        # Get start and goal positions
        try:
            # Transform goal to map frame
            goal_in_map = self.transform_pose_to_frame(
                goal_handle.request.goal.pose, 'map')
        except TransformException:
            self.get_logger().error('Could not transform goal to map frame')
            result = ComputePathToPose.Result()
            result.path = Path()
            goal_handle.succeed()
            return result

        # Get current robot position
        try:
            robot_pose = self.get_robot_pose('map')
        except TransformException:
            self.get_logger().error('Could not get robot pose')
            result = ComputePathToPose.Result()
            result.path = Path()
            goal_handle.succeed()
            return result

        # Plan path using custom humanoid-aware algorithm
        path = self.plan_humanoid_path(
            start=robot_pose,
            goal=goal_in_map,
            tolerance=goal_handle.request.planner_params.tolerance
        )

        # Create result
        result = ComputePathToPose.Result()
        result.path = path

        goal_handle.succeed()
        return result

    def transform_pose_to_frame(self, pose, target_frame):
        """Transform pose to target frame"""
        # This would use TF2 to transform the pose
        # Implementation details omitted for brevity
        return pose

    def get_robot_pose(self, frame):
        """Get current robot pose in specified frame"""
        # This would get the robot's current pose using TF2
        # Implementation details omitted for brevity
        return Pose()

    def plan_humanoid_path(self, start, goal, tolerance=0.5):
        """Plan path considering humanoid-specific constraints"""
        # Convert poses to coordinates
        start_pos = [start.position.x, start.position.y]
        goal_pos = [goal.position.x, goal.position.y]

        # Check if goal is reachable
        if self.is_goal_reachable(goal_pos):
            # Use A* or other path planning algorithm
            path_points = self.a_star_pathfinding(start_pos, goal_pos, tolerance)
        else:
            # Find closest reachable point
            adjusted_goal = self.find_closest_reachable(goal_pos)
            path_points = self.a_star_pathfinding(start_pos, adjusted_goal, tolerance)

        # Create Path message
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for point in path_points:
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = 'map'
            pose_stamped.header.stamp = self.get_clock().now().to_msg()
            pose_stamped.pose.position.x = point[0]
            pose_stamped.pose.position.y = point[1]
            pose_stamped.pose.position.z = 0.0  # Humanoid robots operate on ground level
            pose_stamped.pose.orientation.w = 1.0  # Default orientation

            path_msg.poses.append(pose_stamped)

        return path_msg

    def is_goal_reachable(self, goal_pos):
        """Check if goal position is reachable considering humanoid constraints"""
        if self.costmap is None:
            return True

        # Convert world coordinates to costmap indices
        x_idx = int((goal_pos[0] - self.costmap_origin[0]) / self.costmap_resolution)
        y_idx = int((goal_pos[1] - self.costmap_origin[1]) / self.costmap_resolution)

        # Check bounds
        if x_idx < 0 or x_idx >= self.costmap.shape[1] or y_idx < 0 or y_idx >= self.costmap.shape[0]:
            return False

        # Check if cell is free (0 = free, 100 = occupied)
        return self.costmap[y_idx, x_idx] < 50

    def find_closest_reachable(self, goal_pos):
        """Find closest reachable position to goal"""
        # Search in expanding circles around goal
        search_radius = 0.5  # Start with 0.5m radius
        resolution = 0.1     # 10cm resolution

        while search_radius <= 2.0:  # Max search radius 2m
            for angle in np.arange(0, 2 * math.pi, resolution / search_radius):
                x = goal_pos[0] + search_radius * math.cos(angle)
                y = goal_pos[1] + search_radius * math.sin(angle)

                if self.is_goal_reachable([x, y]):
                    return [x, y]

            search_radius += 0.1

        # If no reachable point found, return original goal
        return goal_pos

    def a_star_pathfinding(self, start, goal, tolerance=0.5):
        """A* pathfinding algorithm with humanoid constraints"""
        # This is a simplified implementation
        # In practice, you'd use a more sophisticated algorithm

        # Create path points in straight line (for simplicity)
        # In practice, this would be a proper A* implementation
        path = []
        current = start.copy()
        direction = [goal[0] - start[0], goal[1] - start[1]]
        distance = math.sqrt(direction[0]**2 + direction[1]**2)

        if distance > tolerance:
            # Normalize direction
            direction[0] /= distance
            direction[1] /= distance

            # Generate path points
            step_size = 0.1  # 10cm steps
            num_steps = int(distance / step_size)

            for i in range(num_steps + 1):
                x = start[0] + direction[0] * step_size * i
                y = start[1] + direction[1] * step_size * i
                path.append([x, y])

        # Add goal point
        path.append(goal)

        return path

def main(args=None):
    rclpy.init(args=args)
    planner = HumanoidPathPlanner()

    try:
        rclpy.spin(planner)
    except KeyboardInterrupt:
        pass
    finally:
        planner.destroy_node()
        rclpy.shutdown()
```

## Nav2 Controllers for Humanoid Robots

### Humanoid-Specific Controller

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import Twist, PoseStamped
from nav2_msgs.action import FollowPath
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import numpy as np
from scipy.interpolate import interp1d
import math

class HumanoidController(Node):
    def __init__(self):
        super().__init__('humanoid_controller')

        # Create action server for path following
        self._action_server = rclpy.action.ActionServer(
            self,
            FollowPath,
            'follow_path',
            self.execute_callback
        )

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.controller_status_pub = self.create_publisher(
            PoseStamped, '/controller_status', 10)

        # TF buffer
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Controller parameters
        self.lookahead_distance = 0.5  # meters
        self.max_linear_speed = 0.3    # m/s
        self.max_angular_speed = 0.5   # rad/s
        self.linear_tolerance = 0.1    # meters
        self.angular_tolerance = 0.1   # radians

        # Internal state
        self.current_path = None
        self.current_path_index = 0

        self.get_logger().info('Humanoid Controller initialized')

    def execute_callback(self, goal_handle):
        """Execute path following request"""
        self.get_logger().info('Received path following request')

        # Store the path
        self.current_path = goal_handle.request.path
        self.current_path_index = 0

        # Follow the path
        success = self.follow_path()

        # Set result
        result = FollowPath.Result()
        result.error_code = 0 if success else 1
        result.error_message = "Success" if success else "Failed to follow path"

        goal_handle.succeed()
        return result

    def follow_path(self):
        """Follow the current path"""
        if self.current_path is None or len(self.current_path.poses) == 0:
            return False

        # Path following loop
        while rclpy.ok():
            # Get current robot pose
            try:
                robot_pose = self.get_robot_pose('map')
            except TransformException:
                self.get_logger().warn('Could not get robot pose')
                continue

            # Find next waypoint
            target_pose = self.get_next_waypoint(robot_pose)
            if target_pose is None:
                # Reached end of path
                self.stop_robot()
                return True

            # Calculate control commands
            cmd_vel = self.calculate_control(robot_pose, target_pose)

            # Publish command
            self.cmd_vel_pub.publish(cmd_vel)

            # Check if reached goal
            if self.is_at_goal(robot_pose, self.current_path.poses[-1].pose):
                self.stop_robot()
                return True

            # Sleep to control loop rate
            self.get_clock().sleep_for(rclpy.duration.Duration(seconds=0.05))

    def get_next_waypoint(self, robot_pose):
        """Find the next waypoint to follow"""
        if self.current_path is None:
            return None

        # Find the closest point on the path
        closest_idx = self.find_closest_waypoint(robot_pose)

        # Look ahead to find target point
        target_idx = min(closest_idx + 10, len(self.current_path.poses) - 1)

        return self.current_path.poses[target_idx].pose

    def find_closest_waypoint(self, robot_pose):
        """Find the closest waypoint to robot"""
        if self.current_path is None:
            return 0

        min_dist = float('inf')
        closest_idx = 0

        for i, pose in enumerate(self.current_path.poses):
            dist = math.sqrt(
                (pose.pose.position.x - robot_pose.position.x)**2 +
                (pose.pose.position.y - robot_pose.position.y)**2
            )
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        return closest_idx

    def calculate_control(self, robot_pose, target_pose):
        """Calculate velocity commands to follow path"""
        cmd = Twist()

        # Calculate desired direction
        dx = target_pose.position.x - robot_pose.position.x
        dy = target_pose.position.y - robot_pose.position.y
        distance = math.sqrt(dx**2 + dy**2)

        # Calculate desired angle
        desired_angle = math.atan2(dy, dx)

        # Calculate current angle from orientation
        current_angle = self.quaternion_to_yaw(robot_pose.orientation)

        # Calculate angle error
        angle_error = desired_angle - current_angle
        while angle_error > math.pi:
            angle_error -= 2 * math.pi
        while angle_error < -math.pi:
            angle_error += 2 * math.pi

        # Simple proportional control
        cmd.linear.x = min(self.max_linear_speed, distance * 0.5)
        cmd.angular.z = angle_error * 1.0  # Proportional gain

        # Apply limits
        cmd.linear.x = max(0, min(cmd.linear.x, self.max_linear_speed))
        cmd.angular.z = max(-self.max_angular_speed, min(cmd.angular.z, self.max_angular_speed))

        return cmd

    def quaternion_to_yaw(self, quat):
        """Convert quaternion to yaw angle"""
        siny_cosp = 2 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1 - 2 * (quat.y * quat.y + quat.z * quat.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def is_at_goal(self, robot_pose, goal_pose):
        """Check if robot is at goal position"""
        distance = math.sqrt(
            (goal_pose.position.x - robot_pose.position.x)**2 +
            (goal_pose.position.y - robot_pose.position.y)**2
        )
        return distance < self.linear_tolerance

    def get_robot_pose(self, frame):
        """Get robot pose in specified frame"""
        # This would use TF2 to get robot pose
        # Implementation details omitted for brevity
        return robot_pose

    def stop_robot(self):
        """Stop the robot"""
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    controller = HumanoidController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()
```

## Nav2 Integration with Isaac ROS

### Isaac ROS Nav2 Bridge

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from tf2_ros import TransformBroadcaster
from std_msgs.msg import Float64MultiArray
import numpy as np
from scipy.spatial.transform import Rotation as R

class IsaacNav2Bridge(Node):
    def __init__(self):
        super().__init__('isaac_nav2_bridge')

        # Isaac Sim publishers
        self.joint_cmd_pub = self.create_publisher(
            Float64MultiArray, '/joint_commands', 10)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)

        # Isaac Sim subscribers
        self.joint_state_sub = self.create_subscription(
            Float64MultiArray, '/joint_states', self.joint_state_callback, 10)
        self.imu_sub = self.create_subscription(
            Float64MultiArray, '/imu_data', self.imu_callback, 10)

        # Nav2 interfaces
        self.nav_goal_pub = self.create_publisher(
            PoseStamped, '/goal_pose', 10)
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10)

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Internal state
        self.joint_positions = np.zeros(28)  # Example: 28 DOF humanoid
        self.joint_velocities = np.zeros(28)
        self.imu_orientation = R.from_quat([0, 0, 0, 1])
        self.robot_position = np.array([0.0, 0.0, 0.0])
        self.robot_velocity = np.array([0.0, 0.0, 0.0])

        # Navigation state
        self.is_navigating = False
        self.current_goal = None

        # Timer for state publishing
        self.state_timer = self.create_timer(0.05, self.publish_robot_state)

        self.get_logger().info('Isaac Nav2 Bridge initialized')

    def joint_state_callback(self, msg):
        """Update joint state from Isaac Sim"""
        self.joint_positions = np.array(msg.data[:28])  # First 28 joints
        if len(msg.data) > 28:
            self.joint_velocities = np.array(msg.data[28:56])  # Next 28 velocities

    def imu_callback(self, msg):
        """Update IMU data from Isaac Sim"""
        if len(msg.data) >= 7:  # Orientation (4) + Angular velocity (3) or similar
            orientation = msg.data[:4]
            self.imu_orientation = R.from_quat(orientation)

    def cmd_vel_callback(self, msg):
        """Process velocity commands from Nav2"""
        # Convert Twist command to joint commands for humanoid
        joint_commands = self.twist_to_joint_commands(msg)

        # Publish to Isaac Sim
        cmd_msg = Float64MultiArray()
        cmd_msg.data = joint_commands.tolist()
        self.joint_cmd_pub.publish(cmd_msg)

    def twist_to_joint_commands(self, twist_cmd):
        """Convert Twist command to joint commands for humanoid"""
        # This is a simplified example - real implementation would
        # generate appropriate joint trajectories for walking
        commands = self.joint_positions.copy()

        # Example: Simple mapping for differential drive
        # In reality, this would generate walking gaits
        linear_x = twist_cmd.linear.x
        angular_z = twist_cmd.angular.z

        # Apply to appropriate joints (simplified)
        # This would involve complex inverse kinematics for humanoid walking
        for i in range(len(commands)):
            # Apply walking pattern based on velocity commands
            # This is highly simplified
            commands[i] += linear_x * 0.01 + angular_z * 0.005

        return commands

    def publish_robot_state(self):
        """Publish robot state for Nav2"""
        # Update robot position based on velocity
        dt = 0.05  # Timer period
        self.robot_position += self.robot_velocity * dt

        # Publish odometry
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link'

        # Set position
        odom_msg.pose.pose.position.x = float(self.robot_position[0])
        odom_msg.pose.pose.position.y = float(self.robot_position[1])
        odom_msg.pose.pose.position.z = float(self.robot_position[2])

        # Set orientation
        quat = self.imu_orientation.as_quat()
        odom_msg.pose.pose.orientation.x = float(quat[0])
        odom_msg.pose.pose.orientation.y = float(quat[1])
        odom_msg.pose.pose.orientation.z = float(quat[2])
        odom_msg.pose.pose.orientation.w = float(quat[3])

        # Set velocities
        odom_msg.twist.twist.linear.x = float(self.robot_velocity[0])
        odom_msg.twist.twist.linear.y = float(self.robot_velocity[1])
        odom_msg.twist.twist.angular.z = 0.0  # Simplified

        self.odom_pub.publish(odom_msg)

        # Broadcast transform
        self.broadcast_transform()

    def broadcast_transform(self):
        """Broadcast TF transforms"""
        t = self.get_clock().now().to_msg()

        # Map to odom
        map_to_odom = TransformStamped()
        map_to_odom.header.stamp = t
        map_to_odom.header.frame_id = 'map'
        map_to_odom.child_frame_id = 'odom'
        map_to_odom.transform.translation.x = 0.0
        map_to_odom.transform.translation.y = 0.0
        map_to_odom.transform.translation.z = 0.0
        map_to_odom.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(map_to_odom)

        # Odom to base_link
        odom_to_base = TransformStamped()
        odom_to_base.header.stamp = t
        odom_to_base.header.frame_id = 'odom'
        odom_to_base.child_frame_id = 'base_link'
        odom_to_base.transform.translation.x = float(self.robot_position[0])
        odom_to_base.transform.translation.y = float(self.robot_position[1])
        odom_to_base.transform.translation.z = float(self.robot_position[2])
        quat = self.imu_orientation.as_quat()
        odom_to_base.transform.rotation.x = float(quat[0])
        odom_to_base.transform.rotation.y = float(quat[1])
        odom_to_base.transform.rotation.z = float(quat[2])
        odom_to_base.transform.rotation.w = float(quat[3])
        self.tf_broadcaster.sendTransform(odom_to_base)

def main(args=None):
    rclpy.init(args=args)
    bridge = IsaacNav2Bridge()

    try:
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.destroy_node()
        rclpy.shutdown()
```

## Nav2 Launch Files for Humanoid Robots

### Launch File Configuration

```python
# launch/humanoid_navigation.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    params_file = LaunchConfiguration('params_file')
    bt_xml_file = LaunchConfiguration('bt_xml_file')
    autostart = LaunchConfiguration('autostart')
    map_topic = LaunchConfiguration('map_topic')

    # Declare launch arguments
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='False',
        description='Use simulation (Gazebo) clock if true'
    )

    declare_params_file = DeclareLaunchArgument(
        'params_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('your_robot_bringup'),
            'config',
            'nav2',
            'nav2_params.yaml'
        ]),
        description='Full path to the ROS2 parameters file to use for all launched nodes'
    )

    declare_bt_xml_file = DeclareLaunchArgument(
        'bt_xml_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('your_robot_bringup'),
            'behavior_trees',
            'humanoid_nav_to_pose_bt.xml'
        ]),
        description='Full path to the behavior tree xml file to use'
    )

    declare_autostart = DeclareLaunchArgument(
        'autostart',
        default_value='True',
        description='Automatically startup the nav2 stack'
    )

    declare_map_topic = DeclareLaunchArgument(
        'map_topic',
        default_value='map',
        description='Topic name for the map'
    )

    # Navigation Server
    navigation_server_cmd = Node(
        package='nav2_bt_navigator',
        executable='bt_navigator',
        name='bt_navigator',
        output='screen',
        parameters=[params_file, {'use_sim_time': use_sim_time}],
        remappings=[('cmd_vel', 'cmd_vel_nav')]
    )

    # Planner Server
    planner_server_cmd = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        output='screen',
        parameters=[params_file, {'use_sim_time': use_sim_time}],
        remappings=[('cmd_vel', 'cmd_vel_nav')]
    )

    # Controller Server
    controller_server_cmd = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        output='screen',
        parameters=[params_file, {'use_sim_time': use_sim_time}],
        remappings=[
            ('cmd_vel', 'cmd_vel_nav'),
            ('odom', 'odom')
        ]
    )

    # Local costmap Server
    local_costmap_cmd = Node(
        package='nav2_costmap_2d',
        executable='nav2_costmap_2d',
        name='local_costmap',
        output='screen',
        parameters=[params_file, {'use_sim_time': use_sim_time}],
        remappings=[
            ('cmd_vel', 'cmd_vel_nav'),
            ('odom', 'odom')
        ]
    )

    # Global costmap Server
    global_costmap_cmd = Node(
        package='nav2_costmap_2d',
        executable='nav2_costmap_2d',
        name='global_costmap',
        output='screen',
        parameters=[params_file, {'use_sim_time': use_sim_time}],
        remappings=[
            ('cmd_vel', 'cmd_vel_nav'),
            ('map', map_topic),
            ('odom', 'odom')
        ]
    )

    # Lifecycle Manager
    lifecycle_manager_cmd = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time},
                   {'autostart': autostart},
                   {'node_names': ['controller_server',
                                  'planner_server',
                                  'bt_navigator',
                                  'local_costmap',
                                  'global_costmap']}]
    )

    # Create the launch description and populate
    ld = LaunchDescription()

    # Declare the launch options
    ld.add_action(declare_use_sim_time)
    ld.add_action(declare_params_file)
    ld.add_action(declare_bt_xml_file)
    ld.add_action(declare_autostart)
    ld.add_action(declare_map_topic)

    # Add the actions to launch all of the navigation nodes
    ld.add_action(navigation_server_cmd)
    ld.add_action(planner_server_cmd)
    ld.add_action(controller_server_cmd)
    ld.add_action(local_costmap_cmd)
    ld.add_action(global_costmap_cmd)
    ld.add_action(lifecycle_manager_cmd)

    return ld
```

## Performance Optimization and Tuning

### Nav2 Parameter Tuning for Humanoid Robots

```bash
# Example parameter tuning commands
ros2 param set /planner_server GridBased.tolerance 0.3
ros2 param set /controller_server FollowPath.xy_goal_tolerance 0.2
ros2 param set /controller_server FollowPath.yaw_goal_tolerance 0.3
ros2 param set /local_costmap.local_costmap.rolling_window true
ros2 param set /local_costmap.local_costmap.width 4.0
ros2 param set /local_costmap.local_costmap.height 4.0
```

### Performance Monitoring

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from std_msgs.msg import Float64
import time

class Nav2PerformanceMonitor(Node):
    def __init__(self):
        super().__init__('nav2_performance_monitor')

        # Publishers
        self.diag_pub = self.create_publisher(DiagnosticArray, '/diagnostics', 10)
        self.planning_time_pub = self.create_publisher(Float64, '/planning_time', 10)

        # Timers
        self.diag_timer = self.create_timer(1.0, self.publish_diagnostics)

        # Performance tracking
        self.planning_times = []
        self.navigation_success_rate = 0.0

        self.get_logger().info('Nav2 Performance Monitor initialized')

    def publish_diagnostics(self):
        """Publish diagnostic information"""
        diag_array = DiagnosticArray()
        diag_array.header.stamp = self.get_clock().now().to_msg()

        # Planning performance
        planning_diag = DiagnosticStatus()
        planning_diag.name = 'Nav2 Planning Performance'
        planning_diag.level = DiagnosticStatus.OK
        planning_diag.message = 'Planning within acceptable time limits'

        # Add key-value pairs for metrics
        if self.planning_times:
            avg_time = sum(self.planning_times) / len(self.planning_times)
            planning_diag.values.append(KeyValue(key='avg_planning_time', value=f'{avg_time:.3f}s'))
            planning_diag.values.append(KeyValue(key='min_planning_time', value=f'{min(self.planning_times):.3f}s'))
            planning_diag.values.append(KeyValue(key='max_planning_time', value=f'{max(self.planning_times):.3f}s'))

        diag_array.status.append(planning_diag)

        # Navigation performance
        nav_diag = DiagnosticStatus()
        nav_diag.name = 'Nav2 Navigation Performance'
        nav_diag.level = DiagnosticStatus.OK
        nav_diag.message = f'Success rate: {self.navigation_success_rate:.2f}'

        nav_diag.values.append(KeyValue(key='success_rate', value=f'{self.navigation_success_rate:.2f}'))
        nav_diag.values.append(KeyValue(key='total_attempts', value='0'))  # Would need tracking

        diag_array.status.append(nav_diag)

        self.diag_pub.publish(diag_array)

def main(args=None):
    rclpy.init(args=args)
    monitor = Nav2PerformanceMonitor()

    try:
        rclpy.spin(monitor)
    except KeyboardInterrupt:
        pass
    finally:
        monitor.destroy_node()
        rclpy.shutdown()
```

## Troubleshooting and Best Practices

### Common Issues and Solutions

1. **Localization Problems**:
   - Ensure proper map quality and resolution
   - Check sensor calibration (LiDAR, IMU)
   - Verify TF tree completeness
   - Adjust AMCL parameters for your environment

2. **Path Planning Failures**:
   - Check costmap inflation parameters
   - Verify robot footprint configuration
   - Ensure proper obstacle detection
   - Adjust planner tolerances

3. **Controller Instability**:
   - Tune controller parameters (P, I, D gains)
   - Check odometry quality
   - Verify robot kinematic model
   - Adjust control frequency

### Best Practices for Humanoid Navigation

1. **Safety First**: Always implement safety limits and emergency stops
2. **Gradual Complexity**: Start with simple navigation tasks
3. **Simulation Testing**: Test extensively in simulation before real deployment
4. **Parameter Tuning**: Carefully tune parameters for your specific robot
5. **Monitoring**: Implement comprehensive monitoring and logging
6. **Fallback Plans**: Have robust recovery behaviors
7. **Human Oversight**: Maintain human-in-the-loop for critical operations

## Summary

In this chapter, you learned:
- The architecture and components of the Nav2 navigation system
- How to configure Nav2 for humanoid robotics applications
- The use of behavior trees for flexible navigation behaviors
- Implementation of custom path planners and controllers for humanoid robots
- Integration of Nav2 with Isaac ROS for enhanced capabilities
- Performance optimization and tuning strategies
- Troubleshooting techniques and best practices
- Safety considerations for humanoid navigation

Nav2 provides a robust foundation for navigation in humanoid robotics, enabling complex autonomous behaviors while maintaining safety and reliability.

---
**Continue to [Chapter 4: Hardware Acceleration](/docs/module-3-nvidia-isaac/chapter-4-hardware-acceleration)**