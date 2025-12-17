---
sidebar_label: 'Chapter 3: Sensor Simulation'
title: 'Chapter 3: Sensor Simulation'
description: 'Understanding sensor simulation for humanoid robotics digital twin applications'
slug: '/module-2-digital-twin/chapter-3-sensor-simulation'
difficulty: 'advanced'
requiredHardware: ['computer', 'sensors']
recommendedHardware: ['nvidia_isaac', 'lidar', 'depth_camera']
---

# Chapter 3: Sensor Simulation

Sensor simulation is a critical component of digital twin systems for humanoid robotics. Accurate sensor simulation allows us to test perception algorithms, navigation systems, and control strategies in a safe virtual environment before deploying them on real robots. This chapter explores the simulation of various sensors commonly used in humanoid robots, including cameras, LiDAR, IMUs, force/torque sensors, and depth sensors.

## Overview of Robot Sensors

Humanoid robots typically use multiple sensor types to perceive their environment:

- **Cameras**: RGB, stereo, fisheye for visual perception
- **LiDAR**: 3D point clouds for mapping and navigation
- **Depth Sensors**: RGB-D cameras for 3D perception
- **IMUs**: Inertial measurement units for orientation and acceleration
- **Force/Torque Sensors**: For contact detection and manipulation
- **Joint Encoders**: For proprioception and control
- **GPS**: For outdoor localization (when applicable)

## Camera Simulation in Gazebo and Unity

### Gazebo Camera Simulation

Gazebo provides realistic camera simulation through plugins that generate synthetic images matching real camera specifications:

```xml
<!-- RGB Camera Plugin Configuration -->
<gazebo reference="camera_link">
  <sensor name="camera" type="camera">
    <update_rate>30</update_rate>
    <camera name="head">
      <horizontal_fov>1.3962634</horizontal_fov> <!-- 80 degrees -->
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>100</far>
      </clip>
      <noise>
        <type>gaussian</type>
        <mean>0.0</mean>
        <stddev>0.007</stddev>
      </noise>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <frame_name>camera_optical_frame</frame_name>
      <min_depth>0.1</min_depth>
      <max_depth>100.0</max_depth>
    </plugin>
  </sensor>
</gazebo>
```

### Stereo Camera Simulation

```xml
<!-- Stereo Camera Setup -->
<gazebo reference="left_camera">
  <sensor name="stereo_left" type="camera">
    <camera name="left">
      <horizontal_fov>1.04719755</horizontal_fov> <!-- 60 degrees -->
      <image>
        <width>1280</width>
        <height>720</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>30</far>
      </clip>
    </camera>
    <plugin name="stereo_left_camera" filename="libgazebo_ros_camera.so">
      <frame_name>left_camera_optical_frame</frame_name>
      <topic_name>left/image_raw</topic_name>
      <hack_baseline>0.075</hack_baseline>
    </plugin>
  </sensor>
</gazebo>

<gazebo reference="right_camera">
  <sensor name="stereo_right" type="camera">
    <camera name="right">
      <horizontal_fov>1.04719755</horizontal_fov> <!-- 60 degrees -->
      <image>
        <width>1280</width>
        <height>720</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>30</far>
      </clip>
    </camera>
    <plugin name="stereo_right_camera" filename="libgazebo_ros_camera.so">
      <frame_name>right_camera_optical_frame</frame_name>
      <topic_name>right/image_raw</topic_name>
      <hack_baseline>0.075</hack_baseline>
    </plugin>
  </sensor>
</gazebo>
```

### Unity Camera Simulation

```csharp
// RobotCamera.cs - Unity camera simulation
using UnityEngine;
using System.Collections;

public class RobotCamera : MonoBehaviour
{
    [Header("Camera Specifications")]
    public float horizontalFOV = 80.0f;
    public int resolutionWidth = 640;
    public int resolutionHeight = 480;
    public float nearClip = 0.1f;
    public float farClip = 100.0f;

    [Header("Noise Simulation")]
    public bool simulateNoise = true;
    public float noiseIntensity = 0.01f;

    [Header("Output Settings")]
    public bool saveImages = false;
    public string imageSavePath = "robot_images/";

    private Camera cam;
    private RenderTexture renderTexture;

    void Start()
    {
        SetupCamera();
    }

    void SetupCamera()
    {
        cam = GetComponent<Camera>();

        // Calculate vertical FOV based on aspect ratio
        float aspectRatio = (float)resolutionWidth / resolutionHeight;
        cam.fieldOfView = 2.0f * Mathf.Atan(Mathf.Tan(horizontalFOV * Mathf.Deg2Rad / 2.0f) / aspectRatio) * Mathf.Rad2Deg;

        cam.nearClipPlane = nearClip;
        cam.farClipPlane = farClip;

        // Create render texture for synthetic image generation
        renderTexture = new RenderTexture(resolutionWidth, resolutionHeight, 24);
        cam.targetTexture = renderTexture;
    }

    public Texture2D CaptureImage()
    {
        // Set the camera to render to the render texture
        RenderTexture.active = renderTexture;
        cam.Render();

        // Create a texture to read the render texture
        Texture2D image = new Texture2D(renderTexture.width, renderTexture.height, TextureFormat.RGB24, false);
        image.ReadPixels(new Rect(0, 0, renderTexture.width, renderTexture.height), 0, 0);
        image.Apply();

        // Apply noise simulation if enabled
        if (simulateNoise)
        {
            AddNoiseToImage(image);
        }

        RenderTexture.active = null;

        return image;
    }

    void AddNoiseToImage(Texture2D image)
    {
        Color[] pixels = image.GetPixels();

        for (int i = 0; i < pixels.Length; i++)
        {
            // Add Gaussian noise
            float noiseX = Random.Range(-noiseIntensity, noiseIntensity);
            float noiseY = Random.Range(-noiseIntensity, noiseIntensity);
            float noiseZ = Random.Range(-noiseIntensity, noiseIntensity);

            pixels[i] = new Color(
                Mathf.Clamp01(pixels[i].r + noiseX),
                Mathf.Clamp01(pixels[i].g + noiseY),
                Mathf.Clamp01(pixels[i].b + noiseZ)
            );
        }

        image.SetPixels(pixels);
        image.Apply();
    }

    void Update()
    {
        if (saveImages && Input.GetKeyDown(KeyCode.Space))
        {
            Texture2D image = CaptureImage();
            byte[] bytes = ImageConversion.EncodeToPNG(image);
            string filename = imageSavePath + "camera_" + Time.time + ".png";
            System.IO.File.WriteAllBytes(filename, bytes);
        }
    }
}
```

## LiDAR Simulation

### Gazebo LiDAR Plugin

```xml
<!-- 3D LiDAR Sensor (HDL-64E equivalent) -->
<gazebo reference="lidar_link">
  <sensor name="velodyne" type="ray">
    <pose>0 0 0 0 0 0</pose>
    <visualize>true</visualize>
    <update_rate>10</update_rate>
    <ray>
      <scan>
        <horizontal>
          <samples>800</samples>
          <resolution>1</resolution>
          <min_angle>-3.141592653589793</min_angle> <!-- -π -->
          <max_angle>3.141592653589793</max_angle>   <!-- π -->
        </horizontal>
        <vertical>
          <samples>64</samples>
          <resolution>1</resolution>
          <min_angle>-0.2617993877991494</min_angle> <!-- -15 degrees -->
          <max_angle>0.2617993877991494</max_angle>  <!-- 15 degrees -->
        </vertical>
      </scan>
      <range>
        <min>0.1</min>
        <max>120.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
    <plugin name="velodyne_controller" filename="libgazebo_ros_laser.so">
      <topicName>/laser_scan</topicName>
      <frameName>lidar_link</frameName>
    </plugin>
  </sensor>
</gazebo>
```

### 2D LiDAR Simulation

```xml
<!-- 2D LiDAR (SICK LMS1xx equivalent) -->
<gazebo reference="laser_link">
  <sensor name="laser" type="ray">
    <pose>0 0 0 0 0 0</pose>
    <visualize>false</visualize>
    <update_rate>50</update_rate>
    <ray>
      <scan>
        <horizontal>
          <samples>720</samples>
          <resolution>1</resolution>
          <min_angle>-2.356194490192345</min_angle> <!-- -135 degrees -->
          <max_angle>2.356194490192345</max_angle>  <!-- 135 degrees -->
        </horizontal>
      </scan>
      <range>
        <min>0.1</min>
        <max>25.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
    <plugin name="laser_controller" filename="libgazebo_ros_laser.so">
      <topicName>/scan</topicName>
      <frameName>laser_link</frameName>
    </plugin>
  </sensor>
</gazebo>
```

### LiDAR Point Cloud Processing

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
import numpy as np
from scipy.spatial.transform import Rotation as R

class LIDARProcessor(Node):
    def __init__(self):
        super().__init__('lidar_processor')

        # Subscribers
        self.scan_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # Publishers
        self.pointcloud_publisher = self.create_publisher(
            PointCloud2,
            '/pointcloud',
            10
        )

        self.get_logger().info('LIDAR Processor initialized')

    def scan_callback(self, scan_msg):
        """Convert LaserScan to PointCloud2"""
        # Convert scan to points
        angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(scan_msg.ranges))
        ranges = np.array(scan_msg.ranges)

        # Filter out invalid ranges
        valid_indices = (ranges >= scan_msg.range_min) & (ranges <= scan_msg.range_max)
        valid_angles = angles[valid_indices]
        valid_ranges = ranges[valid_indices]

        # Convert to Cartesian coordinates
        x = valid_ranges * np.cos(valid_angles)
        y = valid_ranges * np.sin(valid_angles)
        z = np.zeros_like(x)

        # Create point cloud
        points = np.column_stack((x, y, z)).astype(np.float32)

        # Create PointCloud2 message
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = scan_msg.header.frame_id

        fields = [
            point_cloud2.PointField(name='x', offset=0, datatype=point_cloud2.PointField.FLOAT32, count=1),
            point_cloud2.PointField(name='y', offset=4, datatype=point_cloud2.PointField.FLOAT32, count=1),
            point_cloud2.PointField(name='z', offset=8, datatype=point_cloud2.PointField.FLOAT32, count=1)
        ]

        pointcloud_msg = point_cloud2.create_cloud(header, fields, points)
        self.pointcloud_publisher.publish(pointcloud_msg)

def main(args=None):
    rclpy.init(args=args)
    lidar_processor = LIDARProcessor()

    try:
        rclpy.spin(lidar_processor)
    except KeyboardInterrupt:
        pass
    finally:
        lidar_processor.destroy_node()
        rclpy.shutdown()
```

## Depth Sensor Simulation

### RGB-D Camera in Gazebo

```xml
<!-- RGB-D Camera Plugin -->
<gazebo reference="rgbd_camera_link">
  <sensor name="rgbd_camera" type="depth">
    <update_rate>30</update_rate>
    <camera name="rgbd">
      <horizontal_fov>1.04719755</horizontal_fov> <!-- 60 degrees -->
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>10</far>
      </clip>
    </camera>
    <plugin name="rgbd_camera_controller" filename="libgazebo_ros_openni_kinect.so">
      <baseline>0.2</baseline>
      <alwaysOn>true</alwaysOn>
      <updateRate>30.0</updateRate>
      <cameraName>rgbd_camera</cameraName>
      <imageTopicName>/rgb/image_raw</imageTopicName>
      <depthImageTopicName>/depth/image_raw</depthImageTopicName>
      <pointCloudTopicName>/depth/points</pointCloudTopicName>
      <cameraInfoTopicName>/rgb/camera_info</cameraInfoTopicName>
      <depthImageCameraInfoTopicName>/depth/camera_info</depthImageCameraInfoTopicName>
      <frameName>rgbd_camera_depth_optical_frame</frameName>
      <pointCloudCutoff>0.1</pointCloudCutoff>
      <pointCloudCutoffMax>10.0</pointCloudCutoffMax>
      <distortion_k1>0.0</distortion_k1>
      <distortion_k2>0.0</distortion_k2>
      <distortion_k3>0.0</distortion_k3>
      <distortion_t1>0.0</distortion_t1>
      <distortion_t2>0.0</distortion_t2>
      <CxPrime>0.0</CxPrime>
      <Cx>320.5</Cx>
      <Cy>240.5</Cy>
      <focalLength>320.0</focalLength>
      <hackBaseline>0.071</hackBaseline>
    </plugin>
  </sensor>
</gazebo>
```

### Depth Processing in Unity

```csharp
// DepthCamera.cs - Unity depth camera simulation
using UnityEngine;
using System.Collections;

public class DepthCamera : MonoBehaviour
{
    [Header("Depth Camera Settings")]
    public float minDepth = 0.1f;
    public float maxDepth = 10.0f;
    public float depthResolution = 0.01f;

    [Header("Noise Simulation")]
    public bool simulateDepthNoise = true;
    public float depthNoiseSigma = 0.01f;

    private Camera depthCam;
    private RenderTexture depthTexture;

    void Start()
    {
        SetupDepthCamera();
    }

    void SetupDepthCamera()
    {
        depthCam = GetComponent<Camera>();
        depthCam.depthTextureMode = DepthTextureMode.Depth;

        // Create render texture for depth data
        depthTexture = new RenderTexture(Screen.width, Screen.height, 24, RenderTextureFormat.RFloat);
        depthCam.targetTexture = depthTexture;
    }

    public float[,] GetDepthData()
    {
        RenderTexture.active = depthTexture;
        depthCam.Render();

        Texture2D depthTex = new Texture2D(depthTexture.width, depthTexture.height, TextureFormat.RFloat, false);
        depthTex.ReadPixels(new Rect(0, 0, depthTexture.width, depthTexture.height), 0, 0);
        depthTex.Apply();

        float[,] depthData = new float[depthTexture.height, depthTexture.width];

        for (int y = 0; y < depthTexture.height; y++)
        {
            for (int x = 0; x < depthTexture.width; x++)
            {
                float rawDepth = depthTex.GetPixel(x, y).r;

                // Convert from normalized depth to real-world depth
                float depth = ConvertRawDepthToReal(rawDepth);

                // Add noise if enabled
                if (simulateDepthNoise)
                {
                    depth += Random.Range(-depthNoiseSigma, depthNoiseSigma);
                }

                // Clamp to valid range
                depth = Mathf.Clamp(depth, minDepth, maxDepth);

                depthData[y, x] = depth;
            }
        }

        RenderTexture.active = null;
        return depthData;
    }

    float ConvertRawDepthToReal(float rawDepth)
    {
        // Convert from normalized [0,1] to real depth based on camera parameters
        float near = depthCam.nearClipPlane;
        float far = depthCam.farClipPlane;

        // For perspective camera
        float linearDepth = (2.0f * near * far) / (far + near - rawDepth * (far - near));
        return linearDepth;
    }
}
```

## IMU Simulation

### Gazebo IMU Plugin

```xml
<!-- IMU Sensor Plugin -->
<gazebo reference="imu_link">
  <sensor name="imu_sensor" type="imu">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <visualize>false</visualize>
    <imu>
      <angular_velocity>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.001</stddev>
            <bias_mean>0.00001</bias_mean>
            <bias_stddev>0.000001</bias_stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.001</stddev>
            <bias_mean>0.00001</bias_mean>
            <bias_stddev>0.000001</bias_stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.001</stddev>
            <bias_mean>0.00001</bias_mean>
            <bias_stddev>0.000001</bias_stddev>
          </noise>
        </z>
      </angular_velocity>
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.017</stddev>
            <bias_mean>0.00017</bias_mean>
            <bias_stddev>0.000017</bias_stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.017</stddev>
            <bias_mean>0.00017</bias_mean>
            <bias_stddev>0.000017</bias_stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.017</stddev>
            <bias_mean>0.00017</bias_mean>
            <bias_stddev>0.000017</bias_stddev>
          </noise>
        </z>
      </linear_acceleration>
    </imu>
    <plugin name="imu_controller" filename="libgazebo_ros_imu.so">
      <topicName>/imu/data</topicName>
      <serviceName>/imu/service</serviceName>
      <gaussianNoise>0.001</gaussianNoise>
      <frameName>imu_link</frameName>
    </plugin>
  </sensor>
</gazebo>
```

### IMU Data Processing

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3
from std_msgs.msg import Header
import numpy as np
from scipy.spatial.transform import Rotation as R

class IMUProcessor(Node):
    def __init__(self):
        super().__init__('imu_processor')

        self.imu_publisher = self.create_publisher(Imu, '/processed_imu', 10)

        # Simulate IMU data generation
        self.timer = self.create_timer(0.01, self.publish_imu_data)  # 100Hz

        # State variables for simulation
        self.orientation = R.from_quat([0, 0, 0, 1])
        self.angular_velocity = np.array([0.0, 0.0, 0.0])
        self.linear_acceleration = np.array([0.0, 0.0, 9.81])  # Gravity
        self.time = 0.0

        self.get_logger().info('IMU Processor initialized')

    def publish_imu_data(self):
        """Simulate and publish IMU data"""
        # Simulate some movement
        self.time += 0.01

        # Update angular velocity (oscillating motion)
        self.angular_velocity[0] = 0.1 * np.sin(2 * np.pi * 0.5 * self.time)
        self.angular_velocity[1] = 0.05 * np.cos(2 * np.pi * 0.3 * self.time)
        self.angular_velocity[2] = 0.02 * np.sin(2 * np.pi * 0.7 * self.time)

        # Update orientation based on angular velocity
        dt = 0.01
        delta_quat = self.angular_velocity_to_quaternion(self.angular_velocity, dt)
        self.orientation = self.orientation * R.from_quat(delta_quat)

        # Update linear acceleration (with gravity and some motion)
        self.linear_acceleration[0] = 0.5 * np.sin(2 * np.pi * 0.2 * self.time)
        self.linear_acceleration[1] = 0.3 * np.cos(2 * np.pi * 0.4 * self.time)
        self.linear_acceleration[2] = 9.81 + 0.2 * np.sin(2 * np.pi * 0.1 * self.time)

        # Add noise to simulate real sensor
        angular_vel_noisy = self.add_noise_to_vector(self.angular_velocity, 0.001)
        linear_acc_noisy = self.add_noise_to_vector(self.linear_acceleration, 0.017)

        # Create and publish IMU message
        imu_msg = Imu()
        imu_msg.header = Header()
        imu_msg.header.stamp = self.get_clock().now().to_msg()
        imu_msg.header.frame_id = 'imu_link'

        # Set orientation (with noise)
        quat = self.orientation.as_quat()
        orientation_noise = np.random.normal(0, 0.001, 4)
        quat_noisy = quat + orientation_noise
        quat_noisy = quat_noisy / np.linalg.norm(quat_noisy)  # Normalize

        imu_msg.orientation.x = quat_noisy[0]
        imu_msg.orientation.y = quat_noisy[1]
        imu_msg.orientation.z = quat_noisy[2]
        imu_msg.orientation.w = quat_noisy[3]

        # Set angular velocity
        imu_msg.angular_velocity.x = angular_vel_noisy[0]
        imu_msg.angular_velocity.y = angular_vel_noisy[1]
        imu_msg.angular_velocity.z = angular_vel_noisy[2]

        # Set linear acceleration
        imu_msg.linear_acceleration.x = linear_acc_noisy[0]
        imu_msg.linear_acceleration.y = linear_acc_noisy[1]
        imu_msg.linear_acceleration.z = linear_acc_noisy[2]

        # Set covariance matrices (information about uncertainty)
        imu_msg.orientation_covariance = [0.01, 0, 0, 0, 0.01, 0, 0, 0, 0.01]
        imu_msg.angular_velocity_covariance = [0.01, 0, 0, 0, 0.01, 0, 0, 0, 0.01]
        imu_msg.linear_acceleration_covariance = [0.01, 0, 0, 0, 0.01, 0, 0, 0, 0.01]

        self.imu_publisher.publish(imu_msg)

    def angular_velocity_to_quaternion(self, omega, dt):
        """Convert angular velocity to quaternion increment"""
        # Convert angular velocity to quaternion using small angle approximation
        angle = np.linalg.norm(omega) * dt
        if angle == 0:
            return np.array([0, 0, 0, 1])

        axis = omega / np.linalg.norm(omega)
        half_angle = angle / 2
        q = np.array([
            axis[0] * np.sin(half_angle),
            axis[1] * np.sin(half_angle),
            axis[2] * np.sin(half_angle),
            np.cos(half_angle)
        ])
        return q

    def add_noise_to_vector(self, vector, std_dev):
        """Add Gaussian noise to a vector"""
        noise = np.random.normal(0, std_dev, vector.shape)
        return vector + noise

def main(args=None):
    rclpy.init(args=args)
    imu_processor = IMUProcessor()

    try:
        rclpy.spin(imu_processor)
    except KeyboardInterrupt:
        pass
    finally:
        imu_processor.destroy_node()
        rclpy.shutdown()
```

## Force/Torque Sensor Simulation

### Gazebo Force/Torque Sensor

```xml
<!-- Force/Torque Sensor Plugin -->
<gazebo>
  <plugin name="ft_sensor_plugin" filename="libgazebo_ros_ft_sensor.so">
    <updateRate>100</updateRate>
    <topicName>/ft_sensor</topicName>
    <jointName>wrist_joint</jointName>
    <frameName>wrist_frame</frameName>
    <bodyName>wrist_link</bodyName>
    <gaussianNoise>0.01</gaussianNoise>
  </plugin>
</gazebo>
```

### Force/Torque Processing

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Header
import numpy as np

class ForceTorqueProcessor(Node):
    def __init__(self):
        super().__init__('force_torque_processor')

        self.ft_publisher = self.create_publisher(WrenchStamped, '/ft_sensor', 10)
        self.timer = self.create_timer(0.01, self.publish_ft_data)  # 100Hz

        self.time = 0.0
        self.get_logger().info('Force/Torque Processor initialized')

    def publish_ft_data(self):
        """Simulate and publish force/torque data"""
        self.time += 0.01

        # Simulate force and torque (with some oscillating patterns)
        force_x = 5.0 * np.sin(2 * np.pi * 0.5 * self.time)  # 0.5Hz oscillation
        force_y = 2.0 * np.cos(2 * np.pi * 0.3 * self.time)
        force_z = 10.0 + 1.0 * np.sin(2 * np.pi * 0.1 * self.time)  # Gravity + small oscillation

        torque_x = 0.5 * np.sin(2 * np.pi * 1.0 * self.time)
        torque_y = 0.3 * np.cos(2 * np.pi * 0.7 * self.time)
        torque_z = 0.1 * np.sin(2 * np.pi * 1.2 * self.time)

        # Add sensor noise
        force_noise = np.random.normal(0, 0.1, 3)
        torque_noise = np.random.normal(0, 0.01, 3)

        wrench_msg = WrenchStamped()
        wrench_msg.header = Header()
        wrench_msg.header.stamp = self.get_clock().now().to_msg()
        wrench_msg.header.frame_id = 'wrist_frame'

        wrench_msg.wrench.force.x = force_x + force_noise[0]
        wrench_msg.wrench.force.y = force_y + force_noise[1]
        wrench_msg.wrench.force.z = force_z + force_noise[2]

        wrench_msg.wrench.torque.x = torque_x + torque_noise[0]
        wrench_msg.wrench.torque.y = torque_y + torque_noise[1]
        wrench_msg.wrench.torque.z = torque_z + torque_noise[2]

        self.ft_publisher.publish(wrench_msg)

def main(args=None):
    rclpy.init(args=args)
    ft_processor = ForceTorqueProcessor()

    try:
        rclpy.spin(ft_processor)
    except KeyboardInterrupt:
        pass
    finally:
        ft_processor.destroy_node()
        rclpy.shutdown()
```

## Sensor Fusion and Integration

### Multi-Sensor Data Processing Node

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, LaserScan, Image, PointCloud2
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import numpy as np
from scipy.spatial.transform import Rotation as R

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion_node')

        # Subscribers for different sensors
        self.imu_sub = self.create_subscription(Imu, '/imu/data', self.imu_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.odom_pub = self.create_publisher(Odometry, '/fused_odom', 10)

        # Internal state
        self.orientation = R.from_quat([0, 0, 0, 1])
        self.position = np.array([0.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])

        # Timer for publishing fused data
        self.timer = self.create_timer(0.02, self.publish_fused_odom)  # 50Hz

        self.get_logger().info('Sensor Fusion Node initialized')

    def imu_callback(self, msg):
        """Process IMU data for orientation and acceleration"""
        # Extract orientation from IMU
        quat = np.array([
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        ])
        self.orientation = R.from_quat(quat)

        # Integrate angular velocity to get orientation change
        # (In a real system, this would be done more carefully)

        # Extract linear acceleration
        linear_acc = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])

        # Remove gravity and integrate to get velocity change
        # This is a simplified approach - in reality, gravity compensation is more complex
        gravity = np.array([0, 0, 9.81])
        rotated_gravity = self.orientation.apply(gravity)
        net_acceleration = linear_acc - rotated_gravity

        dt = 0.01  # Assuming 100Hz IMU
        self.velocity += net_acceleration * dt

    def scan_callback(self, msg):
        """Process laser scan for position estimation (simplified)"""
        # This is a simplified example - in reality, you'd use more sophisticated
        # SLAM algorithms to estimate position from laser scan data
        pass

    def publish_fused_odom(self):
        """Publish fused odometry based on all sensors"""
        # Update position based on velocity
        dt = 0.02  # 50Hz
        self.position += self.velocity * dt

        # Create odometry message
        odom_msg = Odometry()
        odom_msg.header = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link'

        # Set position
        odom_msg.pose.pose.position.x = float(self.position[0])
        odom_msg.pose.pose.position.y = float(self.position[1])
        odom_msg.pose.pose.position.z = float(self.position[2])

        # Set orientation
        quat = self.orientation.as_quat()
        odom_msg.pose.pose.orientation.x = float(quat[0])
        odom_msg.pose.pose.orientation.y = float(quat[1])
        odom_msg.pose.pose.orientation.z = float(quat[2])
        odom_msg.pose.pose.orientation.w = float(quat[3])

        # Set velocities
        odom_msg.twist.twist.linear.x = float(self.velocity[0])
        odom_msg.twist.twist.linear.y = float(self.velocity[1])
        odom_msg.twist.twist.linear.z = float(self.velocity[2])

        # Set covariance (simplified)
        odom_msg.pose.covariance = [0.1] * 36  # Simplified covariance
        odom_msg.twist.covariance = [0.1] * 36  # Simplified covariance

        self.odom_pub.publish(odom_msg)

def main(args=None):
    rclpy.init(args=args)
    fusion_node = SensorFusionNode()

    try:
        rclpy.spin(fusion_node)
    except KeyboardInterrupt:
        pass
    finally:
        fusion_node.destroy_node()
        rclpy.shutdown()
```

## Sensor Calibration and Validation

### Camera Calibration Simulation

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np

class CameraCalibrationSimulator(Node):
    def __init__(self):
        super().__init__('camera_calibration_simulator')

        self.bridge = CvBridge()
        self.image_pub = self.create_publisher(Image, '/calibration_image', 10)
        self.info_pub = self.create_publisher(CameraInfo, '/camera_info', 10)

        # Camera parameters (simulated calibration)
        self.camera_matrix = np.array([
            [615.0, 0.0, 320.0],
            [0.0, 615.0, 240.0],
            [0.0, 0.0, 1.0]
        ])

        self.distortion_coeffs = np.array([0.1, -0.2, 0.0, 0.0, 0.0])  # Simulated distortion

        # Timer to publish calibration images
        self.timer = self.create_timer(2.0, self.publish_calibration_image)

        self.get_logger().info('Camera Calibration Simulator initialized')

    def publish_calibration_image(self):
        """Generate and publish a calibration pattern image"""
        # Create a checkerboard pattern
        img = np.ones((480, 640, 3), dtype=np.uint8) * 255

        # Draw checkerboard pattern
        square_size = 30
        rows, cols = 6, 8

        for i in range(rows):
            for j in range(cols):
                if (i + j) % 2 == 0:
                    pt1 = (j * square_size, i * square_size)
                    pt2 = ((j + 1) * square_size, (i + 1) * square_size)
                    cv2.rectangle(img, pt1, pt2, (0, 0, 0), -1)

        # Add some distortion to simulate real camera
        img_undistorted = cv2.undistort(img, self.camera_matrix, self.distortion_coeffs)

        # Publish the image
        image_msg = self.bridge.cv2_to_imgmsg(img_undistorted, encoding='bgr8')
        image_msg.header.stamp = self.get_clock().now().to_msg()
        image_msg.header.frame_id = 'camera_optical_frame'

        self.image_pub.publish(image_msg)

        # Publish camera info
        info_msg = CameraInfo()
        info_msg.header = image_msg.header
        info_msg.height = 480
        info_msg.width = 640
        info_msg.distortion_model = 'plumb_bob'
        info_msg.d = self.distortion_coeffs.tolist()
        info_msg.k = self.camera_matrix.flatten().tolist()
        info_msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info_msg.p = [615.0, 0.0, 320.0, 0.0, 0.0, 615.0, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0]

        self.info_pub.publish(info_msg)

def main(args=None):
    rclpy.init(args=args)
    calib_sim = CameraCalibrationSimulator()

    try:
        rclpy.spin(calib_sim)
    except KeyboardInterrupt:
        pass
    finally:
        calib_sim.destroy_node()
        rclpy.shutdown()
```

## Best Practices for Sensor Simulation

1. **Realistic Noise Models**: Include appropriate noise models that match real sensors
2. **Temporal Consistency**: Ensure sensors are synchronized and have appropriate update rates
3. **Physical Accuracy**: Model physical properties like field of view, range limits, and resolution
4. **Calibration Integration**: Include calibration parameters in simulation
5. **Performance Considerations**: Balance simulation accuracy with computational performance
6. **Validation**: Regularly validate simulated sensors against real hardware data
7. **Modular Design**: Keep sensor simulation modular for easy replacement or modification

## Troubleshooting Sensor Simulation

Common issues and solutions:

1. **Synchronization Problems**: Use ROS 2 message filters for proper sensor synchronization
2. **Performance Issues**: Reduce simulation complexity or use approximated models
3. **Calibration Discrepancies**: Verify camera intrinsic and extrinsic parameters
4. **Noise Modeling**: Adjust noise parameters to match real sensor characteristics
5. **Coordinate Frame Issues**: Ensure proper TF transforms between sensor frames

## Summary

In this chapter, you learned:
- How to simulate various robot sensors including cameras, LiDAR, IMUs, and force/torque sensors
- Techniques for realistic noise modeling and sensor calibration
- Integration approaches for multi-sensor systems
- Best practices for accurate sensor simulation in digital twin applications
- Troubleshooting techniques for common sensor simulation issues

Accurate sensor simulation is crucial for effective digital twin systems, enabling comprehensive testing of perception and control algorithms before deployment on real humanoid robots.

---
**Continue to [Chapter 4: Sim-to-Real Transfer Techniques](/docs/module-2-digital-twin/chapter-4-sim-to-real)**